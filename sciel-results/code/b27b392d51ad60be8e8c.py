"""
Strategy: Physical Repulsion + Annealing + Pairwise Local Search.
Instead of smooth approximations (soft-min) and L-BFGS, we use a physics-based 
relaxation where circles repel each other and are attracted to the center.
We anneal the repulsion strength to settle into a local minimum, then perform 
targeted pairwise swaps/moves to improve the minimum distance.
"""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 130.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return max(0.0, _DEADLINE - time.time())

def compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    if len(pts) < 2:
        return 0.0
    # Efficient distance calculation
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    min_dist_to_edge = (1.0 - np.linalg.norm(pts, axis=1)).min()
    return min(min_pair, min_dist_to_edge)

def seed_rings(n_points):
    """
    Generate a good initial configuration using concentric rings.
    This provides a structured starting point that is often better than random.
    """
    points = np.zeros((n_points, 2))
    if n_points == 0:
        return points
    
    # Center point
    points[0] = [0.0, 0.0]
    remaining = n_points - 1
    
    ring_radius = 0.1
    ring_spacing = 0.12
    
    idx = 1
    while remaining > 0 and ring_radius < 0.9:
        # Estimate how many points fit on this ring
        # Circumference / (2 * r) is a rough max count
        circumference = 2 * np.pi * ring_radius
        # We want spacing roughly equal to the ring radius for good packing
        # But we don't know r yet. Let's assume r ~ 0.12
        # max_count ~ 2 * pi * ring_radius / (2 * 0.12)
        max_on_ring = max(1, int(circumference / 0.24))
        count = min(remaining, max_on_ring)
        
        if count > 0:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            # Offset angles for alternating rings to improve packing
            offset = (idx - 1) * np.pi / max(1, count)
            for a in angles:
                if idx >= n_points:
                    break
                points[idx] = [ring_radius * np.cos(a + offset), ring_radius * np.sin(a + offset)]
                idx += 1
            remaining -= count
        
        ring_radius += ring_spacing
    
    # If we still have remaining points, place them randomly or in a spiral
    while remaining > 0 and idx < n_points:
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.sqrt(np.random.uniform(0, 0.9))
        points[idx] = [r * np.cos(angle), r * np.sin(angle)]
        idx += 1
        remaining -= 1
        
    return points

def physics_relaxation(pts, max_time=20.0):
    """
    Relax the configuration using repulsive forces.
    Force = k_rep * (1/d^2) - k_att * r
    """
    n = len(pts)
    pts = pts.copy()
    start = time.time()
    timeout = start + max_time
    
    # Parameters
    k_rep = 1.0
    k_att = 0.1
    dt = 0.01
    temperature = 1.0
    
    while time.time() < timeout:
        # Calculate forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d_vec = pts[i] - pts[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-6:
                    # Random push if overlapping exactly
                    d_vec = np.random.normal(0, 0.01, 2)
                    d = np.linalg.norm(d_vec)
                
                # Repulsive force: 1/d^2 to handle close contacts strongly
                f_mag = k_rep / (d * d)
                f_vec = f_mag * (d_vec / d)
                
                forces[i] += f_vec
                forces[j] -= f_vec
        
        # Attractive force to center (weak) to keep them in
        for i in range(n):
            r_vec = -pts[i]
            r_mag = np.linalg.norm(r_vec)
            if r_mag > 1e-6:
                forces[i] += k_att * (r_vec / r_mag)
        
        # Apply forces with cooling
        # Add noise
        noise = np.random.normal(0, temperature * 0.01, (n, 2))
        pts += (forces + noise) * dt
        
        # Constrain to unit circle
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 0.99
        if np.any(mask):
            pts[mask] = pts[mask] / dists[mask, np.newaxis] * 0.99
            
        # Cool down
        temperature *= 0.995
        if temperature < 1e-6:
            break
            
    return pts

def pairwise_local_search(pts, max_time=30.0):
    """
    Local search by identifying the pair with minimum distance and
    trying to move one of them to improve the minimum distance.
    """
    n = len(pts)
    pts = pts.copy()
    start = time.time()
    timeout = start + max_time
    
    best_r = compute_radius(pts)
    no_improve_count = 0
    step_size = 0.05
    
    while time.time() < timeout:
        # Find critical pair
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        
        min_val = d.min() / 2.0
        min_idx = np.unravel_index(d.argmin(), d.shape)
        i, j = min_idx[0], min_idx[1]
        
        # Also check boundary
        dists_to_edge = 1.0 - np.linalg.norm(pts, axis=1)
        min_edge = dists_to_edge.min()
        
        current_r = min(min_val, min_edge)
        
        if current_r > best_r:
            best_r = current_r
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 20:
                break
                
        # Try to move point i
        d_vec = pts[i] - pts[j]
        d_mag = np.linalg.norm(d_vec)
        
        if d_mag < 1e-6:
            continue
            
        # Direction away from j
        dir_vec = d_vec / d_mag
        
        # Try a few steps
        improved = False
        for step in [step_size, step_size * 0.5, -step_size * 0.5]:
            new_pts = pts.copy()
            new_pts[i] += step * dir_vec
            
            # Check if still inside
            if np.linalg.norm(new_pts[i]) < 0.999:
                new_r = compute_radius(new_pts)
                if new_r > current_r:
                    pts = new_pts
                    improved = True
                    break
        
        # If we didn't improve by moving i, try moving j
        if not improved:
            dir_vec_j = -dir_vec
            for step in [step_size, step_size * 0.5, -step_size * 0.5]:
                new_pts = pts.copy()
                new_pts[j] += step * dir_vec_j
                
                if np.linalg.norm(new_pts[j]) < 0.999:
                    new_r = compute_radius(new_pts)
                    if new_r > current_r:
                        pts = new_pts
                        improved = True
                        break
        
        # Reduce step size if stuck
        if not improved:
            step_size *= 0.9
            if step_size < 1e-4:
                break
                
    return pts

def construct():
    """Construct centers of N equal circles inside unit circle."""
    if _elapsed() > _DEADLINE - 10.0:
        return seed_rings(N)
    
    # 1. Generate initial seed
    pts = seed_rings(N)
    
    # 2. Physics relaxation to get into a good basin
    if _remaining() > 30.0:
        pts = physics_relaxation(pts, max_time=min(20.0, _remaining() - 10.0))
        
    # 3. Pairwise local search to polish
    if _remaining() > 10.0:
        pts = pairwise_local_search(pts, max_time=min(40.0, _remaining() - 5.0))
        
    # 4. Final check and projection
    dists = np.linalg.norm(pts, axis=1)
    mask = dists > 0.999
    if np.any(mask):
        pts[mask] = pts[mask] / dists[mask, np.newaxis] * 0.999
        
    return pts
# EVOLVE-BLOCK-END
