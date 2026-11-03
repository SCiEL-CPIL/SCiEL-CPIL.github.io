import numpy as np
import time
import math

N = 300
DEADLINE = 118.0  # Safety margin from 150s limit

def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, return first N."""
    best_points = None
    # Try a few spacings to find a good initial configuration
    for s in [0.10, 0.102, 0.104, 0.106, 0.108, 0.110, 0.112, 0.115, 0.12]:
        k = int(1.0 / s) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * math.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            # Keep the N closest to center for a compact packing
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            pts = points[idx]
            if best_points is None or len(pts) > len(best_points):
                best_points = pts
                if len(pts) == N:
                    break
    
    if best_points is not None and len(best_points) == N:
        return best_points
    
    # Fallback: very fine grid
    s = 0.05
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * math.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                points.append((x, y))
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind='stable')[:N]
    return points[idx]

def _score(points):
    """Compute the maximum radius r for given centers."""
    # Distance to boundary
    dist_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)

def _md_relax(points, time_limit):
    """
    Perform Molecular Dynamics relaxation.
    Uses Coulombic repulsion (1/d^2) to maximize separation.
    Includes a soft boundary constraint.
    """
    start_time = time.time()
    pts = points.copy()
    v = np.zeros_like(pts)
    
    # Parameters
    dt = 0.005
    k_repel = 10.0
    k_boundary = 100.0
    damping = 0.95
    max_iter = 10000
    
    # Precompute indices for pairs? No, vectorize the force calculation.
    # N=300, N^2 = 90k pairs. Vectorized numpy is fast enough.
    
    for step in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        # 1. Pairwise repulsion
        # diff[i, j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, 1.0) # Avoid div by 0
        
        # Force is proportional to 1/d^2 in direction of diff
        # F_ij = k_repel * (pts_i - pts_j) / |pts_i - pts_j|^3
        # Since we want to maximize distance, force on i due to j is repulsive.
        # Vectorized:
        inv_d3 = 1.0 / (dists_sq * np.sqrt(dists_sq))
        # Set diagonal to 0
        inv_d3[np.diag_indices_from(inv_d3)] = 0.0
        
        # Force on each point i is sum over j of k_repel * diff[i,j] * inv_d3[i,j]
        # diff shape (N, N, 2), inv_d3 shape (N, N)
        # We need to multiply diff by inv_d3[..., np.newaxis] and sum over axis 1
        forces = k_repel * np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        # 2. Boundary force
        # Push points inward if they are close to boundary
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        # If r > 1, force is inward. 
        # Let's use a soft constraint: if r > 0.9, apply force towards center.
        # Actually, better: force = -k_boundary * (r - 1) * (pts / r) if r > 1?
        # Or just a harmonic potential: F = -k * (r - R_target) * radial_vec
        # Let's use a simple radial force that pushes outwards if r < 1? No, we want them inside.
        # The repulsion pushes them out. The boundary should push them in.
        # F_boundary = -k_boundary * pts * max(0, r_pts - 0.99)
        # This only activates when close to boundary.
        
        # More robust: Use a potential that penalizes being outside or too close.
        # Let's just clamp and apply a restoring force if outside.
        # But we want them to stay inside.
        # Let's add a force that pulls them towards the center if they are near the edge.
        # F = -k_boundary * pts / r_pts * (r_pts - 0.95) * (r_pts > 0.95)
        
        r_safe = np.maximum(r_pts, 1e-6)
        radial_vec = pts / r_safe[:, np.newaxis]
        
        # Apply force only if r > 0.95
        mask = r_pts > 0.95
        boundary_forces = np.zeros_like(pts)
        if np.any(mask):
            # Linear spring pulling back to r=0.95
            dist_over = r_pts[mask] - 0.95
            boundary_forces[mask] = -k_boundary * dist_over[:, np.newaxis] * radial_vec[mask]
            
        total_forces = forces + boundary_forces
        
        # Update velocity
        v = v * damping + total_forces * dt
        pts += v * dt
        
        # Hard clamp to unit circle to ensure validity
        r_new = np.hypot(pts[:, 0], pts[:, 1])
        mask_out = r_new > 1.0
        if np.any(mask_out):
            pts[mask_out] = pts[mask_out] / r_new[mask_out, np.newaxis] * (1.0 - 1e-9)
            # Kill velocity component normal to boundary
            # Simplified: just zero out velocity for those points? Or reflect?
            # For stability, just dampen velocity significantly for boundary points
            v[mask_out] *= 0.5
            
    return pts

def construct():
    start_time = time.time()
    
    # Generate initial hexagonal lattice
    points = _hex_lattice_init()
    if points is None or len(points) != N:
        # Emergency fallback
        points = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 0.5
            points[i, 0] = r * math.cos(angle)
            points[i, 1] = r * math.sin(angle)
    
    best_points = points.copy()
    best_score = _score(points)
    
    np.random.seed(42)
    
    # Multi-restart strategy with MD
    # We will run MD for a few seconds, score, keep best.
    # Then perturb and run again.
    
    max_restarts = 10
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 2.0
        if remaining < 5.0:
            break
        
        if restart == 0:
            current = points.copy()
            time_limit = min(remaining * 0.4, 10.0)
        else:
            # Perturb from best found
            # Use a larger perturbation for later restarts to escape local minima
            magnitude = 0.05 * (1 + 0.5 * restart)
            current = best_points + np.random.normal(0, magnitude, size=best_points.shape)
            # Clamp to disk
            r = np.hypot(current[:, 0], current[:, 1])
            mask = r > 0.95
            current[mask] = current[mask] / r[mask, np.newaxis] * 0.95
            time_limit = min(remaining * 0.3, 8.0)
            
        optimized = _md_relax(current, time_limit)
        
        # Post-process: Simple local polish using gradient ascent on min distance?
        # Or just score. MD usually gets close.
        score = _score(optimized)
        
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
            
        # Check time
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Final check: ensure validity
    r = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(r > 1.0):
        mask = r > 1.0
        best_points[mask] = best_points[mask] / r[mask, np.newaxis] * (1.0 - 1e-10)
    
    # Ensure no coincident points
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(best_points[i] - best_points[j]) < 1e-10:
                    best_points[j, 0] += 1e-6
    
    return best_points
