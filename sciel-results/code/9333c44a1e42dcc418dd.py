"""
Strategy: Rigid-body collision dynamics simulation using Hertzian contact law.
Points are initialized on a hexagonal lattice and evolved via Verlet integration.
Repulsive forces are proportional to overlap depth to the fourth power (Hertzian),
ensuring a smooth, differentiable transition from free motion to hard-contact equilibrium.
A boundary spring force keeps points inside the unit disk.
"""
import numpy as np
import time
import math

N = 300
DEADLINE = 140.0  # Safety margin from 150s limit

def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, return first N."""
    best_points = None
    # Try a range of spacings to find a good initial dense packing
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

def _simulate(points, time_limit=10.0, max_steps=50000, dt=0.005, damping=0.995, k_rep=100.0, k_bound=50.0):
    """
    Run Verlet integration with Hertzian repulsive forces and boundary springs.
    Returns the best configuration found during the simulation.
    """
    start_time = time.time()
    pts = points.copy()
    n = len(pts)
    
    # Initialize velocities to zero
    v = np.zeros_like(pts)
    
    best_pts = pts.copy()
    best_score = _score(pts)
    best_kinetic = 0.0
    
    # Precompute indices for pairs? No, vectorized is better for N=300.
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise Hertzian repulsion: F = k * (2r - d)^4 * direction, if d < 2r
        # We use a "virtual" radius r_virtual that is slightly larger than current estimate to ensure separation
        # But for dynamics, we just push apart if they overlap. 
        # Let's use a fixed reference radius for the force calculation to stabilize.
        # Actually, better to use the current minimum radius estimate? 
        # No, standard collision detection: if dist < 2*r_min, push.
        # But we want to maximize r_min. So we should push them apart until they are at distance 2*r_current.
        # Let's use a dynamic r_ref = max(current_score, 0.05) to ensure we are pushing to a reasonable size.
        # A better approach: use a fixed target radius for the force, e.g., r_target = 0.055.
        # This forces the system to pack as if the circles were size 0.055.
        # If they can't fit, they will be compressed, but the dynamics will find the best local arrangement.
        r_target = 0.055  # Slightly above best known to encourage expansion
        
        diff = pts[:, np.newaxis, :] - points[np.newaxis, :, :] # Wait, pts is current, points is initial? No, pts is updated.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 1e-10)
        
        # Overlap depth
        overlap = 2.0 * r_target - dists
        overlap[overlap < 0] = 0.0
        
        # Force magnitude: k * overlap^4
        force_mag = k_rep * (overlap ** 4)
        np.fill_diagonal(force_mag, 0.0)
        
        # Force direction: normalized difference vector
        # F_ij = force_mag * (p_j - p_i) / dist
        # But we need to sum over j for each i.
        # F_i = sum_j (k * overlap_ij^4 * (p_j - p_i) / dist_ij)
        
        # Vectorized computation of forces
        # dir_vec = diff / dists (but careful with zero dist)
        dir_vec = diff / dists[:, :, np.newaxis]
        
        # Force on i from j is -k * overlap^4 * dir_vec_ij (since dir_vec_ij points i->j? No, diff = i-j, so dir is i->j? 
        # diff[i,j] = pts[i] - pts[j]. So direction from j to i is diff[i,j]/dist.
        # Force on i should be away from j. So direction is (pts[i] - pts[j])/dist = diff[i,j]/dist.
        # So F_i += k * overlap^4 * (pts[i] - pts[j])/dist
        
        forces += np.sum(force_mag[:, :, np.newaxis] * dir_vec, axis=1)
        
        # Boundary force: if |p| > 1 - r_target, push inward
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        bound_overlap = r_pts - (1.0 - r_target)
        bound_overlap[bound_overlap < 0] = 0.0
        
        # Force magnitude
        bound_force_mag = k_bound * (bound_overlap ** 4)
        
        # Direction: inward (towards origin)
        # dir_bound = -pts / r_pts
        dir_bound = -pts / r_pts[:, np.newaxis]
        # Handle r_pts == 0
        dir_bound[0] = 0.0 # If at origin, no direction
        
        forces += bound_force_mag[:, np.newaxis] * dir_bound
        
        # Verlet integration
        # v = v * damping + forces * dt
        v = v * damping + forces * dt
        pts = pts + v * dt
        
        # Keep inside hard boundary (safety)
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        mask = r_pts > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / r_pts[mask, np.newaxis] * (1.0 - 1e-6)
            v[mask] = 0.0 # Kill velocity if hitting hard wall
        
        # Check score periodically
        if step % 100 == 0:
            current_score = _score(pts)
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
            
            # Check for stagnation in kinetic energy to stop early if settled
            kinetic = np.sum(v * v)
            if kinetic < 1e-10:
                break
            best_kinetic = kinetic

    return best_pts, best_score

def _perturb(points, magnitude=0.02):
    """Perturb points to escape local minima."""
    pts = points.copy()
    noise = np.random.normal(0, magnitude, size=pts.shape)
    pts += noise
    # Ensure points stay inside the unit disk
    r = np.hypot(pts[:, 0], pts[:, 1])
    mask = r > 0.95
    pts[mask] = pts[mask] / r[mask, np.newaxis] * 0.95
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
            points[i, 0] = r * np.cos(angle)
            points[i, 1] = r * np.sin(angle)
    
    best_points = points.copy()
    best_score = _score(points)
    
    # Multi-restart dynamics
    max_restarts = 5
    np.random.seed(42)
    
    for restart in range(max_restarts):
        # Time check
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 5.0  # 5s safety margin
        if remaining < 5.0:
            break
        
        # For first restart, use original; for subsequent, perturb
        if restart == 0:
            current = points.copy()
            # Slight randomization to break symmetry
            current += np.random.normal(0, 0.001, size=current.shape)
        else:
            # Perturb from best found so far
            magnitude = 0.01 + 0.01 * restart
            current = _perturb(best_points, magnitude)
        
        # Run dynamics
        time_limit_for_this = min(remaining * 0.8, 20.0)
        relaxed, score = _simulate(current, time_limit=time_limit_for_this)
        
        if score > best_score:
            best_score = score
            best_points = relaxed.copy()
        
        # If we've hit a good score, we can stop early
        if best_score > 0.0531:
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
