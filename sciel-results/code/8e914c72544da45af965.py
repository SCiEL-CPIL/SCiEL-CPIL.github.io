"""Packing 100 equal spheres in a unit cube via penalty method + force-directed refinement."""
import numpy as np
from scipy.spatial.distance import pdist, cdist
import time

N = 100
CUBE_HALF = 0.5

def _compute_radius(centres):
    """Compute the maximum radius for given centres."""
    # Minimum distance between any pair
    if N > 1:
        d = pdist(centres)
        min_pair = d.min() if len(d) > 0 else np.inf
    else:
        min_pair = np.inf
    
    # Minimum distance to boundary
    dists_to_boundary = np.min(np.abs(centres), axis=1)
    min_boundary = CUBE_HALF - dists_to_boundary.max() if N > 0 else np.inf
    
    # Wait, that's wrong. Let me recalculate.
    # For each sphere, the distance to the nearest face is min over axes of (CUBE_HALF - |c_ik|)
    # Actually, distance to boundary for sphere i is min over k of (CUBE_HALF - |c_i[k]|)
    # But we need the MINIMUM over all spheres of their distance to the boundary
    # So: min_i (min_k (CUBE_HALF - |c_i[k]|))
    
    # Let me redo this properly
    # dist_to_boundary[i] = min_k (CUBE_HALF - |c_i[k]|)
    abs_centres = np.abs(centres)
    dist_to_faces = CUBE_HALF - abs_centres  # shape (N, 3)
    min_dist_to_boundary = dist_to_faces.min()  # minimum over all i and k
    
    radius = min(min_pair / 2.0, min_dist_to_boundary)
    return radius

def _force_directed_refine(centres, max_iters=500, dt=0.001):
    """Refine centres using force-directed method to increase minimum distance."""
    centres = centres.copy()
    for iteration in range(max_iters):
        # Compute pairwise distances
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Compute forces
        forces = np.zeros_like(centres)
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 1e-10:
                    # Random perturbation
                    forces[i] += np.random.randn(3) * 0.01
                    forces[j] -= np.random.randn(3) * 0.01
                    continue
                # Repulsive force (inverse square)
                f = 1.0 / (d * d)
                direction = (centres[i] - centres[j]) / d
                forces[i] += f * direction
                forces[j] -= f * direction
        
        # Add boundary repulsion
        for i in range(N):
            for k in range(3):
                if centres[i, k] > 0:
                    dist_to_pos = CUBE_HALF - centres[i, k]
                    if dist_to_pos < 0.1:
                        forces[i, k] -= 10.0 / (dist_to_pos + 0.01) ** 2
                else:
                    dist_to_neg = CUBE_HALF + centres[i, k]
                    if dist_to_neg < 0.1:
                        forces[i, k] += 10.0 / (dist_to_neg + 0.01) ** 2
        
        # Update positions with damping
        centres += dt * forces
        # Clamp to cube
        centres = np.clip(centres, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
        
        # Decrease step size
        dt *= 0.995
    
    return centres

def _penalty_solve(centres, r_target, max_iters=200, mu_init=1.0):
    """Solve for centres that satisfy distance constraints >= 2*r_target using penalty method."""
    centres = centres.copy()
    mu = mu_init
    
    for iteration in range(max_iters):
        # Compute gradient of penalty function
        # Penalty = sum_{i<j} max(0, (2*r_target)^2 - ||c_i - c_j||^2)^2
        # + sum_i max(0, CUBE_HALF - |c_ik| - r_target)^2 for boundary
        
        forces = np.zeros_like(centres)
        
        # Pairwise constraints
        for i in range(N):
            for j in range(i + 1, N):
                diff = centres[i] - centres[j]
                dist_sq = np.dot(diff, diff)
                min_dist_sq = (2 * r_target) ** 2
                if dist_sq < min_dist_sq:
                    # Violation: need to increase distance
                    # Gradient of max(0, min_dist_sq - dist_sq)^2 w.r.t. c_i
                    # = 2 * (min_dist_sq - dist_sq) * (-2 * diff) for c_i
                    # = 4 * (min_dist_sq - dist_sq) * diff
                    violation = min_dist_sq - dist_sq
                    f = 4.0 * violation * diff
                    forces[i] += f
                    forces[j] -= f
        
        # Boundary constraints
        for i in range(N):
            for k in range(3):
                c = centres[i, k]
                # Distance to nearest face in axis k
                if c >= 0:
                    dist_to_face = CUBE_HALF - c
                    if dist_to_face < r_target:
                        violation = r_target - dist_to_face
                        forces[i, k] -= 4.0 * violation  # push inward (negative direction)
                else:
                    dist_to_face = CUBE_HALF + c
                    if dist_to_face < r_target:
                        violation = r_target - dist_to_face
                        forces[i, k] += 4.0 * violation  # push inward (positive direction)
        
        # Update
        step = 0.01 / (1.0 + 0.01 * iteration)
        centres += step * forces
        centres = np.clip(centres, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
        
        # Check if all constraints satisfied
        min_dist_sq = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.dot(centres[i] - centres[j], centres[i] - centres[j])
                min_dist_sq = min(min_dist_sq, d)
        
        min_boundary = np.inf
        for i in range(N):
            for k in range(3):
                d = CUBE_HALF - abs(centres[i, k])
                min_boundary = min(min_boundary, d)
        
        current_r = min(min_dist_sq ** 0.5 / 2.0, min_boundary)
        if current_r >= r_target - 1e-8:
            break
        
        # Increase penalty parameter
        mu *= 1.5
    
    return centres

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit with safety margin
    
    # Initial configuration: simple grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    # Force-directed refinement to get a better starting point
    centres = _force_directed_refine(centres, max_iters=300, dt=0.005)
    
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    # Binary search on radius with penalty solver
    r_low = best_r
    r_high = 0.15  # Upper bound
    
    while (time.time() < deadline - 30.0) and (r_high - r_low > 1e-7):
        r_mid = (r_low + r_high) / 2.0
        candidate = _penalty_solve(centres, r_mid, max_iters=100, mu_init=1.0)
        r_achieved = _compute_radius(candidate)
        
        if r_achieved >= r_mid - 1e-8:
            r_low = r_mid
            best_centres = candidate
            best_r = r_achieved
        else:
            r_high = r_mid
        
        # Use the best candidate as starting point for next iteration
        centres = candidate
    
    # Final refinement with force-directed method
    if time.time() < deadline - 10.0:
        final_centres = _force_directed_refine(best_centres, max_iters=500, dt=0.002)
        final_r = _compute_radius(final_centres)
        if final_r > best_r:
            best_centres = final_centres
            best_r = final_r
    
    # Ensure valid output
    best_centres = np.clip(best_centres, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
    
    return best_centres
