"""
Optimized packing of 50 equal circles in a unit circle.
Strategy:
1. Geometry-informed ring initialization.
2. Smooth overlap energy objective with analytic gradients.
3. Multi-stage L-BFGS-B annealing on target radius.
4. Final local polish.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 50
R_KNOWN = 0.1258254895304

def _score(points):
    """Calculate the exact packing score."""
    if len(points) != N:
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    
    # Check if points are inside
    dists_to_origin = np.linalg.norm(points, axis=1)
    min_boundary_dist = 1.0 - np.max(dists_to_origin)
    
    # Calculate min pair distance
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _energy_and_grad(P, r):
    """
    Calculate the overlap energy and its gradient.
    Energy = 0.5 * (sum(overlap_cc^2) + sum(overlap_cb^2))
    overlap_cc = max(2r - dist, 0)
    overlap_cb = max(|p| + r - 1, 0)
    """
    P = P.reshape(N, 2)
    grad = np.zeros_like(P)
    
    # Circle-Circle interactions
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    overlap_cc = np.maximum(2.0 * r - dist, 0.0)
    
    # Gradient of sum(overlap_cc^2) / 2
    # d/dP_i sum_j max(2r - |P_i-P_j|, 0)^2 / 2
    # = sum_j (2r - |P_i-P_j|) * (P_i - P_j) / |P_i-P_j|  for |P_i-P_j| < 2r
    # = sum_j overlap_cc[i,j] * (P_i - P_j) / dist[i,j]
    
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 0)
    
    # W[i,j] = overlap_cc[i,j] / dist[i,j]
    W = overlap_cc * inv_dist
    
    # grad_i = sum_j W[i,j] * (P_i - P_j) = P_i * sum_j W[i,j] - sum_j W[i,j] P_j
    sum_W = np.sum(W, axis=1)
    term1 = P * sum_W[:, np.newaxis]
    term2 = (P.T @ W).T  # This is sum_j W[i,j] P_j? No.
    # Let's check dimensions.
    # W is (N, N). P is (N, 2).
    # (P.T @ W) is (2, N).
    # We want for each i: sum_j W[i,j] P_j.
    # This is (W @ P).T ?
    # (W @ P) is (N, 2). Row i is sum_j W[i,j] P_j.
    # So term2 should be W @ P.
    
    term2 = W @ P
    
    grad += term1 - term2
    
    # Circle-Boundary interactions
    dist_origin = np.linalg.norm(P, axis=1)
    overlap_cb = np.maximum(dist_origin + r - 1.0, 0.0)
    
    # Gradient of sum(overlap_cb^2) / 2
    # d/dP_i max(|P_i| + r - 1, 0)^2 / 2
    # = overlap_cb[i] * P_i / |P_i|
    
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
    
    grad += (overlap_cb * inv_dist_origin)[:, np.newaxis] * P
    
    energy = 0.5 * (np.sum(overlap_cc**2) + np.sum(overlap_cb**2))
    
    return energy, grad.flatten()

def _ring_initialization(r_est=0.125):
    """
    Initialize with a ring structure matching the known optimal packing.
    """
    centers = []
    centers.append([0.0, 0.0])
    
    # Ring 1
    ring1_r = 2.0 * r_est
    ring1_n = 6
    for i in range(ring1_n):
        angle = 2.0 * np.pi * i / ring1_n
        centers.append([ring1_r * np.cos(angle), ring1_r * np.sin(angle)])
    
    # Ring 2
    ring2_r = 2.0 * np.sqrt(3.0) * r_est
    ring2_n = 12
    for i in range(ring2_n):
        angle = 2.0 * np.pi * i / ring2_n
        centers.append([ring2_r * np.cos(angle), ring2_r * np.sin(angle)])
    
    # Ring 3
    ring3_r = 4.0 * r_est
    ring3_n = 18
    for i in range(ring3_n):
        angle = 2.0 * np.pi * i / ring3_n
        centers.append([ring3_r * np.cos(angle), ring3_r * np.sin(angle)])
    
    # Ring 4
    ring4_r = 5.5 * r_est
    ring4_n = 13
    for i in range(ring4_n):
        angle = 2.0 * np.pi * i / ring4_n + np.pi / ring4_n
        centers.append([ring4_r * np.cos(angle), ring4_r * np.sin(angle)])
    
    pts = np.array(centers[:N])
    # Ensure inside
    norms = np.linalg.norm(pts, axis=1)
    for i in range(len(pts)):
        if norms[i] > 1.0 - r_est:
            pts[i] *= (1.0 - r_est) / norms[i]
            
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start with ring initialization
    pts = _ring_initialization(r_est=0.125)
    current_score = _score(pts)
    best_score = current_score
    best_points = pts.copy()
    
    # Define objective and gradient for L-BFGS-B
    def objective(x):
        e, _ = _energy_and_grad(x, r_target)
        return e
        
    def gradient(x):
        _, g = _energy_and_grad(x, r_target)
        return g
        
    # Annealing loop: increase target radius
    # Start slightly below known best to ensure feasibility
    r_start = 0.120
    r_end = 0.126
    steps = 10
    
    for i in range(steps):
        if time.time() > deadline - 20.0:
            break
            
        r_target = r_start + (r_end - r_start) * i / steps
        
        # Update objective/gradient closures
        # Since r_target is captured by reference in the nested functions,
        # we just need to update the variable.
        # However, L-BFGS-B might cache? No, it calls the function each time.
        # But we need to make sure the functions see the new r_target.
        # Python closures capture the variable, not the value.
        # So this works.
        
        try:
            res = minimize(
                objective, 
                pts.flatten(), 
                jac=gradient, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            
            new_pts = res.x.reshape(N, 2)
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_pts.copy()
                
            # Always move to the new position, even if score didn't improve immediately
            # because we are optimizing for a larger r_target, which might temporarily
            # lower the "current" score if the config is distorted, but it's necessary
            # to reach the higher r.
            # However, if the score drops significantly, it might be better to keep the old one?
            # Usually, annealing works best if you follow the optimization path.
            pts = new_pts
            
        except Exception:
            pass
            
    # Final Polish: Use the current best points and try to optimize the exact score
    # We can't easily use gradients for the min function, but we can use L-BFGS-B
    # on a soft-min of the exact constraints if we have time, or just return the best found.
    # Given the high quality of the energy optimization, the score should be very close to optimal.
    
    # One more quick polish with a slightly higher r_target if time permits
    if time.time() < deadline - 5.0:
        r_target = best_score * 1.01  # Try to push a bit further
        # Reset objective/gradient to use this new r_target?
        # The closure uses r_target. We can just change r_target.
        r_target = 0.1265
        try:
            res = minimize(
                objective, 
                best_points.flatten(), 
                jac=gradient, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
            )
            new_pts = res.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_points = new_pts.copy()
        except Exception:
            pass

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
    print(f"Shape: {pts.shape}")
