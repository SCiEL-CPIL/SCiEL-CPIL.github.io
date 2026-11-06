import numpy as np
import time
from scipy.optimize import minimize
import itertools

def construct():
    start_time = time.time()
    timeout = 140  # Safety margin
    
    # Objective: Maximize product of distances, equivalent to maximizing sum of log distances
    # Subject to: max distance <= 2 (and distinct points)
    
    def log_product_dist(pts):
        # pts: (8, 2)
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        # Avoid log(0)
        d = np.maximum(d, 1e-12)
        # Exclude self-distances (diagonal)
        mask = ~np.eye(8, dtype=bool)
        d_masked = d[mask]
        return np.sum(np.log(d_masked))

    def diameter(pts):
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        return np.max(d)

    def objective_and_grad(x):
        pts = x.reshape(8, 2)
        # Compute log product
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        d = np.maximum(d, 1e-12)
        mask = ~np.eye(8, dtype=bool)
        
        # Gradient of sum log(d_ij) w.r.t pts
        # d/dp_i log(d_ij) = (p_i - p_j) / d_ij^2
        grad = np.zeros_like(pts)
        for i in range(8):
            for j in range(8):
                if i == j: continue
                diff = pts[i] - pts[j]
                dist2 = np.dot(diff, diff)
                if dist2 < 1e-12: continue
                grad[i] += diff / dist2
        
        # Penalty for diameter > 2
        # To make it differentiable, we use a soft max or just penalize the max pair
        # Let's find the pair with max distance
        d_matrix = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        max_idx = np.unravel_index(np.argmax(d_matrix), d_matrix.shape)
        i, j = max_idx
        if i == j:
            i, j = 0, 1
            
        # If diameter > 2, add penalty
        pen_lambda = 100.0
        diam = d_matrix[i, j]
        if diam > 2.0:
            # Gradient of (diam - 2)^2 w.r.t pts
            # d(diam)/dpts[i] = (pts[i] - pts[j]) / diam
            # d(diam)/dpts[j] = (pts[j] - pts[i]) / diam
            diff = pts[i] - pts[j]
            grad_pen_i = 2 * pen_lambda * (diam - 2.0) * diff / diam
            grad_pen_j = 2 * pen_lambda * (diam - 2.0) * (-diff) / diam
            grad[i] += grad_pen_i
            grad[j] += grad_pen_j
            obj_val = -log_product_dist(pts) + pen_lambda * (diam - 2.0)**2
        else:
            obj_val = -log_product_dist(pts)
            
        return obj_val, grad.flatten()

    def refine_from_start(start_pts):
        # Use L-BFGS-B
        x0 = start_pts.flatten()
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
        best_pts = res.x.reshape(8, 2)
        best_score = np.exp(log_product_dist(best_pts))
        
        # Normalize to diameter 2 for final score calculation consistency
        # The problem says "after scaling the set to diameter 2"
        # So we must scale the result to diameter 2 before scoring
        d_max = diameter(best_pts)
        if d_max > 1e-9:
            best_pts_scaled = best_pts * (2.0 / d_max)
        else:
            best_pts_scaled = best_pts
            
        return best_pts_scaled, np.exp(log_product_dist(best_pts_scaled))

    # Candidate Topologies
    candidates = []
    
    # 1. Regular Octagon (Baseline)
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    candidates.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Two Concentric Squares
    # Outer square radius 1, Inner square radius r
    for r in [0.2, 0.5, 0.8]:
        angles_out = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        angles_in = np.linspace(0, 2*np.pi, 4, endpoint=False)
        p_out = np.column_stack([r_out*np.cos(angles_out), r_out*np.sin(angles_out)]) if False else np.column_stack([1.0*np.cos(angles_out), 1.0*np.sin(angles_out)])
        p_in = np.column_stack([r*np.cos(angles_in), r*np.sin(angles_in)])
        candidates.append(np.vstack([p_out, p_in]))
        
    # 3. Square + Center + Antipode? No, distinct points.
    # 4. Rectangle + Rectangle?
    
    # 5. Random Restarts
    np.random.seed(42)
    for _ in range(20):
        # Random points in unit disk
        r = np.sqrt(np.random.rand(8))
        theta = np.random.rand(8) * 2 * np.pi
        pts = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
        candidates.append(pts)

    # 6. Structured "Distorted" Octagon
    # Perturb radii of regular octagon
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for dist in [0.1, 0.3, 0.5]:
        radii = 1.0 + dist * np.sin(2 * angles) # Alternating radii
        candidates.append(np.column_stack([radii*np.cos(angles), radii*np.sin(angles)]))
        
    # 7. The "1.115" hint suggests a specific structure. Let's try a square with corners moved.
    # Or two squares, one rotated 22.5 deg (octagon) vs 45 deg (square+square).
    
    best_pts = None
    best_score = 0.0
    
    for start_pts in candidates:
        if time.time() - start_time > timeout:
            break
            
        try:
            # Ensure distinct points
            d_min = np.min(np.linalg.norm(start_pts[:, np.newaxis, :] - start_pts[np.newaxis, :, :], axis=2) * (~np.eye(8, dtype=bool)))
            if d_min < 1e-5:
                # Add small noise
                start_pts += np.random.normal(0, 1e-3, start_pts.shape)
                
            pts, score = refine_from_start(start_pts)
            if score > best_score:
                best_score = score
                best_pts = pts
        except Exception:
            continue

    # Final check and return
    if best_pts is None:
        # Fallback to regular octagon
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Ensure coordinates are within limits
    # Scale to diameter 2
    d_max = diameter(best_pts)
    if d_max > 1e-9:
        best_pts = best_pts * (2.0 / d_max)
        
    return best_pts.tolist()
