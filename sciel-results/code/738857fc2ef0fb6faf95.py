import numpy as np
from math import log, exp, sqrt
import time

def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 100.0  # 100s buffer
    
    # Regular octagon initialization
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0 = np.cos(angles)
    y0 = np.sin(angles)
    pts_init = np.column_stack((x0, y0))
    
    best_score = -np.inf
    best_pts = pts_init.copy()
    
    # Soft max parameter
    p = 50.0
    
    def compute_J_and_grad(pts):
        n = pts.shape[0]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        
        J = 0.0
        grad = np.zeros_like(pts)
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        d_flat = d[mask]
        
        # Term 1: sum log(d_ij)
        J += np.sum(np.log(d_flat))
        
        # Term 2: -28 * log(D_soft)
        d_p = d_flat**p
        S = np.sum(d_p)
        
        # Gradient accumulation
        # We need to map back to indices
        # This loop is O(n^2)
        for i in range(n):
            for j in range(i + 1, n):
                dij = d[i, j]
                if dij < 1e-10:
                    continue
                diff_ij = pts[i] - pts[j]
                
                grad_log = diff_ij / (dij**2)
                
                w = (dij**(p-1)) / S
                grad_soft = w * (diff_ij / dij)
                
                grad[i] += grad_log - 28.0 * grad_soft
                grad[j] -= grad_log - 28.0 * grad_soft
                
        J -= 28.0 * (1.0/p) * np.log(S)
        
        return J, grad

    def optimize(pts, steps=2000):
        current_J, _ = compute_J_and_grad(pts)
        best_local_J = current_J
        best_local_pts = pts.copy()
        
        step_size = 0.05
        for it in range(steps):
            if time.time() > deadline:
                break
                
            J, grad = compute_J_and_grad(pts)
            if J > best_local_J:
                best_local_J = J
                best_local_pts = pts.copy()
                
            # Rescale if too large
            norm = np.linalg.norm(pts)
            if norm > 10:
                pts /= 10
                # J is invariant, so best_local_J is still valid relative to new pts?
                # No, J(pts/10) = J(pts). So yes.
                
            pts += step_size * grad
            step_size *= 0.995
            
        return best_local_J, best_local_pts

    # Run optimization from regular octagon
    J_opt, pts_opt = optimize(pts_init)
    # Score = 16 * exp(J)
    score_opt = 16.0 * exp(J_opt)
    if score_opt > best_score:
        best_score = score_opt
        best_pts = pts_opt.copy()
        
    # Try random restarts
    np.random.seed(42)
    for _ in range(30):
        if time.time() > deadline:
            break
            
        # Mix of perturbations and random
        if np.random.rand() < 0.5:
            perturbed = pts_init + np.random.normal(0, 0.2, pts_init.shape)
        else:
            perturbed = np.random.uniform(-1, 1, (n, 2))
            
        J_r, pts_r = optimize(perturbed, steps=1000)
        score_r = 16.0 * exp(J_r)
        if score_r > best_score:
            best_score = score_r
            best_pts = pts_r.copy()
            
    return best_pts.tolist()
