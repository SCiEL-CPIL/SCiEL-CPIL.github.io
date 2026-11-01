import numpy as np
import time
from scipy.optimize import minimize

N = 32
TIME_LIMIT = 120.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # 1. Initialization: Hexagonal lattice
    points = []
    for row in range(10):
        for col in range(10):
            x = col + (0.5 if row % 2 else 0)
            y = row * np.sqrt(3) / 2
            points.append((x, y))
    
    pts = np.array(points)
    center = pts.mean(axis=0)
    pts = pts - center
    
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:N]
    init_pts = pts[idx].copy()
    
    # Scale to fit
    min_x, min_y = init_pts.min(axis=0)
    max_x, max_y = init_pts.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    scale = 0.9 / max(width, height)
    
    init_pts[:, 0] = (init_pts[:, 0] - min_x) * scale
    init_pts[:, 1] = (init_pts[:, 1] - min_y) * scale
    
    # Center
    current_w = init_pts.max(axis=0) - init_pts.min(axis=0)
    offset_x = (1.0 - current_w[0]) / 2
    offset_y = (1.0 - current_w[1]) / 2
    init_pts[:, 0] += offset_x
    init_pts[:, 1] += offset_y
    
    # Initial radii
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(init_pts[i] - init_pts[j])
            if d < min_dist:
                min_dist = d
    init_r = min_dist / 2 * 0.9
    
    # Variables
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = init_pts[i, 0]
        x0[3*i+1] = init_pts[i, 1]
        x0[3*i+2] = init_r
        
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-5, 0.5))
        
    # Weight holder for adaptive penalty
    weight_ref = [100.0]
    
    def objective(vars):
        X = vars[0::3]
        Y = vars[1::3]
        R = vars[2::3]
        
        obj = -np.sum(R)
        
        # Pairwise distances
        dx = X[:, None] - X[None, :]
        dy = Y[:, None] - Y[None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        
        # Overlap penalty
        overlap = R[:, None] + R[None, :] - dist
        overlap = np.where(overlap > 0, overlap, 0)
        
        # Sum over upper triangle i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pen_pairs = np.sum(overlap[mask] ** 2)
        
        # Boundary penalty
        b_x1 = np.where(X - R < 0, (X - R)**2, 0)
        b_x2 = np.where(1.0 - X - R < 0, (1.0 - X - R)**2, 0)
        b_y1 = np.where(Y - R < 0, (Y - R)**2, 0)
        b_y2 = np.where(1.0 - Y - R < 0, (1.0 - Y - R)**2, 0)
        pen_bounds = np.sum(b_x1) + np.sum(b_x2) + np.sum(b_y1) + np.sum(b_y2)
        
        return obj + weight_ref[0] * (pen_pairs + pen_bounds)

    def objective_grad(vars):
        X = vars[0::3]
        Y = vars[1::3]
        R = vars[2::3]
        
        grad = np.zeros_like(vars)
        grad[2::3] = -1.0
        
        dx = X[:, None] - X[None, :]
        dy = Y[:, None] - Y[None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        
        # Avoid division by zero
        dist_safe = np.where(dist > 1e-8, dist, 1.0)
        
        overlap = R[:, None] + R[None, :] - dist
        overlap = np.where(overlap > 0, overlap, 0)
        
        # Weights for gradient
        # W_ij = 2 * overlap_ij / dist_ij
        W = 2.0 * overlap / dist_safe
        W = np.where(overlap > 0, W, 0.0)
        
        # Gradient for Radii
        # For objective sum_{i<j} o_ij^2, grad_R_k = sum_{j!=k} 2 o_kj
        grad_R_pen = 2.0 * np.sum(overlap, axis=1)
        grad[2::3] += weight_ref[0] * grad_R_pen
        
        # Gradient for X
        # sum_j W_ij * (X_i - X_j)
        sum_W = np.sum(W, axis=1)
        W_X = W @ X
        grad_X_pen = 2.0 * (X * sum_W - W_X)
        grad[0::3] += weight_ref[0] * grad_X_pen
        
        # Gradient for Y
        sum_W_y = np.sum(W, axis=1)
        W_Y = W @ Y
        grad_Y_pen = 2.0 * (Y * sum_W_y - W_Y)
        grad[1::3] += weight_ref[0] * grad_Y_pen
        
        # Boundary Penalties
        mask_x1 = X - R < 0
        grad_x1 = np.where(mask_x1, 2.0 * (X - R), 0.0)
        grad[0::3] += weight_ref[0] * grad_x1
        grad[2::3] += weight_ref[0] * np.where(mask_x1, -2.0 * (X - R), 0.0)
        
        mask_x2 = 1.0 - X - R < 0
        grad_x2 = np.where(mask_x2, -2.0 * (1.0 - X - R), 0.0)
        grad[0::3] += weight_ref[0] * grad_x2
        grad[2::3] += weight_ref[0] * np.where(mask_x2, -2.0 * (1.0 - X - R), 0.0)
        
        mask_y1 = Y - R < 0
        grad_y1 = np.where(mask_y1, 2.0 * (Y - R), 0.0)
        grad[1::3] += weight_ref[0] * grad_y1
        grad[2::3] += weight_ref[0] * np.where(mask_y1, -2.0 * (Y - R), 0.0)
        
        mask_y2 = 1.0 - Y - R < 0
        grad_y2 = np.where(mask_y2, -2.0 * (1.0 - Y - R), 0.0)
        grad[1::3] += weight_ref[0] * grad_y2
        grad[2::3] += weight_ref[0] * np.where(mask_y2, -2.0 * (1.0 - Y - R), 0.0)
        
        return grad

    best_x = x0.copy()
    best_val = np.inf
    current_x = x0.copy()
    
    # Adaptive Penalty Loop
    for weight_iter in range(5):
        weight_ref[0] = 100.0 * (10 ** weight_iter)
        
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT * 0.8:
            break
            
        try:
            res = minimize(objective, current_x, method='L-BFGS-B', jac=objective_grad, bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success or res.status == 1:
                current_x = res.x
                val = objective(current_x)
                if val < best_val:
                    best_val = val
                    best_x = current_x.copy()
        except Exception:
            pass
            
        # Check if penalty is small enough
        X = current_x[0::3]
        Y = current_x[1::3]
        R = current_x[2::3]
        dx = X[:, None] - X[None, :]
        dy = Y[:, None] - Y[None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        overlap = R[:, None] + R[None, :] - dist
        overlap = np.where(overlap > 0, overlap, 0)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pen_pairs = np.sum(overlap[mask] ** 2)
        
        b_x1 = np.where(X - R < 0, (X - R)**2, 0)
        b_x2 = np.where(1.0 - X - R < 0, (1.0 - X - R)**2, 0)
        b_y1 = np.where(Y - R < 0, (Y - R)**2, 0)
        b_y2 = np.where(1.0 - Y - R < 0, (1.0 - Y - R)**2, 0)
        pen_bounds = np.sum(b_x1) + np.sum(b_x2) + np.sum(b_y1) + np.sum(b_y2)
        
        if pen_pairs + pen_bounds < 1e-8:
            break
            
    # Final Polish: Maximize radii given positions
    X = best_x[0::3].copy()
    Y = best_x[1::3].copy()
    R = best_x[2::3].copy()
    
    for _ in range(10):
        for i in range(N):
            min_dist_others = np.inf
            for j in range(N):
                if i == j: continue
                d = np.linalg.norm(np.array([X[i]-X[j], Y[i]-Y[j]]))
                min_dist_others = min(min_dist_others, d - R[j])
            min_dist_bound = min(X[i], 1-X[i], Y[i], 1-Y[i])
            R[i] = max(1e-6, min(min_dist_others, min_dist_bound) - 1e-7)
            
    result = np.zeros((N, 3))
    result[:, 0] = X
    result[:, 1] = Y
    result[:, 2] = R
    
    return result
# EVOLVE-BLOCK-END
