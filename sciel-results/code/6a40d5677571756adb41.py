import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    P /= np.linalg.norm(P, axis=1, keepdims=True)

    # 2. Generate Evaluation Grid
    # Use a Fibonacci spiral for the grid too, but denser
    N_grid = 20000
    i_grid = np.arange(N_grid) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / N_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    Y = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)

    # 3. Optimization Loop
    # We minimize J(P) = sum_y (min_i d(y, P_i))^p
    # Start with moderate p, increase later
    
    p = 4.0
    lr = 0.01
    best_P = P.copy()
    best_score = np.inf

    def compute_score(P):
        # Compute min distance from each grid point to any P
        # dist(y, p) = arccos(y . p)
        # For large p, we can optimize on cos(dist) to avoid arccos?
        # d = arccos(c). d^p.
        # Let's compute distances in radians.
        
        # Dot products: (N_grid, N)
        dots = Y @ P.T
        # Clip for numerical stability
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots) # (N_grid, N)
        min_dists = np.min(dists, axis=1) # (N_grid,)
        # The covering radius is approx max(min_dists)
        # But our objective is sum(min_dists^p)
        return np.sum(min_dists ** p)

    def compute_gradient(P, min_dists, nearest_idx):
        # min_dists: (N_grid,)
        # nearest_idx: (N_grid,)
        grad = np.zeros_like(P)
        
        # Weights for each grid point
        # dJ/dP_i = sum_{y: nearest(y)==i} p * min_dists[y]^(p-1) * d(dist(y, P_i))/dP_i
        # d(dist)/dP_i = - (y - (y.P_i)P_i) / dist(y, P_i)
        
        # Vectorized calculation
        # For each y, we need to update P[nearest_idx[y]]
        
        # Let's loop over unique nearest indices or use bincount-like logic
        # Since N is small (50), we can loop over i
        for i in range(N):
            mask = (nearest_idx == i)
            if not np.any(mask):
                continue
            
            y_subset = Y[mask]
            d_subset = min_dists[mask]
            p_i = P[i]
            
            # Gradient term for each y in subset
            # term = p * d^(p-1) * (- (y - (y.p_i)p_i) / d)
            #      = - p * d^(p-2) * (y - (y.p_i)p_i)
            
            dots = y_subset @ p_i
            # Project y onto tangent plane at p_i: y_tan = y - (y.p_i)p_i
            y_tan = y_subset - dots[:, np.newaxis] * p_i[np.newaxis, :]
            
            # Weights
            weights = p * (d_subset ** (p - 2))
            
            # Sum of weighted y_tan
            grad_i = -np.sum(weights[:, np.newaxis] * y_tan, axis=0)
            
            grad[i] = grad_i
            
        return grad

    # Optimization Phases
    # Phase 1: Coarse optimization
    phases = [
        (4.0, 0.05, 100),   # p, lr, iters
        (8.0, 0.02, 100),
        (16.0, 0.01, 100),
        (32.0, 0.005, 100),
        (64.0, 0.002, 100)
    ]
    
    for p_val, lr_val, iters in phases:
        p = p_val
        for it in range(iters):
            if time.time() > deadline:
                break
            
            # Compute nearest and min dist
            dots = Y @ P.T
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            min_dists = np.min(dists, axis=1)
            nearest_idx = np.argmin(dists, axis=1)
            
            # Current objective
            current_obj = np.sum(min_dists ** p)
            
            # Compute gradient
            grad = compute_gradient(P, min_dists, nearest_idx)
            
            # Update
            P_new = P - lr_val * grad
            # Project back to sphere
            norms = np.linalg.norm(P_new, axis=1, keepdims=True)
            P_new = P_new / norms
            
            # Line search (simple backtracking)
            obj_new = compute_score(P_new)
            
            if obj_new < current_obj:
                P = P_new
                # Update best if improved
                # We track the actual covering radius (max min dist) for scoring
                # But the objective is a surrogate.
                # Let's compute the true max min dist for tracking
                true_max_min_dist = np.max(min_dists)
                if true_max_min_dist < best_score:
                    best_score = true_max_min_dist
                    best_P = P.copy()
            else:
                # Reduce learning rate if stuck
                lr_val *= 0.5

        # Increase p for next phase if time allows
        if time.time() > deadline:
            break

    # Final Polish: Use the best_P found
    # Do a few more iterations with very high p and small lr
    p = 128.0
    lr = 0.001
    for it in range(50):
        if time.time() > deadline:
            break
        
        dots = Y @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        nearest_idx = np.argmin(dists, axis=1)
        
        current_obj = np.sum(min_dists ** p)
        grad = compute_gradient(P, min_dists, nearest_idx)
        
        P_new = P - lr * grad
        norms = np.linalg.norm(P_new, axis=1, keepdims=True)
        P_new = P_new / norms
        
        obj_new = compute_score(P_new)
        if obj_new < current_obj:
            P = P_new
            true_max_min_dist = np.max(min_dists)
            if true_max_min_dist < best_score:
                best_score = true_max_min_dist
                best_P = P.copy()
        else:
            lr *= 0.5

    # Return the best configuration found
    # Ensure it's normalized
    best_P /= np.linalg.norm(best_P, axis=1, keepdims=True)
    return best_P
