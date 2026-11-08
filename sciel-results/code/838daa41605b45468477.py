"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Precompute indices for vectorized penalty
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def penalty(circles):
        """
        circles: (N, 3) array of (x, y, r)
        Returns: sum of squared constraint violations (to be minimized)
        """
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Wall constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
        # Violations: max(0, r - x), max(0, r - (1-x)), etc.
        v1 = np.maximum(0, r - x)
        v2 = np.maximum(0, r - (1 - x))
        v3 = np.maximum(0, r - y)
        v4 = np.maximum(0, r - (1 - y))
        
        wall_penalty = np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2) + np.sum(v4**2)
        
        # Overlap constraints: dist(i,j) >= r_i + r_j
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist_sq = dx**2 + dy**2
        
        r_sum = r[i_idx] + r[j_idx]
        # Violation if dist_sq < r_sum^2, i.e., r_sum^2 - dist_sq > 0
        overlap_viol = np.maximum(0, r_sum**2 - dist_sq)
        overlap_penalty = np.sum(overlap_viol**2)
        
        return wall_penalty + overlap_penalty
    
    def is_valid(circles, tol=1e-8):
        """Check if circles satisfy constraints within tolerance."""
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        if np.any(r < -tol):
            return False
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx**2 + dy**2)
        
        if np.any(dist < (r[i_idx] + r[j_idx]) - tol):
            return False
            
        return True
    
    def refine(circles, max_time=10.0):
        """
        Refine a valid packing by maximizing sum of radii using scipy.optimize.
        Variables: (x_0...x_{N-1}, y_0...y_{N-1}, r_0...r_{N-1})
        """
        current_time = time.time()
        if current_time > deadline:
            return circles
        
        x0 = circles[:, 0].copy()
        y0 = circles[:, 1].copy()
        r0 = circles[:, 2].copy()
        
        x0 = np.clip(x0, r0, 1 - r0)
        y0 = np.clip(y0, r0, 1 - r0)
        
        init_vec = np.concatenate([x0, y0, r0])
        
        def obj(vec):
            x = vec[:N]
            y = vec[N:2*N]
            r = vec[2*N:]
            circles = np.column_stack([x, y, r])
            return penalty(circles)
        
        def gradient(vec):
            # Numerical gradient with central differences
            eps = 1e-6
            grad = np.zeros_like(vec)
            for i in range(len(vec)):
                vec_plus = vec.copy()
                vec_minus = vec.copy()
                vec_plus[i] += eps
                vec_minus[i] -= eps
                grad[i] = (obj(vec_plus) - obj(vec_minus)) / (2 * eps)
            return grad
        
        best_circles = circles.copy()
        best_sum_r = np.sum(circles[:, 2])
        
        # Try a few local optimizations with different settings
        # SLSQP is good for constrained problems, but we use penalty method
        # We can also try L-BFGS-B which is faster for large N
        
        methods = ['SLSQP', 'L-BFGS-B']
        
        for method in methods:
            if time.time() > deadline:
                break
                
            try:
                # Run optimization
                res = minimize(obj, init_vec, method=method, jac=gradient,
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                                        'maxls': 50})
                
                if res.success or res.fun < obj(init_vec):
                    x_opt = res.x[:N]
                    y_opt = res.x[N:2*N]
                    r_opt = res.x[2*N:]
                    
                    # Clip to valid range
                    x_opt = np.clip(x_opt, 0, 1)
                    y_opt = np.clip(y_opt, 0, 1)
                    r_opt = np.clip(r_opt, 1e-6, 0.5)
                    
                    opt_circles = np.column_stack([x_opt, y_opt, r_opt])
                    
                    # Check validity
                    if is_valid(opt_circles):
                        sum_r = np.sum(r_opt)
                        if sum_r > best_sum_r:
                            best_sum_r = sum_r
                            best_circles = opt_circles
                            
            except Exception:
                continue
                
        return best_circles

    # Initializations
    initializations = []
    
    # 1. Grid-based initialization (5x5 grid, take 26)
    side = 5
    rng = np.random.RandomState(42)
    x = (np.arange(N) % side + 0.5 + rng.randn(N) * 0.05) / side
    y = (np.arange(N) // side + 0.5 + rng.randn(N) * 0.05) / side
    r = np.full(N, 0.5 / side * 0.9)
    x = np.clip(x, 0.02, 0.98)
    y = np.clip(y, 0.02, 0.98)
    initializations.append(np.column_stack([x, y, r]))
    
    # 2. Random initialization
    for seed in range(3):
        rng = np.random.RandomState(100 + seed)
        x = rng.rand(N) * 0.8 + 0.1
        y = rng.rand(N) * 0.8 + 0.1
        r = rng.rand(N) * 0.04 + 0.04
        initializations.append(np.column_stack([x, y, r]))
        
    # 3. Spiral initialization
    angles = 2 * np.pi * np.arange(N) / N * 1.618
    x = 0.5 + 0.4 * np.cos(angles)
    y = 0.5 + 0.4 * np.sin(angles)
    r = np.full(N, 0.08)
    x = np.clip(x, 0.02, 0.98)
    y = np.clip(y, 0.02, 0.98)
    initializations.append(np.column_stack([x, y, r]))
    
    best_circles = None
    best_sum_r = 0.0
    
    for init_circles in initializations:
        if time.time() > deadline:
            break
        
        # First, make it valid by shrinking radii if necessary
        x, y, r = init_circles[:, 0].copy(), init_circles[:, 1].copy(), init_circles[:, 2].copy()
        
        # Fix wall violations
        for i in range(N):
            r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
            r[i] = max(r[i], 1e-6)
        
        # Fix overlaps by shrinking radii
        for _ in range(50):
            fixed = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist - 1e-10:
                        overlap = min_dist - dist
                        shrink = overlap / 2 + 0.001
                        r[i] = max(1e-6, r[i] - shrink)
                        r[j] = max(1e-6, r[j] - shrink)
                        fixed = True
            if not fixed:
                break
        
        valid_circles = np.column_stack([x, y, r])
        
        if not is_valid(valid_circles):
            continue
            
        sum_r = np.sum(r)
        if sum_r > best_sum_r:
            best_sum_r = sum_r
            best_circles = valid_circles
            
        # Refine
        refined = refine(valid_circles)
        sum_r_ref = np.sum(refined[:, 2])
        if sum_r_ref > best_sum_r:
            best_sum_r = sum_r_ref
            best_circles = refined
            
        if time.time() > deadline:
            break
            
    # Fallback if no valid packing found
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01])
                    count += 1
        best_circles = np.array(cells)
        
    return best_circles
# EVOLVE-BLOCK-END
