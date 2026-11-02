"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s kill (or 150s)
    
    # Objective: maximize sum of radii
    # Constraints:
    # 1. Circles inside rectangle (0,w) x (0,h) with 2(w+h) <= 4
    # 2. No overlap: dist(i,j) >= r_i + r_j
    
    # We optimize over: positions (x_i, y_i) for i in 0..N-1, radii r_i for i in 0..N-1, and aspect ratio a = h/w
    # w = 2/(1+a), h = 2a/(1+a)
    
    # Penalty method:
    # Minimize: -sum(r_i) + lambda * (overlap_penalty + boundary_penalty)
    
    def objective(vars):
        # vars layout: [x_0, y_0, ..., x_{N-1}, y_{N-1}, r_0, ..., r_{N-1}, a]
        # Total vars: 2*N + N + 1 = 3N + 1
        n = N
        xy = vars[:2*n].reshape(n, 2)
        r = vars[2*n:3*n]
        a = vars[3*n]
        
        # Ensure a is positive
        if a <= 0.01:
            a = 0.01
        if a > 100:
            a = 100
        
        w = 2.0 / (1.0 + a)
        h = 2.0 * a / (1.0 + a)
        
        # Ensure radii positive
        r = np.maximum(r, 0.001)
        
        # Sum of radii (we minimize negative)
        obj = -np.sum(r)
        
        # Boundary penalty: circles must be inside [0,w] x [0,h]
        # dist to left: x_i, right: w - x_i, bottom: y_i, top: h - y_i
        bound_pen = 0.0
        for i in range(n):
            d_left = xy[i, 0] - r[i]
            d_right = w - xy[i, 0] - r[i]
            d_bottom = xy[i, 1] - r[i]
            d_top = h - xy[i, 1] - r[i]
            for d in [d_left, d_right, d_bottom, d_top]:
                if d < 0:
                    bound_pen += d * d
        obj += 100.0 * bound_pen
        
        # Overlap penalty
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist * min_dist:
                    diff = min_dist - np.sqrt(dist_sq)
                    if diff > 0:
                        overlap_pen += diff * diff
        obj += 100.0 * overlap_pen
        
        return obj
    
    def gradient(vars):
        # Numerical gradient for simplicity, or implement analytic
        # For speed, let's use finite differences with small step
        eps = 1e-6
        grad = np.zeros_like(vars)
        n = N
        for k in range(len(vars)):
            old = vars[k]
            vars[k] = old + eps
            f_plus = objective(vars)
            vars[k] = old - eps
            f_minus = objective(vars)
            vars[k] = old
            grad[k] = (f_plus - f_minus) / (2 * eps)
        return grad
    
    best_val = -1e10
    best_vars = None
    
    # Initial guess: grid-like arrangement
    side = int(np.ceil(np.sqrt(N)))
    
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Initialize with grid + noise
        a_init = 1.0 + (np.random.rand() - 0.5) * 0.5  # aspect ratio near 1
        w_init = 2.0 / (1.0 + a_init)
        h_init = 2.0 * a_init / (1.0 + a_init)
        
        x = np.array([(i + 0.5) * w_init / side for i in range(N)])
        y = np.array([(j + 0.5) * h_init / side for j in range(N)])
        
        r_init = 0.5 * min(w_init, h_init) / side * (0.5 + 0.5 * np.random.rand(N))
        
        vars0 = np.concatenate([x, y, r_init, [a_init]])
        
        # Optimization
        try:
            res = minimize(objective, vars0, method='BFGS', jac=gradient, 
                          options={'maxiter': 200, 'gtol': 1e-5})
            
            if res.success:
                # Validate
                v = res.x
                xy = v[:2*N].reshape(N, 2)
                r = v[2*N:3*N]
                a = v[3*N]
                
                if a <= 0 or a > 100:
                    continue
                w = 2.0 / (1.0 + a)
                h = 2.0 * a / (1.0 + a)
                
                # Check validity
                valid = True
                # Boundary check
                for i in range(N):
                    if xy[i,0] - r[i] < -1e-6 or w - xy[i,0] - r[i] < -1e-6:
                        valid = False
                        break
                    if xy[i,1] - r[i] < -1e-6 or h - xy[i,1] - r[i] < -1e-6:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Overlap check
                for i in range(N):
                    for j in range(i+1, N):
                        dx = xy[i,0] - xy[j,0]
                        dy = xy[i,1] - xy[j,1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < r[i] + r[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                val = np.sum(r)
                if val > best_val:
                    best_val = val
                    best_vars = v.copy()
        except:
            pass
    
    # If no valid solution found, return seed-like result
    if best_vars is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Construct output: (x, y, r) for each circle
    xy = best_vars[:2*N].reshape(N, 2)
    r = best_vars[2*N:3*N]
    
    # Ensure radii are positive
    r = np.maximum(r, 1e-4)
    
    return np.column_stack([xy[:, 0], xy[:, 1], r])

# EVOLVE-BLOCK-END
