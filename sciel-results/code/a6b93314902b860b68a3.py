import numpy as np
from scipy.optimize import minimize
import time

N = 26

def objective_and_grad(params, lam_wall=1000.0, lam_overlap=1000.0):
    # params: array of shape (3*N,), [x0, y0, r0, x1, y1, r1, ...]
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum(r))/dr
    
    # Wall penalties
    # For each circle i
    # P_i = max(0, r_i - x_i)^2 + max(0, r_i - (1-x_i))^2 + max(0, r_i - y_i)^2 + max(0, r_i - (1-y_i))^2
    for i in range(N):
        # x terms
        v1 = max(0.0, r[i] - x[i])
        v2 = max(0.0, r[i] - (1.0 - x[i]))
        # y terms
        v3 = max(0.0, r[i] - y[i])
        v4 = max(0.0, r[i] - (1.0 - y[i]))
        
        val += lam_wall * (v1**2 + v2**2 + v3**2 + v4**2)
        
        # Grads for x[i]
        # d/dx (r-x)^2 = 2(r-x)(-1) = -2(r-x)
        # d/dx (r-(1-x))^2 = 2(r-(1-x))(1) = 2(r-1+x)
        grad[3*i + 0] += lam_wall * (-2.0 * v1 + 2.0 * v2)
        
        # Grads for y[i]
        grad[3*i + 1] += lam_wall * (-2.0 * v3 + 2.0 * v4)
        
        # Grads for r[i]
        # d/dr (r-x)^2 = 2(r-x)
        grad[3*i + 2] += lam_wall * (2.0 * v1 + 2.0 * v2 + 2.0 * v3 + 2.0 * v4)

    # Overlap penalties
    # P_ij = max(0, (r_i + r_j)^2 - d_ij^2)^2
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx*dx + dy*dy
            rs = r[i] + r[j]
            rs2 = rs*rs
            u = rs2 - d2
            if u > 0:
                p = u * u
                val += lam_overlap * p
                
                # Grads
                # dp/dr_i = 2 * u * 2 * rs
                # dp/dx_i = 2 * u * (-2 * dx)
                # dp/dy_i = 2 * u * (-2 * dy)
                
                factor_r = lam_overlap * 4.0 * u * rs
                factor_d = lam_overlap * 4.0 * u
                
                grad[3*i + 2] += factor_r
                grad[3*j + 2] += factor_r
                
                grad[3*i + 0] += factor_d * (-dx)
                grad[3*j + 0] += factor_d * (dx)
                
                grad[3*i + 1] += factor_d * (-dy)
                grad[3*j + 1] += factor_d * (dy)

    return val, grad

def is_valid(circles, tol=1e-5):
    # circles: (N, 3) array of [x, y, r]
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check square
    if np.any(x < -tol) or np.any(x > 1 + tol):
        return False
    if np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(r < -tol):
        return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
        
    # Check overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx*dx + dy*dy
            rs = r[i] + r[j]
            if d2 < rs*rs - tol:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_val = -np.inf
    best_circles = None
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Try multiple random starts
    np.random.seed(42) # Reproducibility for debugging, but maybe random is better? 
    # Let's use random seed each time? No, deterministic is safer for evaluation if it fails.
    # But we want to explore. Let's use a fixed seed for the first few, then random?
    # Actually, the evaluator might run this multiple times. 
    # Let's just use random initialization.
    
    n_starts = 20
    
    for k in range(n_starts):
        if time.time() > deadline:
            break
            
        # Random initialization
        x0 = np.random.uniform(0.1, 0.9, N)
        y0 = np.random.uniform(0.1, 0.9, N)
        r0 = np.random.uniform(0.01, 0.1, N)
        
        params0 = np.zeros(3 * N)
        params0[0::3] = x0
        params0[1::3] = y0
        params0[2::3] = r0
        
        try:
            res = minimize(
                lambda p: objective_and_grad(p)[0],
                params0,
                jac=lambda p: objective_and_grad(p)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
        except Exception:
            continue
            
        if res.success:
            params = res.x
            circles = params.reshape(N, 3)
            
            # Check validity
            if is_valid(circles):
                val = np.sum(circles[:, 2])
                if val > best_val:
                    best_val = val
                    best_circles = circles.copy()
        
        # Time check
        if time.time() > deadline - 20.0:
             break

    # If no valid result found, return a trivial valid one (grid)
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
        
    return best_circles
