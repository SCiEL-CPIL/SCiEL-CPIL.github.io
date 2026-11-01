import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius(X, Y):
    # X: (N, 3), Y: (M, 3)
    # Compute dot products
    # Clip to [-1, 1] to avoid domain errors in arccos
    dots = np.clip(Y @ X.T, -1.0, 1.0)
    # Max dot product corresponds to min angle
    max_dots = np.max(dots, axis=1)
    angles = np.arccos(max_dots)
    return np.max(angles)

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Initial guess
    X0 = _fibonacci_spiral(N)
    
    # Test points for covering radius estimation
    # Use a dense set of points to approximate the true covering radius
    # 10,000 points is a good balance between accuracy and speed
    M = 10000
    Y = _fibonacci_spiral(M)
    
    def objective(flat_X):
        X = flat_X.reshape(N, 3)
        X = _normalize(X)
        return _covering_radius(X, Y)
    
    best_X = X0
    best_val = objective(X0.flatten())
    
    # Try Nelder-Mead optimization
    # We optimize the 150 coordinates
    # Nelder-Mead is robust to non-smoothness
    try:
        res = minimize(
            objective, 
            X0.flatten(), 
            method='Nelder-Mead',
            options={
                'xatol': 1e-8,
                'fatol': 1e-8,
                'maxiter': 2000,
                'maxfev': 2000,
                'adaptive': True
            }
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_X = _normalize(res.x.reshape(N, 3))
            
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < timeout - 10:
        # Try a second round with a different initial guess or perturbed start
        # Perturb the best found solution
        noise = np.random.normal(0, 0.01, (N, 3))
        X1 = _normalize(best_X + noise)
        
        def objective2(flat_X):
            X = flat_X.reshape(N, 3)
            X = _normalize(X)
            return _covering_radius(X, Y)
        
        try:
            res2 = minimize(
                objective2, 
                X1.flatten(), 
                method='Nelder-Mead',
                options={
                    'xatol': 1e-9,
                    'fatol': 1e-9,
                    'maxiter': 2000,
                    'maxfev': 2000,
                    'adaptive': True
                }
            )
            
            if res2.fun < best_val:
                best_val = res2.fun
                best_X = _normalize(res2.x.reshape(N, 3))
                
        except Exception:
            pass
            
    return best_X
