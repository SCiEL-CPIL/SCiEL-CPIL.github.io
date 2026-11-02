import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 180  # Safety margin
    
    # Grid for evaluation
    x = np.linspace(-10, 10, 2001)
    dx = x[1] - x[0]
    
    # Precompute Hermite polynomials times Gaussian
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 1280x^6 + 4320x^4 - 3360x^2 + 420
    # H12(x) = 4096x^12 - 49152x^10 + 230400x^8 - 403200x^6 + 252000x^4 - 45360x^2 + 10395
    
    # We use physicists' Hermite polynomials
    # Recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
    H = np.zeros((13, len(x)))
    H[0, :] = 1
    H[1, :] = 2 * x
    for n in range(1, 12):
        H[n+1, :] = 2 * x * H[n, :] - 2 * n * H[n, :]
        
    gauss = np.exp(-x**2 / 2)
    # v(x) components
    v0 = H[0, :] * gauss
    v4 = H[4, :] * gauss
    v8 = H[8, :] * gauss
    v12 = H[12, :] * gauss
    
    # Matrix A such that c_full = A @ c_048
    # c12 = (c0 - 3c4 + 105c8) / 10395
    # c0 = c0
    # c4 = c4
    # c8 = c8
    # c12 = ...
    
    # w(x) = A^T v(x)
    # w0(x) = v0(x) + (1/10395) * v12(x)
    # w4(x) = v4(x) - (3/10395) * v12(x)
    # w8(x) = v8(x) + (105/10395) * v12(x)
    
    w0 = v0 + (1/10395) * v12
    w4 = v4 - (3/10395) * v12
    w8 = v8 + (105/10395) * v12
    
    def objective(c):
        c0, c4, c8 = c
        g = c0 * w0 + c4 * w4 + c8 * w8
        max_g = np.max(np.abs(g))
        norm_g_sq = np.sum(g**2) * dx
        if norm_g_sq < 1e-12:
            return 1e12
        return (max_g**2) / norm_g_sq
        
    # Initial guess from seed
    c_seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_c = c_seed
    best_score = objective(c_seed)
    
    # Run optimization
    try:
        res = minimize(objective, c_seed, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.success and res.fun < best_score:
            best_c = res.x
            best_score = res.fun
            
        # Try a few other starting points if time permits
        if time.time() < deadline - 60:
            start_points = [
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 0, 1]),
                np.array([1, 1, 1]),
                np.array([-1, 1, -1]),
            ]
            for sp in start_points:
                if time.time() > deadline - 60:
                    break
                res2 = minimize(objective, sp, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res2.success and res2.fun < best_score:
                    best_c = res2.x
                    best_score = res2.fun
                    
    except Exception as e:
        pass

    return np.array(best_c)
