import numpy as np
from scipy.optimize import minimize
import time

N = 400

def compute_value(h):
    s = np.sum(h)
    if s == 0:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / s**2)

def construct():
    start_time = time.time()
    deadline = 130
    i = np.arange(N)
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Chirp: c + A*cos(2*pi*(alpha*i + beta*i^2)/N + phi)
    def chirp_obj(params):
        c, A, alpha, beta, phi = params
        h = c + A*np.cos(2*np.pi*(alpha*i + beta*i**2)/N + phi)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        return abs(2*N*np.max(conv)/s**2)
    
    best_chirp_val = best_val
    best_chirp_params = None
    
    inits = [
        [0.5, 1, 0, 1, 0],
        [0.5, 1, 0, 2, 0],
        [0.5, 1, 0, 5, 0],
        [0.3, 1, 0, 1, 0],
        [0.5, 1, 0.5, 1, 0],
        [0.5, 1, 0, 1, 1],
        [0.2, 1, 0, 3, 0],
        [0.5, 1, 0, 0.5, 0],
    ]
    
    for init in inits:
        if time.time() > deadline - 40:
            break
        try:
            res = minimize(chirp_obj, init, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9})
            if res.fun < best_chirp_val:
                best_chirp_val = res.fun
                best_chirp_params = res.x
        except:
            pass
    
    if best_chirp_params is not None and best_chirp_val < best_val:
        best_val = best_chirp_val
        c, A, alpha, beta, phi = best_chirp_params
        best_h = c + A*np.cos(2*np.pi*(alpha*i + beta*i**2)/N + phi)
    
    # Random search
    rng = np.random.RandomState(42)
    for _ in range(100):
        if time.time() > deadline - 20:
            break
        h = rng.randn(N)
        s = np.sum(h)
        if abs(s) < 1e-10:
            continue
        conv = np.convolve(h, h)
        val = abs(2*N*np.max(conv)/s**2)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Local optimization
    if time.time() < deadline - 10:
        lr = 0.01
        h_curr = best_h.copy()
        f_curr = compute_value(h_curr)
        for it in range(300):
            if time.time() > deadline - 5:
                break
            d = rng.randn(N)
            d /= np.linalg.norm(d)
            best_t = 0
            best_f = f_curr
            for t in [lr, lr/2, lr/4]:
                h_test = h_curr + t*d
                f_test = compute_value(h_test)
                if f_test < best_f:
                    best_f = f_test
                    best_t = t
            if best_t > 0:
                h_curr = h_curr + best_t*d
                f_curr = best_f
                if f_curr < best_val:
                    best_val = f_curr
                    best_h = h_curr.copy()
            else:
                lr /= 2
                if lr < 1e-9:
                    break
    
    return best_h
