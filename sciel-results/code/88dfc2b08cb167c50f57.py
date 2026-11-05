import numpy as np
import scipy.optimize
import time

N = 400

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100s buffer
    
    # Initial guess
    h = np.ones(N)
    # Add small noise to break symmetry
    np.random.seed(42)
    h += 0.01 * np.random.randn(N)
    
    def objective(h_vec):
        # Compute convolution using FFT
        M = 2 * N
        H = np.fft.fft(h_vec, M)
        C = np.fft.ifft(H * H, M)
        # C is complex due to numerical errors, take real part
        C = np.real(C)
        # The full convolution has length 2N-1. C has length M=2N.
        # The last element C[-1] is effectively 0 (or wrap-around error).
        # We take max of the first 2N-1 elements.
        return np.max(C[:2*N-1])
        
    best_h = h
    best_val = objective(h)
    
    # Try optimization
    try:
        res = scipy.optimize.minimize(
            objective, 
            h, 
            method='L-BFGS-B',
            options={'maxiter': 50, 'maxfun': 100}
        )
        if res.fun < best_val:
            best_h = res.x
            best_val = res.fun
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline:
        return best_h
        
    # Try a few random restarts if time permits
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        h_init = np.random.randn(N)
        # Normalize to avoid sum=0 issues during optimization?
        # Actually, objective is scale invariant in terms of relative flatness,
        # but magnitude matters for the gradient.
        # Let's just run it.
        try:
            res = scipy.optimize.minimize(
                objective, 
                h_init, 
                method='L-BFGS-B',
                options={'maxiter': 20, 'maxfun': 50}
            )
            if res.fun < best_val:
                best_h = res.x
                best_val = res.fun
        except Exception:
            pass
            
    return best_h
