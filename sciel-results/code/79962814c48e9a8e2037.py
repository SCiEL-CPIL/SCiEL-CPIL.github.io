import numpy as np
from scipy.optimize import minimize
import time

N = 400

def autoconv(x):
    # Efficient autoconvolution using FFT
    # c(k) = sum x_t x_{t+k}
    # Using FFT: IFFT(FFT(x)^2) gives circular autoconvolution.
    # We need linear autoconvolution.
    # Pad x to 2N
    x_padded = np.zeros(2 * N)
    x_padded[:N] = x
    X = np.fft.fft(x_padded)
    C = np.fft.ifft(X * X)
    # C[0:N] are c(0), c(1), ..., c(N-1)
    # C[N:2N] are c(-N+1), ..., c(-1)
    # We want max absolute value for k != 0
    # c(0) is C[0]. c(k) for k>0 is C[k]. c(k) for k<0 is C[N+k].
    # Due to symmetry for real x, C[k] = C[N-k] (conjugate, but real so equal).
    # So we just need to check C[1:N].
    return C.real

def score_continuous(x, alpha=10.0, eps=1e-6):
    c = autoconv(x)
    S = np.sum(x)
    S_sq = S * S + eps
    
    # Get coefficients for k != 0
    # c[1:N] corresponds to lags 1 to N-1
    c_off = c[1:N]
    
    # Soft max of absolute values
    abs_c = np.abs(c_off)
    # To avoid overflow in exp
    max_val = np.max(abs_c)
    if max_val > 100:
        # Scale down
        soft_max = max_val + np.log(np.sum(np.exp((abs_c - max_val) * alpha)) / alpha)
    else:
        soft_max = np.log(np.sum(np.exp(abs_c * alpha)) / alpha)
        
    return soft_max / S_sq

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Initial guess: random
    x0 = np.random.uniform(-1, 1, N)
    
    best_score = np.inf
    best_x = None
    
    # Try a few random restarts
    for i in range(5):
        if time.time() > deadline - 20:
            break
            
        x0 = np.random.uniform(-1, 1, N)
        
        # Use L-BFGS-B
        # Bounds: [-1, 1] for each x
        bounds = [(-1, 1)] * N
        
        try:
            res = minimize(
                score_continuous, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 2000}
            )
            
            if res.success or res.fun < best_score:
                # Project to +/- 1
                x_int = np.sign(res.x)
                # Handle zeros
                x_int[x_int == 0] = 1
                
                # Evaluate discrete score
                c = autoconv(x_int)
                S = np.sum(x_int)
                if S == 0:
                    # Avoid division by zero, penalize heavily
                    val = 1e9
                else:
                    c_off = c[1:N]
                    max_off = np.max(np.abs(c_off))
                    val = max_off / (S * S)
                
                if val < best_score:
                    best_score = val
                    best_x = x_int.copy()
                    
        except Exception:
            pass

    # If no optimization succeeded or time is running out, fall back to seed
    if best_x is None:
        return np.ones(N)
        
    return best_x
