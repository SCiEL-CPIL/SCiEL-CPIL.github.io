import numpy as np
import time
from scipy.optimize import minimize

N = 600

def compute_autoconv_max(h):
    """
    Compute the maximum of the autoconvolution of h.
    h is a 1D numpy array.
    """
    # Linear autoconvolution
    # Using FFT for efficiency
    n = len(h)
    size = 2 * n - 1
    # Zero pad to avoid circular convolution
    h_fft = np.fft.fft(h, size=size)
    ac_fft = h_fft * np.conjugate(h_fft)
    ac = np.real(np.fft.ifft(ac_fft))
    return np.max(ac)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial guess: Uniform
    h0 = np.ones(N) / N
    
    # Objective function: Max autoconvolution
    # We minimize the log-sum-exp approximation of the max to make it smooth
    def objective(h_flat):
        h = np.maximum(h_flat, 0) # Enforce non-negativity softly or clip
        # Normalize to sum=1
        s = np.sum(h)
        if s == 0:
            return 1e10
        h = h / s
        
        # Compute autoconvolution
        size = 2 * N - 1
        h_fft = np.fft.fft(h, size=size)
        ac = np.real(np.fft.ifft(h_fft * np.conjugate(h_fft)))
        
        # Log-sum-exp for max
        T = 10.0
        max_ac = np.max(ac)
        # If max is very small, log-sum-exp is stable
        # We want to minimize the max value
        # Using log-sum-exp(T * ac) / T
        # To avoid overflow, subtract max
        ac_shifted = ac - max_ac
        lse = np.log(np.sum(np.exp(T * ac_shifted))) / T + max_ac
        return lse

    # Constraints: sum(h) = 1 is handled by normalization inside objective?
    # It's better to use constraints in optimizer if possible, but L-BFGS-B handles bounds.
    # We can just normalize in the objective and use bounds [0, 1].
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Try L-BFGS-B
    best_h = h0
    best_val = compute_autoconv_max(h0)
    
    try:
        # Initial optimization
        res = minimize(
            objective, 
            h0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-6}
        )
        if time.time() < deadline:
            h_opt = np.maximum(res.x, 0)
            s = np.sum(h_opt)
            if s > 0:
                h_opt = h_opt / s
                val = compute_autoconv_max(h_opt)
                if val < best_val:
                    best_h = h_opt
                    best_val = val
        
        # Second round with more iterations if time permits
        if time.time() < deadline - 20:
            res2 = minimize(
                objective, 
                best_h, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            if time.time() < deadline:
                h_opt2 = np.maximum(res2.x, 0)
                s2 = np.sum(h_opt2)
                if s2 > 0:
                    h_opt2 = h_opt2 / s2
                    val2 = compute_autoconv_max(h_opt2)
                    if val2 < best_val:
                        best_h = h_opt2
                        best_val = val2
                        
    except Exception as e:
        pass

    return list(best_h)
