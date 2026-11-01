import numpy as np
from scipy.optimize import minimize
import time

N = 400

def _autocorr_fft(x):
    """Compute linear autocorrelation of x using FFT."""
    n_fft = 2 * len(x)
    X = np.fft.fft(x, n=n_fft)
    P = X * np.conj(X)
    R = np.fft.ifft(P).real
    return R[:len(x)]

def _max_sidelobe(x):
    R = _autocorr_fft(x)
    return np.max(np.abs(R[1:]))

def _objective_and_grad(x, beta=50.0):
    """
    Objective function for continuous relaxation.
    Minimizes the LogSumExp of the absolute values of off-peak autocorrelations.
    """
    R = _autocorr_fft(x)
    off_peak = np.abs(R[1:])
    
    m = np.max(off_peak)
    if m < 1e-10:
        return 0.0, np.zeros_like(x)
    
    # Compute LSE
    shifted = beta * (off_peak - m)
    exp_terms = np.exp(shifted)
    sum_exp = np.sum(exp_terms)
    val = m + np.log(sum_exp) / beta
    
    # Gradient
    weights = exp_terms / sum_exp
    signs = np.sign(R[1:])
    S = np.zeros(N)
    S[1:] = weights * signs
    
    grad = np.zeros(N)
    for tau in range(1, N):
        w = S[tau]
        if w == 0:
            continue
        grad[tau:] += w * x[:-tau]
        grad[:N-tau] += w * x[tau:]
        
    return val, grad

def _local_search(h, time_limit=5.0, max_iterations=100000):
    """
    Fast local search using FFT for evaluation.
    """
    start_time = time.time()
    ac = _autocorr_fft(h)
    current_val = np.max(np.abs(ac[1:N]))
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping a random bit
        i = np.random.randint(N)
        old_h_i = h[i]
        h[i] = -old_h_i
        
        ac_new = _autocorr_fft(h)
        new_val = np.max(np.abs(ac_new[1:N]))
        
        if new_val < current_val:
            current_val = new_val
            ac = ac_new
        else:
            # Revert
            h[i] = old_h_i
            # ac remains unchanged (old ac)
            
    return h, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation with multiple starts
    num_continuous_starts = 5
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 10.0:
            break
            
        # Initialize x in [-1, 1]
        # Start near binary to help rounding
        x0 = np.random.choice([-1.0, 1.0], size=N) * 0.9
        
        # Use L-BFGS-B
        try:
            res = minimize(_objective_and_grad, x0, jac=True, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 50, 'ftol': 1e-5, 'gtol': 1e-5})
        except Exception:
            continue
        
        # Round to binary
        h_bin = np.sign(res.x)
        # Handle zeros
        zeros = h_bin == 0
        if np.any(zeros):
            h_bin[zeros] = np.random.choice([-1.0, 1.0], size=np.sum(zeros))
        
        # Evaluate binary solution
        val = _max_sidelobe(h_bin)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
            
        # Refine the best binary solution found so far using local search
        if best_h is not None:
            if time.time() < deadline - 10.0:
                h_ref, val_ref = _local_search(best_h.copy(), time_limit=2.0)
                if val_ref < best_val:
                    best_val = val_ref
                    best_h = h_ref
    
    # Phase 2: Final Discrete Search
    remaining_time = deadline - time.time()
    if remaining_time > 5.0 and best_h is not None:
        # Use the best_h as starting point for a longer local search
        h_final, val_final = _local_search(best_h.copy(), time_limit=remaining_time - 2.0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
