"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    N = len(h)
    one_minus_h = 1.0 - h
    # Use FFT for circular cross-correlation
    H = np.fft.fft(h)
    O = np.fft.fft(one_minus_h)
    # overlap[k] = sum_i h[i] * one_minus_h[(i+k) % N]
    # This is ifft(fft(h) * conj(fft(one_minus_h)))?
    # Let's verify with a small example.
    # h = [1, 0], one_minus_h = [0, 1]
    # overlap[0] = 1*0 + 0*1 = 0
    # overlap[1] = 1*1 + 0*0 = 1
    # fft([1,0]) = [1, 1]
    # fft([0,1]) = [1, -1]
    # conj(fft(one_minus_h)) = [1, -1]
    # product = [1, -1]
    # ifft([1, -1]) = [0, 1]
    # So ifft(fft(h) * conj(fft(one_minus_h))) gives [0, 1].
    # overlap[0]=0, overlap[1]=1. Correct.
    overlaps = np.real(np.fft.ifft(H * np.conj(O)))
    return np.max(overlaps)

def _log_sum_exp_overlap(h, beta=10.0):
    """Smooth approximation of max overlap for gradient-based optimization."""
    N = len(h)
    one_minus_h = 1.0 - h
    H = np.fft.fft(h)
    O = np.fft.fft(one_minus_h)
    overlaps = np.real(np.fft.ifft(H * np.conj(O)))
    # log-sum-exp: log(sum(exp(beta * x))) / beta
    # To avoid overflow, subtract max
    max_val = np.max(overlaps)
    if max_val > 1e6:
        return max_val
    lse = np.log(np.sum(np.exp(beta * (overlaps - max_val)))) + beta * max_val
    return lse / beta

def _project(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    # Simple projection: scale and clip, then adjust
    # This is a heuristic, SLSQP handles constraints directly
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Start with the known optimal constant solution
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Try SLSQP from a few random starts
    np.random.seed(42)
    
    for attempt in range(5):
        if time.time() > deadline - 20:
            break
            
        # Random initialization
        h0 = np.random.rand(N)
        # Scale to sum N/2
        h0 = h0 * (N / 2.0) / np.sum(h0)
        h0 = np.clip(h0, 0, 1)
        # Adjust sum exactly
        diff = N / 2.0 - np.sum(h0)
        h0[0] += diff
        h0 = np.clip(h0, 0, 1)
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0},
        ]
        bounds = [(0, 1)] * N
        
        try:
            res = minimize(
                _log_sum_exp_overlap, 
                h0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            if res.success:
                h_opt = res.x
                # Ensure exact sum
                diff = N / 2.0 - np.sum(h_opt)
                # Adjust the element with most room
                if diff > 0:
                    idx = np.argmin(1.0 - h_opt)
                    h_opt[idx] += diff
                else:
                    idx = np.argmax(h_opt)
                    h_opt[idx] += diff
                h_opt = np.clip(h_opt, 0, 1)
                
                score = _max_overlap(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt
        except Exception:
            pass

    # Final check
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.argmin(1.0 - best_h)
            best_h[idx] += diff
        else:
            idx = np.argmax(best_h)
            best_h[idx] += diff
        best_h = np.clip(best_h, 0, 1)

    return best_h
# EVOLVE-BLOCK-END
