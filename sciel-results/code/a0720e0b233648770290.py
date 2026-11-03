"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _autocorr(s):
    """Compute autocorrelation of binary sequence s using FFT."""
    n = len(s)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(s, n=nfft)
    ac = np.fft.irfft(f * np.conj(f), n=nfft)[:n]
    return ac

def _max_abs_autocorr(s):
    """Maximum absolute value of autocorrelation."""
    ac = _autocorr(s)
    return np.max(np.abs(ac))

def _eval_score(s):
    """Score: max |autoconv| for a = s (unit L2 norm scaling).
    For a = s / ||s||, autoconv = (s*s)/N, so max|autoconv| = max|s*s|/N."""
    ac = _autocorr(s)
    return np.max(np.abs(ac)) / N

def construct():
    deadline = time.time() + 120.0
    rng = np.random.default_rng(42)
    
    # Start with random binary sequence
    best_s = rng.choice([-1, 1], size=N)
    best_score = _eval_score(best_s)
    
    # Try a few structured initializations
    # Alternating
    alt = np.array([1 if i % 2 == 0 else -1 for i in range(N)])
    alt_score = _eval_score(alt)
    if alt_score < best_score:
        best_s = alt.copy()
        best_score = alt_score
    
    # Block pattern
    block = np.zeros(N)
    block[:N//2] = 1
    block[N//2:] = -1
    block_s = np.sign(block)
    block_score = _eval_score(block_s)
    if block_score < best_score:
        best_s = block_s.copy()
        best_score = block_score
    
    # Local search with simulated annealing
    current_s = best_s.copy()
    current_score = best_score
    
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    while time.time() < deadline - 5.0:
        # Pick a random position to flip
        i = rng.integers(0, N)
        old_val = current_s[i]
        current_s[i] = -old_val
        
        new_score = _eval_score(current_s)
        
        if new_score < current_score or rng.random() < np.exp((current_score - new_score) / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_s = current_s.copy()
        else:
            current_s[i] = old_val
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final gradient-based polish on amplitudes given best sign pattern
    # For fixed signs, optimize amplitudes to minimize max|autoconv|
    # Use L-BFGS-B to minimize the max autoconvolution
    s_best = best_s
    
    def objective(a):
        """a are the amplitudes, s_best are the signs. Compute max|autoconv|."""
        x = a * s_best
        ac = _autocorr(x)
        return np.max(np.abs(ac))
    
    # Initial amplitudes: all 1
    a0 = np.ones(N)
    
    # Constraint: L2 norm = 1
    from scipy.optimize import NonlinearConstraint
    constraint = NonlinearConstraint(lambda a: np.sum(a**2), 1.0, 1.0)
    
    try:
        result = minimize(objective, a0, method='L-BFGS-B', 
                         constraints=[constraint],
                         options={'maxiter': 500, 'ftol': 1e-10})
        if result.success or result.fun < best_score:
            a_opt = result.x
            # Normalize to unit L2
            norm = np.sqrt(np.sum(a_opt**2))
            a_opt = a_opt / norm
            final_score = _eval_score(a_opt * s_best)
            if final_score < best_score:
                best_score = final_score
                best_s = a_opt * s_best
    except:
        pass
    
    # Return the optimized sequence
    # The task asks for "signed step heights" - return the amplitudes
    # Ensure the return is a sequence of signed step heights
    return best_s
# EVOLVE-BLOCK-END
