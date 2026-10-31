"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _autoconv_bound(f):
    """Compute max_t |f*f(t)| for a step function f on [-1/4,1/4] with N intervals.
    f is an array of N heights. Interval width h = 1/200.
    The continuous autoconvolution (f*f)(t) = int f(s) f(t-s) ds.
    """
    h = 1.0 / 200.0
    # Pad to length >= 2N for linear convolution
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    F = np.zeros(nfft)
    F[:N] = f
    conv = np.fft.irfft(np.fft.rfft(F) ** 2) * h
    return float(np.max(np.abs(conv)))

def _soft_max_sidelobe(f, temperature=0.1):
    """Smooth surrogate for max |f*f(t)| using log-sum-exp.
    This differentiable approximation helps gradient-based optimization.
    """
    h = 1.0 / 200.0
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    F = np.zeros(nfft)
    F[:N] = f
    conv = np.fft.irfft(np.fft.rfft(F) ** 2) * h
    abs_conv = np.abs(conv)
    # Log-sum-exp for numerical stability
    max_val = np.max(abs_conv)
    if max_val < 1e-10:
        return 0.0
    shifted = abs_conv / temperature
    log_sum_exp = max_val / temperature + np.log(np.sum(np.exp(shifted - max_val / temperature))) * temperature
    return float(log_sum_exp)

def _candidate_m_sequence():
    """Generate an m-sequence of length N."""
    degree = 9  # 2^9 - 1 = 511 >= 400
    state = 1
    seq = []
    for _ in range(2**degree - 1):
        seq.append(1.0 if (state & 1) else -1.0)
        feedback = (state & 1) ^ ((state >> 4) & 1)
        state = (state >> 1) | (feedback << (degree - 1))
    return np.array(seq[:N])

def _candidate_legendre():
    """Legendre sequence based on quadratic residues mod p."""
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    p = N
    while not is_prime(p):
        p += 1
    seq = np.zeros(N)
    for i in range(N):
        if i == 0:
            seq[i] = 1.0
        else:
            if pow(i, (p - 1) // 2, p) == 1:
                seq[i] = 1.0
            else:
                seq[i] = -1.0
    return seq

def _candidate_alternating():
    return np.where(np.arange(N) % 2 == 0, 1.0, -1.0)

def _candidate_random(seed):
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=N)

def _local_search_bit_flip(f, max_sweeps=100):
    """Greedy local search: flip bits to reduce max autoconvolution."""
    f = f.copy()
    best_val = _autoconv_bound(f)
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE:
            break
        improved = False
        indices = np.random.permutation(N)
        for i in indices:
            if time.time() > _DEADLINE:
                break
            f[i] = -f[i]
            val = _autoconv_bound(f)
            if val < best_val:
                best_val = val
                improved = True
            else:
                f[i] = -f[i]
        if not improved:
            break
    return f, best_val

def _gradient_descent_smooth(f_init, max_iter=200):
    """Gradient descent on smooth surrogate, then project back to ±1."""
    f = f_init.astype(float).copy()
    best_f = f.copy()
    best_val = _autoconv_bound(f)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        # Compute gradient of soft-max objective
        # Numerical gradient for simplicity
        eps = 0.01
        grad = np.zeros(N)
        for i in range(N):
            f_plus = f.copy()
            f_plus[i] += eps
            f_minus = f.copy()
            f_minus[i] -= eps
            grad[i] = (_soft_max_sidelobe(f_plus) - _soft_max_sidelobe(f_minus)) / (2 * eps)
        
        # Gradient step
        lr = 0.05
        f -= lr * grad
        
        # Project to ±1 by thresholding
        f_binary = np.where(f >= 0, 1.0, -1.0)
        val = _autoconv_bound(f_binary)
        
        if val < best_val:
            best_val = val
            best_f = f_binary.copy()
            f = f_binary.copy()
    
    return best_f, best_val

def construct():
    # Generate diverse initial candidates
    candidates = [
        _candidate_m_sequence(),
        _candidate_legendre(),
        _candidate_alternating(),
    ]
    # Add random candidates
    for s in range(10):
        candidates.append(_candidate_random(1000 + s))
    
    best = None
    best_val = float('inf')
    
    # First pass: evaluate all candidates and do local search
    for c in candidates:
        if time.time() > _DEADLINE:
            break
        c_opt, val = _local_search_bit_flip(c, max_sweeps=50)
        if val < best_val:
            best_val = val
            best = c_opt
    
    # Second pass: try gradient descent from best candidate
    if best is not None and time.time() < _DEADLINE - 20:
        gd_result, gd_val = _gradient_descent_smooth(best, max_iter=100)
        if gd_val < best_val:
            best_val = gd_val
            best = gd_result
    
    # Final local search polish
    if best is not None and time.time() < _DEADLINE - 5:
        best_final, final_val = _local_search_bit_flip(best, max_sweeps=50)
        if final_val < best_val:
            best_val = final_val
            best = best_final
    
    if best is None:
        return np.ones(N)
    return best
# EVOLVE-BLOCK-END
