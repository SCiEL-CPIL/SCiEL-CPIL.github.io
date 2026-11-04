import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 140.0  # 10s safety margin before 150s kill

def _compute_ratio(h):
    """Compute ||f||_2^2 / (||f||_1 * ||f*f||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = float(np.sum(h * h))
    l1 = float(np.sum(h))
    if l1 < 1e-15:
        return 0.0
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(F * F.conj(), n=n_fft)
    conv_inf = float(np.max(np.abs(conv)))
    if conv_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def _neg_ratio(params):
    h = np.exp(params)
    return -_compute_ratio(h)

def _get_sidon_set(n, max_val):
    """Generate a Sidon set of size n within [0, max_val]."""
    S = [0]
    sums = set([0])
    for i in range(1, max_val + 1):
        if len(S) >= n:
            break
        new_sums = set()
        valid = True
        for s in S:
            new_sum = i + s
            if new_sum in sums or new_sum in new_sums:
                valid = False
                break
            new_sums.add(new_sum)
        two_i = 2 * i
        if two_i in sums or two_i in new_sums:
            valid = False
        
        if valid:
            S.append(i)
            sums.update(new_sums)
            sums.add(two_i)
    return S

def _rectangular_init(L):
    """Create a rectangular window of length L, L1-normalized."""
    h = np.zeros(N)
    h[:L] = 1.0 / L
    return h

def _dft_init(z):
    """Create h from DFT parameterization: h = |ifft(z)|^2, L1-normalized."""
    h = np.abs(np.fft.irfft(z, n=N)) ** 2
    l1 = np.sum(h)
    if l1 < 1e-15:
        h = np.zeros(N)
        h[0] = 1.0
    else:
        h = h / l1
    return h

def _local_optimize(h0, max_time):
    """Local optimization from h0 using log-parameterization."""
    t_start = time.time()
    h0 = np.asarray(h0, dtype=np.float64)
    # Clip to avoid log(0)
    h0_clipped = np.maximum(h0, 1e-10)
    x0 = np.log(h0_clipped)
    
    best_h = h0.copy()
    best_score = _compute_ratio(h0)
    
    def callback(xk):
        nonlocal best_h, best_score
        h_k = np.exp(xk)
        s_k = _compute_ratio(h_k)
        if s_k > best_score:
            best_score = s_k
            best_h = h_k.copy()
        return time.time() - t_start > max_time
    
    try:
        res = minimize(_neg_ratio, x0, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8,
                               'maxfun': 1000, 'callback': callback})
    except Exception:
        pass
    
    # Try Nelder-Mead as well if time permits
    if time.time() - t_start < max_time - 5:
        try:
            res2 = minimize(_neg_ratio, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'maxfev': 500, 'callback': callback})
        except Exception:
            pass
    
    return best_h, best_score

def construct():
    t_start = time.time()
    max_time = _DEADLINE - (time.time() - t_start)
    
    # Generate initializations
    inits = []
    
    # 1. Sidon set
    sidon_indices = _get_sidon_set(10, N - 1)
    h_sidon = np.zeros(N)
    for idx in sidon_indices:
        h_sidon[idx] = 1.0
    l1 = np.sum(h_sidon)
    if l1 > 0:
        h_sidon = h_sidon / l1
    inits.append(h_sidon)
    
    # 2. Rectangular windows of various lengths
    for L in [5, 8, 10, 12, 15, 20, 25]:
        if L <= N:
            inits.append(_rectangular_init(L))
    
    # 3. DFT-based initializations with random phases
    np.random.seed(42)
    for _ in range(5):
        z = np.random.randn(N) + 1j * np.random.randn(N)
        inits.append(_dft_init(z))
    
    # Evaluate all initializations
    best_h = None
    best_score = 0.0
    for h_init in inits:
        s = _compute_ratio(h_init)
        if s > best_score:
            best_score = s
            best_h = h_init.copy()
    
    # Local optimization from the best initialization
    if time.time() - t_start < _DEADLINE - 10:
        remaining = _DEADLINE - (time.time() - t_start)
        h_opt, s_opt = _local_optimize(best_h, remaining - 5)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Also try local opt from second-best if time permits
    if time.time() - t_start < _DEADLINE - 15:
        # Find second best init
        scores = [(_compute_ratio(h), h) for h in inits]
        scores.sort(key=lambda x: -x[0])
        if len(scores) > 1:
            h2, s2 = scores[1]
            if s2 > best_score * 0.95:  # Only if competitive
                remaining = _DEADLINE - (time.time() - t_start)
                if remaining > 10:
                    h_opt2, s_opt2 = _local_optimize(h2, remaining - 5)
                    if s_opt2 > best_score:
                        best_score = s_opt2
                        best_h = h_opt2
    
    return [float(x) for x in best_h]
