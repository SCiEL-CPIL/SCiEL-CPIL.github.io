"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 150  # seconds
START_TIME = time.time()

def _time_remaining():
    return DEADLINE - (time.time() - START_TIME)

def _autoconvolution(h):
    """Compute autoconvolution of height array h."""
    n = len(h)
    conv = np.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            conv[i + j] += h[i] * h[j]
    return conv

def _autoconvolution_fast(h):
    """Compute autoconvolution using FFT for speed."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    h_padded = np.zeros(size)
    h_padded[:n] = h
    fft_h = np.fft.fft(h_padded)
    conv = np.fft.ifft(fft_h * fft_h).real
    return conv[:2 * n - 1]

def _score(h):
    """Compute the ratio: (sum h^2) / (sum h * max autoconvolution)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        h = np.maximum(h, 0)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    conv = _autoconvolution_fast(h)
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _negative_log_ratio(h):
    """Negative of log of the ratio for minimization."""
    h = np.maximum(h, 0)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 1e10
    conv = _autoconvolution_fast(h)
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 1e10
    ratio = l2_sq / (l1 * sup_conv)
    if ratio <= 0:
        return 1e10
    return -np.log(ratio)

def _find_sidon_set(n, size):
    """Find a Sidon set of given size from {0, 1, ..., n-1}."""
    # Use a greedy approach to find a Sidon set
    # A Sidon set has all pairwise sums distinct
    # Try to find one of the desired size
    
    # Start with a known good Sidon set structure
    # For small sizes, we can use the Bose-Chowla construction or greedy
    
    def is_sidon(s):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                s_val = s[i] + s[j]
                if s_val in sums:
                    return False
                sums.add(s_val)
        return True
    
    # Greedy approach
    sidon = []
    for candidate in range(n):
        trial = sidon + [candidate]
        if is_sidon(trial):
            sidon.append(candidate)
        if len(sidon) == size:
            break
    
    if len(sidon) < size:
        # Fallback: use a simple spread
        sidon = list(range(0, n, n // size))[:size]
    
    return sidon

def _optimize_amplitudes(support_indices, h0, max_iter=200):
    """Optimize amplitudes on given support using gradient descent."""
    n_support = len(support_indices)
    
    # Create initial heights
    h = np.zeros(N)
    for i, idx in enumerate(support_indices):
        h[idx] = h0[i]
    
    best_h = h.copy()
    best_score = _score(h)
    
    # Use scipy minimize with log-barrier for positivity
    def objective(x):
        if _time_remaining() < 5:
            return 1e10
        h_temp = np.zeros(N)
        for i, idx in enumerate(support_indices):
            h_temp[idx] = x[i]
        return _negative_log_ratio(h_temp)
    
    # Initial values
    x0 = np.array([h[idx] for idx in support_indices], dtype=float)
    
    # Bounds: nonnegative
    bounds = [(1e-10, None)] * n_support
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if result.fun < 1e9:
            h_new = np.zeros(N)
            for i, idx in enumerate(support_indices):
                h_new[idx] = max(0, result.x[i])
            new_score = _score(h_new)
            if new_score > best_score:
                best_h = h_new
                best_score = new_score
    except Exception:
        pass
    
    return best_h, best_score

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    global START_TIME
    START_TIME = time.time()
    
    # Find a Sidon set of size 50 from {0, 1, ..., 49}
    # Actually, we need to use all 50 positions, so let's think differently
    # The heights can be zero, so we choose which positions to use
    
    # For N=50, let's try different support sizes
    # A Sidon set of size k from {0,...,49}
    
    best_h = None
    best_score = 0.0
    
    # Try various support sizes
    for k in [15, 20, 25, 30, 35, 40, 45, 50]:
        if _time_remaining() < 10:
            break
        
        # Find a Sidon set of size k
        support = _find_sidon_set(N, k)
        if len(support) < k:
            continue
        
        # Initial amplitudes: equal
        h0 = np.ones(k) / np.sqrt(k)
        
        # Optimize
        h_opt, score = _optimize_amplitudes(support, h0, max_iter=100)
        
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
        
        if _time_remaining() < 15:
            break
    
    # If no good result, fall back to uniform
    if best_h is None or best_score < 0.1:
        best_h = np.ones(N) / N
        best_score = _score(best_h)
    
    # Final polish with more iterations if time allows
    if _time_remaining() > 30 and best_h is not None:
        # Identify the support from best_h
        support = [i for i in range(N) if best_h[i] > 1e-8]
        if len(support) > 0:
            h0 = np.array([best_h[idx] for idx in support])
            h_opt, score = _optimize_amplitudes(support, h0, max_iter=300)
            if score > best_score:
                best_h = h_opt
    
    return list(best_h)

# EVOLVE-BLOCK-START
def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    global START_TIME
    START_TIME = time.time()
    
    best_h = None
    best_score = 0.0
    
    # Try various support sizes with Sidon sets
    for k in [15, 20, 25, 30, 35, 40, 45, 50]:
        if _time_remaining() < 10:
            break
        
        support = _find_sidon_set(N, k)
        if len(support) < k:
            continue
        
        h0 = np.ones(k) / np.sqrt(k)
        h_opt, score = _optimize_amplitudes(support, h0, max_iter=100)
        
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
        
        if _time_remaining() < 15:
            break
    
    if best_h is None or best_score < 0.1:
        best_h = np.ones(N) / N
        best_score = _score(best_h)
    
    if _time_remaining() > 30 and best_h is not None:
        support = [i for i in range(N) if best_h[i] > 1e-8]
        if len(support) > 0:
            h0 = np.array([best_h[idx] for idx in support])
            h_opt, score = _optimize_amplitudes(support, h0, max_iter=300)
            if score > best_score:
                best_h = h_opt
    
    return list(best_h)
# EVOLVE-BLOCK-END
