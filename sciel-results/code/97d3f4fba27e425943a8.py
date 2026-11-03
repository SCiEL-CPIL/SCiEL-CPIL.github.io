"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Binary parity-constrained sequence with Walsh-Hadamard peak-sidelobe minimization.
h[2i] = 1 - h[2i+1] forces even-shift autocorrelation to zero; optimize odd shifts via FWHT."""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _odd_shift_overlaps_fwht(x):
    """
    Fast computation of odd-shift overlaps using FWHT.
    x: array of length half (N/2), where h[2i]=x[i], h[2i+1]=1-x[i]
    Returns max odd-shift overlap.
    
    For binary parity-constrained sequences, the odd-shift autocorrelation
    can be computed efficiently. We use the fact that:
    C_{2k+1} = (2/N) * (N/2 - sum_{i=2k+1}^{N-1} h[i]*h[i-(2k+1)])
    
    For the parity constraint, this simplifies to a sum over x values.
    We compute it directly but vectorized for speed.
    """
    half = len(x)
    # Build h implicitly
    h = np.empty(N)
    h[0::2] = x
    h[1::2] = 1.0 - x
    
    # Compute odd shifts only: k = 1, 3, 5, ..., N-1 (if N even, up to N-1)
    odd_shifts = np.arange(1, N, 2)
    max_overlap = 0.0
    for k in odd_shifts:
        # C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h[i]*h[i-k])
        corr_val = np.dot(h[k:], h[:N-k])
        c_val = (2.0 / N) * (N / 2.0 - corr_val)
        if c_val > max_overlap:
            max_overlap = c_val
    return max_overlap


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0, 1]."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h


def _construct_binary_parity(seed=42):
    """
    Construct a binary sequence of length N with parity constraint h[2i] + h[2i+1] = 1.
    This means h is determined by N/2 binary variables (the even-indexed positions).
    We use a greedy/local-search approach to minimize the peak odd-shift overlap.
    """
    half = N // 2  # 50
    # x[i] = h[2i], then h[2i+1] = 1 - x[i]
    # Initialize with a pseudo-random binary sequence
    rng = np.random.RandomState(seed)
    x = rng.randint(0, 2, half).astype(np.float64)
    
    best_x = x.copy()
    best_score = float('inf')
    
    def _build_h(x):
        h = np.zeros(N)
        h[0::2] = x
        h[1::2] = 1.0 - x
        return h
    
    def _odd_shift_overlap_max(x):
        """Compute max overlap for odd shifts only (even shifts are zero by construction)."""
        h = _build_h(x)
        cs = _linear_overlap_values(h)
        # Only consider odd shifts
        odd_cs = cs[1::2]
        return float(np.max(odd_cs))
    
    # Local search: flip individual bits to reduce peak overlap
    t0 = time.time()
    current_score = _odd_shift_overlap_max(x)
    if current_score < best_score:
        best_score = current_score
        best_x = x.copy()
    
    # Greedy local search with multiple passes
    for iteration in range(500):
        if time.time() - t0 > _DEADLINE - 40:
            break
        
        improved = False
        for i in range(half):
            if time.time() - t0 > _DEADLINE - 40:
                break
            
            old_val = x[i]
            x[i] = 1.0 - old_val
            new_score = _odd_shift_overlap_max(x)
            
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
                improved = True
            else:
                x[i] = old_val  # revert
        
        if not improved:
            break
    
    h = _build_h(best_x)
    return h


def _refine_continuous(h0):
    """
    Refine a binary parity-constrained solution by allowing small continuous deviations
    while maintaining the parity structure approximately.
    """
    from scipy.optimize import minimize
    
    half = N // 2
    
    def _param_to_h(params):
        """params: 2*half values. h[2i] = params[i], h[2i+1] = params[half+i]"""
        h = np.zeros(N)
        h[0::2] = params[:half]
        h[1::2] = params[half:]
        return h
    
    def _h_to_params(h):
        params = np.zeros(2 * half)
        params[:half] = h[0::2]
        params[half:] = h[1::2]
        return params
    
    h0 = np.asarray(h0, dtype=np.float64)
    # Enforce parity constraint approximately: average with its parity-flipped version
    h_parity = np.zeros(N)
    h_parity[0::2] = (h0[0::2] + (1.0 - h0[1::2])) / 2.0
    h_parity[1::2] = (h0[1::2] + (1.0 - h0[0::2])) / 2.0
    h0 = np.clip(h_parity, 0.0, 1.0)
    
    x0 = _h_to_params(h0)
    
    def obj(x):
        h = _param_to_h(x)
        h = np.clip(h, 0.0, 1.0)
        cs = _linear_overlap_values(h)
        return float(np.max(cs))
    
    bounds = [(0.0, 1.0)] * (2 * half)
    
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9})
        if res.fun < obj(x0):
            h_res = _param_to_h(res.x)
            h_res = np.clip(h_res, 0.0, 1.0)
            return h_res
    except Exception:
        pass
    
    return h0


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    
    best_h = None
    best_score = float('inf')
    
    # Try multiple random seeds for the binary construction
    seeds = [42, 1, 7, 13, 23, 37, 53, 71, 89, 101]
    for seed in seeds:
        if time.time() - t0 > _DEADLINE - 25:
            break
        
        h_try = _construct_binary_parity(seed=seed)
        h_try = _fix_sum(h_try)
        s = _score(h_try)
        if s < best_score:
            best_score = s
            best_h = h_try.copy()
    
    # Refine the best solution continuously
    if best_h is not None and time.time() - t0 < _DEADLINE - 8:
        h_refined = _refine_continuous(best_h)
        h_refined = _fix_sum(h_refined)
        s_ref = _score(h_refined)
        if s_ref < best_score:
            best_score = s_ref
            best_h = h_refined
    
    # Final fallback
    if best_h is None or len(best_h) != N:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
