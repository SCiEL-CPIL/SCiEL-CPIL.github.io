import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import warnings

N = 50
dw = 1.0 / N
warnings.filterwarnings('ignore')

def _compute_score(h):
    """Compute the ratio L2^2 / (L1 * sup|autoconv|) for step heights h."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    if np.sum(h) < 1e-15:
        return 0.0
    
    # L1 norm (integral)
    L1 = np.sum(h) * dw
    
    # L2 norm squared (integral of h^2)
    L2_sq = np.sum(h**2) * dw
    
    # Compute autoconvolution on a fine grid
    # Support of f is [-1/4, 1/4], so support of f*f is [-1/2, 1/2]
    n_fine = 4 * N
    t_fine = np.linspace(-0.5, 0.5, n_fine)
    
    x = np.linspace(-0.25, 0.25, N + 1)
    
    autoconv = np.zeros(n_fine)
    for idx, t in enumerate(t_fine):
        total = 0.0
        for i in range(N):
            if h[i] == 0:
                continue
            a = x[i]
            b = x[i+1]
            # s in [a, b] and t-s in [-1/4, 1/4] => s in [t-1/4, t+1/4]
            lo = max(a, t - 0.25)
            hi = min(b, t + 0.25)
            if lo >= hi:
                continue
            # For s in [lo, hi], f(s) = h[i], f(t-s) = h[j] where t-s in [x_j, x_{j+1})
            ts_lo = t - hi
            ts_hi = t - lo
            for j in range(N):
                if h[j] == 0:
                    continue
                c = x[j]
                d = x[j+1]
                olo = max(c, ts_lo)
                ohi = min(d, ts_hi)
                if olo < ohi:
                    total += h[i] * h[j] * (ohi - olo)
        autoconv[idx] = total
    
    sup_conv = np.max(np.abs(autoconv))
    
    if sup_conv < 1e-15:
        return 0.0
    
    return L2_sq / (L1 * sup_conv)

def _score_neg(h):
    return -_compute_score(h)

def _generate_sidon_set(n, size):
    """Generate a Sidon subset of {0,...,n-1} of given size using greedy."""
    S = []
    sums = set()
    for i in range(n):
        if len(S) >= size:
            break
        # Check if adding i creates a duplicate sum
        new_sums = set()
        valid = True
        for j in S + [i]:
            for k in S + [i]:
                if j <= k:
                    s = j + k
                    if s in sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
        if valid:
            S.append(i)
            sums |= new_sums
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try various Sidon set sizes
    for size in range(2, 14):
        if time.time() > deadline - 10:
            break
        
        S = _generate_sidon_set(N, size)
        if len(S) < size:
            continue
        
        # Initialize with equal heights
        h0 = np.zeros(N)
        h0[S] = 1.0 / size
        
        score0 = _compute_score(h0)
        if score0 > best_score:
            best_score = score0
            best_h = h0.copy()
        
        # Optimize the heights
        try:
            bounds = [(0, 10)] * N
            result = minimize(_score_neg, h0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-12})
            h_trial = np.maximum(result.x, 0.0)
            # Only keep nonzero at Sidon positions
            h_trial[~np.isin(np.arange(N), S)] = 0.0
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_score = score_trial
                best_h = h_trial.copy()
        except Exception:
            pass
    
    # Also try the original 2-element support
    h0 = np.zeros(N)
    h0[0] = 2.0/3.0
    h0[1] = 1.0/3.0
    score0 = _compute_score(h0)
    if score0 > best_score:
        best_score = score0
        best_h = h0.copy()
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            result = minimize(_score_neg, best_h, method='L-BFGS-B',
                           bounds=[(0, 10)] * N,
                           options={'maxiter': 500, 'ftol': 1e-14})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_score = score_trial
                best_h = h_trial.copy()
        except Exception:
            pass
    
    return list(best_h)
