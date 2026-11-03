import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50
warnings.filterwarnings('ignore')

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|) using discrete convolution."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    if np.all(h < 1e-15):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    # Autoconvolution via discrete convolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _smooth_ratio(h, beta=100.0):
    """Compute a smooth approximation of the ratio using log-sum-exp for the max."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    if np.all(h < 1e-15):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    # Soft-max via log-sum-exp for smoothness
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0
    # log-sum-exp: max + log(sum(exp(x - max)))
    shifted = conv / max_conv
    log_sum_exp = np.log(np.sum(np.exp(shifted - 1.0))) + 1.0
    soft_max = max_conv * np.exp(log_sum_exp)
    if soft_max < 1e-15:
        return 0.0
    return l2_sq / (l1 * soft_max)

def _score_neg_smooth(h, beta=100.0):
    return -_smooth_ratio(h, beta)

def _generate_sidon_set(n, size):
    """Generate a Sidon subset of {0,...,n-1} of given size using greedy."""
    S = []
    sums = set()
    for i in range(n):
        if len(S) >= size:
            break
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
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try various Sidon set sizes
    for size in range(2, 15):
        if time.time() > deadline - 15:
            break
        
        S = _generate_sidon_set(N, size)
        if len(S) < size:
            continue
        
        # Initialize with equal heights at Sidon positions
        h0 = np.zeros(N)
        h0[S] = 1.0 / size
        
        score0 = _compute_ratio(h0)
        if score0 > best_score:
            best_score = score0
            best_h = h0.copy()
        
        # Optimize the heights with L-BFGS-B using smooth surrogate
        if time.time() > deadline - 10:
            continue
        try:
            bounds = [(0, 10)] * N
            # Use smooth objective during optimization
            result = minimize(lambda x: -_smooth_ratio(x, beta=200.0), h0, 
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            h_trial = np.maximum(result.x, 0.0)
            # Only keep nonzero at Sidon positions to maintain structure
            mask = np.zeros(N, dtype=bool)
            mask[S] = True
            h_trial[~mask] = 0.0
            score_trial = _compute_ratio(h_trial)
            if score_trial > best_score:
                best_score = score_trial
                best_h = h_trial.copy()
        except Exception:
            pass
    
    # Also try the original 2-element support as baseline
    if time.time() < deadline - 5:
        h0 = np.zeros(N)
        h0[0] = 2.0/3.0
        h0[1] = 1.0/3.0
        score0 = _compute_ratio(h0)
        if score0 > best_score:
            best_score = score0
            best_h = h0.copy()
    
    # Final polish with tighter tolerance using exact objective
    if time.time() < deadline - 3:
        try:
            result = minimize(_score_neg, best_h, method='L-BFGS-B',
                           bounds=[(0, 10)] * N,
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_ratio(h_trial)
            if score_trial > best_score:
                best_score = score_trial
                best_h = h_trial.copy()
        except Exception:
            pass
    
    # Additional polish: try to refine with even tighter settings
    if time.time() < deadline - 1:
        try:
            result = minimize(_score_neg, best_h, method='L-BFGS-B',
                           bounds=[(0, 10)] * N,
                           options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_ratio(h_trial)
            if score_trial > best_score:
                best_score = score_trial
                best_h = h_trial.copy()
        except Exception:
            pass
    
    return list(best_h)
