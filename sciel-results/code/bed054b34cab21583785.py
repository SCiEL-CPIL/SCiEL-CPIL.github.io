import numpy as np
from scipy.optimize import minimize
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
    
    # Compute autoconvolution on a fine grid using vectorized approach
    # Support of f is [-1/4, 1/4], so support of f*f is [-1/2, 1/2]
    n_fine = 200  # fine grid points
    t_fine = np.linspace(-0.5, 0.5, n_fine)
    
    x = np.linspace(-0.25, 0.25, N + 1)
    # midpoints of intervals
    x_mid = 0.5 * (x[:-1] + x[1:])
    
    # For each t in t_fine, compute (f*f)(t) = integral f(s)f(t-s) ds
    # f(s) = h[i] for s in [x[i], x[i+1])
    # f(t-s) = h[j] for t-s in [x[j], x[j+1]) => s in [t-x[j+1], t-x[j])
    
    # Vectorized: for each pair (i,j), the contribution to (f*f)(t) is
    # h[i]*h[j] * length of intersection of [x[i],x[i+1]] and [t-x[j+1], t-x[j]]
    
    # For a step function, (f*f)(t) = sum_{i,j} h[i]*h[j] * overlap_length(i,j,t)
    # overlap_length(i,j,t) = max(0, min(x[i+1], t-x[j]) - max(x[i], t-x[j+1]))
    
    # This is piecewise linear in t, so we can compute it on the fine grid
    # For each (i,j) pair, the overlap as function of t is trapezoidal
    
    # Precompute: for pair (i,j), t range where overlap > 0 is
    # t in [x[i] + x[j], x[i+1] + x[j+1]]
    
    # Let's compute using the fine grid directly but vectorized over (i,j)
    # t_fine shape: (n_fine,)
    # x[i] for i in range(N): shape (N,)
    # x[j] for j in range(N): shape (N,)
    
    # t_fine[:, None, None] - x[None, :, None] gives (n_fine, N, 1)
    # t_fine[:, None, None] - x[None, None, :] gives (n_fine, 1, N)
    
    # For pair (i,j): overlap = max(0, min(x[i+1], t - x[j]) - max(x[i], t - x[j+1]))
    
    # Let's do this in chunks to avoid memory issues
    # Actually, N=50, n_fine=200, so 50*50*200 = 500K elements, that's fine
    
    # t - x[j] for all t, j: shape (n_fine, N)
    t_minus_xj = t_fine[:, None] - x[None, :]  # (n_fine, N)
    t_minus_xj1 = t_fine[:, None] - x[1:][None, :]  # (n_fine, N), x[j+1]
    
    # For each i, j: 
    # lo = max(x[i], t - x[j+1])
    # hi = min(x[i+1], t - x[j])
    # overlap = max(0, hi - lo)
    
    # x[i] shape (N,), x[i+1] shape (N,)
    # t_minus_xj1 shape (n_fine, N) -> for each i, this is t - x[j+1]
    # t_minus_xj shape (n_fine, N) -> for each j, this is t - x[j]
    
    # We need for each (i,j): max(x[i], t_minus_xj1[:,j]) and min(x[i+1], t_minus_xj[:,j])
    
    # Let's compute the autoconvolution as:
    # (f*f)(t) = sum_i sum_j h[i]*h[j] * max(0, min(x[i+1], t-x[j]) - max(x[i], t-x[j+1]))
    
    # Vectorize over i: for fixed j, sum_i h[i] * max(0, min(x[i+1], t-x[j]) - max(x[i], t-x[j+1]))
    # This is like a convolution of the step function with itself
    
    # Alternative: use the fact that for step functions, the autoconvolution can be computed
    # by noting it's piecewise linear with breakpoints at x[i]+x[j]
    
    # Let me try a different approach: compute using cumulative sums
    # For step functions on uniform grid, (f*f)(t) can be computed efficiently
    
    # Actually, let's just do the direct computation but vectorized:
    # Create arrays of shape (n_fine, N, N)
    
    # lo[i,j] = max(x[i], t - x[j+1]) for all t,i,j
    # hi[i,j] = min(x[i+1], t - x[j]) for all t,i,j
    
    # x_i = x[:-1] shape (N,)
    # x_i1 = x[1:] shape (N,)
    
    x_i = x[:-1]  # (N,)
    x_i1 = x[1:]  # (N,)
    x_j = x[:-1]  # (N,)
    x_j1 = x[1:]  # (N,)
    
    # t_minus_xj: (n_fine, N) where column j is t - x[j]
    # t_minus_xj1: (n_fine, N) where column j is t - x[j+1]
    
    # For pair (i,j): lo = max(x[i], t - x[j+1])
    # x[i] broadcasts over (n_fine, N_j)
    # t - x[j+1] is t_minus_xj1[:, j]
    
    # lo[i,j,t] = max(x_i[i], t_minus_xj1[t, j])
    # hi[i,j,t] = min(x_i1[i], t_minus_xj[t, j])
    
    # Let's compute this in a memory-efficient way by iterating over i
    autoconv = np.zeros(n_fine)
    for i in range(N):
        if h[i] == 0:
            continue
        # For this i, compute sum_j h[j] * max(0, min(x_i1[i], t-x[j]) - max(x_i[i], t-x[j+1]))
        # lo_j = max(x_i[i], t_minus_xj1[:, j]) shape (n_fine,)
        # hi_j = min(x_i1[i], t_minus_xj[:, j]) shape (n_fine,)
        
        lo = np.maximum(x_i[i], t_minus_xj1)  # (n_fine, N)
        hi = np.minimum(x_i1[i], t_minus_xj)   # (n_fine, N)
        overlap = np.maximum(0, hi - lo)  # (n_fine, N)
        
        # Sum over j: sum_j h[j] * overlap[:,j]
        contrib = overlap @ h  # (n_fine,)
        autoconv += h[i] * contrib
    
    sup_conv = np.max(np.abs(autoconv))
    
    if sup_conv < 1e-15:
        return 0.0
    
    return L2_sq / (L1 * sup_conv)


def _score_neg(h):
    return -_compute_score(h)


def _score_neg_smooth(h, beta=50.0):
    """Smooth surrogate using soft-max for the sup norm."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    if np.sum(h) < 1e-15:
        return 0.0
    
    L1 = np.sum(h) * dw
    L2_sq = np.sum(h**2) * dw
    
    # Compute autoconvolution
    n_fine = 200
    t_fine = np.linspace(-0.5, 0.5, n_fine)
    x = np.linspace(-0.25, 0.25, N + 1)
    x_i = x[:-1]
    x_i1 = x[1:]
    x_j = x[:-1]
    x_j1 = x[1:]
    
    t_minus_xj = t_fine[:, None] - x_j[None, :]
    t_minus_xj1 = t_fine[:, None] - x_j1[None, :]
    
    autoconv = np.zeros(n_fine)
    for i in range(N):
        if h[i] == 0:
            continue
        lo = np.maximum(x_i[i], t_minus_xj1)
        hi = np.minimum(x_i1[i], t_minus_xj)
        overlap = np.maximum(0, hi - lo)
        contrib = overlap @ h
        autoconv += h[i] * contrib
    
    # Soft-max for sup norm: (1/beta) * log(sum(exp(beta * |autoconv|)))
    abs_ac = np.abs(autoconv)
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        return 0.0
    
    # Soft max
    soft_max = max_val + np.log(np.sum(np.exp(beta * (abs_ac - max_val)))) / beta
    
    if soft_max < 1e-15:
        return 0.0
    
    return L2_sq / (L1 * soft_max)


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
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try various Sidon set sizes
    for size in range(2, 14):
        if time.time() > deadline - 15:
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
        
        # Optimize using smooth surrogate first, then exact
        try:
            bounds = [(0, 10)] * N
            # First pass with smooth surrogate
            result = minimize(lambda x: -_score_neg_smooth(x, beta=50.0), h0,
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-10})
            h_trial = np.maximum(result.x, 0.0)
            h_trial[~np.isin(np.arange(N), S)] = 0.0
            
            # Second pass with exact objective
            result2 = minimize(_score_neg, h_trial, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-12})
            h_trial2 = np.maximum(result2.x, 0.0)
            h_trial2[~np.isin(np.arange(N), S)] = 0.0
            
            score_trial = _compute_score(h_trial2)
            if score_trial > best_score:
                best_score = score_trial
                best_h = h_trial2.copy()
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
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        try:
            result = minimize(_score_neg, best_h, method='L-BFGS-B',
                           bounds=[(0, 10)] * N,
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_score = score_trial
                best_h = h_trial.copy()
        except Exception:
            pass
    
    return list(best_h)
