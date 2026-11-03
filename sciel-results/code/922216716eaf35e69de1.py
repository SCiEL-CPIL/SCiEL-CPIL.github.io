import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50
warnings.filterwarnings('ignore')

# Precompute the autoconvolution kernel for step functions
# For step functions on [-1/4, 1/4] with N intervals of width dw=1/N:
# f(x) = h_i for x in [x_i, x_{i+1}) where x_i = -1/4 + i*dw
# (f*f)(t) = sum over i,j of h_i * h_j * overlap_length(i, j, t)
# where overlap is the measure of s such that s in [x_i, x_{i+1}) and t-s in [x_j, x_{j+1})
# This equals the measure of intersection of [x_i, x_{i+1}) and [t - x_{j+1}, t - x_j]

def _precompute_conv_matrix():
    """Precompute the autoconvolution kernel for step functions using vectorized ops."""
    dw = 1.0 / N
    x = np.linspace(-0.25, 0.25, N + 1)
    
    n_pts = 4 * N
    t_grid = np.linspace(-0.5, 0.5, n_pts + 1)
    midpoints = (t_grid[:-1] + t_grid[1:]) / 2.0
    
    # Vectorized: K[i,j,k] = max(0, min(x[i+1], t_k - x[j]) - max(x[i], t_k - x[j+1]))
    # Shape: i (N), j (N), k (n_pts)
    xi = x[:-1][:, None, None]  # (N, 1, 1)
    xi1 = x[1:][:, None, None]  # (N, 1, 1)
    xj = x[:-1][None, :, None]  # (1, N, 1)
    xj1 = x[1:][None, :, None]  # (1, N, 1)
    tk = midpoints[None, None, :]  # (1, 1, n_pts)
    
    lo = np.maximum(xi, tk - xj1)
    hi = np.minimum(xi1, tk - xj)
    K = np.maximum(0.0, hi - lo)
    
    return K, midpoints

_K, _t_mid = _precompute_conv_matrix()

def _compute_score(h):
    """Compute the ratio L2^2 / (L1 * sup|autoconv|) for step heights h."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    if np.sum(h) < 1e-15:
        return 0.0
    
    dw = 1.0 / N
    
    # L1 norm (integral)
    L1 = np.sum(h) * dw
    
    # L2 norm squared (integral of h^2)
    L2_sq = np.sum(h**2) * dw
    
    # Compute autoconvolution using precomputed kernel
    # autoconv[k] = sum_{i,j} h_i * h_j * K[i,j,k]
    # Use matrix multiplication: autoconv = (h @ K_reshaped)
    K_flat = _K.reshape(N * N, -1)
    h_outer = np.outer(h, h).flatten()
    autoconv = h_outer @ K_flat
    
    sup_conv = np.max(np.abs(autoconv))
    
    if sup_conv < 1e-15:
        return 0.0
    
    return L2_sq / (L1 * sup_conv)


def _score_neg(h):
    """Negative score for minimization."""
    return -_compute_score(h)

def _score_neg_smooth(h, beta=100.0):
    """Negative smooth surrogate score using log-sum-exp for the supremum."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    if np.sum(h) < 1e-15:
        return 0.0
    
    dw = 1.0 / N
    L1 = np.sum(h) * dw
    L2_sq = np.sum(h**2) * dw
    
    K_flat = _K.reshape(N * N, -1)
    h_outer = np.outer(h, h).flatten()
    autoconv = h_outer @ K_flat
    
    # Smooth max using log-sum-exp
    abs_ac = np.abs(autoconv)
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        return 0.0
    
    # Shift for numerical stability
    shifted = beta * (abs_ac - max_val)
    log_sum = max_val + np.log(np.sum(np.exp(shifted))) / beta
    
    if log_sum < 1e-15:
        return 0.0
    
    return -(L2_sq / (L1 * log_sum))


def _subgradient_step(h, step_size=0.01):
    """Perform a subgradient step on the non-smooth objective."""
    h = np.maximum(h, 0.0)
    if np.sum(h) < 1e-15:
        return h.copy()
    
    dw = 1.0 / N
    L1 = np.sum(h) * dw
    L2_sq = np.sum(h**2) * dw
    
    K_flat = _K.reshape(N * N, -1)
    h_outer = np.outer(h, h).flatten()
    autoconv = h_outer @ K_flat
    
    abs_ac = np.abs(autoconv)
    sup_conv = np.max(abs_ac)
    if sup_conv < 1e-15:
        return h.copy()
    
    # Subgradient of -L2_sq/(L1 * sup_conv) w.r.t. h
    # d/dh_i [L2_sq/(L1 * sup)] where sup = max_k |autoconv_k|
    # Let S = sup_conv, then obj = L2_sq / (L1 * S)
    # d(obj)/dh_i = [2*h_i*dw * L1*S - L2_sq * (dh_i(L1)*S + L1*dh_i(S))] / (L1*S)^2
    # dh_i(L1) = dw
    # dh_i(S) = sum_k sign(autoconv_k) * d(autoconv_k)/dh_i * 1[k in argmax]
    
    # Find argmax indices
    max_idx = np.where(abs_ac >= sup_conv - 1e-12)[0]
    
    # d(autoconv_k)/dh_i = sum_j h_j * K[i,j,k] + sum_j h_i * K[j,i,k]
    # = 2 * sum_j h_j * K[i,j,k] (since K is symmetric in i,j for autoconv)
    # Actually K[i,j,k] is not necessarily symmetric, but autoconv uses h_i*h_j*K[i,j,k]
    # d(autoconv_k)/dh_i = sum_j (h_j * K[i,j,k] + h_j * K[j,i,k])
    
    # Compute gradient of autoconv w.r.t. h
    # autoconv_k = sum_{i,j} h_i h_j K[i,j,k]
    # d(autoconv_k)/dh_i = sum_j (h_j K[i,j,k] + h_j K[j,i,k])
    
    # Using the precomputed K, we can compute this efficiently
    # Let A[i,j,k] = K[i,j,k]
    # d(autoconv_k)/dh_i = sum_j h_j (K[i,j,k] + K[j,i,k])
    
    # Precompute: for each k, the matrix M_k[i,j] = K[i,j,k] + K[j,i,k]
    # Then d(autoconv_k)/dh_i = (h @ M_k)_i
    
    # But we only need it for k in max_idx
    # Let's compute gradient components
    
    score = L2_sq / (L1 * sup_conv)
    
    # Compute d(autoconv_k)/dh for k in max_idx
    K3 = _K  # (N, N, n_pts)
    
    # For each max index k, compute grad_k[i] = sum_j h_j (K[i,j,k] + K[j,i,k])
    # = sum_j h_j K[i,j,k] + sum_j h_j K[j,i,k]
    # = (h @ K[:,:,k])_i + (K[:,:,k].T @ h)_i
    
    grad_ac = np.zeros((N, len(max_idx)))
    for mi, k in enumerate(max_idx):
        Kk = K3[:, :, k]
        grad_ac[:, mi] = h @ Kk + Kk.T @ h
    
    # dh_i(S) = sum_{k in argmax} sign(autoconv_k) * d(autoconv_k)/dh_i / |argmax|
    # (averaging over argmax for subgradient)
    signs = np.sign(autoconv[max_idx])
    dh_S = (grad_ac * signs[None, :]).sum(axis=1) / len(max_idx)
    
    # d(obj)/dh_i = [2*h_i*dw * L1*S - L2_sq*(dw*S + L1*dh_S_i)] / (L1*S)^2
    # = [2*h_i*dw - L2_sq*dw/(L1*S) - L2_sq*dh_S_i/S] / L1
    # Let's simplify: obj = L2_sq/(L1*S)
    # d(obj)/dh_i = (1/(L1*S)) * d(L2_sq)/dh_i - (L2_sq/(L1*S^2)) * d(S)/dh_i - (L2_sq/(L1^2*S)) * d(L1)/dh_i
    # = (1/(L1*S)) * 2*h_i*dw - (L2_sq/(L1*S^2)) * dh_S_i - (L2_sq/(L1^2*S)) * dw
    
    grad_obj = (2.0 * h * dw) / (L1 * sup_conv) \
             - (L2_sq / (L1 * sup_conv**2)) * dh_S \
             - (L2_sq * dw) / (L1**2 * sup_conv)
    
    # We want to maximize obj, so take step in direction of grad_obj
    # But h must stay nonnegative
    h_new = h + step_size * grad_obj
    h_new = np.maximum(h_new, 0.0)
    
    return h_new


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (150s limit)
    
    # Start from known good solution
    h0 = np.zeros(N)
    h0[0] = 2.0 / 3.0
    h0[1] = 1.0 / 3.0
    
    best_h = h0.copy()
    best_score = _compute_score(best_h)
    
    # Use smooth surrogate for gradient-based optimization
    # This helps L-BFGS-B navigate the non-smooth landscape
    
    # Phase 1: Smooth surrogate optimization from multiple starts
    starts = [
        h0.copy(),
        np.array([1.0/3.0, 1.0/3.0, 1.0/3.0] + [0.0]*(N-3)),
        np.array([0.5, 0.5] + [0.0]*(N-2)),
        np.array([1.0] + [0.0]*(N-1)),
        np.array([0.8, 0.2] + [0.0]*(N-2)),
        np.array([0.6, 0.3, 0.1] + [0.0]*(N-3)),
    ]
    
    for s in starts:
        if time.time() > deadline - 30:
            break
        try:
            # Use smooth objective with moderate beta
            result = minimize(_score_neg_smooth, s, method='L-BFGS-B',
                             bounds=[(0, 10)] * N,
                             args=(50.0,),
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_h = h_trial
                best_score = score_trial
        except Exception:
            pass
    
    # Phase 2: Increase beta for sharper approximation
    for beta in [100.0, 200.0, 500.0, 1000.0]:
        if time.time() > deadline - 20:
            break
        try:
            result = minimize(_score_neg_smooth, best_h, method='L-BFGS-B',
                             bounds=[(0, 10)] * N,
                             args=(beta,),
                             options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-10})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_h = h_trial
                best_score = score_trial
        except Exception:
            pass
    
    # Phase 3: Subgradient steps to navigate non-smooth landscape
    if time.time() < deadline - 15:
        for ss in [0.05, 0.02, 0.01, 0.005, 0.002]:
            if time.time() > deadline - 10:
                break
            for _ in range(200):
                if time.time() > deadline - 10:
                    break
                h_trial = _subgradient_step(best_h, step_size=ss)
                score_trial = _compute_score(h_trial)
                if score_trial > best_score:
                    best_h = h_trial
                    best_score = score_trial
    
    # Phase 4: Direct optimization with exact (non-smooth) objective
    if time.time() < deadline - 10:
        try:
            result = minimize(_score_neg, best_h, method='L-BFGS-B',
                             bounds=[(0, 10)] * N,
                             options={'maxiter': 1000, 'ftol': 1e-13, 'gtol': 1e-10})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_h = h_trial
                best_score = score_trial
        except Exception:
            pass
    
    # Phase 5: Nelder-Mead polish (derivative-free, handles non-smoothness)
    if time.time() < deadline - 5:
        try:
            result = minimize(_score_neg, best_h, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12,
                                     'initial_simplex_size': 1e-4})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_h = h_trial
                best_score = score_trial
        except Exception:
            pass
    
    # Phase 6: Final high-precision polish
    if time.time() < deadline - 2:
        try:
            result = minimize(_score_neg, best_h, method='L-BFGS-B',
                             bounds=[(0, 10)] * N,
                             options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_h = h_trial
                best_score = score_trial
        except Exception:
            pass
    
    # Ensure nonnegativity and clip tiny negatives
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)
