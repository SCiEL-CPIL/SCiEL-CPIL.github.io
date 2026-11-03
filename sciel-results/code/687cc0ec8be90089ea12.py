"""Rewritten program: smooth-annealed gradient search for 50-circle packing."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 50
_T0 = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _pairwise(c):
    # c: (N,2) -> (N, N) distance matrix
    diff = c[:, None, :] - c[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))

def _distances(c):
    d = _pairwise(c)
    # boundary distances: 1/2 - |c_ik|
    b = 0.5 - np.abs(c)
    return d, b

def _softmin(d, tau):
    # d: (N,N) full matrix; take upper triangle only (i<j)
    iu = np.triu_indices(N, k=1)
    vals = d[iu]
    return -logsumexp(-vals / tau) * tau

def _boundary_min(c, tau):
    b = 0.5 - np.abs(c)
    return -logsumexp(-b.ravel() / tau) * tau

def _objective(c, tau):
    d, b = _distances(c)
    # We want to MAXIMIZE min(pairwise, boundary).
    # Use a combined soft-min: min of all pairwise and all boundary distances.
    iu = np.triu_indices(N, k=1)
    all_d = d[iu]
    all_b = b.ravel()
    all_vals = np.concatenate([all_d, all_b])
    # soft-min of all_vals
    return -logsumexp(-all_vals / tau) * tau

def _grad_obj(c, tau):
    # Gradient of -softmin = -d/dc [ -tau * logsumexp(-vals/tau) ]
    # = tau * d/dc logsumexp(-vals/tau)
    # = sum_k p_k * (1) * d(vals_k)/dc  where p_k = softmax(-vals_k/tau)
    # Actually: d/dc softmin = d/dc [-tau logsumexp(-v/tau)]
    #   = -tau * (1/logsumexp) * sum_k (-1/tau) exp(-v_k/tau) dv_k/dc
    #   = sum_k p_k dv_k/dc  where p_k = exp(-v_k/tau)/sum
    # We want gradient of -softmin (since we minimize -softmin) = -sum p_k dv_k/dc
    d, b = _distances(c)
    iu = np.triu_indices(N, k=1)
    all_d = d[iu]
    all_b = b.ravel()
    all_vals = np.concatenate([all_d, all_b])
    p = np.exp(-all_vals / tau)
    p /= p.sum()
    # gradient w.r.t. c
    grad = np.zeros_like(c)
    # pairwise part
    n_pairs = all_d.shape[0]
    pd = p[:n_pairs]
    # for pair (i,j), d_ij = |c_i - c_j|, d/dc_i d_ij = (c_i - c_j)/d_ij
    i_idx = iu[0]
    j_idx = iu[1]
    diff = c[i_idx] - c[j_idx]
    # avoid div by zero
    safe_d = np.where(all_d > 1e-12, all_d, 1e-12)
    unit = diff / safe_d[:, None]
    # contribution to grad[i] += p * unit, grad[j] -= p * unit
    np.add.at(grad, i_idx, pd[:, None] * unit)
    np.add.at(grad, j_idx, -pd[:, None] * unit)
    # boundary part: b = 0.5 - |c|, db/dc = -sign(c)
    pb = p[n_pairs:]
    sign_c = np.sign(c)
    # db/dc = -sign(c), so d(boundary_vals)/dc = -sign(c)
    # gradient of -softmin boundary = -sum pb * (-sign(c)) = sum pb * sign(c)
    # Wait: we're computing gradient of -softmin(all_vals).
    # -softmin = tau * logsumexp(-v/tau)
    # d/dc [-softmin] = tau * (1/sum) sum_k (-1/tau) exp(-v_k/tau) dv_k/dc
    # = -sum_k p_k dv_k/dc
    # For boundary: v_k = 0.5 - |c_ik|, dv/dc_ik = -sign(c_ik)
    # So -sum p_k * (-sign(c_ik)) = sum p_k * sign(c_ik)
    grad += np.sum(pb[:, None], axis=0) * sign_c
    return grad

def _initial_layout():
    # 5 columns x 10 rows
    cols = 5
    rows = 10
    xs = (np.arange(cols) + 0.5) / cols - 0.5
    ys = (np.arange(rows) + 0.5) / rows - 0.5
    grid = np.array([(x, y) for y in ys for x in xs])
    return grid[:N].copy()

def _score(c):
    d, b = _distances(c)
    iu = np.triu_indices(N, k=1)
    min_pair = d[iu].min()
    min_bound = b.min()
    return min(min_pair / 2.0, min_bound)

def construct():
    t_start = time.time()
    c0 = _initial_layout()
    best_c = c0.copy()
    best_score = _score(c0)

    # Annealing schedule: tau from large to small
    tau_schedule = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

    for tau in tau_schedule:
        if time.time() - t_start > _DEADLINE - 10:
            break
        # Run L-BFGS-B
        def obj_func(x):
            return -_objective(x, tau)
        def grad_func(x):
            return -_grad_obj(x, tau)

        try:
            res = minimize(
                obj_func,
                c0.ravel(),
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            c_new = res.x.reshape(N, 2)
            # Clip to stay inside square (safety)
            c_new = np.clip(c_new, -0.5, 0.5)
            s = _score(c_new)
            if s > best_score:
                best_score = s
                best_c = c_new.copy()
                c0 = c_new.copy()
        except Exception:
            pass

    # Final polish: very small tau, more iterations
    if time.time() - t_start < _DEADLINE - 5:
        tau_final = 0.001
        def obj_func(x):
            return -_objective(x, tau_final)
        def grad_func(x):
            return -_grad_obj(x, tau_final)
        try:
            res = minimize(
                obj_func,
                best_c.ravel(),
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            c_new = res.x.reshape(N, 2)
            c_new = np.clip(c_new, -0.5, 0.5)
            s = _score(c_new)
            if s > best_score:
                best_c = c_new.copy()
        except Exception:
            pass

    # Ensure validity: finite, inside square
    best_c = np.clip(best_c, -0.5, 0.5)
    if not np.all(np.isfinite(best_c)):
        best_c = _initial_layout()
    return best_c
