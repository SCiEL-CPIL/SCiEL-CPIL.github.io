# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DL = 600.0
_MARGIN = 25.0


def _deadline_ok():
    return (time.time() - _T0) < (_DL - _MARGIN)


def _keich(N):
    pos = int(round(np.log2(N)))
    j_terms = np.arange(1, pos + 1, dtype=float)
    out = np.empty(N, dtype=np.float64)
    for idx in range(N):
        bits = [int(b) for b in bin(idx)[2:]]
        bits = [0] * (pos - len(bits)) + bits
        out[idx] = np.sum((1.0 - j_terms) * np.array(bits) * (2.0 ** -j_terms) / pos)
    return out


def _keich_positions(N):
    keich = _keich(N)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    base_x = keich[i_grid.flatten()]
    base_y = keich[j_grid.flatten()]
    return np.stack([base_x, base_y], axis=1)


def _union_length_1d(p, N, c_vals):
    """
    Compute the length of the union of intervals [p_i + c*i/N, p_i + c*i/N + 1/N]
    for a given set of c values.
    p: array of shape (N,)
    N: grid size
    c_vals: array of shape (M,)
    Returns: array of shape (M,) with union lengths.
    """
    M = len(c_vals)
    # L[i, k] = p[i] + c[k] * i / N
    # We want union length for each k.
    # N is small (8), M can be large.
    # Vectorized approach:
    # Sort intervals for each c? 
    # Alternative: Since N is small, we can just compute the union length by 
    # checking coverage on a fine grid? No, that's what we want to avoid.
    # Let's use the sweep line algorithm vectorized over c.
    
    # L[i, k] and R[i, k]
    slopes = np.arange(N) / N
    # p is (N,), c_vals is (M,)
    # L[i, k] = p[i] + c_vals[k] * slopes[i]
    L = p[:, None] + c_vals[None, :] * slopes[:, None]
    R = L + 1.0 / N
    
    # For each k (column), we have N intervals [L[i,k], R[i,k]].
    # We need the length of their union.
    # Since N is small, we can sort them.
    # But sorting along an axis in numpy is easy.
    
    # Sort L and R along axis 0 (for each k)
    # np.argsort(L, axis=0) gives indices.
    order = np.argsort(L, axis=0)
    Ls = L[order, np.arange(M)[None, :]] # Gather Ls
    Rs = R[order, np.arange(M)[None, :]] # Gather Rs
    
    # Now merge intervals for each k.
    # This is hard to vectorize perfectly with variable merging, but N is small.
    # We can iterate through the sorted intervals.
    # lengths = np.zeros(M)
    # cur_l = Ls[0]
    # cur_r = Rs[0]
    # for i in range(1, N):
    #    if Ls[i] <= cur_r:
    #       cur_r = np.maximum(cur_r, Rs[i])
    #    else:
    #       lengths += cur_r - cur_l
    #       cur_l = Ls[i]
    #       cur_r = Rs[i]
    # lengths += cur_r - cur_l
    
    # Let's implement this loop. It's fast for N=8.
    lengths = np.zeros(M)
    cur_l = Ls[0].copy()
    cur_r = Rs[0].copy()
    for i in range(1, N):
        # If they overlap
        mask = Ls[i] <= cur_r
        # Update cur_r where they overlap
        cur_r = np.where(mask, np.maximum(cur_r, Rs[i]), cur_r)
        # Where they don't overlap, add the current segment and start new
        lengths += np.where(mask, 0.0, cur_r - cur_l)
        cur_l = np.where(mask, cur_l, Ls[i])
        cur_r = np.where(mask, cur_r, Rs[i])
    lengths += cur_r - cur_l
    return lengths


def _objective(p, N, c_vals):
    """
    Objective: Integral of L(c)^2 dc.
    """
    L = _union_length_1d(p, N, c_vals)
    # Trapezoidal rule
    return np.trapz(L ** 2, c_vals)


def _mc_volume(pos, N, seed=12345):
    """Exact Monte Carlo estimate used by the evaluator (fixed sample, seed 12345)."""
    a_ = pos[:, 0]
    b_ = pos[:, 1]
    i_idx = np.arange(N * N) // N
    j_idx = np.arange(N * N) % N
    X0 = np.min(a_)
    X1 = np.max(a_ + (i_idx + 1) / N)
    Y0 = np.min(b_)
    Y1 = np.max(b_ + (j_idx + 1) / N)
    if not (np.isfinite(X0) and np.isfinite(X1) and np.isfinite(Y0) and np.isfinite(Y1)):
        return np.nan
    area = (X1 - X0) * (Y1 - Y0)
    if area <= 1e-9:
        return np.nan
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, 500000)
    b = rng.uniform(Y0, Y1, 500000)
    c = rng.uniform(0.0, 1.0, 500000)
    a2 = a[:, None]
    b2 = b[:, None]
    c2 = c[:, None]
    Lx = a_[None, :] + c2 * (i_idx[None, :] / N)
    Rx = Lx + 1.0 / N
    Ly = b_[None, :] + c2 * (j_idx[None, :] / N)
    Ry = Ly + 1.0 / N
    inside = (a2 >= Lx) & (a2 < Rx) & (b2 >= Ly) & (b2 < Ry)
    frac = inside.any(axis=1).mean()
    return frac * area


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    # Initial guess from Keich
    p_keich = _keich(N)
    
    # Grid for integration
    M = 4096
    c_vals = np.linspace(0.0, 1.0, M)
    
    best_p = p_keich.copy()
    best_obj = _objective(best_p, N, c_vals)
    
    # Optimization
    # Bounds: The positions can be anywhere, but to keep the box small, 
    # they should be within [0, 1] roughly. 
    # Actually, the relative positions matter. 
    # Let's fix p[0] = 0 to remove translation invariance?
    # No, the box X0, X1 depends on absolute positions.
    # But the volume is translation invariant.
    # However, the MC estimator uses the box defined by the points.
    # So we can optimize freely.
    
    # We will optimize p directly.
    # To avoid drift, we can center p or just let it go.
    # L-BFGS-B with bounds is safer.
    # Let's use bounds [-1, 2] to allow some flexibility.
    bounds = [(-1.0, 2.0)] * N
    
    # Try a few seeds
    for seed in (0, 1, 2, 3, 4, 5, 6, 7):
        if not _deadline_ok():
            break
        rng = np.random.default_rng(seed)
        p0 = p_keich + rng.uniform(-0.1, 0.1, N)
        
        try:
            res = minimize(
                _objective, 
                p0, 
                args=(N, c_vals),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success or res.fun < best_obj:
                if np.all(np.isfinite(res.x)):
                    # Check if this is better
                    obj_val = res.fun
                    if obj_val < best_obj:
                        best_obj = obj_val
                        best_p = res.x.copy()
        except Exception:
            pass
    
    # Construct final positions
    # X_i = best_p[i], Y_j = best_p[j]
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    base_x = best_p[i_grid.flatten()]
    base_y = best_p[j_grid.flatten()]
    positions = np.stack([base_x, base_y], axis=1)
    
    # Verify with MC
    v = _mc_volume(positions, N)
    if not np.isfinite(v) or v <= 0:
        # Fallback to Keich
        return _keich_positions(N)
        
    return positions

# EVOLVE-BLOCK-END
