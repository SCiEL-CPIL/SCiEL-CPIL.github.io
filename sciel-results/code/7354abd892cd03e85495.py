# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _union_len_at_c(motion, base, c, w=1.0/8.0):
    """Compute union length at a specific c value."""
    K = len(base)
    lo = base + c * motion
    hi = lo + w
    order = np.argsort(lo)
    s_lo = lo[order]
    s_hi = hi[order]
    
    L = 0.0
    cur_max_hi = -np.inf
    for t in range(K):
        if s_hi[t] > cur_max_hi:
            start = max(s_lo[t], cur_max_hi)
            L += s_hi[t] - start
            cur_max_hi = s_hi[t]
    return L


def _exact_union_len_sq(motion, base, w=1.0/8.0):
    """
    Exact integral of squared 1D union length over c in [0,1].
    Uses piecewise linear approximation of L(c) with exact breakpoints.
    """
    K = len(base)
    if K == 0:
        return 0.0
    
    # Find all breakpoints where the combinatorial structure changes
    breakpoints = {0.0, 1.0}
    for a in range(K):
        for b in range(a + 1, K):
            dm = motion[a] - motion[b]
            if abs(dm) > 1e-15:
                # lo_a = hi_b
                c1 = (base[b] + w - base[a]) / dm
                if 0.0 < c1 < 1.0:
                    breakpoints.add(c1)
                # hi_a = lo_b
                c2 = (base[b] - base[a] - w) / dm
                if 0.0 < c2 < 1.0:
                    breakpoints.add(c2)
                # lo_a = lo_b
                c3 = (base[b] - base[a]) / dm
                if 0.0 < c3 < 1.0:
                    breakpoints.add(c3)
                # hi_a = hi_b
                c4 = (base[b] - base[a]) / dm
                if 0.0 < c4 < 1.0:
                    breakpoints.add(c4)
    
    breakpoints = sorted(breakpoints)
    
    total_integral = 0.0
    for idx in range(len(breakpoints) - 1):
        c_lo = breakpoints[idx]
        c_hi = breakpoints[idx + 1]
        h = c_hi - c_lo
        if h < 1e-15:
            continue
        
        L_lo = _union_len_at_c(motion, base, c_lo, w)
        L_hi = _union_len_at_c(motion, base, c_hi, w)
        
        # L(c) is linear between breakpoints
        m = (L_hi - L_lo) / h
        # Integral of (L_lo + m*u)^2 du from 0 to h
        integral_segment = L_lo * L_lo * h + L_lo * m * h * h + m * m * h * h * h / 3.0
        total_integral += integral_segment
    
    return total_integral


def _project_nonoverlap(base, w=1.0/8.0):
    """Project to enforce non-overlapping base intervals in sorted order."""
    K = len(base)
    order = np.argsort(base)
    base_sorted = base[order].copy()
    
    # Enforce base_sorted[i] + w <= base_sorted[i+1]
    for i in range(K - 1):
        min_next = base_sorted[i] + w
        if base_sorted[i + 1] < min_next:
            base_sorted[i + 1] = min_next
    
    result = np.empty(K)
    result[order] = base_sorted
    return result


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / 8.0
    motion = np.arange(N, dtype=np.float64) / N  # c*i/8
    
    # Initial guess: evenly spaced
    base_x = np.linspace(0.0, (N - 1) * w, N)
    base_y = np.linspace(0.0, (N - 1) * w, N)
    
    def objective_2d(params):
        bx = params[:N]
        by = params[N:]
        bx = _project_nonoverlap(bx, w)
        by = _project_nonoverlap(by, w)
        return _exact_union_len_sq(motion, bx) + _exact_union_len_sq(motion, by)
    
    x0 = np.concatenate([base_x, base_y])
    
    try:
        res = minimize(objective_2d, x0, method="L-BFGS-B",
                       options={"maxiter": 1000, "ftol": 1e-14, "gtol": 1e-12, "maxls": 100})
        base_x_opt = res.x[:N]
        base_y_opt = res.x[N:]
    except Exception:
        base_x_opt = base_x
        base_y_opt = base_y
    
    base_x_opt = _project_nonoverlap(base_x_opt, w)
    base_y_opt = _project_nonoverlap(base_y_opt, w)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    positions = np.stack([base_x_opt[i_flat], base_y_opt[j_flat]], axis=1)
    return positions


# EVOLVE-BLOCK-END
