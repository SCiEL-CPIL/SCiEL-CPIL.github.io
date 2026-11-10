# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _keich_offsets(N):
    """Keich binary construction for N = 2^k base positions."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _compute_1d_union_length(positions, N):
    """Compute the 1D union length of N intervals of width 1/N, each sheared by c*i/N."""
    # For a fixed c, interval i is [pos[i] + c*i/N, pos[i] + c*i/N + 1/N]
    # We need the union length integrated over c in [0,1]
    # Sample c values and compute union length at each
    n_samples = 200
    c_vals = np.linspace(0, 1, n_samples)
    total = 0.0
    for c in c_vals:
        # Compute all intervals
        starts = positions + c * np.arange(N) / N
        ends = starts + 1.0 / N
        # Sort by start
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        # Merge intervals
        union_len = 0.0
        cur_start = s_sorted[0]
        cur_end = e_sorted[0]
        for k in range(1, N):
            if s_sorted[k] > cur_end:
                union_len += cur_end - cur_start
                cur_start = s_sorted[k]
                cur_end = e_sorted[k]
            else:
                cur_end = max(cur_end, e_sorted[k])
        union_len += cur_end - cur_start
        total += union_len
    return total / n_samples


def _optimize_1d(N, keich_init, time_budget):
    """Optimize 1D positions using Nelder-Mead, with time budget."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    x0 = keich_init.copy()
    
    def cost(x):
        return _compute_1d_union_length(x, N)
    
    best_x = x0.copy()
    best_cost = cost(x0)
    
    # Nelder-Mead optimization
    try:
        result = minimize(
            cost, x0, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if result.fun < best_cost:
            best_x = result.x.copy()
            best_cost = result.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    
    # If we have time, try a few random restarts
    if remaining > 10:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline:
                break
            x_restart = best_x + rng.normal(0, 0.01, size=N)
            try:
                result = minimize(
                    cost, x_restart, method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_cost:
                    best_x = result.x.copy()
                    best_cost = result.fun
            except Exception:
                pass
    
    return best_x


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    """
    N = 8
    
    # Get initial Keich construction
    keich = _keich_offsets(N)
    
    # Estimate time budget: leave 30s safety margin
    total_budget = 600
    safety_margin = 30
    opt_budget = (total_budget - safety_margin) / 2.0  # Split between x and y optimization
    
    # Optimize x positions
    x_opt = _optimize_1d(N, keich, opt_budget)
    
    # Optimize y positions (same problem, but we can reuse the same optimization)
    # Since x and y are independent, use same optimized positions
    y_opt = x_opt.copy()
    
    # Build the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
