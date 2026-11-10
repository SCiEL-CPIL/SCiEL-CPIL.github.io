# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_1d_union_length(x, N=8):
    """
    Compute the integral from 0 to 1 of the union length of intervals
    [x_i + c*i/N, x_i + 1/N + c*i/N] for i=0..N-1.
    
    Uses exact breakpoint analysis.
    """
    # Endpoints: L_i(c) = x_i + (i/N)*c, R_i(c) = x_i + 1/N + (i/N)*c
    # Breakpoints occur when any L_i(c) = L_j(c) or L_i(c) = R_j(c) or R_i(c) = R_j(c)
    # L_i(c) = L_j(c) => x_i + (i/N)c = x_j + (j/N)c => c = (x_j - x_i) * N / (i - j) if i != j
    # L_i(c) = R_j(c) => x_i + (i/N)c = x_j + 1/N + (j/N)c => c = (x_j + 1/N - x_i) * N / (i - j) if i != j
    # R_i(c) = R_j(c) => same as L_i = L_j since both have same slope offset
    # Actually R_i(c) - R_j(c) = (x_i - x_j) + (i-j)/N * c, same form as L_i - L_j
    
    # So breakpoints from:
    # Type 1: x_i + (i/N)c = x_j + (j/N)c for i != j => c = (x_j - x_i)*N/(i-j)
    # Type 2: x_i + (i/N)c = x_j + 1/N + (j/N)c for i != j => c = (x_j + 1/N - x_i)*N/(i-j)
    
    breakpoints = set([0.0, 1.0])
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            denom = i - j
            # Type 1: L_i = L_j
            c1 = (x[j] - x[i]) * N / denom
            if 0.0 < c1 < 1.0:
                breakpoints.add(c1)
            # Type 2: L_i = R_j
            c2 = (x[j] + 1.0/N - x[i]) * N / denom
            if 0.0 < c2 < 1.0:
                breakpoints.add(c2)
    
    breakpoints = sorted(breakpoints)
    
    # Integrate piecewise: between consecutive breakpoints, the union structure is constant
    # At each subinterval, compute union length at midpoint, multiply by width
    total = 0.0
    for k in range(len(breakpoints) - 1):
        c0 = breakpoints[k]
        c1 = breakpoints[k + 1]
        if c1 <= c0:
            continue
        cmid = 0.5 * (c0 + c1)
        # Compute union of intervals at height cmid
        intervals = []
        for i in range(N):
            left = x[i] + (i / N) * cmid
            right = x[i] + 1.0 / N + (i / N) * cmid
            intervals.append((left, right))
        # Merge intervals
        intervals.sort()
        merged = [intervals[0]]
        for left, right in intervals[1:]:
            if left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            else:
                merged.append((left, right))
        union_len = sum(r - l for l, r in merged)
        total += union_len * (c1 - c0)
    
    return total


def _keich_offsets(N=8):
    """Keich binary construction for N=2^k."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _optimize_1d(x0, N=8, max_iters=50, tol=1e-10):
    """
    Optimize 1D base positions to minimize union volume.
    Uses Nelder-Mead on the exact cost function.
    """
    def cost(x):
        return _compute_1d_union_length(x, N)
    
    # Use multiple restarts with different initial points
    best_x = x0.copy()
    best_cost = cost(x0)
    
    # Try a few different initializations
    inits = [x0.copy()]
    # Perturbed versions
    for _ in range(3):
        perturbed = x0 + 0.01 * np.random.default_rng(42 + _).standard_normal(N)
        inits.append(perturbed)
    
    for x_init in inits:
        try:
            result = minimize(cost, x_init, method='Nelder-Mead',
                            options={'maxiter': max_iters, 'xatol': tol, 'fatol': tol})
            if result.fun < best_cost:
                best_cost = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
    
    return best_x


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Returns positions of shape (64, 2).
    """
    deadline = time.time() + 580  # Safety margin before 600s limit
    
    N = 8
    keich = _keich_offsets(N)
    
    # Optimize 1D x-positions
    x_opt = _optimize_1d(keich, N=N, max_iters=100, tol=1e-10)
    
    # Check time
    if time.time() > deadline - 30:
        x_opt = keich
    
    # By symmetry, use same optimized positions for y
    y_opt = x_opt.copy()
    
    # Build full 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
