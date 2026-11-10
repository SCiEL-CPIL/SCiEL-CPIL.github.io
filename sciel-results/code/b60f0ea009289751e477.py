# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 64
K = int(np.log2(N))  # 6
WIDTH = 1.0 / N
# Precompute a_i = (i+1)/64 for i in 0..63
A = (np.arange(N) + 1) / N
# Quadrature heights: y_m = (m + 0.5) / 20000
M = 20000
Y = (np.arange(M) + 0.5) / M  # shape (M,)
# For each triangle i, L_i(y) = x_i + a_i * y
# w(y) = (1 - y) / 64


def _keich_baseline():
    """Classical Keich construction for n=64."""
    positions = np.zeros(N)
    for i in range(N):
        # Binary representation of i, padded to K bits
        bits = [(i >> b) & 1 for b in range(K)]  # b=0 is LSB
        # Keich: x_i = sum_{j=1}^{k} ((1-j)/k) * eps_j * 2^{-j}
        # where eps_j is the j-th bit (1-indexed from LSB? or MSB?)
        # In the original: a_i = i/n, expand in binary a_i = sum_j eps_j 2^{-j}
        # For i in 0..n-1, a_i = (i+1)/n... wait, the task says a_i = (i+1)/64
        # But the Keich construction in the comment uses a_i = i/n.
        # Let me re-read: "expand a_i = i/n in binary" - but task says a_i=(i+1)/64
        # The seed code uses bin(i) for i in range(n), so it uses i, not i+1.
        # Let's just replicate the seed code's logic which scores 1.0
        x = 0.0
        for j in range(1, K + 1):
            eps = (i >> (j - 1)) & 1  # j-th bit, 1-indexed from LSB
            x += ((1 - j) / K) * eps * (2.0 ** -j)
        positions[i] = x
    return positions


def _compute_area(x, y_vals=None):
    """Compute the area of the union of triangles using quadrature."""
    if y_vals is None:
        y_vals = Y
    # L_i(y) = x_i + a_i * y for i=0..N-1
    # Shape: (N, M)
    L = x[:, None] + A[:, None] * y_vals[None, :]
    w = (1.0 - y_vals) / N  # shape (M,)
    # For each height, sort L and compute union length
    # Union length = w + sum_{j} min(L_{j+1} - L_j, w) over consecutive sorted pairs
    L_sorted = np.sort(L, axis=0)  # shape (N, M)
    diffs = np.diff(L_sorted, axis=0)  # shape (N-1, M)
    # Union length at each height
    union_len = w[None, :] + np.sum(np.minimum(diffs, w[None, :]), axis=0)  # shape (M,)
    area = np.mean(union_len)
    return area


def _soft_min(x, y_vals, beta):
    """
    Smooth surrogate for the union area using soft-min.
    We want to minimize the area of the union.
    
    The union length at height y is:
    w + sum_j min(L_{j+1} - L_j, w)
    
    where L are sorted. This is hard to differentiate.
    
    Alternative approach: The area of the union of triangles can be written as
    the integral over y of the length of the union of intervals [L_i(y), L_i(y)+w(y)].
    
    A smooth surrogate: instead of min, use soft-min.
    But the sorting is the main issue.
    
    Let me think of a different smooth surrogate.
    
    Actually, a common approach for union area minimization:
    The area of union = integral_y [max_i (L_i + w) - min_i L_i] ... no, that's the bounding box.
    
    Let me use a different formulation. The union of intervals [L_i, L_i+w] has length:
    If we sort L_i, the union length = (max L_i + w) - (min L_i) - sum of gaps that are > w.
    Actually no. The union length = w + sum of min(gap_j, w) for consecutive sorted intervals.
    
    This is inherently nonsmooth due to sorting.
    
    For a smooth surrogate, I'll use a coordinate descent / L-BFGS-B approach
    with the exact (nonsmooth) objective, relying on L-BFGS-B's ability to handle
    piecewise linear objectives.
    """
    pass


def _optimize_with_lbfgsb(x0, max_time):
    """Use L-BFGS-B to optimize the area."""
    start_time = time.time()
    
    def objective(x):
        return _compute_area(x)
    
    # Run L-BFGS-B
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 2000, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    elapsed = time.time() - start_time
    return result.x, result.fun, elapsed


def _coordinate_descent(x, max_time, n_passes_max=50):
    """
    Adaptive coordinate descent with shrinking scan range.
    """
    start_time = time.time()
    current_area = _compute_area(x)
    best_x = x.copy()
    best_area = current_area
    
    # Initial step sizes
    step_sizes = np.ones(N) * 0.01
    
    for pass_num in range(n_passes_max):
        elapsed = time.time() - start_time
        if elapsed > max_time - 10:  # Safety margin
            break
        
        improved = False
        for i in range(N):
            elapsed = time.time() - start_time
            if elapsed > max_time - 5:
                    break
            
            # Try to optimize x_i
            # Use a local search around x_i
            best_local = x[i]
            best_local_area = current_area
            
            # Adaptive range: start wide, shrink
            for scale in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]:
                step = step_sizes[i] * scale
                # Try moving in both directions
                for direction in [1, -1]:
                    x_new = x.copy()
                    x_new[i] = x[i] + direction * step
                    area_new = _compute_area(x_new)
                    if area_new < best_local_area - 1e-15:
                        best_local = x_new[i]
                        best_local_area = area_new
                        improved = True
            
            if improved:
                x[i] = best_local
                current_area = best_local_area
                if current_area < best_area - 1e-15:
                    best_area = current_area
                    best_x = x.copy()
                # Shrink step size for this coordinate
                step_sizes[i] *= 0.8
            else:
                # No improvement, shrink step size
                step_sizes[i] *= 0.5
        
        # Check if we should stop
        if not improved:
            break
    
    return best_x, best_area


def _final_polish(x, max_time):
    """Final high-precision polish pass."""
    start_time = time.time()
    current_area = _compute_area(x)
    
    # Use very small steps for coordinate descent
    for i in range(N):
        elapsed = time.time() - start_time
        if elapsed > max_time - 2:
            break
        
        best_local = x[i]
        best_local_area = current_area
        
        for step in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]:
            for direction in [1, -1]:
                x_new = x.copy()
                x_new[i] = x[i] + direction * step
                area_new = _compute_area(x_new)
                if area_new < best_local_area - 1e-18:
                    best_local = x_new[i]
                    best_local_area = area_new
        
        if best_local != x[i]:
            x[i] = best_local
            current_area = best_local_area
    
    return x, current_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem. For a fixed n, choose
    base positions x = (x_0, ..., x_{n-1}). For position index i (0-based) let
    a_i = (i + 1) / n and define the triangle R_i with vertices
        (x_i, 0), (x_i + 1/n, 0), (x_i + a_i, 1).
    Every triangle has base width 1/n on the line y = 0 and its apex on the line
    y = 1. The objective is to choose x so that the AREA of the union of the n
    triangles R_i is as small as possible. Here n = 64.
    
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    total_deadline = time.time() + 620  # 620s out of 630s budget
    
    # Start with Keich baseline
    x = _keich_baseline()
    best_x = x.copy()
    best_area = _compute_area(x)
    
    # Phase 1: L-BFGS-B optimization
    try:
        x_opt, area_opt, elapsed = _optimize_with_lbfgsb(x, max_time=300)
        if area_opt < best_area:
            best_x = x_opt
            best_area = area_opt
    except Exception:
        pass
    
    # Check time
    if time.time() > total_deadline - 60:
        # Not enough time for more phases
        pass
    else:
        # Phase 2: Coordinate descent
        try:
            x_cd, area_cd = _coordinate_descent(best_x, max_time=200)
            if area_cd < best_area:
                best_x = x_cd
                best_area = area_cd
        except Exception:
            pass
    
    # Check time
    if time.time() > total_deadline - 20:
        pass
    else:
        # Phase 3: Final polish
        try:
            x_polish, area_polish = _final_polish(best_x, max_time=15)
            if area_polish < best_area:
                best_x = x_polish
                best_area = area_polish
        except Exception:
            pass
    
    # Ensure valid output
    best_x = np.asarray(best_x, dtype=np.float64)
    if np.any(~np.isfinite(best_x)):
        # Fallback to Keich baseline
        best_x = _keich_baseline()
    
    return best_x


# EVOLVE-BLOCK-END
