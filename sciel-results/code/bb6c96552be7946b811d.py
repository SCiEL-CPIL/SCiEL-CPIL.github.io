# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_area(x, n=64, M=20000):
    """Compute the union area via quadrature over M height samples."""
    a = (np.arange(n) + 1) / n
    w = 1.0 / n
    y = (np.arange(M) + 0.5) / M  # shape (M,)
    
    # L_i = x_i + a_i * y, shape (n, M)
    L = x[:, None] + a[:, None] * y[None, :]
    
    # Sort L along axis 0 for each y
    L_sorted = np.sort(L, axis=0)  # shape (n, M)
    
    # Consecutive differences
    diffs = L_sorted[1:, :] - L_sorted[:-1, :]  # shape (n-1, M)
    
    # Union length = w + sum(min(diffs, w))
    union_lengths = w + np.sum(np.minimum(diffs, w), axis=0)  # shape (M,)
    
    area = np.mean(union_lengths)
    return area


def _keich_baseline(n=64):
    """Generate the classical Keich construction."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def _coordinate_descent(x0, n=64, M=20000, max_iters=500, tol=1e-10, time_limit=580.0):
    """
    Coordinate descent to minimize union area.
    For each coordinate i, find the optimal x_i given all others fixed.
    """
    x = x0.copy()
    a = (np.arange(n) + 1) / n
    w = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    
    best_area = _compute_area(x, n, M)
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                    break
            
            # Fix all other coordinates, optimize x_i
            # L_i(y) = x_i + a_i * y
            # We need to find x_i that minimizes union area
            
            # Get current L values for all triangles
            L_others = np.sort(x[:i, None] + a[:i, None] * y[None, :], axis=0)
            L_after = np.sort(x[i+1:, None] + a[i+1:, None] * y[None, :], axis=0)
            
            # The optimal x_i should place L_i(y) = x_i + a_i*y to minimize overlap
            # Strategy: try candidate positions around current and neighbors
            
            current_xi = x[i]
            best_xi = current_xi
            best_local_area = best_area
            
            # Try shifting x_i to align with other triangles
            # Key insight: the area is piecewise linear in x_i (for fixed other x's)
            # So optimal is at a breakpoint where L_i(y) crosses some L_j(y)
            
            # Collect candidate x_i values
            candidates = [current_xi]
            
            # For each other triangle j and each y, find x_i where L_i(y) = L_j(y)
            # x_i + a_i * y = x_j + a_j * y => x_i = x_j + (a_j - a_i) * y
            for j in range(n):
                if j == i:
                    continue
                # x_i = x[j] + (a[j] - a[i]) * y
                cand = x[j] + (a[j] - a[i]) * y
                # Add quantized versions to keep search tractable
                candidates.extend([np.median(cand), np.min(cand), np.max(cand)])
            
            # Also try small perturbations
            for delta in [-w, -w/2, w/2, w]:
                candidates.append(current_xi + delta)
            
            # Deduplicate and sort
            candidates = sorted(set([round(c, 8) for c in candidates]))
            
            # Evaluate each candidate
            for xi in candidates:
                x[i] = xi
                area = _compute_area(x, n, M)
                if area < best_local_area - tol:
                    best_local_area = area
                    best_xi = xi
            
            if best_xi != current_xi:
                x[i] = best_xi
                improved = True
                if best_local_area < best_area:
                    best_area = best_local_area
            
        if not improved:
            break
    
    return x, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem. For a fixed n, choose
    base positions x = (x_0, ..., x_{n-1}). For position index i (0-based) let
    a_i = (i + 1) / n and define the triangle R_i with vertices
        (x_i, 0), (x_i + 1/n, 0), (x_i + a_i, 1).
    Every triangle has base width 1/n on the line y = 0 and its apex on the line
    y = 1. The objective is to choose x so that the AREA of the union of the n
    triangles R_i is as small as possible. Here n = 64.

    Uses Keich construction as initial guess, then refines with coordinate descent.

    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    x0 = _keich_baseline(n)
    x, area = _coordinate_descent(x0, n=n, M=20000, max_iters=200, tol=1e-10, time_limit=580.0)
    return x


# EVOLVE-BLOCK-END
