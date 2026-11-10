# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 625.0
_SAFETY_MARGIN = 5.0


def _area_from_positions(x, num_points=20000):
    """
    Compute the area of the union of the 64 triangles using the prescribed quadrature.
    x: array of shape (64,)
    Returns: float area
    """
    n = 64
    m = np.arange(num_points, dtype=np.float64)
    y = (m + 0.5) / num_points

    a = (np.arange(n, dtype=np.float64) + 1.0) / n

    L = x[None, :] + a[None, :] * y[:, None]
    L_sorted = np.sort(L, axis=1)

    w = (1.0 - y) / n

    diffs = np.diff(L_sorted, axis=1)
    capped_diffs = np.minimum(diffs, w[:, None])
    union_len = w + capped_diffs.sum(axis=1)

    area = np.mean(union_len)
    return float(area)


def _keich_baseline():
    """
    Classical Keich construction for n a power of 2.
    """
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def _area_proxy(x, num_points=2000):
    """
    A faster proxy for the area, using fewer quadrature points.
    Used during optimization for speed.
    """
    return _area_from_positions(x, num_points=num_points)


def _coordinate_descent_scan(x, max_passes=20, opt_points=2000):
    """
    Coordinate descent with a 1D scan over each coordinate.
    For each coordinate i, we scan a range of values and pick the one that
    minimizes the proxy area. This is more robust than simple +/- step.
    """
    n = 64
    x = x.copy()
    current_area = _area_proxy(x, num_points=opt_points)

    for _pass in range(max_passes):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break

        improved = False
        for i in range(n):
            if time.time() > _DEADLINE - _SAFETY_MARGIN:
                break

            x_i_current = x[i]
            # Adaptive search range: start wide, narrow down
            # The optimal x_i is likely near the current value
            # Let's search in a range of +/- 0.5 initially, then narrow
            # But to be fast, let's use a smaller range like +/- 0.1
            # and refine in subsequent passes
            
            # Estimate a good range based on the spread of x
            # Or just use a fixed range that covers the likely optimum
            # Given the problem structure, x_i should be in [-1, 1] roughly
            # But we want to be local.
            
            # Let's try a local search: +/- 0.05 with 50 points
            # This is fast: 50 * 64 * 20 passes = 64000 evals of proxy
            # Each proxy eval is ~ 2000 * 64 ops + sort. Should be fast.
            
            search_range = np.linspace(x_i_current - 0.05, x_i_current + 0.05, 50)
            
            best_xi = x_i_current
            best_local_area = current_area
            
            for xi in search_range:
                x_temp = x.copy()
                x_temp[i] = xi
                area = _area_proxy(x_temp, num_points=opt_points)
                if area < best_local_area - 1e-12:
                    best_local_area = area
                    best_xi = xi
            
            if best_xi != x_i_current:
                x[i] = best_xi
                current_area = best_local_area
                improved = True
        
        if not improved:
            break

    return x


def _scipy_optimize(x0, num_points=1000):
    """
    Use scipy.optimize.minimize (L-BFGS-B) to refine the solution.
    This uses gradient estimation internally (or we can provide gradients, but
    L-BFGS-B with numerical gradients is often sufficient for small n).
    """
    def objective(x):
        return _area_from_positions(x, num_points=num_points)
    
    # L-BFGS-B is good for large scale, but n=64 is small.
    # We can try BFGS or L-BFGS-B.
    # Since we don't have an analytical gradient easily, let's use numerical.
    # However, numerical gradient for 64 variables requires 65 function evals per step.
    # That might be slow.
    
    # Alternative: use the coordinate descent result as a good starting point
    # and let scipy polish it.
    
    try:
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 100, 'gtol': 1e-8}
        )
        return result.x
    except Exception:
        return x0


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
    # Start with the Keich baseline
    x = _keich_baseline()
    best_x = x.copy()
    best_area = _area_from_positions(x)

    # Phase 1: Coordinate descent with scan (fast, robust)
    # Use a moderate number of points for the proxy
    x_cd = _coordinate_descent_scan(x, max_passes=30, opt_points=2000)
    area_cd = _area_from_positions(x_cd)
    if area_cd < best_area:
        best_x = x_cd
        best_area = area_cd

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 2: Try to further refine with scipy
    # Use a coarser grid for scipy to make it faster
    x_scipy = _scipy_optimize(best_x, num_points=500)
    area_scipy = _area_from_positions(x_scipy)
    if area_scipy < best_area:
        best_x = x_scipy
        best_area = area_scipy

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 3: Final fine-tuning with coordinate descent on full resolution
    # Use the full 20000 points for the proxy in this phase?
    # That might be too slow. Let's stick to 2000 but do more passes.
    # Or use 5000 points.
    x_final = _coordinate_descent_scan(best_x, max_passes=50, opt_points=5000)
    area_final = _area_from_positions(x_final)
    if area_final < best_area:
        best_x = x_final
        best_area = area_final

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 4: Random restarts + local search
    rng = np.random.default_rng(42)
    for _ in range(10):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break

        # Perturb the best solution
        x_pert = best_x + rng.normal(0, 0.01, size=64)
        x_pert = _coordinate_descent_scan(x_pert, max_passes=10, opt_points=2000)
        area_pert = _area_from_positions(x_pert)
        if area_pert < best_area:
            best_x = x_pert
            best_area = area_pert

    return best_x


# EVOLVE-BLOCK-END
