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
    Coordinate descent with adaptive 1D scan over each coordinate.
    Uses a golden-section-like refinement: coarse scan first, then fine scan
    around the best point found.
    """
    n = 64
    x = x.copy()
    current_area = _area_proxy(x, num_points=opt_points)

    for _pass in range(max_passes):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break

        improved = False
        # Adaptive range: shrink as we converge
        base_range = 0.1 / (1.0 + 0.1 * _pass)
        
        for i in range(n):
            if time.time() > _DEADLINE - _SAFETY_MARGIN:
                break

            x_i_current = x[i]
            
            # Coarse scan: wider range, fewer points
            coarse_range = base_range * 2.0
            coarse_points = np.linspace(x_i_current - coarse_range, x_i_current + coarse_range, 20)
            
            best_xi = x_i_current
            best_local_area = current_area
            
            for xi in coarse_points:
                x_temp = x.copy()
                x_temp[i] = xi
                area = _area_proxy(x_temp, num_points=opt_points)
                if area < best_local_area - 1e-12:
                    best_local_area = area
                    best_xi = xi
            
            # Fine scan: narrow range around the best coarse point
            if best_xi != x_i_current:
                fine_range = base_range * 0.5
                fine_points = np.linspace(best_xi - fine_range, best_xi + fine_range, 30)
                for xi in fine_points:
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


def _scipy_optimize(x0, num_points=2000):
    """
    Use scipy.optimize.minimize (L-BFGS-B) to refine the solution.
    Uses a moderate number of quadrature points for a better gradient signal.
    """
    def objective(x):
        return _area_from_positions(x, num_points=num_points)
    
    try:
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12}
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
    # Use a moderate grid for scipy to get better gradient signal
    x_scipy = _scipy_optimize(best_x, num_points=2000)
    area_scipy = _area_from_positions(x_scipy)
    if area_scipy < best_area:
        best_x = x_scipy
        best_area = area_scropy if False else area_scipy

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

    # Phase 4: Random restarts with varied perturbation scales
    rng = np.random.default_rng(42)
    for _ in range(15):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break

        # Use varying perturbation scales to escape local optima
        scale = 0.005 * (1.0 + _ / 5.0)  # gradually increase perturbation
        x_pert = best_x + rng.normal(0, scale, size=64)
        x_pert = _coordinate_descent_scan(x_pert, max_passes=15, opt_points=2000)
        area_pert = _area_from_positions(x_pert)
        if area_pert < best_area:
            best_x = x_pert
            best_area = area_pert

    # Phase 5: Final fine-tuning with tighter tolerance
    if time.time() < _DEADLINE - _SAFETY_MARGIN:
        x_final2 = _coordinate_descent_scan(best_x, max_passes=30, opt_points=5000)
        area_final2 = _area_from_positions(x_final2)
        if area_final2 < best_area:
            best_x = x_final2
            best_area = area_final2

    return best_x


# EVOLVE-BLOCK-END
