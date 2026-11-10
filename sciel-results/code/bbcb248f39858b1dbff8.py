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


def _soft_min_union_area(x, num_points=2000, beta=50.0):
    """
    Smooth surrogate for the union area using soft-min (log-sum-exp) to handle
    the nonsmooth min in the union length computation.
    """
    n = 64
    m = np.arange(num_points, dtype=np.float64)
    y = (m + 0.5) / num_points

    a = (np.arange(n, dtype=np.float64) + 1.0) / n

    L = x[None, :] + a[None, :] * y[:, None]  # shape (num_points, 64)

    w = (1.0 - y) / n  # shape (num_points,)

    # For each row, we need to compute the union length.
    # The union of intervals [L_j, L_j + w] for j=0..n-1.
    # If we sort L, the union length is w + sum_{j=1}^{n-1} min(L_{(j+1)} - L_{(j)}, w)
    # where (j) denotes the j-th order statistic.
    
    # Soft approach: for each pair of consecutive order statistics, use soft-min.
    # But sorting is nonsmooth. Instead, we can use a different formulation.
    
    # Alternative: The union length of intervals can be computed as:
    # max_j (L_j + w) - min_j L_j - (gaps that are not covered)
    # This is complex. Let's stick with the sorted approach but use a smooth approximation.
    
    # Actually, let's use a different smooth surrogate:
    # The area is the integral of the union length. The union length at height y is:
    # w + sum_{j=1}^{n-1} min(d_j, w) where d_j are the gaps between consecutive sorted L values.
    
    # A simpler smooth surrogate: use the fact that for intervals of equal width w,
    # the union length = w + sum of min(gap, w) over consecutive sorted positions.
    # We can approximate min(a, w) with a soft-min: -1/beta * log(exp(-beta*a) + exp(-beta*w)) / 2 ... 
    # Actually, soft_min(a, b) = -1/beta * log(exp(-beta*a) + exp(-beta*b))
    # But min(d, w) where d can be negative? No, d >= 0 after sorting.
    
    # Let me try a different approach: compute the union area directly using a 
    # smooth approximation of the indicator function.
    
    # For now, let's just use the exact area but with fewer points as a proxy.
    # The soft-min idea is good but implementation is tricky. Let me try a simpler 
    # smooth surrogate: the L1 norm of the "excess" beyond the union.
    
    # Actually, let's just return the exact area for now and use it with scipy.
    # The key insight is that scipy with a good starting point and enough iterations
    # can handle the nonsmooth objective reasonably well.
    return _area_from_positions(x, num_points=num_points)


def _coordinate_descent_scan(x, max_passes=20, opt_points=2000):
    """
    Coordinate descent with adaptive 1D scan over each coordinate.
    Uses a golden-section-like refinement: coarse scan first, then fine scan
    around the best point found.
    """
    n = 64
    x = x.copy()
    current_area = _area_from_positions(x, num_points=opt_points)

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
                area = _area_from_positions(x_temp, num_points=opt_points)
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
                    area = _area_from_positions(x_temp, num_points=opt_points)
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


def _subgradient_and_value(x, num_points=2000):
    """
    Compute the subgradient of the area with respect to x.
    The area is a sum of min functions, which are convex but nonsmooth.
    We compute a subgradient using the sorted order at the current point.
    """
    n = 64
    m = np.arange(num_points, dtype=np.float64)
    y = (m + 0.5) / num_points

    a = (np.arange(n, dtype=np.float64) + 1.0) / n

    L = x[None, :] + a[None, :] * y[:, None]  # (num_points, 64)
    L_sorted = np.sort(L, axis=1)  # (num_points, 64)
    
    w = (1.0 - y) / n  # (num_points,)

    diffs = np.diff(L_sorted, axis=1)  # (num_points, 63)
    capped_diffs = np.minimum(diffs, w[:, None])
    union_len = w + capped_diffs.sum(axis=1)  # (num_points,)
    area = np.mean(union_len)
    
    # Subgradient computation
    # For each y_m, the union length is w + sum_{j=1}^{63} min(L_{(j+1)} - L_{(j)}, w)
    # The subgradient of min(d, w) with respect to d is 1 if d < w, 0 if d > w, [0,1] if d == w
    # d_j = L_{(j+1)} - L_{(j)}
    # L_{(j)} = x_{i_j} + a_{i_j} * y_m where i_j is the index that gives the j-th order statistic
    
    # For each y_m, we need to know which x_i contributes to each order statistic.
    # This is complex. Let's use a simpler approach: finite differences.
    
    # Actually, let's just return the area and use finite differences for the gradient.
    return float(area)


def _finite_difference_gradient(x, num_points=2000, eps=1e-6):
    """
    Compute the gradient using central finite differences.
    """
    n = 64
    grad = np.zeros(n, dtype=np.float64)
    base_area = _area_from_positions(x, num_points=num_points)
    
    for i in range(n):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        area_plus = _area_from_positions(x_plus, num_points=num_points)
        area_minus = _area_from_positions(x_minus, num_points=num_points)
        grad[i] = (area_plus - area_minus) / (2 * eps)
    
    return grad


def _scipy_optimize(x0, num_points=2000, maxiter=300):
    """
    Use scipy.optimize.minimize (L-BFGS-B) to refine the solution.
    """
    def objective(x):
        return _area_from_positions(x, num_points=num_points)
    
    try:
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-12}
        )
        return result.x
    except Exception:
        return x0


def _scipy_slsqp(x0, num_points=2000, maxiter=200):
    """
    Use SLSQP method which may handle nonsmoothness differently.
    """
    def objective(x):
        return _area_from_positions(x, num_points=num_points)
    
    try:
        result = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        return result.x
    except Exception:
        return x0


def _trust_constr(x0, num_points=2000, maxiter=200):
    """
    Use trust-constr method with finite difference gradients.
    """
    def objective(x):
        return _area_from_positions(x, num_points=num_points)
    
    try:
        result = minimize(
            objective, 
            x0, 
            method='trust-constr',
            options={'maxiter': maxiter, 'gtol': 1e-10, 'xtol': 1e-12}
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
    x_cd = _coordinate_descent_scan(x, max_passes=30, opt_points=2000)
    area_cd = _area_from_positions(x_cd)
    if area_cd < best_area:
        best_x = x_cd
        best_area = area_cd

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 2: Try SLSQP which may handle the nonsmooth structure better
    x_slsqp = _scipy_slsqp(best_x, num_points=2000, maxiter=150)
    area_slsqp = _area_from_positions(x_slsqp)
    if area_slsqp < best_area:
        best_x = x_slsqp
        best_area = area_slsqp

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 3: Try trust-constr
    x_tc = _trust_constr(best_x, num_points=2000, maxiter=100)
    area_tc = _area_from_positions(x_tc)
    if area_tc < best_area:
        best_x = x_tc
        best_area = area_tc

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 4: L-BFGS-B refinement
    x_lbfgs = _scipy_optimize(best_x, num_points=2000, maxiter=200)
    area_lbfgs = _area_from_positions(x_lbfgs)
    if area_lbfgs < best_area:
        best_x = x_lbfgs
        best_area = area_lbfgs

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 5: Final fine-tuning with coordinate descent on higher resolution
    x_final = _coordinate_descent_scan(best_x, max_passes=40, opt_points=5000)
    area_final = _area_from_positions(x_final)
    if area_final < best_area:
        best_x = x_final
        best_area = area_final

    # Check deadline
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return best_x

    # Phase 6: Random restarts with varied perturbation scales
    rng = np.random.default_rng(42)
    for _ in range(12):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break

        scale = 0.005 * (1.0 + _ / 6.0)
        x_pert = best_x + rng.normal(0, scale, size=64)
        x_pert = _coordinate_descent_scan(x_pert, max_passes=15, opt_points=2000)
        area_pert = _area_from_positions(x_pert)
        if area_pert < best_area:
            best_x = x_pert
            best_area = area_pert

    # Phase 7: Final polish with SLSQP on the best found
    if time.time() < _DEADLINE - _SAFETY_MARGIN:
        x_polish = _scipy_slsqp(best_x, num_points=5000, maxiter=100)
        area_polish = _area_from_positions(x_polish)
        if area_polish < best_area:
            best_x = x_polish
            best_area = area_polish

    # Phase 8: Final coordinate descent polish
    if time.time() < _DEADLINE - _SAFETY_MARGIN:
        x_final2 = _coordinate_descent_scan(best_x, max_passes=25, opt_points=5000)
        area_final2 = _area_from_positions(x_final2)
        if area_final2 < best_area:
            best_x = x_final2
            best_area = area_final2

    return best_x


# EVOLVE-BLOCK-END
