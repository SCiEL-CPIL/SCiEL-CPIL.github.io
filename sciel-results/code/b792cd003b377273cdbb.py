# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _union_length_at_heights(x, y_vals, a, n):
    """Compute union lengths at given heights for positions x."""
    L = x[None, :] + a[None, :] * y_vals[:, None]  # shape (M, n)
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)  # shape (M, n-1)
    w_vec = (1.0 - y_vals) / n  # shape (M,)
    union_lens = w_vec + np.sum(np.minimum(gaps, w_vec[:, None]), axis=1)
    return union_lens


def _area(x, y_vals, a, n):
    """Compute area of union of triangles."""
    union_lens = _union_length_at_heights(x, y_vals, a, n)
    return np.mean(union_lens)


def _keich_construction(n):
    """Keich construction for initial guess."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    a = np.array([(i + 1) / n for i in range(n)], dtype=np.float64)
    y_vals = np.array([(m + 0.5) / 20000 for m in range(20000)], dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 610  # safety margin before 630s limit
    
    x0 = _keich_construction(n)
    best_x = x0.copy()
    best_area = _area(x0, y_vals, a, n)
    
    # Strategy 1: Nelder-Mead from Keich
    try:
        result = minimize(
            _area, x0, args=(y_vals, a, n),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11}
        )
        if np.isfinite(result.fun) and result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x
    
    # Strategy 2: L-BFGS-B from best
    try:
        result2 = minimize(
            _area, best_x, args=(y_vals, a, n),
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        if np.isfinite(result2.fun) and result2.fun < best_area:
            best_area = result2.fun
            best_x = result2.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x
    
    # Strategy 3: Coordinate descent - optimize each x_i one at a time
    # For fixed other x_j, x_i appears linearly in L_i(y) = x_i + a_i*y
    # The union length is piecewise linear in x_i, so we can find optimal x_i
    # by checking critical points where L_i(y) = L_j(y) +/- w(y) for some j
    
    improved = True
    coord_iter = 0
    while improved and time.time() < deadline and coord_iter < 50:
        improved = False
        coord_iter += 1
        for i in range(n):
            # Try to optimize x_i
            x_trial = best_x.copy()
            
            # Sample critical points for x_i
            # L_i(y) = x_i + a_i*y, L_j(y) = x_j + a_j*y
            # Critical when L_i(y) = L_j(y) + w(y) or L_i(y) = L_j(y) - w(y)
            # x_i + a_i*y = x_j + a_j*y + (1-y)/n
            # x_i = x_j + (a_j - a_i)*y + (1-y)/n
            # x_i = x_j + (a_j - a_i - 1/n)*y + 1/n
            
            critical_points = set()
            # Current value
            critical_points.add(best_x[i])
            
            # Add critical points from each pair
            for j in range(n):
                if j == i:
                    continue
                # x_i = x_j + (a_j - a_i - 1/n)*y + 1/n
                # Sample at several y values
                for y_sample in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    cp1 = best_x[j] + (a[j] - a[i] - 1.0/n) * y_sample + 1.0/n
                    cp2 = best_x[j] + (a[j] - a_i + 1.0/n) * y_sample - 1.0/n
                    critical_points.add(cp1)
                    critical_points.add(cp2)
            
            # Also try small perturbations around current best
            for delta in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                critical_points.add(best_x[i] + delta)
            
            # Evaluate all candidates
            best_local = best_x[i]
            best_local_area = best_area
            
            for cp in critical_points:
                x_trial[i] = cp
                trial_area = _area(x_trial, y_vals, a, n)
                if trial_area < best_local_area:
                    best_local_area = trial_area
                    best_local = cp
            
            if best_local != best_x[i] and best_local_area < best_area:
                best_x[i] = best_local
                best_area = best_local_area
                improved = True
    
    if time.time() > deadline:
        return best_x
    
    # Strategy 4: Another Nelder-Mead from improved position
    try:
        result3 = minimize(
            _area, best_x, args=(y_vals, a, n),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if np.isfinite(result3.fun) and result3.fun < best_area:
            best_area = result3.fun
            best_x = result3.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x
    
    # Strategy 5: Multiple random restarts with Nelder-Mead
    rng = np.random.default_rng(42)
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Perturb from best
        x_restart = best_x + rng.normal(0, 0.01, size=n)
        
        try:
            result_r = minimize(
                _area, x_restart, args=(y_vals, a, n),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            if np.isfinite(result_r.fun) and result_r.fun < best_area:
                best_area = result_r.fun
                best_x = result_r.x.copy()
        except Exception:
            pass
    
    # Ensure all values are finite
    if not np.all(np.isfinite(best_x)):
        return x0.copy()
    
    return best_x


# EVOLVE-BLOCK-END
