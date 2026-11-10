# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, positive sides, non-overlapping. Vectorized."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Inside unit square
    if np.any(centers - sides[:, None] / 2.0 < -tol):
        return False
    if np.any(centers + sides[:, None] / 2.0 > 1.0 + tol):
        return False
    # Non-overlap: for i < j, must not have |dx| < t AND |dy| < t where t = (s_i+s_j)/2 - tol
    dx = np.abs(centers[:, 0][:, None] - centers[:, 0][None, :])
    dy = np.abs(centers[:, 1][:, None] - centers[:, 1][None, :])
    t = (sides[:, None] + sides[None, :]) / 2.0 - tol
    iu = np.triu_indices(n, k=1)
    if np.any((dx[iu] < t[iu]) & (dy[iu] < t[iu])):
        return False
    return True


def _objective(centers, sides):
    """Sum of side lengths."""
    return float(np.sum(sides))


def _max_side_at(cx, cy, centers, sides, i, tol=1e-7):
    """
    Compute the maximum side length for square i centered at (cx,cy)
    given the positions and sizes of all other squares. Vectorized.
    """
    max_s = min(2.0 * cx, 2.0 * (1.0 - cx), 2.0 * cy, 2.0 * (1.0 - cy))
    if max_s < 0:
        max_s = 0.0

    mask = np.arange(len(sides)) != i
    if not np.any(mask):
        return max_s
    dx = np.abs(cx - centers[mask, 0])
    dy = np.abs(cy - centers[mask, 1])
    sj = sides[mask]
    bounds = np.maximum(2.0 * (dx + tol) - sj, 2.0 * (dy + tol) - sj)
    if len(bounds) > 0:
        max_s = min(max_s, float(np.min(bounds)))

    if max_s < 0:
        max_s = 0.0
    return max_s


def _max_side_all(centers, sides, tol=1e-7):
    """Compute max side for all squares given current positions and other squares' sizes."""
    n = len(sides)
    result = np.zeros(n)
    for i in range(n):
        result[i] = _max_side_at(centers[i, 0], centers[i, 1], centers, sides, i, tol)
    return result


def _soft_min_values(vals, beta=50.0):
    """
    Smooth approximation to min(vals) using log-sum-exp.
    For a set of constraint values where we want the minimum (bottleneck),
    soft_min = -logsumexp(-beta * vals) / beta
    """
    if len(vals) == 0:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    scaled = -beta * np.array(vals)
    m = np.max(scaled)
    if m > 700:
        return -m / beta
    log_sum = m + math.log(np.sum(np.exp(scaled - m)))
    return -log_sum / beta


def _smooth_side_i(cx, cy, centers, sides, i, tol=1e-7, beta=50.0):
    """
    Smooth approximation of max side for square i using soft-min over all constraints.
    Constraints: boundary limits and pairwise separation limits.
    """
    # Boundary constraints
    boundary_vals = [2.0 * cx, 2.0 * (1.0 - cx), 2.0 * cy, 2.0 * (1.0 - cy)]
    
    # Pairwise constraints
    pairwise_vals = []
    n = len(sides)
    for j in range(n):
        if j == i:
            continue
        dx = abs(cx - centers[j, 0])
        dy = abs(cy - centers[j, 1])
        sj = sides[j]
        # Max side from this pair: min(2*(dx+tol) - sj, 2*(dy+tol) - sj)
        # Use soft-min for smoothness
        pair_val = _soft_min_values([2.0 * (dx + tol) - sj, 2.0 * (dy + tol) - sj], beta)
        pairwise_vals.append(pair_val)
    
    all_vals = boundary_vals + pairwise_vals
    return _soft_min_values(all_vals, beta)


def _smooth_total_side(centers, sides, tol=1e-7, beta=50.0):
    """Smooth total side length using soft-min for each square."""
    n = len(sides)
    total = 0.0
    for i in range(n):
        total += _smooth_side_i(centers[i, 0], centers[i, 1], centers, sides, i, tol, beta)
    return total


def _clamp_to_unit(centers, sides):
    """Clamp centers so squares stay inside [0,1]^2."""
    for i in range(len(sides)):
        half = sides[i] / 2.0
        centers[i, 0] = max(half, min(1.0 - half, centers[i, 0]))
        centers[i, 1] = max(half, min(1.0 - half, centers[i, 1]))
    return centers


def _resolve_overlap(centers, sides, tol=1e-7):
    """
    Attempt to resolve overlaps by pushing squares apart.
    """
    n = len(sides)
    for iteration in range(500):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                if t < 0:
                    continue
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                if abs(dx) < t and abs(dy) < t:
                    if abs(dx) < abs(dy):
                        push = (t - abs(dx)) / 2.0 + 2*tol
                        if dx >= 0:
                            centers[j, 0] += push
                            centers[i, 0] -= push
                        else:
                            centers[j, 0] -= push
                            centers[i, 0] += push
                    else:
                        push = (t - abs(dy)) / 2.0 + 2*tol
                        if dy >= 0:
                            centers[j, 1] += push
                            centers[i, 1] -= push
                        else:
                            centers[j, 1] -= push
                            centers[i, 1] += push
                    moved = True
        if not moved:
            break
    return centers


def _grow_all_squares(centers, sides, tol=1e-7, max_rounds=100):
    """
    Iteratively grow all squares to their maximum given current positions.
    Repeat until convergence.
    """
    n = len(sides)
    for _ in range(max_rounds):
        improved = False
        for i in range(n):
            new_s = _max_side_at(centers[i, 0], centers[i, 1], centers, sides, i, tol)
            if new_s > sides[i] + 1e-10:
                sides[i] = new_s
                improved = True
        if not improved:
            break
    return sides


def _grow_all_squares_fast(centers, sides, tol=1e-7, max_rounds=200):
    """Faster grow using vectorized computation."""
    n = len(sides)
    for _ in range(max_rounds):
        improved = False
        new_sides = np.zeros(n)
        for i in range(n):
            new_sides[i] = _max_side_at(centers[i, 0], centers[i, 1], centers, sides, i, tol)
        if np.any(new_sides > sides + 1e-10):
            sides = new_sides
            improved = True
        if not improved:
            break
    return sides


def _density_energy(centers, sides, tol=1e-7, beta=30.0):
    """
    Smooth density-based energy: high when squares are too close or overlapping.
    """
    n = len(sides)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            if t < 0:
                continue
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            a = -beta * (t - dx)
            b = -beta * (t - dy)
            m = max(a, b)
            if m > 700:
                soft_min = m / beta
            else:
                soft_min = (m - math.log(math.exp(a - m) + math.exp(b - m))) / beta
            if soft_min > 0:
                energy += soft_min ** 2
    # Boundary penalty
    for i in range(n):
        half = sides[i] / 2.0
        for coord in [0, 1]:
            lo = centers[i, coord] - half
            hi = centers[i, coord] + half
            if lo < -tol:
                energy += (lo + tol) ** 2 * 100
            if hi > 1.0 + tol:
                energy += (hi - 1.0 - tol) ** 2 * 100
    return energy


def _scipy_optimize(centers, sides, tol=1e-7, max_iter=200, deadline=None):
    """
    Use scipy.optimize.minimize to find a better layout.
    """
    n = len(sides)
    
    def combined_obj(x):
        c = x[:2*n].reshape(n, 2)
        s = x[2*n:]
        s = np.maximum(s, 1e-8)
        obj = -np.sum(s)
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if t < 0:
                    continue
                dx = abs(c[i,0] - c[j,0])
                dy = abs(c[i,1] - c[j,1])
                if dx < t and dy < t:
                    penalty = max(0, t - dx) ** 2 + max(0, t - dy) ** 2
                    obj += penalty * 50.0
        for i in range(n):
            half = s[i] / 2.0
            for coord in [0, 1]:
                lo = c[i, coord] - half
                hi = c[i, coord] + half
                if lo < -tol:
                    obj += (lo + tol) ** 2 * 100
                if hi > 1.0 + tol:
                    obj += (hi - 1.0 - tol) ** 2 * 100
        return obj
    
    x0 = np.concatenate([centers.ravel(), sides])
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((1e-8, 1.0))
    
    try:
        result = minimize(combined_obj, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter, 'maxls': 50})
        
        if result.success:
            c_new = result.x[:2*n].reshape(n, 2)
            s_new = result.x[2*n:]
            s_new = np.maximum(s_new, 1e-8)
            c_new = _clamp_to_unit(c_new.copy(), s_new.copy())
            s_grown = s_new.copy()
            s_grown = _grow_all_squares(c_new, s_grown, tol=tol)
            
            if _is_valid(c_new, s_grown, tol):
                return c_new, s_grown
            else:
                c_fixed = c_new.copy()
                c_fixed = _resolve_overlap(c_fixed, s_grown, tol)
                c_fixed = _clamp_to_unit(c_fixed, s_grown)
                if _is_valid(c_fixed, s_grown, tol):
                    return c_fixed, s_grown
    except Exception:
        pass
    
    return centers, sides


def _gradient_optimize(centers, sides, tol=1e-7, max_iter=300, beta=50.0, deadline=None):
    """
    Gradient-based optimization using smooth soft-min surrogate for side lengths.
    Optimizes center positions to maximize sum of smooth max-sides.
    """
    n = len(sides)
    
    def neg_smooth_obj(x):
        c = x.reshape(n, 2)
        # Clamp to valid range
        c = np.clip(c, 0.01, 0.99)
        total = 0.0
        for i in range(n):
            total += _smooth_side_i(c[i, 0], c[i, 1], c, sides, i, tol, beta)
        return -total
    
    x0 = centers.ravel().copy()
    
    bounds = [(0.01, 0.99)] * (2 * n)
    
    try:
        result = minimize(neg_smooth_obj, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter, 'maxls': 30, 'ftol': 1e-12})
        
        c_new = result.x.reshape(n, 2)
        c_new = _clamp_to_unit(c_new.copy(), sides.copy())
        
        # Grow to exact max sides
        s_new = sides.copy()
        s_new = _grow_all_squares(c_new, s_new, tol=tol)
        
        if _is_valid(c_new, s_new, tol):
            return c_new, s_new
        else:
            c_fixed = c_new.copy()
            c_fixed = _resolve_overlap(c_fixed, s_new, tol)
            c_fixed = _clamp_to_unit(c_fixed, s_new)
            if _is_valid(c_fixed, s_new, tol):
                return c_fixed, s_new
    except Exception:
        pass
    
    return centers, sides


def _local_search(centers, sides, tol=1e-7, n_iter=200, rng=None, deadline=None):
    """
    Local search: perturb positions, then grow all squares.
    Keep the best result.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n = len(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = _objective(best_c, best_s)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        k = int(rng.integers(1, min(n, 5)))
        idx = rng.choice(n, size=k, replace=False)
        trial_c = best_c.copy()
        for i in idx:
            step = rng.uniform(0.01, 0.08)
            trial_c[i, 0] += rng.uniform(-step, step)
            trial_c[i, 1] += rng.uniform(-step, step)
        
        trial_c = _clamp_to_unit(trial_c, best_s)
        
        if not _is_valid(trial_c, best_s, tol):
            trial_c2 = trial_c.copy()
            trial_c2 = _resolve_overlap(trial_c2, best_s, tol)
            trial_c2 = _clamp_to_unit(trial_c2, best_s)
            if _is_valid(trial_c2, best_s, tol):
                trial_c = trial_c2
            else:
                continue
        
        trial_s = best_s.copy()
        trial_s = _grow_all_squares(trial_c, trial_s, tol)
        
        if _is_valid(trial_c, trial_s, tol):
            obj = _objective(trial_c, trial_s)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_c = trial_c.copy()
                best_s = trial_s.copy()
    
    return best_c, best_s


def _swap_local_search(centers, sides, tol=1e-7, n_iter=100, rng=None, deadline=None):
    """
    Local search with pairwise swap moves: exchange positions of two squares.
    """
    if rng is None:
        rng = np.random.default_rng(555)
    
    n = len(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = _objective(best_c, best_s)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        move_type = rng.random()
        
        if move_type < 0.4:
            # Perturb 1-3 squares
            k = int(rng.integers(1, 4))
            idx = rng.choice(n, size=k, replace=False)
            trial_c = best_c.copy()
            for i in idx:
                step = rng.uniform(0.005, 0.05)
                trial_c[i, 0] += rng.uniform(-step, step)
                trial_c[i, 1] += rng.uniform(-step, step)
        elif move_type < 0.7:
            # Swap centers of two squares
            i, j = rng.choice(n, size=2, replace=False)
            trial_c = best_c.copy()
            trial_c[i], trial_c[j] = trial_c[j].copy(), trial_c[i].copy()
        else:
            # Swap and perturb both
            i, j = rng.choice(n, size=2, replace=False)
            trial_c = best_c.copy()
            trial_c[i], trial_c[j] = trial_c[j].copy(), trial_c[i].copy()
            for idx in [i, j]:
                step = rng.uniform(0.005, 0.03)
                trial_c[idx, 0] += rng.uniform(-step, step)
                trial_c[idx, 1] += rng.uniform(-step, step)
        
        trial_c = _clamp_to_unit(trial_c, best_s)
        
        if not _is_valid(trial_c, best_s, tol):
            trial_c2 = trial_c.copy()
            trial_c2 = _resolve_overlap(trial_c2, best_s, tol)
            trial_c2 = _clamp_to_unit(trial_c2, best_s)
            if _is_valid(trial_c2, best_s, tol):
                trial_c = trial_c2
            else:
                continue
        
        trial_s = best_s.copy()
        trial_s = _grow_all_squares(trial_c, trial_s, tol)
        
        if _is_valid(trial_c, trial_s, tol):
            obj = _objective(trial_c, trial_s)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_c = trial_c.copy()
                best_s = trial_s.copy()
    
    return best_c, best_s


def _simulated_annealing(centers, sides, tol=1e-7, n_iter=800, rng=None,
                         t_start=0.15, t_end=0.003, deadline=None):
    """
    Simulated annealing using smooth repulsion energy as the objective landscape.
    """
    if rng is None:
        rng = np.random.default_rng(777)
    
    n = len(sides)
    cooling = (t_end / t_start) ** (1.0 / max(n_iter - 1, 1))
    
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = _objective(best_c, best_s)
    
    cur_c = centers.copy()
    cur_s = sides.copy()
    cur_energy = _density_energy(cur_c, cur_s, tol=tol, beta=30.0)
    cur_obj = _objective(cur_c, cur_s)
    
    T = t_start
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        T *= cooling
        
        k = int(rng.integers(1, 4))
        idx = rng.choice(n, size=k, replace=False)
        trial_c = cur_c.copy()
        for i in idx:
            step = rng.uniform(0.005, 0.10) * (1.0 + 0.5 * T / t_start)
            trial_c[i, 0] += rng.uniform(-step, step)
            trial_c[i, 1] += rng.uniform(-step, step)
        
        trial_c = _clamp_to_unit(trial_c, cur_s)
        
        trial_s = np.empty(n)
        for i in range(n):
            trial_s[i] = _max_side_at(trial_c[i, 0], trial_c[i, 1], trial_c, cur_s, i, tol)
        
        if not _is_valid(trial_c, trial_s, tol):
            trial_c2 = trial_c.copy()
            trial_c2 = _resolve_overlap(trial_c2, trial_s, tol)
            trial_c2 = _clamp_to_unit(trial_c2, trial_s)
            if not _is_valid(trial_c2, trial_s, tol):
                continue
            trial_c = trial_c2
            for i in range(n):
                trial_s[i] = _max_side_at(trial_c[i, 0], trial_c[i, 1], trial_c, trial_s, i, tol)
        
        if not _is_valid(trial_c, trial_s, tol):
            continue
        
        trial_obj = _objective(trial_c, trial_s)
        trial_energy = _density_energy(trial_c, trial_s, tol=tol, beta=30.0)
        
        delta_obj = trial_obj - cur_obj
        delta_energy = trial_energy - cur_energy
        
        if delta_obj > 0 or delta_energy < 0:
            accept = True
        else:
            delta = -delta_obj + 0.1 * delta_energy
            if T > 1e-10:
                accept_prob = math.exp(-delta / T)
                accept = rng.random() < accept_prob
            else:
                accept = False
        
        if accept:
            cur_c = trial_c.copy()
            cur_s = trial_s.copy()
            cur_obj = trial_obj
            cur_energy = trial_energy
            
            if cur_obj > best_obj + 1e-10:
                best_obj = cur_obj
                best_c = cur_c.copy()
                best_s = cur_s.copy()
    
    return best_c, best_s


def _bisection_side(cx, cy, centers, sides, i, tol=1e-7, n_bisect=80):
    """
    Use bisection to find the exact maximum side for square i at position (cx,cy).
    Binary search on side length s such that square i with side s doesn't overlap
    any other square and stays in bounds.
    """
    # Upper bound from boundaries
    s_hi = min(2.0 * cx, 2.0 * (1.0 - cx), 2.0 * cy, 2.0 * (1.0 - cy))
    if s_hi <= 0:
        return 0.0
    s_lo = 0.0
    
    for _ in range(n_bisect):
        s_mid = (s_lo + s_hi) / 2.0
        # Check if square i with side s_mid is valid
        valid = True
        half_i = s_mid / 2.0
        # Boundary check
        if cx - half_i < -tol or cx + half_i > 1.0 + tol:
            valid = False
        if cy - half_i < -tol or cy + half_i > 1.0 + tol:
            valid = False
        if not valid:
            s_hi = s_mid
            continue
        # Overlap check with all other squares
        for j in range(len(sides)):
            if j == i:
                continue
            half_j = sides[j] / 2.0
            dx = abs(cx - centers[j, 0])
            dy = abs(cy - centers[j, 1])
            t = (s_mid + sides[j]) / 2.0 - tol
            if dx < t and dy < t:
                valid = False
                break
        if valid:
            s_lo = s_mid
        else:
            s_hi = s_mid
    
    return s_lo


def _grow_all_squares_bisection(centers, sides, tol=1e-7, max_rounds=50, n_bisect=80):
    """Grow squares using bisection for exact max side."""
    n = len(sides)
    for _ in range(max_rounds):
        improved = False
        new_sides = np.zeros(n)
        for i in range(n):
            new_sides[i] = _bisection_side(centers[i, 0], centers[i, 1], centers, sides, i, tol, n_bisect)
        if np.any(new_sides > sides + 1e-12):
            sides = new_sides
            improved = True
        if not improved:
            break
    return sides


def _high_precision_polish(centers, sides, tol=1e-9, max_rounds=500, deadline=None):
    """
    Final high-precision polish: use bisection for exact side computation
    plus small-step local search.
    """
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = _objective(best_c, best_s)
    
    # Bisection-based grow with tight tolerance
    trial_s = best_s.copy()
    trial_s = _grow_all_squares_bisection(best_c, trial_s, tol=1e-9, max_rounds=50, n_bisect=80)
    if _is_valid(best_c, trial_s, tol=1e-7):
        obj = _objective(best_c, trial_s)
        if obj > best_obj:
            best_obj = obj
            best_s = trial_s.copy()
    
    # Small-step local search with bisection grow
    rng = np.random.default_rng(99999)
    for it in range(150):
        if deadline is not None and time.time() > deadline:
            break
        k = int(rng.integers(1, 4))
        idx = rng.choice(10, size=k, replace=False)
        trial_c = best_c.copy()
        for i in idx:
            step = rng.uniform(0.001, 0.01)
            trial_c[i, 0] += rng.uniform(-step, step)
            trial_c[i, 1] += rng.uniform(-step, step)
        
        trial_c = _clamp_to_unit(trial_c, best_s)
        
        if not _is_valid(trial_c, best_s, tol=1e-7):
            trial_c2 = trial_c.copy()
            trial_c2 = _resolve_overlap(trial_c2, best_s, tol=1e-7)
            trial_c2 = _clamp_to_unit(trial_c2, best_s)
            if _is_valid(trial_c2, best_s, tol=1e-7):
                trial_c = trial_c2
            else:
                continue
        
        trial_s = best_s.copy()
        trial_s = _grow_all_squares_bisection(trial_c, trial_s, tol=1e-9, max_rounds=50, n_bisect=80)
        
        if _is_valid(trial_c, trial_s, tol=1e-7):
            obj = _objective(trial_c, trial_s)
            if obj > best_obj + 1e-10:
                best_obj = obj
                best_c = trial_c.copy()
                best_s = trial_s.copy()
    
    return best_c, best_s


def _init_symmetric_pattern1():
    """
    Pattern: 4 large squares in corners, 6 smaller filling the cross.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    
    s_large = 0.38
    centers[0] = [s_large/2, s_large/2]
    centers[1] = [1 - s_large/2, s_large/2]
    centers[2] = [s_large/2, 1 - s_large/2]
    centers[3] = [1 - s_large/2, 1 - s_large/2]
    sides[0:4] = s_large
    
    s_small = 0.20
    centers[4] = [0.5, s_large + s_small/2]
    centers[5] = [0.5, 1 - s_large - s_small/2]
    centers[6] = [s_large + s_small/2, 0.5]
    centers[7] = [1 - s_large - s_small/2, 0.5]
    centers[8] = [0.5, 0.5]
    sides[4:9] = s_small
    
    centers[9] = [0.5, 0.5 + s_small]
    sides[9] = s_small * 0.8
    
    return centers, sides


def _init_symmetric_pattern2():
    """
    2x5 grid: 2 rows of 5.
    Side = 1/5 = 0.2
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.full(10, 0.2, dtype=np.float64)
    idx = 0
    for row in range(2):
        for col in range(5):
            centers[idx, 0] = (col + 0.5) * 0.2
            centers[idx, 1] = (row + 0.5) * 0.5
            idx += 1
    return centers, sides


def _init_symmetric_pattern3():
    """
    One large (0.5) in bottom-left, 3x3 grid of small (1/6) in the L-shape.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    side_s = 1.0/6.0
    idx = 1
    for col in range(6):
        centers[idx, 0] = (col + 0.5) * side_s
        centers[idx, 1] = 0.5 + 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    for col in range(3):
        centers[idx, 0] = 0.5 + (col + 0.5) * side_s
        centers[idx, 1] = 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_symmetric_pattern4():
    """
    1 large (0.44) center, 8 around.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.5, 0.5]
    sides[0] = 0.44
    s_small = 0.28
    positions = [
        [0.14, 0.14], [0.86, 0.14], [0.14, 0.86], [0.86, 0.86],
        [0.5, 0.14], [0.5, 0.86], [0.14, 0.5], [0.86, 0.5]
    ]
    for i, pos in enumerate(positions):
        centers[i+1] = pos
        sides[i+1] = s_small
    return centers, sides


def _init_symmetric_pattern5():
    """
    5 large in a cross, 5 small in corners.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.5, 0.5]
    sides[0] = 0.25
    centers[1] = [0.5, 0.25]
    centers[2] = [0.5, 0.75]
    centers[3] = [0.25, 0.5]
    centers[4] = [0.75, 0.5]
    sides[1:5] = 0.25
    s_small = 0.15
    centers[5] = [0.125, 0.125]
    centers[6] = [0.875, 0.125]
    centers[7] = [0.125, 0.875]
    centers[8] = [0.875, 0.875]
    centers[9] = [0.5, 0.05]
    sides[5:10] = s_small
    return centers, sides


def _init_symmetric_pattern6():
    """
    3x3 grid of small squares + 1 large in center gap.
    3x3 grid of side 1/3 fills most, but we need 10 squares.
    Try: 4 corners (0.3), 4 edges (0.2), 2 center (0.15)
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    
    # 4 corners
    s1 = 0.30
    centers[0] = [s1/2, s1/2]
    centers[1] = [1 - s1/2, s1/2]
    centers[2] = [s1/2, 1 - s1/2]
    centers[3] = [1 - s1/2, 1 - s1/2]
    sides[0:4] = s1
    
    # 4 edge middles
    s2 = 0.22
    centers[4] = [0.5, s1 + s2/2]
    centers[5] = [0.5, 1 - s1 - s2/2]
    centers[6] = [s1 + s2/2, 0.5]
    centers[7] = [1 - s1 - s2/2, 0.5]
    sides[4:8] = s2
    
    # 2 center
    s3 = 0.14
    centers[8] = [0.5 - s3/2 - 0.01, 0.5]
    centers[9] = [0.5 + s3/2 + 0.01, 0.5]
    sides[8:10] = s3
    
    return centers, sides


def _init_random(rng, n=10):
    """Generate a random initial configuration."""
    centers = rng.uniform(0.1, 0.9, size=(n, 2))
    sides = rng.uniform(0.1, 0.3, size=n)
    return centers, sides


def squares_in_square10() -> tuple:
    """
    Maximize the sum of side lengths of 10 non-overlapping axis-aligned squares in the unit square.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    
    inits = [
        _init_symmetric_pattern1(),
        _init_symmetric_pattern2(),
        _init_symmetric_pattern3(),
        _init_symmetric_pattern4(),
        _init_symmetric_pattern5(),
        _init_symmetric_pattern6(),
    ]
    
    best_centers = None
    best_sides = None
    best_obj = -1.0
    
    for init_idx, (centers, sides) in enumerate(inits):
        if time.time() > deadline - 60:
            break
        
        centers = centers.copy()
        sides = sides.copy()
        
        # Ensure validity
        centers = _clamp_to_unit(centers, sides)
        if not _is_valid(centers, sides, tol=1e-7):
            centers = _resolve_overlap(centers, sides, tol=1e-7)
            centers = _clamp_to_unit(centers, sides)
            if not _is_valid(centers, sides, tol=1e-7):
                continue
        
        # Grow all squares
        sides = _grow_all_squares(centers, sides, tol=1e-7)
        if not _is_valid(centers, sides, tol=1e-7):
            continue
        
        obj = _objective(centers, sides)
        if obj > best_obj:
            best_obj = obj
            best_centers = centers.copy()
            best_sides = sides.copy()
        
        # Gradient-based optimization
        if time.time() < deadline - 80:
            grad_c, grad_s = _gradient_optimize(centers, sides, tol=1e-7, max_iter=200, beta=50.0,
                                                deadline=deadline - 30)
            if _is_valid(grad_c, grad_s, tol=1e-7):
                obj_grad = _objective(grad_c, grad_s)
                if obj_grad > best_obj:
                    best_obj = obj_grad
                    best_centers = grad_c.copy()
                    best_sides = grad_s.copy()
        
        # Local search
        if time.time() < deadline - 60:
            rng = np.random.default_rng(init_idx * 7919 + 1)
            opt_c, opt_s = _local_search(centers, sides, tol=1e-7, n_iter=150, rng=rng,
                                         deadline=deadline - 40)
            if _is_valid(opt_c, opt_s, tol=1e-7):
                obj2 = _objective(opt_c, opt_s)
                if obj2 > best_obj:
                    best_obj = obj2
                    best_centers = opt_c.copy()
                    best_sides = opt_s.copy()
        
        # Swap-based local search
        if time.time() < deadline - 50:
            rng_swap = np.random.default_rng(init_idx * 104729 + 3)
            swap_c, swap_s = _swap_local_search(best_centers, best_sides, tol=1e-7, n_iter=100,
                                                rng=rng_swap, deadline=deadline - 30)
            if _is_valid(swap_c, swap_s, tol=1e-7):
                obj_swap = _objective(swap_c, swap_s)
                if obj_swap > best_obj:
                    best_obj = obj_swap
                    best_centers = swap_c.copy()
                    best_sides = swap_s.copy()
        
        # Simulated annealing
        if time.time() < deadline - 40:
            rng_sa = np.random.default_rng(init_idx * 104729 + 5)
            sa_c, sa_s = _simulated_annealing(centers, sides, tol=1e-7, n_iter=500, rng=rng_sa,
                                              t_start=0.12, t_end=0.003, deadline=deadline - 20)
            if _is_valid(sa_c, sa_s, tol=1e-7):
                obj3 = _objective(sa_c, sa_s)
                if obj3 > best_obj:
                    best_obj = obj3
                    best_centers = sa_c.copy()
                    best_sides = sa_s.copy()
        
        # SciPy optimization
        if time.time() < deadline - 30:
            sp_c, sp_s = _scipy_optimize(best_centers, best_sides, tol=1e-7, max_iter=80,
                                         deadline=deadline - 15)
            if _is_valid(sp_c, sp_s, tol=1e-7):
                obj_sp = _objective(sp_c, sp_s)
                if obj_sp > best_obj:
                    best_obj = obj_sp
                    best_centers = sp_c.copy()
                    best_sides = sp_s.copy()
    
    # If nothing valid, fallback
    if best_centers is None:
        centers, sides = _init_symmetric_pattern2()
        centers = _clamp_to_unit(centers, sides)
        sides = _grow_all_squares(centers, sides, tol=1e-7)
        best_centers = centers
        best_sides = sides
    else:
        # Additional refinement on best result
        if time.time() < deadline - 30:
            rng_sa2 = np.random.default_rng(987654)
            sa_c, sa_s = _simulated_annealing(best_centers, best_sides, tol=1e-7, n_iter=400, rng=rng_sa2,
                                              t_start=0.06, t_end=0.002, deadline=deadline - 20)
            if _is_valid(sa_c, sa_s, tol=1e-7):
                obj_sa = _objective(sa_c, sa_s)
                if obj_sa > best_obj:
                    best_obj = obj_sa
                    best_centers = sa_c.copy()
                    best_sides = sa_s.copy()
        
        # Gradient optimization on best
        if time.time() < deadline - 25:
            grad_c, grad_s = _gradient_optimize(best_centers, best_sides, tol=1e-7, max_iter=150,
                                                beta=80.0, deadline=deadline - 15)
            if _is_valid(grad_c, grad_s, tol=1e-7):
                obj_grad = _objective(grad_c, grad_s)
                if obj_grad > best_obj:
                    best_obj = obj_grad
                    best_centers = grad_c.copy()
                    best_sides = grad_s.copy()
        
        # High-precision final polish with bisection
        if time.time() < deadline - 10:
            polished_c, polished_s = _high_precision_polish(best_centers, best_sides, tol=1e-9,
                                                            max_rounds=500, deadline=deadline - 2)
            if _is_valid(polished_c, polished_s, tol=1e-7):
                obj_polished = _objective(polished_c, polished_s)
                if obj_polished > best_obj:
                    best_centers = polished_c
                    best_sides = polished_s
                    best_obj = obj_polished
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
