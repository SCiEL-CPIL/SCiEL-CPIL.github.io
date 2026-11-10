import numpy as np
from scipy.spatial.distance import pdist
import time
import scipy.optimize as so
from scipy.optimize import LinearConstraint, NonlinearConstraint

def _compute_score(points):
    """Compute R = (d_min / d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    if d_min == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _pair_indices(n):
    """Return indices for all pairs (i, j) with i < j."""
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return np.array(pairs)

def _soft_max(vals, beta=100.0):
    """Compute soft-max of squared distances using log-sum-exp for smoothness."""
    # vals are squared distances
    max_val = np.max(vals)
    if max_val < 1e-15:
        return 0.0
    exp_vals = np.exp(-beta * (vals - max_val))
    log_sum = np.log(np.sum(exp_vals))
    return max_val + log_sum / beta

def _soft_min(vals, beta=100.0):
    """Compute soft-min of squared distances using log-sum-exp for smoothness."""
    min_val = np.min(vals)
    if min_val < 1e-15:
        return 0.0
    exp_vals = np.exp(beta * (vals - min_val))
    log_sum = np.log(np.sum(exp_vals))
    return min_val - log_sum / beta

def _trust_constr_smooth(points, max_iter=200, tol=1e-8, beta=50.0):
    """
    Maximize the minimum pairwise distance subject to max pairwise distance <= 1.
    Uses trust-constr with smooth surrogate objectives and analytical gradients.
    """
    n = len(points)
    x0 = points.flatten()
    pairs = _pair_indices(n)
    m = len(pairs)  # 120 for n=16

    def objective(v):
        pts = v.reshape(n, 2)
        d_sq = pdist(pts) ** 2
        # We want to maximize min distance, so minimize -soft_min
        return -_soft_min(d_sq, beta=beta)

    def objective_grad(v):
        pts = v.reshape(n, 2)
        d_sq = pdist(pts) ** 2
        d = np.sqrt(d_sq + 1e-20)

        # For soft-min, the gradient is weighted by the exp terms
        min_val = np.min(d_sq)
        exp_vals = np.exp(beta * (d_sq - min_val))
        weights = exp_vals / np.sum(exp_vals)

        grad = np.zeros(2 * n)
        for k, (i, j) in enumerate(pairs):
            w = weights[k]
            diff = pts[i] - pts[j]
            dist = d[k]
            if dist > 1e-12:
                grad[2 * i: 2 * i + 2] -= w * 2 * diff / dist
                grad[2 * j: 2 * j + 2] += w * 2 * diff / dist
        return grad

    # Constraint: all pairwise squared distances <= 1
    def max_dist_constraint(v):
        pts = v.reshape(n, 2)
        d_sq = pdist(pts) ** 2
        return 1.0 - d_sq  # Should be >= 0

    def max_dist_jac(v):
        pts = v.reshape(n, 2)
        d_sq = pdist(pts) ** 2
        d = np.sqrt(d_sq + 1e-20)
        jac = np.zeros((m, 2 * n))
        for k, (i, j) in enumerate(pairs):
            diff = pts[i] - pts[j]
            dist = d[k]
            if dist > 1e-12:
                jac[k, 2 * i: 2 * i + 2] = -2 * diff / dist
                jac[k, 2 * j: 2 * j + 2] = 2 * diff / dist
        return jac

    cons = NonlinearConstraint(max_dist_constraint, 0, np.inf, jac=max_dist_jac)
    bounds = [(-2, 2)] * (2 * n)

    try:
        res = so.minimize(objective, x0, method='trust-constr', jac=objective_grad,
                          bounds=bounds, constraints=[cons],
                          options={'maxiter': max_iter, 'gtol': tol, 'xtol': tol,
                                   'barrier_tol': tol, 'barrier_alpha': 1e-4})
        return res.x
    except Exception:
        return x0

def _sqp_maximize_min_dist(points, max_dist_sq, max_iter=100, tol=1e-8):
    """
    Maximize the minimum distance between points, subject to the constraint 
    that all pairwise distances are <= max_dist_sq.
    Uses SLSQP with smooth log-sum-exp surrogate for both objective and constraints.
    """
    n = len(points)
    x0 = points.flatten()
    pairs = _pair_indices(n)
    m = len(pairs)

    def objective(v):
        pts = v.reshape(n, 2)
        d_sq = pdist(pts) ** 2
        # Minimize negative soft-min of squared distances
        return -_soft_min(d_sq, beta=30.0)

    def jac(v):
        pts = v.reshape(n, 2)
        d_sq = pdist(pts) ** 2
        d = np.sqrt(d_sq + 1e-20)
        beta = 30.0
        min_val = np.min(d_sq)
        exp_vals = np.exp(beta * (d_sq - min_val))
        weights = exp_vals / np.sum(exp_vals)

        grad = np.zeros(2 * n)
        for k, (i, j) in enumerate(pairs):
            w = weights[k]
            diff = pts[i] - pts[j]
            dist = d[k]
            if dist > 1e-12:
                grad[2 * i: 2 * i + 2] -= w * 2 * diff / dist
                grad[2 * j: 2 * j + 2] += w * 2 * diff / dist
        return grad

    # Smooth constraint: soft-max of squared distances <= max_dist_sq
    def max_dist_constraint(v):
        pts = v.reshape(n, 2)
        d_sq = pdist(pts) ** 2
        return max_dist_sq - _soft_max(d_sq, beta=30.0)

    def max_dist_jac(v):
        pts = v.reshape(n, 2)
        d_sq = pdist(pts) ** 2
        d = np.sqrt(d_sq + 1e-20)
        beta = 30.0
        max_val = np.max(d_sq)
        exp_vals = np.exp(-beta * (d_sq - max_val))
        weights = exp_vals / np.sum(exp_vals)

        jac = np.zeros(2 * n)
        for k, (i, j) in enumerate(pairs):
            w = weights[k]
            diff = pts[i] - pts[j]
            dist = d[k]
            if dist > 1e-12:
                jac[2 * i: 2 * i + 2] += w * 2 * diff / dist
                jac[2 * j: 2 * j + 2] -= w * 2 * diff / dist
        return jac

    constraints = [{'type': 'ineq', 'fun': max_dist_constraint, 'jac': max_dist_jac}]
    bounds = [(-10, 10)] * (2 * n)

    try:
        res = so.minimize(objective, x0, method='SLSQP', jac=jac, bounds=bounds,
                          constraints=constraints, options={'maxiter': max_iter, 'ftol': tol})
        if res.success:
            return res.x
        else:
            return x0
    except Exception:
        return x0

def _bisection_solve(initial_points, time_limit=500.0):
    """
    Use trust-constr and SLSQP to maximize the min/max distance ratio.
    """
    n = len(initial_points)
    best_points = initial_points.copy()
    best_ratio = 0.0

    start_time = time.time()

    for _ in range(20):
        if time.time() - start_time > time_limit:
            break

        d = pdist(best_points)
        d_max = d.max()
        if d_max > 1e-9:
            normalized_points = best_points / d_max
        else:
            normalized_points = best_points

        # Try smooth trust-constr first
        try:
            result_x = _trust_constr_smooth(normalized_points, max_iter=150, tol=1e-8, beta=40.0)
            new_points = result_x.reshape(n, 2)

            d_new = pdist(new_points)
            d_min_new = d_new.min()
            d_max_new = d_new.max()

            if d_max_new <= 1.0 + 1e-4 and d_min_new > 1e-9:
                ratio_new = d_min_new / d_max_new
                if ratio_new > best_ratio:
                    best_ratio = ratio_new
                    best_points = new_points.copy()
        except Exception:
            pass

        # Also try SLSQP
        if time.time() - start_time > time_limit - 5.0:
            break

        try:
            result_x = _sqp_maximize_min_dist(normalized_points, 1.0, max_iter=80, tol=1e-7)
            if result_x is not None and len(result_x) == 2 * n:
                new_points = result_x.reshape(n, 2)

                d_new = pdist(new_points)
                d_min_new = d_new.min()
                d_max_new = d_new.max()

                if d_max_new <= 1.0 + 1e-5 and d_min_new > 1e-9:
                    ratio_new = d_min_new / d_max_new
                    if ratio_new > best_ratio:
                        best_ratio = ratio_new
                        best_points = new_points.copy()
        except Exception:
            pass

    return best_points, best_ratio

def _generate_initial_points(seed=0):
    """Generate multiple structured initial configurations."""
    n = 16
    rng = np.random.RandomState(seed)
    candidates = []

    # 1. Regular polygon + center (8-gon + inner 8-gon scaled)
    angles_outer = 2 * np.pi * np.arange(8) / 8
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    angles_inner = 2 * np.pi * np.arange(8) / 8 + np.pi / 8
    inner = 0.5 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    candidates.append(np.vstack([outer, inner]))

    # 2. 4x4 grid
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid = np.column_stack(np.meshgrid(xs, ys, indexing='ij')).reshape(-1, 2)
    candidates.append(grid)

    # 3. Two concentric 8-gons
    angles1 = 2 * np.pi * np.arange(8) / 8
    angles2 = 2 * np.pi * np.arange(8) / 8 + np.pi / 8
    r1 = 1.0
    r2 = 0.6
    p1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
    p2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    candidates.append(np.vstack([p1, p2]))

    # 4. Hexagonal-ish arrangement: 1 center + 6 ring + 9 outer
    center = np.array([[0.0, 0.0]])
    ring1_angles = 2 * np.pi * np.arange(6) / 6
    ring1 = 0.8 * np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    ring2_angles = 2 * np.pi * np.arange(9) / 9
    ring2 = 1.5 * np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    candidates.append(np.vstack([center, ring1, ring2]))

    # 5. Random
    candidates.append(rng.randn(n, 2) * 0.5)

    # 6. Another structured: 2x8 rectangle with jitter
    xs = np.tile(np.array([-1, 1]), 8)
    ys = np.linspace(-1, 1, 16)
    rect = np.column_stack([xs, ys])
    candidates.append(rect + rng.randn(16, 2) * 0.05)

    # 7. Random with larger spread
    candidates.append(rng.randn(n, 2) * 2.0)

    return candidates

def _simulated_annealing(points, time_budget=100.0, seed=42):
    """
    Simulated annealing to maximize R = (d_min/d_max)^2.
    """
    n = len(points)
    rng = np.random.RandomState(seed)

    # Normalize so max distance is ~1
    d = pdist(points)
    d_max = d.max()
    if d_max > 1e-9:
        current_points = points / d_max
    else:
        current_points = points.copy()

    current_score = _compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score

    start_time = time.time()
    T = 0.05  # Initial temperature
    alpha = 0.999  # Cooling rate
    step_size = 0.05

    while time.time() - start_time < time_budget:
        if T < 1e-8:
            break

        # Generate a perturbation
        idx = rng.randint(0, n)
        delta = rng.randn(2) * step_size
        new_points = current_points.copy()
        new_points[idx] += delta

        new_score = _compute_score(new_points)
        delta_score = new_score - current_score

        # Accept or reject
        if delta_score > 0 or rng.random() < np.exp(delta_score / T):
            current_points = new_points
            current_score = new_score

            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()

        T *= alpha

    return best_points, best_score

def _polish_trust_constr(points, time_budget=50.0):
    """
    Final polish using trust-constr with tighter tolerances.
    """
    n = len(points)
    d = pdist(points)
    d_max = d.max()
    if d_max > 1e-9:
        normalized = points / d_max
    else:
        normalized = points.copy()

    start_time = time.time()
    best = normalized.copy()
    best_score = _compute_score(best)

    # Multiple restarts with slight perturbations
    rng = np.random.RandomState(7)
    for attempt in range(12):
        if time.time() - start_time > time_budget:
            break

        if attempt == 0:
            x0 = normalized.flatten()
        else:
            x0 = best.flatten() + rng.randn(2 * n) * 0.005

        try:
            result_x = _trust_constr_smooth(x0.reshape(n, 2), max_iter=200, tol=1e-10, beta=60.0)
            new_points = result_x.reshape(n, 2)

            d_new = pdist(new_points)
            d_min_new = d_new.min()
            d_max_new = d_new.max()

            if d_max_new <= 1.0 + 1e-4 and d_min_new > 1e-9:
                new_score = (d_min_new / d_max_new) ** 2
                if new_score > best_score:
                    best_score = new_score
                    best = new_points.copy()
        except Exception:
            pass

    return best, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    deadline = time.time() + 610.0  # Safety margin before 630s kill

    # Generate initial candidates
    initial_points_list = _generate_initial_points(seed=42)

    best_points = initial_points_list[0].copy()
    best_score = _compute_score(best_points)

    # Phase 1: Optimization from each initial configuration
    for i, init_pts in enumerate(initial_points_list):
        if time.time() > deadline - 80.0:
            break

        time_budget = min(40.0, deadline - time.time() - 60.0)
        if time_budget < 5.0:
            break

        try:
            result_pts, result_ratio = _bisection_solve(init_pts, time_limit=time_budget)
            result_score = result_ratio ** 2
            if result_score > best_score:
                best_points = result_pts
                best_score = result_score
        except Exception as e:
            pass

    # Phase 2: Simulated annealing from the best point found
    remaining = deadline - time.time() - 20.0
    if remaining > 30.0:
        try:
            sa_points, sa_score = _simulated_annealing(best_points, time_budget=remaining - 20.0, seed=123)
            if sa_score > best_score:
                best_points = sa_points
                best_score = sa_score
        except Exception:
            pass

    # Phase 3: Final trust-constr polish
    remaining = deadline - time.time() - 5.0
    if remaining > 15.0:
        try:
            polished_pts, polished_score = _polish_trust_constr(best_points, time_budget=remaining - 5.0)
            if polished_score > best_score:
                best_points = polished_pts
                best_score = polished_score
        except Exception:
            pass

    # Phase 4: Additional SLSQP polish from best point
    remaining = deadline - time.time() - 3.0
    if remaining > 10.0:
        try:
            n = len(best_points)
            d = pdist(best_points)
            d_max = d.max()
            if d_max > 1e-9:
                normalized = best_points / d_max
            else:
                normalized = best_points

            result_x = _sqp_maximize_min_dist(normalized, 1.0, max_iter=100, tol=1e-9)
            if result_x is not None and len(result_x) == 2 * n:
                new_points = result_x.reshape(n, 2)
                d_new = pdist(new_points)
                d_min_new = d_new.min()
                d_max_new = d_new.max()
                if d_max_new <= 1.0 + 1e-4 and d_min_new > 1e-9:
                    new_score = (d_min_new / d_max_new) ** 2
                    if new_score > best_score:
                        best_points = new_points
                        best_score = new_score
        except Exception:
            pass

    return best_points
