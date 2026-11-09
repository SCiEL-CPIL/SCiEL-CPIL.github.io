import numpy as np
import time
from scipy.optimize import minimize


def _is_feasible(centers, sides, tol=1e-7):
    """Check if the configuration is feasible."""
    n = len(sides)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return False
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(s)):
            return False
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True


def _make_2big_8small():
    """Construct 2 large (0.5) + 8 small (0.25) tiling. Sum = 3.0."""
    centers = np.array([
        [0.25, 0.75],
        [0.75, 0.75],
        [0.125, 0.125], [0.375, 0.125], [0.625, 0.125], [0.875, 0.125],
        [0.125, 0.375], [0.375, 0.375], [0.625, 0.375], [0.875, 0.375]
    ])
    sides = np.array([0.5, 0.5] + [0.25]*8)
    return centers, sides


def _make_4_3_3_packing():
    """
    Try 4 squares of size ~0.4 in a 2x2 grid at bottom,
    3 squares of size ~0.3 in middle, 3 squares of size ~0.25 at top.
    """
    s1 = 0.4
    s2 = 0.3
    s3 = 0.25

    centers = np.array([
        [s1/2, s1/2], [1-s1/2, s1/2],
        [s1/2, 1-s1/2], [1-s1/2, 1-s1/2],
        [s2/2, s1 + s2/2], [0.5, s1 + s2/2], [1-s2/2, s1 + s2/2],
        [s3/2, 1-s3/2], [0.5, 1-s3/2], [1-s3/2, 1-s3/2]
    ])
    sides = np.array([s1]*4 + [s2]*3 + [s3]*3)
    return centers, sides


def _make_staggered_packing():
    """Staggered arrangement."""
    centers = np.array([
        [0.2, 0.2], [0.8, 0.2],
        [0.2, 0.6], [0.8, 0.6],
        [0.5, 0.4], [0.5, 0.8],
        [0.1, 0.85], [0.9, 0.85],
        [0.35, 0.1], [0.65, 0.1]
    ])
    sides = np.array([0.35, 0.35, 0.35, 0.35, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2])
    return centers, sides


def _push_apart(centers, sides, tol=1e-7, max_iter=1000, deadline=None):
    """
    Iteratively push overlapping squares apart and grow them.
    """
    n = len(sides)
    c = centers.copy()
    s = sides.copy()

    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        improved = False

        # First, ensure all squares are within bounds
        for i in range(n):
            si = s[i]
            c[i, 0] = np.clip(c[i, 0], si/2, 1 - si/2)
            c[i, 1] = np.clip(c[i, 1], si/2, 1 - si/2)

        # Try to resolve overlaps by pushing squares apart
        for i in range(n):
            for j in range(i+1, n):
                si, sj = s[i], s[j]
                t = (si + sj)/2

                dx = c[j, 0] - c[i, 0]
                dy = c[j, 1] - c[i, 1]

                if abs(dx) < t and abs(dy) < t:
                    pen_x = t - abs(dx)
                    pen_y = t - abs(dy)

                    if pen_x < pen_y:
                        push = pen_x / 2 + 0.001
                        if dx >= 0:
                            c[i, 0] -= push
                            c[j, 0] += push
                        else:
                            c[i, 0] += push
                            c[j, 0] -= push
                    else:
                        push = pen_y / 2 + 0.001
                        if dy >= 0:
                            c[i, 1] -= push
                            c[j, 1] += push
                        else:
                            c[i, 1] += push
                            c[j, 1] -= push

                    improved = True

        # Clip to bounds after pushing
        for i in range(n):
            si = s[i]
            c[i, 0] = np.clip(c[i, 0], si/2, 1 - si/2)
            c[i, 1] = np.clip(c[i, 1], si/2, 1 - si/2)

        # Try to grow each square
        for i in range(n):
            si = s[i]
            max_s = 2 * min(c[i, 0], 1 - c[i, 0], c[i, 1], 1 - c[i, 1])

            for j in range(n):
                if i == j:
                    continue
                sj = s[j]
                dx = abs(c[i, 0] - c[j, 0])
                dy = abs(c[i, 1] - c[j, 1])

                max_from_x = 2*dx - si - sj
                max_from_y = 2*dy - si - sj
                max_s = min(max_s, max_from_x, max_from_y)

            if max_s > si + 1e-8:
                new_s = min(max_s, si + 0.02)
                s[i] = new_s
                improved = True

        if not improved:
            break

    return c, s


def _local_refine_grow(centers, sides, tol=1e-7, max_iter=3000, deadline=None):
    """
    Local refinement: try to grow squares while maintaining feasibility.
    Uses multiple delta scales for fine-tuning.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)

    deltas = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]

    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        improved = False
        for i in range(n):
            for delta in deltas:
                new_sides = best_sides.copy()
                new_sides[i] = best_sides[i] + delta
                new_centers = best_centers.copy()

                s = new_sides[i]
                cx = np.clip(best_centers[i, 0], s/2, 1 - s/2)
                cy = np.clip(best_centers[i, 1], s/2, 1 - s/2)
                new_centers[i] = [cx, cy]

                if _is_feasible(new_centers, new_sides, tol):
                    best_centers = new_centers
                    best_sides = new_sides
                    best_val = np.sum(new_sides)
                    improved = True
                    break

        if not improved:
            break

    return best_centers, best_sides


def _smooth_refine(centers, sides, tol=1e-7, deadline=None):
    """
    Smooth SLSQP-based refinement using a soft penalty surrogate.
    Variables: [cx_0..cx_9, cy_0..cy_9, s_0..s_9] (30 vars)
    Objective: maximize sum(s_i)
    Uses soft-abs via log-sum-exp for smooth non-overlap constraints.
    """
    n = len(sides)
    x0 = np.concatenate([centers.flatten(), sides])

    k = 50.0  # Soft-abs sharpness

    def soft_abs(a, k=k):
        """Smooth approximation of |a| using log-sum-exp."""
        return (a + np.logaddexp(0, -k*a)) / k

    def objective(x):
        s = x[2*n:]
        return -np.sum(s)

    def make_overlap_constraint(i, j):
        def c(x):
            cx_i, cy_i = x[i], x[n+i]
            cx_j, cy_j = x[j], x[n+j]
            s_i, s_j = x[2*n+i], x[2*n+j]
            dx = cx_i - cx_j
            dy = cy_i - cy_j
            # Non-overlap: |dx| + |dy| >= (s_i + s_j)/2 - tol
            # Using soft-abs for smoothness
            return soft_abs(dx) + soft_abs(dy) - (s_i + s_j)/2 + tol
        return c

    def make_boundary_constraint(i, axis):
        """Constraint: s_i/2 <= coord_i <= 1 - s_i/2"""
        if axis == 0:
            def c(x):
                cx = x[i]
                s = x[2*n+i]
                return np.array([cx - s/2, 1 - s/2 - cx])
            return c
        else:
            def c(x):
                cy = x[n+i]
                s = x[2*n+i]
                return np.array([cy - s/2, 1 - s/2 - cy])
            return c

    def make_side_positive(i):
        def c(x):
            return x[2*n+i] - 0.01
        return c

    # Build constraints
    constraints = []
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1)})
        constraints.append({'type': 'ineq', 'fun': make_side_positive(i)})

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.01, 1.0))  # cx
    for i in range(n):
        bounds.append((0.01, 1.0))  # cy
    for i in range(n):
        bounds.append((0.01, 1.0))  # s

    try:
        result = minimize(
            objective, x0, method='SLSQP', bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-12}
        )

        if np.all(np.isfinite(result.x)):
            c = result.x[:2*n].reshape(n, 2)
            s = result.x[2*n:]
            for i in range(n):
                s[i] = max(0.01, s[i])
                c[i, 0] = np.clip(c[i, 0], s[i]/2, 1 - s[i]/2)
                c[i, 1] = np.clip(c[i, 1], s[i]/2, 1 - s[i]/2)

            if _is_feasible(c, s):
                return c, s
    except Exception:
        pass

    return centers, sides


def _local_search_with_push(centers, sides, tol=1e-7, deadline=None, max_iter=300):
    """
    Local search that uses push-apart to resolve overlaps after random moves.
    """
    n = len(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    best_val = np.sum(sides)

    rng = np.random.default_rng(42)

    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        i = rng.integers(n)

        dx = rng.uniform(-0.05, 0.05)
        dy = rng.uniform(-0.05, 0.05)
        ds = rng.uniform(-0.02, 0.05)

        new_c = best_c.copy()
        new_s = best_s.copy()

        new_s[i] = np.clip(new_s[i] + ds, 0.05, 0.6)
        new_c[i, 0] = np.clip(new_c[i, 0] + dx, new_s[i]/2, 1 - new_s[i]/2)
        new_c[i, 1] = np.clip(new_c[i, 1] + dy, new_s[i]/2, 1 - new_s[i]/2)

        new_c, new_s = _push_apart(new_c, new_s, tol, max_iter=100, deadline=deadline)

        if _is_feasible(new_c, new_s):
            new_val = np.sum(new_s)
            if new_val > best_val + 1e-10:
                best_c = new_c
                best_s = new_s
                best_val = new_val

    return best_c, best_s


def _nelder_mead_refine(centers, sides, deadline=None):
    """
    Use Nelder-Mead to refine the packing.
    """
    n = 10

    def flatten(c, s):
        return np.concatenate([c.flatten(), s])

    def unflatten(x):
        c = x[:2*n].reshape(n, 2)
        s = x[2*n:3*n]
        return c, s

    def objective(x):
        c, s = unflatten(x)

        for i in range(n):
            s[i] = max(0.01, s[i])
            c[i, 0] = np.clip(c[i, 0], s[i]/2, 1 - s[i]/2)
            c[i, 1] = np.clip(c[i, 1], s[i]/2, 1 - s[i]/2)

        penalty = 0
        for i in range(n):
            cx, cy = c[i]
            si = s[i]
            if cx - si/2 < 0:
                penalty += abs(cx - si/2)
            if cx + si/2 > 1:
                penalty += abs(cx + si/2 - 1)
            if cy - si/2 < 0:
                penalty += abs(cy - si/2)
            if cy + si/2 > 1:
                penalty += abs(cy + si/2 - 1)

        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j])/2
                dx = abs(c[i, 0] - c[j, 0])
                dy = abs(c[i, 1] - c[j, 1])
                if dx < t and dy < t:
                    penalty += min(t - dx, t - dy)

        return -np.sum(s) + 100 * penalty

    x0 = flatten(centers, sides)

    try:
        result = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )

        if np.all(np.isfinite(result.x)):
            c, s = unflatten(result.x)
            for i in range(n):
                s[i] = max(0.01, s[i])
                c[i, 0] = np.clip(c[i, 0], s[i]/2, 1 - c[i, 0] if c[i, 0] > 0.5 else 1 - s[i]/2)
                c[i, 1] = np.clip(c[i, 1], s[i]/2, 1 - s[i]/2)

            if _is_feasible(c, s):
                return c, s
    except Exception:
        pass

    return centers, sides


def _multi_start_optimize(deadline):
    """
    Try multiple initial configurations and optimize each.
    """
    best_c = None
    best_s = None
    best_val = -np.inf

    configs = [
        _make_2big_8small(),
        _make_4_3_3_packing(),
        _make_staggered_packing()
    ]

    for c, s in configs:
        if not _is_feasible(c, s):
            continue

        val = np.sum(s)

        # Try to improve with push-apart
        if time.time() < deadline - 80:
            c2, s2 = _push_apart(c, s, deadline=deadline - 70)
            if _is_feasible(c2, s2):
                val2 = np.sum(s2)
                if val2 > val:
                    c, s = c2, s2
                    val = val2

        # Try local search
        if time.time() < deadline - 50:
            c3, s3 = _local_search_with_push(c, s, deadline=deadline - 40)
            if _is_feasible(c3, s3):
                val3 = np.sum(s3)
                if val3 > val:
                    c, s = c3, s3
                    val = val3

        if val > best_val:
            best_val = val
            best_c = c.copy()
            best_s = s.copy()

    return best_c, best_s


def squares_in_square10():
    start_time = time.time()
    deadline = start_time + 580

    # Start with multi-start optimization using push-apart
    best_c, best_s = _multi_start_optimize(deadline - 200)

    if best_c is None:
        best_c, best_s = _make_2big_8small()

    # Smooth SLSQP refinement
    if time.time() < deadline - 150:
        try:
            c2, s2 = _smooth_refine(best_c, best_s, deadline=deadline - 130)
            if _is_feasible(c2, s2) and np.sum(s2) > np.sum(best_s):
                best_c, best_s = c2, s2
        except Exception:
            pass

    # Nelder-Mead refinement
    if time.time() < deadline - 100:
        try:
            c3, s3 = _nelder_mead_refine(best_c, best_s, deadline=deadline - 80)
            if _is_feasible(c3, s3) and np.sum(s3) > np.sum(best_s):
                best_c, best_s = c3, s3
        except Exception:
            pass

    # Local refinement with grow
    if time.time() < deadline - 30:
        try:
            c4, s4 = _local_refine_grow(best_c, best_s, deadline=deadline - 10)
            if _is_feasible(c4, s4) and np.sum(s4) > np.sum(best_s):
                best_c, best_s = c4, s4
        except Exception:
            pass

    # Final push-apart pass
    if time.time() < deadline - 5:
        try:
            c5, s5 = _push_apart(best_c, best_s, deadline=deadline - 2)
            if _is_feasible(c5, s5) and np.sum(s5) > np.sum(best_s):
                best_c, best_s = c5, s5
        except Exception:
            pass

    return best_c, best_s
