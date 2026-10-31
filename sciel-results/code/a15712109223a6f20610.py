"""Circle packing in perimeter-4 rectangle: SLSQP + Advanced Expansion Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Boundary constraints
    # x_i >= r_i, x_i <= w - r_i, y_i >= r_i, y_i <= h - r_i
    viol_x_lo = np.maximum(r - x, 0.0)
    viol_x_hi = np.maximum(r - (w - x), 0.0)
    viol_y_lo = np.maximum(r - y, 0.0)
    viol_y_hi = np.maximum(r - (h - y), 0.0)
    boundary_viol = viol_x_lo + viol_x_hi + viol_y_lo + viol_y_hi
    obj += penalty_weight * np.sum(boundary_viol * boundary_viol)

    return obj


def _is_valid(circles, w, h, tol=1e-7):
    """Check if circles are valid (disjoint and within bounds)."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    if np.any(r < -tol):
        return False
    r = np.maximum(r, 0.0)

    if np.any(x - r < -tol) or np.any(x + r > w + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > h + tol):
        return False

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_pairs = r_sum[iu]

    if np.any(d_pairs < r_pairs - tol):
        return False

    return True


def _value(circles, w, h):
    """Compute the value (sum of radii) if valid, else 0."""
    if not _is_valid(circles, w, h):
        return 0.0
    return np.sum(circles[:, 2])


def _force_relax_unified(circles, w, h, max_iter=500, tol=1e-8):
    """
    Force relax overlapping circles by pushing them apart and shrinking radii.
    Returns (circles, w, h, success).
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for _ in range(max_iter):
        # Check overlaps and boundaries
        moved = False

        # Boundary constraints
        for i in range(N):
            if x[i] < r[i]:
                x[i] = r[i]
                moved = True
            if x[i] > w - r[i]:
                x[i] = w - r[i]
                moved = True
            if y[i] < r[i]:
                y[i] = r[i]
                moved = True
            if y[i] > h - r[i]:
                y[i] = h - r[i]
                moved = True

        # Pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]

                if d < min_d - tol:
                    # Push apart
                    if d < 1e-10:
                        # Random direction
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        d = 1.0
                    else:
                        dx /= d
                        dy /= d

                    overlap = min_d - d
                    # Distribute push proportionally to radii (or equally)
                    push = overlap / 2.0 + 1e-4
                    x[i] -= dx * push
                    y[i] -= dy * push
                    x[j] += dx * push
                    y[j] += dy * push
                    moved = True

        # Re-check boundaries after pushing
        for i in range(N):
            if x[i] < r[i]:
                x[i] = r[i]
            if x[i] > w - r[i]:
                x[i] = w - r[i]
            if y[i] < r[i]:
                y[i] = r[i]
            if y[i] > h - r[i]:
                y[i] = h - r[i]

        if not moved:
            return np.column_stack((x, y, r)), w, h, True

    return np.column_stack((x, y, r)), w, h, False


def _polish_result(circles, w, h):
    """
    Polish the result by trying to expand radii and adjust positions.
    Uses SLSQP to maximize sum of radii given fixed w, h.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    # Initial state must be valid
    if not _is_valid(np.column_stack((x, y, r)), w, h):
        return np.column_stack((x, y, r)), w, h

    # Objective: maximize sum(r)
    def obj(z):
        xi = z[0::3][:N]
        yi = z[1::3][:N]
        ri = z[2::3][:N]
        return -np.sum(ri)

    # Constraints:
    # 1. Non-overlap: d_ij >= r_i + r_j
    # 2. Boundaries: r_i <= x_i <= w - r_i, r_i <= y_i <= h - r_i

    constraints = []

    # Non-overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            def non_overlap(z, i=i, j=j):
                xi = z[0::3][:N]
                yi = z[1::3][:N]
                ri = z[2::3][:N]
                dx = xi[j] - xi[i]
                dy = yi[j] - yi[i]
                d = np.sqrt(dx * dx + dy * dy)
                return d - ri[i] - ri[j]
            constraints.append({'type': 'ineq', 'fun': non_overlap})

    # Boundary constraints
    for i in range(N):
        # x_i - r_i >= 0
        def bx_lo(z, i=i):
            xi = z[0::3][:N]
            ri = z[2::3][:N]
            return xi[i] - ri[i]
        constraints.append({'type': 'ineq', 'fun': bx_lo})

        # w - x_i - r_i >= 0
        def bx_hi(z, i=i):
            xi = z[0::3][:N]
            ri = z[2::3][:N]
            return w - xi[i] - ri[i]
        constraints.append({'type': 'ineq', 'fun': bx_hi})

        # y_i - r_i >= 0
        def by_lo(z, i=i):
            yi = z[1::3][:N]
            ri = z[2::3][:N]
            return yi[i] - ri[i]
        constraints.append({'type': 'ineq', 'fun': by_lo})

        # h - y_i - r_i >= 0
        def by_hi(z, i=i):
            yi = z[1::3][:N]
            ri = z[2::3][:N]
            return h - yi[i] - ri[i]
        constraints.append({'type': 'ineq', 'fun': by_hi})

    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, w))  # x
        bounds.append((0, h))  # y
        bounds.append((0, min(w, h) / 2))  # r

    z0 = np.zeros(3 * N)
    z0[0::3][:N] = x
    z0[1::3][:N] = y
    z0[2::3][:N] = r

    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 1000, 'ftol': 1e-12})

        if res.success:
            xi = res.x[0::3][:N]
            yi = res.x[1::3][:N]
            ri = res.x[2::3][:N]
            new_circles = np.column_stack((xi, yi, ri))
            if _is_valid(new_circles, w, h) and np.sum(ri) > np.sum(r) - 1e-6:
                return new_circles, w, h
    except Exception:
        pass

    return np.column_stack((x, y, r)), w, h


def _expansion_relaxation_polish(circles, w, h, max_iter=100):
    """
    Iteratively try to expand radii and relax positions.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for _ in range(max_iter):
        improved = False
        for i in range(N):
            # Try to expand r_i
            old_r = r[i]
            # Max possible r_i given position
            max_r = min(x[i], w - x[i], y[i], h - y[i])
            # Also limited by neighbors
            for j in range(N):
                if i == j:
                    continue
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                d = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, d - r[j])

            if max_r > old_r + 1e-6:
                # Try to set r_i to max_r
                new_r = max_r
                # Check if this causes overlap (shouldn't by construction, but verify)
                valid = True
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < new_r + r[j] - 1e-8:
                        valid = False
                        break
                if valid and new_r > old_r + 1e-8:
                    r[i] = new_r
                    improved = True

        if not improved:
            break

    return np.column_stack((x, y, r)), w, h


def _max_radius_for_circle(i, x, y, r, w, h):
    """Calculate the maximum possible radius for circle i given positions of others."""
    max_r = min(x[i], w - x[i], y[i], h - y[i])
    for j in range(N):
        if i == j:
            continue
        dx = x[j] - x[i]
        dy = y[j] - y[i]
        d = np.sqrt(dx * dx + dy * dy)
        max_r = min(max_r, d - r[j])
    return max(0, max_r)


def _is_valid_move(x, y, r, i, new_r, w, h, tol=1e-8):
    """Check if setting r[i] to new_r is valid."""
    if new_r < -tol:
        return False
    if x[i] - new_r < -tol or x[i] + new_r > w + tol:
        return False
    if y[i] - new_r < -tol or y[i] + new_r > h + tol:
        return False
    for j in range(N):
        if i == j:
            continue
        dx = x[j] - x[i]
        dy = y[j] - y[i]
        d = np.sqrt(dx * dx + dy * dy)
        if d < new_r + r[j] - tol:
            return False
    return True


def _constraint_refine(circles, w, h, max_iter=100):
    """
    Refine by adjusting positions to allow larger radii.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for _ in range(max_iter):
        improved = False
        for i in range(N):
            # Try small perturbations in x, y to allow larger r
            best_r = r[i]
            best_x, best_y = x[i], y[i]

            for dx in np.linspace(-0.01, 0.01, 5):
                for dy in np.linspace(-0.01, 0.01, 5):
                    nx = x[i] + dx
                    ny = y[i] + dy
                    if nx < 0 or nx > w or ny < 0 or ny > h:
                        continue
                    # Check if this position allows a larger radius
                    new_r = _max_radius_for_circle(i, np.concatenate([x[:i], [nx], x[i+1:]]),
                                                   np.concatenate([y[:i], [ny], y[i+1:]]),
                                                   r, w, h)
                    if new_r > best_r + 1e-6:
                        # Verify validity
                        if _is_valid_move(np.concatenate([x[:i], [nx], x[i+1:]]),
                                          np.concatenate([y[:i], [ny], y[i+1:]]),
                                          np.concatenate([r[:i], [new_r], r[i+1:]]),
                                          i, new_r, w, h):
                            best_r = new_r
                            best_x, best_y = nx, ny

            if best_r > r[i] + 1e-8:
                x[i] = best_x
                y[i] = best_y
                r[i] = best_r
                improved = True

        if not improved:
            break

    return np.column_stack((x, y, r)), w, h


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin

    # Initial guess: grid
    side = int(np.ceil(np.sqrt(N)))
    w = 1.0
    h = 1.0

    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)

    for i in range(N):
        ix = i % side
        iy = i // side
        x[i] = (ix + 0.5) / side * w
        y[i] = (iy + 0.5) / side * h
        r[i] = 0.5 / side * min(w, h) - 0.01

    circles = np.column_stack((x, y, r))

    # Force relax to ensure validity
    circles, w, h, success = _force_relax_unified(circles, w, h)

    # Polish
    circles, w, h = _polish_result(circles, w, h)
    circles, w, h = _expansion_relaxation_polish(circles, w, h)
    circles, w, h = _constraint_refine(circles, w, h)

    # Final polish
    circles, w, h = _polish_result(circles, w, h)

    # Ensure validity
    if not _is_valid(circles, w, h):
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            ix = i % side
            iy = i // side
            x[i] = (ix + 0.5) / side * w
            y[i] = (iy + 0.5) / side * h
            r[i] = 0.5 / side * min(w, h) - 0.01
        circles = np.column_stack((x, y, r))
        circles, w, h, success = _force_relax_unified(circles, w, h)

    return circles
