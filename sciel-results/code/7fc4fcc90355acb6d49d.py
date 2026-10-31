"""
Circle packing in perimeter-4 rectangle.
Strategy:
1. Use a unified SLSQP approach with W as a variable (from inspiration 1).
2. Use vectorized penalties for speed (from inspiration 1).
3. Add a force-directed relaxation step before SLSQP to resolve initial overlaps.
4. Add a high-precision polish phase using constraint-based SLSQP on the best result.
5. Multi-start with diverse aspect ratios and random initializations.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 145.0  # Safety margin before 150s kill

def _validate(x, y, r, w, h, tol=1e-6):
    """Check if the packing is valid."""
    if np.any(r < 0):
        return False
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - tol:
                return False
    return True

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
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj += boundary_penalty

    # Keep aspect ratio in reasonable range
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj

def _force_relax(z, max_iter=200):
    """Force-directed relaxation to resolve initial overlaps."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    
    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 100
            if x[i] + r[i] - w > 0:
                fx[i] -= (x[i] + r[i] - w) * 100
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 100
            if y[i] + r[i] - h > 0:
                fy[i] -= (y[i] + r[i] - h) * 100

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-8:
                    force = (min_dist - dist) * 50.0
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist

        # Update positions
        x += dt * fx
        y += dt * fy

        # Clamp to bounds
        for i in range(n):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])

        z[0::3][:n] = x
        z[1::3][:n] = y

    return z

def _constraint_refine(z, time_budget):
    """High-precision polish using constraint-based SLSQP."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    # Objective: Maximize sum(r) => Minimize -sum(r)
    def objective(vars_):
        r_ = vars_[2::3]
        return -np.sum(r_)

    # Constraints
    def cons_inside(vars_):
        x_ = vars_[0::3]
        y_ = vars_[1::3]
        r_ = vars_[2::3]
        return np.concatenate([
            x_ - r_,
            w - x_ - r_,
            y_ - r_,
            h - y_ - r_
        ])

    def cons_non_overlap(vars_):
        x_ = vars_[0::3]
        y_ = vars_[1::3]
        r_ = vars_[2::3]
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = x_[i] - x_[j]
                dy = y_[i] - y_[j]
                dist = np.sqrt(dx * dx + dy * dy)
                cons.append(dist - r_[i] - r_[j])
        return np.array(cons)

    def cons_positive_r(vars_):
        r_ = vars_[2::3]
        return r_

    constraints = [
        {'type': 'ineq', 'fun': cons_inside},
        {'type': 'ineq', 'fun': cons_non_overlap},
        {'type': 'ineq', 'fun': cons_positive_r}
    ]

    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3 * i] = x[i]
        x0[3 * i + 1] = y[i]
        x0[3 * i + 2] = r[i]

    try:
        result = minimize(
            objective, x0, method='SLSQP',
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10}
        )

        x_res = result.x[0::3]
        y_res = result.x[1::3]
        r_res = result.x[2::3]

        if _validate(x_res, y_res, r_res, w, h):
            return np.column_stack((x_res, y_res, r_res))
        else:
            # Try to repair by shrinking radii slightly
            r_res_fixed = r_res * 0.95
            if _validate(x_res, y_res, r_res_fixed, w, h):
                return np.column_stack((x_res, y_res, r_res_fixed))
            return None
    except Exception:
        return None

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None

    # Aspect ratios to try
    aspect_configs = [
        (1.0, 1.0),
        (1.2, 0.8),
        (0.8, 1.2),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.1, 0.9),
        (0.9, 1.1),
        (1.3, 0.7),
        (0.7, 1.3),
        (1.5, 0.5),
        (0.5, 1.5)
    ]

    max_restarts_per_aspect = 2

    for w, h in aspect_configs:
        if time.time() - start_time > DEADLINE - 20:
            break

        for restart in range(max_restarts_per_aspect):
            if time.time() - start_time > DEADLINE - 10:
                break

            # Initialize
            rng = np.random.RandomState(42 + restart * 100 + int(w * 100))

            # Grid initialization
            cols = int(np.sqrt(N * w / h))
            if cols < 2: cols = 2
            rows = int(np.ceil(N / cols))

            while cols * rows < N:
                if cols <= rows:
                    cols += 1
                else:
                    rows += 1

            positions = np.zeros((N, 2))
            idx = 0
            for r_idx in range(rows):
                for c_idx in range(cols):
                    if idx >= N:
                        break
                    x = (c_idx + 0.5) * w / cols
                    y = (r_idx + 0.5) * h / rows

                    if restart > 0:
                        jitter = 0.1 * min(w / cols, h / rows)
                        x += rng.uniform(-jitter, jitter)
                        y += rng.uniform(-jitter, jitter)

                    x = np.clip(x, 0.01, w - 0.01)
                    y = np.clip(y, 0.01, h - 0.01)

                    positions[idx] = [x, y]
                    idx += 1

            while idx < N:
                positions[idx] = [rng.uniform(0.1, w - 0.1), rng.uniform(0.1, h - 0.1)]
                idx += 1

            # Calculate initial radii
            min_dist = np.inf
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.sqrt((positions[i, 0] - positions[j, 0])**2 + (positions[i, 1] - positions[j, 1])**2)
                    min_dist = min(min_dist, d)
            for i in range(N):
                min_dist = min(min_dist, positions[i, 0], w - positions[i, 0], positions[i, 1], h - positions[i, 1])

            r0 = 0.4 * min_dist

            # Flatten variables
            z0 = np.zeros(3 * N + 1)
            for i in range(N):
                z0[3 * i] = positions[i, 0]
                z0[3 * i + 1] = positions[i, 1]
                z0[3 * i + 2] = r0
            z0[-1] = w

            # Force-directed relaxation
            z0 = _force_relax(z0, max_iter=100)

            # SLSQP optimization
            try:
                result = minimize(
                    lambda z: _vectorized_objective(z, penalty_weight=1000.0),
                    z0, method='SLSQP',
                    bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                    options={'maxiter': 150, 'ftol': 1e-10}
                )

                z_res = result.x
                x_res = z_res[0::3][:N]
                y_res = z_res[1::3][:N]
                r_res = z_res[2::3][:N]
                w_res = z_res[-1]
                h_res = 2.0 - w_res

                # Validation
                if _validate(x_res, y_res, r_res, w_res, h_res):
                    score = np.sum(r_res)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((x_res, y_res, r_res))
                else:
                    # Try to repair by shrinking radii slightly
                    r_res_fixed = r_res * 0.95
                    if _validate(x_res, y_res, r_res_fixed, w_res, h_res):
                        score = np.sum(r_res_fixed)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack((x_res, y_res, r_res_fixed))
            except Exception:
                continue

            # Early exit if we find a very good solution
            if best_score > 2.4:
                break

    # Fallback: Simple grid packing
    if best_result is None or best_score < 2.0:
        side = 5
        r = 0.5 / side - 1e-3
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                cells.append([(i + 0.5) / side, (j + 0.5) / side, r])
                count += 1
        best_result = np.array(cells)
        best_score = np.sum(best_result[:, 2])

    # High-precision polish on best result
    if best_result is not None and time.time() - start_time < DEADLINE - 5:
        x = best_result[:, 0]
        y = best_result[:, 1]
        r = best_result[:, 2]
        w = 1.0  # Use square for polish
        h = 1.0

        # Scale to fit in 1x1 square
        max_x = np.max(x)
        max_y = np.max(y)
        scale = min(1.0 / max_x, 1.0 / max_y) if max_x > 0 and max_y > 0 else 1.0
        x *= scale
        y *= scale
        r *= scale

        z = np.zeros(3 * N + 1)
        for i in range(N):
            z[3 * i] = x[i]
            z[3 * i + 1] = y[i]
            z[3 * i + 2] = r[i]
        z[-1] = 1.0

        polished = _constraint_refine(z, DEADLINE - (time.time() - start_time))
        if polished is not None:
            score = np.sum(polished[:, 2])
            if score > best_score:
                best_score = score
                best_result = polished

    return best_result

# EVOLVE-BLOCK-END
