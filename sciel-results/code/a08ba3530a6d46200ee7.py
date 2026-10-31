"""Circle packing in perimeter-4 rectangle: SLSQP with grid seeds and local refinement."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 21
DEADLINE_SECONDS = 145.0  # safety margin before 150s kill


def construct():
    start_time = time.time()
    best_value = -np.inf
    best_result = None
    best_dims = None

    def elapsed():
        return time.time() - start_time

    def is_feasible(circles, width, height, tol=1e-8):
        """circles: (n, 3) array of [x, y, r]. Returns True if valid."""
        if circles.shape[0] != N:
            return False
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]

        if np.any(rs < 1e-9):
            return False
        if np.any(xs < -tol) or np.any(ys < -tol):
            return False
        if np.any(xs + rs > width + tol) or np.any(ys + rs > height + tol):
            return False

        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = rs[:, None] + rs[None, :]
        mask = ~np.eye(N, dtype=bool)
        if np.any(dists[mask] < min_dists[mask] - tol):
            return False

        return True

    def sum_radii(circles):
        return np.sum(circles[:, 2])

    def grid_seed(width, height):
        """Generate a feasible grid-based initial configuration."""
        best_seed = None
        best_score = -1e9

        for cols in range(1, N + 1):
            rows = (N + cols - 1) // cols
            if cols * rows < N:
                continue

            cw = width / cols
            ch = height / rows
            r = 0.5 * min(cw, ch) * 0.999

            circles = np.zeros((N, 3))
            idx = 0
            for j in range(rows):
                for i in range(cols):
                    if idx >= N:
                        break
                    cx = (i + 0.5) * cw
                    cy = (j + 0.5) * ch
                    circles[idx] = [cx, cy, r]
                    idx += 1

            if is_feasible(circles, width, height):
                score = sum_radii(circles)
                if score > best_score:
                    best_score = score
                    best_seed = circles.copy()

        if best_seed is None:
            # Fallback: tiny circles in a line
            r = 0.01
            circles = np.zeros((N, 3))
            for i in range(N):
                circles[i] = [r + i * (width - 2*r) / (N-1), height/2, r]
            best_seed = circles

        return best_seed

    def make_objective(width, height):
        """Returns objective function that minimizes negative sum of radii with penalties."""
        def objective(params):
            if elapsed() > DEADLINE_SECONDS:
                return 1e10

            circles = params.reshape(N, 3)
            xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]

            # Boundary penalties
            pen = 0.0
            pen += np.sum(np.maximum(0, -xs)**2)
            pen += np.sum(np.maximum(0, -ys)**2)
            pen += np.sum(np.maximum(0, xs + rs - width)**2)
            pen += np.sum(np.maximum(0, ys + rs - height)**2)

            # Pairwise distance penalties
            dx = xs[:, None] - xs[None, :]
            dy = ys[:, None] - ys[None, :]
            dists = np.sqrt(dx**2 + dy**2)
            min_dists = rs[:, None] + rs[None, :]
            mask = ~np.eye(N, dtype=bool)
            violations = np.maximum(0, min_dists[mask] - dists[mask])
            pen += np.sum(violations**2)

            # Small penalty for very small radii
            pen += 1e-4 * np.sum(np.maximum(0, 0.01 - rs)**2)

            return -np.sum(rs) + 100.0 * pen

        return objective

    def make_bounds(width, height):
        """Bounds for each circle."""
        max_r = min(width, height) / 2.0
        bounds = []
        for _ in range(N):
            bounds.append((0.0, width))
            bounds.append((0.0, height))
            bounds.append((0.001, max_r))
        return bounds

    def slsqp_refine(circles, width, height):
        """Refine using SLSQP with explicit constraints."""
        x0 = circles.flatten()
        h = height

        cons = []
        # r_i >= 1e-4
        for i in range(N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[3*i+2] - 1e-4,
            })
        # cx_i - r_i >= 0
        for i in range(N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[3*i] - x[3*i+2],
            })
        # (w - cx_i) - r_i >= 0
        for i in range(N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: (width - x[3*i]) - x[3*i+2],
            })
        # cy_i - r_i >= 0
        for i in range(N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[3*i+1] - x[3*i+2],
            })
        # (h - cy_i) - r_i >= 0
        for i in range(N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: (h - x[3*i+1]) - x[3*i+2],
            })
        # pairwise: dist^2 - (r_i + r_j)^2 >= 0
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: (
                        (x[3*i] - x[3*j])**2 +
                        (x[3*i+1] - x[3*j+1])**2 -
                        (x[3*i+2] + x[3*j+2])**2
                    ),
                })

        def obj(x):
            if elapsed() > DEADLINE_SECONDS:
                return 1e10
            return -np.sum(x[2::3])

        bounds = []
        for i in range(N):
            bounds.append((0.0, width))
            bounds.append((0.0, height))
            bounds.append((1e-4, min(width, height) / 2.0))

        try:
            result = minimize(
                obj, x0, method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            return result.x.reshape(N, 3)
        except Exception:
            return circles

    def lbfgs_refine(circles, width, height):
        """Refine using L-BFGS-B with penalty method."""
        x0 = circles.flatten()
        obj = make_objective(width, height)
        bnds = make_bounds(width, height)

        try:
            result = minimize(
                obj, x0, method='L-BFGS-B',
                bounds=bnds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            return result.x.reshape(N, 3)
        except Exception:
            return circles

    def repair(circles, width, height):
        """Try to repair infeasibility by shrinking radii and re-clamping."""
        for _ in range(10):
            if elapsed() > DEADLINE_SECONDS:
                break
            if is_feasible(circles, width, height):
                return circles
            # Shrink all radii slightly
            circles = circles.copy()
            circles[:, 2] *= 0.98
            # Re-clamp positions
            circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], width - circles[:, 2])
            circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], height - circles[:, 2])
        return circles

    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.66, 1.5, 0.5, 2.0, 0.75, 1.33, 1.1, 0.9, 1.2, 0.85, 1.4, 0.7]

    for aspect in aspect_ratios:
        if elapsed() > DEADLINE_SECONDS - 10:
            break

        height = 2.0 / (1.0 + aspect)
        width = 2.0 * aspect / (1.0 + aspect)

        if width < 0.1 or height < 0.1:
            continue

        # Generate grid seed
        seed = grid_seed(width, height)

        # Run SLSQP refinement
        refined = slsqp_refine(seed, width, height)
        if elapsed() > DEADLINE_SECONDS - 5:
            break

        # Repair if needed
        refined = repair(refined, width, height)

        if is_feasible(refined, width, height):
            val = sum_radii(refined)
            if val > best_value:
                best_value = val
                best_result = refined.copy()
                best_dims = (width, height)

        # Run L-BFGS-B refinement from the SLSQP result
        if elapsed() > DEADLINE_SECONDS - 15:
            break

        refined2 = lbfgs_refine(refined, width, height)
        refined2 = repair(refined2, width, height)

        if is_feasible(refined2, width, height):
            val2 = sum_radii(refined2)
            if val2 > best_value:
                best_value = val2
                best_result = refined2.copy()
                best_dims = (width, height)

    # If no valid result found, fall back to simple grid
    if best_result is None:
        width, height = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        cw = width / side
        ch = height / side
        r = 0.5 * min(cw, ch) * 0.9
        best_result = np.zeros((N, 3))
        idx = 0
        for j in range(side):
            for i in range(side):
                if idx >= N:
                    break
                best_result[idx] = [(i + 0.5) * cw, (j + 0.5) * ch, r]
                idx += 1

    # Final safety: ensure all radii are positive
    best_result[:, 2] = np.maximum(best_result[:, 2], 1e-4)

    return best_result
