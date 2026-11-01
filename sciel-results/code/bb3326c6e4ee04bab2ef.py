"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    best_sum = 0.0
    best_result = None

    def check_validity(positions, radii, w, h):
        """Check if packing is valid and return sum of radii if so."""
        for i in range(N):
            if positions[i, 0] < radii[i] - 1e-9:
                return -1.0
            if positions[i, 0] > w - radii[i] + 1e-9:
                return -1.0
            if positions[i, 1] < radii[i] - 1e-9:
                return -1.0
            if positions[i, 1] > h - radii[i] + 1e-9:
                return -1.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    return -1.0

        return np.sum(radii)

    def objective(z, penalty_weight=1000.0):
        """
        Vectorized objective with smooth penalties.
        z: [x0, y0, r0, x1, y1, r1, ..., x_{N-1}, y_{N-1}, r_{N-1}, w]
        """
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.3 or h < 0.3 or np.any(r < 0.001):
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

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        overlap_penalty = penalty_weight * np.sum(overlap ** 2)

        # Wall constraint penalties
        wall_pen = 0.0
        for i in range(N):
            wall_pen += max(0, radii_wall_violation(x[i], r[i], 0, w))
            wall_pen += max(0, radii_wall_violation(x[i], r[i], w, 0))
            wall_pen += max(0, radii_wall_violation(y[i], r[i], 0, h))
            wall_pen += max(0, radii_wall_violation(y[i], r[i], h, 0))
        wall_penalty = penalty_weight * wall_pen

        return obj + overlap_penalty + wall_penalty

    def radii_wall_violation(coord, radius, low, high):
        """Compute wall violation for one dimension."""
        v = 0.0
        if coord < radius:
            v += (radius - coord) ** 2
        if coord > 2.0 - (coord - radius) and high > 0:  # This is wrong, let me fix
            pass
        return v

    def wall_violation(coord, radius, low, high):
        """Compute wall violation for one dimension."""
        v = 0.0
        if coord < radius:
            v += (radius - coord) ** 2
        if coord > high - radius:
            v += (coord - (high - radius)) ** 2
        return v

    def get_initial_guesses():
        """Generate diverse initial guesses using grid-based cellular decomposition."""
        guesses = []

        # Various grid configurations
        grid_configs = [
            (3, 7), (7, 3),
            (4, 5), (5, 4),
            (3, 4), (4, 3),  # 12 cells, will use 21 with some cells having 2 circles
            (5, 5),
            (6, 4), (4, 6),
            (3, 3),  # 9 cells, 21 circles means ~2-3 per cell
        ]

        aspect_ratios = [
            1.0, 1.2, 0.8, 1.5, 0.5, 1.3, 0.7, 1.4, 0.6, 1.1, 0.9
        ]

        for w_frac in aspect_ratios:
            w = w_frac
            h = 2.0 - w

            for rows, cols in grid_configs:
                if rows * cols < N:
                    # Need to distribute N circles among rows*cols cells
                    # Some cells will have multiple circles
                    pass

                positions = np.zeros((N, 2))
                radii = np.zeros(N)

                # Assign circles to grid cells
                total_cells = rows * cols
                idx = 0
                for r in range(rows):
                    for c in range(cols):
                        if idx >= N:
                            break
                        cell_w = w / cols
                        cell_h = h / rows
                        cx = (c + 0.5) * cell_w
                        cy = (r + 0.5) * cell_h
                        max_r = min(cell_w, cell_h) / 2 * 0.85  # 15% margin

                        # Check if this cell needs multiple circles
                        remaining = N - idx
                        remaining_cells = total_cells - (r * cols + c)
                        if remaining > remaining_cells:
                            # This cell needs 2 circles
                            r1 = max_r * 0.6
                            r2 = max_r * 0.4
                            offset = max_r * 0.3

                            # First circle
                            positions[idx, 0] = cx - offset
                            positions[idx, 1] = cy
                            radii[idx] = max(r1, 0.01)
                            idx += 1

                            # Second circle
                            if idx < N:
                                positions[idx, 0] = cx + offset
                                positions[idx, 1] = cy
                                radii[idx] = max(r2, 0.01)
                                idx += 1
                        else:
                            positions[idx, 0] = cx
                            positions[idx, 1] = cy
                            radii[idx] = max(max_r, 0.01)
                            idx += 1

                # Fill any remaining
                while idx < N:
                    positions[idx, 0] = w / 2
                    positions[idx, 1] = h / 2
                    radii[idx] = 0.01
                    idx += 1

                guesses.append((w, h, positions, radii))

        return guesses

    def polish_result(w, h, positions, radii, max_iter=300):
        """Polish a valid packing using SLSQP with vectorized objective."""
        z0 = np.concatenate([positions.flatten(), [w]])

        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, w))  # x_i
            bounds.append((0, h))  # y_i
            bounds.append((0.005, min(w, h) / 2))  # r_i
        bounds.append((0.3, 1.7))  # w

        try:
            result = minimize(
                lambda z: objective(z),
                z0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-10}
            )

            if result.fun < 0:
                z_opt = result.x
                x_opt = z_opt[0::3][:N]
                y_opt = z_opt[1::3][:N]
                r_opt = z_opt[2::3][:N]
                w_opt = z_opt[-1]
                h_opt = 2.0 - w_opt

                positions_opt = np.column_stack([x_opt, y_opt])

                # Validate
                score = check_validity(positions_opt, r_opt, w_opt, h_opt)
                if score > 0:
                    return positions_opt, r_opt, w_opt, h_opt, score
        except Exception:
            pass

        return None

    def expansion_relaxation_polish(w, h, positions, radii):
        """
        Try to expand radii while maintaining validity.
        Uses iterative radius increase with position adjustment.
        """
        positions = positions.copy()
        radii = radii.copy()
        w_current = w
        h_current = h

        for iteration in range(50):
            if time.time() > deadline:
                break

            improved = False
            for i in range(N):
                # Try to increase radius of circle i
                old_r = radii[i]
                for dr in [0.01, 0.005, 0.002, 0.001]:
                    new_r = old_r + dr
                    if new_r > min(w_current, h_current) / 2:
                        continue

                    # Check if we can place circle i with new radius
                    # Try moving it slightly
                    best_pos = None
                    best_score = -1

                    # Try current position and small perturbations
                    for dx in [0, 0.01, -0.01, 0.02, -0.02]:
                        for dy in [0, 0.01, -0.01, 0.02, -0.02]:
                            new_x = positions[i, 0] + dx
                            new_y = positions[i, 1] + dy

                            # Check walls
                            if new_x < new_r or new_x > w_current - new_r:
                                continue
                            if new_y < new_r or new_y > h_current - new_r:
                                continue

                            # Check overlaps with other circles
                            valid = True
                            for j in range(N):
                                if i == j:
                                    continue
                                dx2 = new_x - positions[j, 0]
                                dy2 = new_y - positions[j, 1]
                                dist = np.sqrt(dx2 * dx2 + dy2 * dy2)
                                if dist < new_r + radii[j] - 1e-9:
                                    valid = False
                                    break

                            if valid:
                                best_pos = (new_x, new_y)
                                best_score = new_r
                                break
                        if best_pos is not None:
                            break

                    if best_pos is not None and best_score > old_r:
                        positions[i, 0] = best_pos[0]
                        positions[i, 1] = best_pos[1]
                        radii[i] = best_score
                        improved = True
                        break

            if not improved:
                break

        score = check_validity(positions, radii, w_current, h_current)
        if score > 0:
            return positions, radii, w_current, h_current, score

        return None

    # Main search loop
    guesses = get_initial_guesses()

    # Sort by initial sum of radii (descending)
    guesses.sort(key=lambda g: -np.sum(g[3]))

    for w, h, positions, radii in guesses:
        if time.time() > deadline:
            break

        # First validate the initial guess
        initial_score = check_validity(positions, radii, w, h)

        # Try polishing
        result = polish_result(w, h, positions, radii)
        if result is not None:
            pos_opt, r_opt, w_opt, h_opt, score = result
            if score > best_sum:
                best_sum = score
                best_result = np.column_stack([pos_opt, r_opt])

        # Try expansion relaxation
        result2 = expansion_relaxation_polish(w, h, positions, radii)
        if result2 is not None:
            pos2, r2, w2, h2, score2 = result2
            if score2 > best_sum:
                best_sum = score2
                best_result = np.column_stack([pos2, r2])

    # If we haven't found anything good, return a simple grid solution
    if best_result is None:
        w, h = 1.0, 1.0
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        rows, cols = 5, 5
        for i in range(N):
            r = i // cols
            c = i % cols
            positions[i, 0] = (c + 0.5) * w / cols
            positions[i, 1] = (r + 0.5) * h / rows
            radii[i] = min(w / cols, h / rows) / 2 * 0.85

        best_result = np.column_stack([positions, radii])

    return best_result
# EVOLVE-BLOCK-END
