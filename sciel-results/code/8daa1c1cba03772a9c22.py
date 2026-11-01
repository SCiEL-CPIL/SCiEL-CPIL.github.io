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

    def smooth_objective(z, penalty_weight=500.0):
        """
        Smooth vectorized objective with penalty terms.
        z: [x0, y0, r0, x1, y1, r1, ..., x_{N-1}, y_{N-1}, r_{N-1}, w]
        """
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        # Soft constraints via smooth max
        if w < 0.2 or h < 0.2:
            return 1e8

        # Clip radii to valid range
        r_clipped = np.clip(r, 0.001, min(w, h) / 2 - 0.001)

        obj = -np.sum(r_clipped)

        # Pairwise overlap penalty (smooth)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r_clipped[:, None] + r_clipped[None, :]

        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-15))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        overlap_penalty = penalty_weight * np.sum(overlap ** 2)

        # Wall constraint penalties (vectorized)
        # x violations
        x_low_viol = np.maximum(r_clipped - x, 0.0)
        x_high_viol = np.maximum(x - (w - r_clipped), 0.0)
        # y violations
        y_low_viol = np.maximum(r_clipped - y, 0.0)
        y_high_viol = np.maximum(y - (h - r_clipped), 0.0)

        wall_penalty = penalty_weight * (np.sum(x_low_viol ** 2) + np.sum(x_high_viol ** 2) +
                                         np.sum(y_low_viol ** 2) + np.sum(y_high_viol ** 2))

        # Penalize if any radius is too small
        small_r_penalty = penalty_weight * np.sum(np.maximum(0.01 - r_clipped, 0.0) ** 2)

        return obj + overlap_penalty + wall_penalty + small_r_penalty

    def get_grid_initialization(rows, cols, w, h):
        """
        Grid-based cellular decomposition with optimized circle placement.
        Uses hexagonal packing where possible for better density.
        """
        positions = np.zeros((N, 2))
        radii = np.zeros(N)

        cell_w = w / cols
        cell_h = h / rows

        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= N:
                    break
                cx = (c + 0.5) * cell_w
                cy = (r + 0.5) * cell_h
                max_r = min(cell_w, cell_h) / 2.0 * 0.92

                # Determine how many circles go in this cell
                remaining_circles = N - idx
                remaining_cells = (rows - r) * cols - c

                if remaining_circles > remaining_cells * 1.5:
                    # This cell needs 2-3 circles - use hexagonal arrangement
                    if cell_w >= cell_h:
                        # Horizontal hexagonal: 2 circles side by side
                        r1 = max_r * 0.58
                        r2 = max_r * 0.58
                        offset = (r1 + r2) * 0.52
                        positions[idx, 0] = cx - offset
                        positions[idx, 1] = cy
                        radii[idx] = max(r1, 0.01)
                        idx += 1
                        if idx < N:
                            positions[idx, 0] = cx + offset
                            positions[idx, 1] = cy
                            radii[idx] = max(r2, 0.01)
                            idx += 1
                    else:
                        # Vertical hexagonal
                        r1 = max_r * 0.58
                        r2 = max_r * 0.58
                        offset = (r1 + r2) * 0.52
                        positions[idx, 0] = cx
                        positions[idx, 1] = cy - offset
                        radii[idx] = max(r1, 0.01)
                        idx += 1
                        if idx < N:
                            positions[idx, 0] = cx
                            positions[idx, 1] = cy + offset
                            radii[idx] = max(r2, 0.01)
                            idx += 1
                elif remaining_circles > remaining_cells:
                    # This cell needs exactly 2 circles
                    r1 = max_r * 0.62
                    r2 = max_r * 0.62
                    offset = (r1 + r2) * 0.50
                    if cell_w >= cell_h:
                        positions[idx, 0] = cx - offset
                        positions[idx, 1] = cy
                        radii[idx] = max(r1, 0.01)
                        idx += 1
                        if idx < N:
                            positions[idx, 0] = cx + offset
                            positions[idx, 1] = cy
                            radii[idx] = max(r2, 0.01)
                            idx += 1
                    else:
                        positions[idx, 0] = cx
                        positions[idx, 1] = cy - offset
                        radii[idx] = max(r1, 0.01)
                        idx += 1
                        if idx < N:
                            positions[idx, 0] = cx
                            positions[idx, 1] = cy + offset
                            radii[idx] = max(r2, 0.01)
                            idx += 1
                else:
                    # Single circle in cell
                    positions[idx, 0] = cx
                    positions[idx, 1] = cy
                    radii[idx] = max(max_r, 0.01)
                    idx += 1

        # Fill any remaining with tiny circles at center
        while idx < N:
            positions[idx, 0] = w / 2.0
            positions[idx, 1] = h / 2.0
            radii[idx] = 0.01
            idx += 1

        return positions, radii

    def get_initial_guesses():
        """Generate diverse initial guesses using grid-based cellular decomposition."""
        guesses = []

        # Grid configurations that can accommodate 21 circles
        grid_configs = [
            (3, 7), (7, 3),   # 21 cells exactly
            (4, 6), (6, 4),   # 24 cells
            (5, 5),           # 25 cells
            (3, 8), (8, 3),   # 24 cells
            (4, 5), (5, 4),   # 20 cells, need some 2-per-cell
            (2, 11), (11, 2), # 22 cells
            (3, 5), (5, 3),   # 15 cells, need 2-3 per cell
            (2, 10), (10, 2), # 20 cells
            (4, 4),           # 16 cells
        ]

        # Aspect ratios: w + h = 2.0, focus on near-square and moderate ratios
        aspect_ratios = [
            1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 0.7, 1.4, 0.6, 1.5, 0.5,
            1.6, 0.4, 1.15, 0.85, 1.25, 0.75, 1.35, 0.65
        ]

        for w_frac in aspect_ratios:
            w = w_frac
            h = 2.0 - w
            if w < 0.25 or h < 0.25:
                continue

            for rows, cols in grid_configs:
                positions, radii = get_grid_initialization(rows, cols, w, h)

                # Check if this is a valid starting point
                score = check_validity(positions, radii, w, h)
                if score > 0:
                    guesses.append((w, h, positions, radii))

        # Add some hand-crafted near-optimal configurations
        # Square-ish configurations with known good packings
        for w_val in [0.95, 1.0, 1.05]:
            h_val = 2.0 - w_val
            # 3x7 grid for square-ish
            positions, radii = get_grid_initialization(3, 7, w_val, h_val)
            score = check_validity(positions, radii, w_val, h_val)
            if score > 0:
                guesses.append((w_val, h_val, positions, radii))

        return guesses

    def polish_result(w, h, positions, radii, max_iter=300, ftol=1e-10):
        """Polish a valid packing using SLSQP with smooth objective.
        Uses tighter bounds around current solution for faster convergence."""
        z0 = np.concatenate([positions.flatten(), [w]])

        # Tighter bounds around current positions for faster convergence
        bounds = []
        for i in range(N):
            x_lo = max(0.001, positions[i, 0] - 0.15)
            x_hi = min(w - 0.001, positions[i, 0] + 0.15)
            y_lo = max(0.001, positions[i, 1] - 0.15)
            y_hi = min(h - 0.001, positions[i, 1] + 0.15)
            r_lo = max(0.005, radii[i] * 0.7)
            r_hi = min(min(w, h) / 2.0 - 0.001, radii[i] * 1.3)
            bounds.append((x_lo, x_hi))
            bounds.append((y_lo, y_hi))
            bounds.append((r_lo, r_hi))
        bounds.append((max(0.25, w - 0.2), min(1.75, w + 0.2)))

        try:
            result = minimize(
                lambda z: smooth_objective(z, penalty_weight=800.0),
                z0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol}
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

    def high_precision_polish(w, h, positions, radii, max_iter=500, ftol=1e-12):
        """Final high-precision polish pass with tighter tolerance and local search."""
        z0 = np.concatenate([positions.flatten(), [w]])

        # Very tight bounds for final refinement
        bounds = []
        for i in range(N):
            bounds.append((max(0.001, positions[i, 0] - 0.03), min(w - 0.001, positions[i, 0] + 0.03)))
            bounds.append((max(0.001, positions[i, 1] - 0.03), min(h - 0.001, positions[i, 1] + 0.03)))
            bounds.append((max(0.005, radii[i] * 0.92), min(min(w, h) / 2.0 - 0.001, radii[i] * 1.08)))
        bounds.append((max(0.25, w - 0.05), min(1.75, w + 0.05)))

        best_local = None
        best_local_score = 0.0

        # Try multiple penalty weights for robustness
        for pw in [1500.0, 2500.0, 4000.0]:
            try:
                result = minimize(
                    lambda z, pw=pw: smooth_objective(z, penalty_weight=pw),
                    z0,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': ftol}
                )

                if result.fun < 0:
                    z_opt = result.x
                    x_opt = z_opt[0::3][:N]
                    y_opt = z_opt[1::3][:N]
                    r_opt = z_opt[2::3][:N]
                    w_opt = z_opt[-1]
                    h_opt = 2.0 - w_opt

                    positions_opt = np.column_stack([x_opt, y_opt])

                    score = check_validity(positions_opt, r_opt, w_opt, h_opt)
                    if score > best_local_score:
                        best_local_score = score
                        best_local = (positions_opt, r_opt, w_opt, h_opt, score)
            except Exception:
                pass

        return best_local

    def expansion_relaxation_polish(w, h, positions, radii):
        """
        Try to expand radii while maintaining validity.
        Uses iterative radius increase with adaptive position adjustment.
        """
        positions = positions.copy()
        radii = radii.copy()
        w_current = w
        h_current = h

        for iteration in range(100):
            if time.time() > deadline - 5.0:
                break

            improved = False
            # Try circles in order of current radius (smaller first, more room to grow)
            order = np.argsort(radii)
            for i in order:
                old_r = radii[i]
                # Adaptive step sizes based on current radius
                for dr in [0.004, 0.002, 0.001, 0.0005, 0.0002]:
                    new_r = old_r + dr
                    if new_r > min(w_current, h_current) / 2.0 - 0.001:
                        continue

                    # Try current position and adaptive perturbations
                    best_pos = None
                    pert_scale = min(0.02, new_r * 0.3)
                    for dx in [0, pert_scale, -pert_scale, pert_scale*2, -pert_scale*2]:
                        for dy in [0, pert_scale, -pert_scale, pert_scale*2, -pert_scale*2]:
                            new_x = positions[i, 0] + dx
                            new_y = positions[i, 1] + dy

                            # Check walls
                            if new_x < new_r - 1e-10 or new_x > w_current - new_r + 1e-10:
                                continue
                            if new_y < new_r - 1e-10 or new_y > h_current - new_r + 1e-10:
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
                                break
                        if best_pos is not None:
                            break

                    if best_pos is not None:
                        positions[i, 0] = best_pos[0]
                        positions[i, 1] = best_pos[1]
                        radii[i] = new_r
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

    # Limit number of guesses to try based on time budget
    max_guesses = min(len(guesses), 30)

    for idx, (w, h, positions, radii) in enumerate(guesses[:max_guesses]):
        if time.time() > deadline - 10.0:
            break

        # Try polishing
        result = polish_result(w, h, positions, radii)
        if result is not None:
            pos_opt, r_opt, w_opt, h_opt, score = result
            if score > best_sum:
                best_sum = score
                best_result = np.column_stack([pos_opt, r_opt])

                # Try high-precision polish on the best result
                if time.time() < deadline - 20.0:
                    hp_result = high_precision_polish(w_opt, h_opt, pos_opt, r_opt)
                    if hp_result is not None:
                        pos_hp, r_hp, w_hp, h_hp, score_hp = hp_result
                        if score_hp > best_sum:
                            best_sum = score_hp
                            best_result = np.column_stack([pos_hp, r_hp])

        # Try expansion relaxation on the best result so far
        if best_result is not None and time.time() < deadline - 8.0:
            pos_exp = best_result[:, 0:2]
            r_exp = best_result[:, 2]
            # Estimate w from bounding box
            x_min = np.min(pos_exp[:, 0])
            x_max = np.max(pos_exp[:, 0])
            y_min = np.min(pos_exp[:, 1])
            y_max = np.max(pos_exp[:, 1])
            w_est = x_max - x_min
            h_est = y_max - y_min
            if w_est > 0 and h_est > 0 and abs(w_est + h_est - 2.0) < 0.3:
                result2 = expansion_relaxation_polish(w_est, h_est, pos_exp, r_exp)
                if result2 is not None:
                    pos2, r2, w2, h2, score2 = result2
                    if score2 > best_sum:
                        best_sum = score2
                        best_result = np.column_stack([pos2, r2])

    # Final expansion pass on best result
    if best_result is not None and time.time() < deadline - 3.0:
        pos_final = best_result[:, 0:2]
        r_final = best_result[:, 2]
        x_min = np.min(pos_final[:, 0])
        x_max = np.max(pos_final[:, 0])
        y_min = np.min(pos_final[:, 1])
        y_max = np.max(pos_final[:, 1])
        w_est = x_max - x_min
        h_est = y_max - y_min
        if w_est > 0 and h_est > 0:
            result_final = expansion_relaxation_polish(w_est, h_est, pos_final, r_final)
            if result_final is not None:
                pos_f, r_f, w_f, h_f, score_f = result_final
                if score_f > best_sum:
                    best_sum = score_f
                    best_result = np.column_stack([pos_f, r_f])

    # If we haven't found anything good, return a simple grid solution
    if best_result is None:
        w, h = 1.0, 1.0
        positions, radii = get_grid_initialization(3, 7, w, h)
        best_result = np.column_stack([positions, radii])

    return best_result
# EVOLVE-BLOCK-END
