"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
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


def _force_relax(z, max_iter=300, dt=0.01):
    """Force-directed relaxation to resolve overlaps and wall violations."""
    z = z.copy()
    for it in range(max_iter):
        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        h = 2.0 - w

        fx = np.zeros(N)
        fy = np.zeros(N)

        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-10:
                    dx, dy = 0.01, 0.01
                    d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d:
                    overlap = min_d - d
                    nx, ny = dx / d, dy / d
                    force = overlap * 5.0
                    fx[i] += nx * force
                    fy[i] += ny * force
                    fx[j] -= nx * force
                    fy[j] -= ny * force

        # Wall forces
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 10.0
            if x[i] > w - r[i]:
                fx[i] -= (x[i] - (w - r[i])) * 10.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 10.0
            if y[i] > h - r[i]:
                fy[i] -= (y[i] - (h - r[i])) * 10.0

        # Apply forces with damping
        x += dt * fx
        y += dt * fy

        # Clamp to walls
        for i in range(N):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])

        z[0::3][:N] = x
        z[1::3][:N] = y

    return z


def _vectorized_expansion_polish(z, deadline):
    """Vectorized expansion: try to grow each radius while adjusting position."""
    x = z[0::3][:N].copy()
    y = z[1::3][:N].copy()
    r = z[2::3][:N].copy()
    w = z[-1]
    h = 2.0 - w

    for iteration in range(120):
        if time.time() > deadline:
            break

        improved = False
        for i in range(N):
            old_r = r[i]
            # Try expanding in small steps
            for dr in [0.008, 0.004, 0.002, 0.001]:
                new_r = old_r + dr
                if new_r > min(w, h) / 2.0 - 0.001:
                    continue

                # Check if we can fit at current position or small perturbations
                found = False
                for dx in [0, 0.003, -0.003, 0.006, -0.006, 0.01, -0.01]:
                    for dy in [0, 0.003, -0.003, 0.006, -0.006, 0.01, -0.01]:
                        nx = x[i] + dx
                        ny = y[i] + dy

                        if nx < new_r or nx > w - new_r:
                            continue
                        if ny < new_r or ny > h - new_r:
                            continue

                        # Check all other circles
                        valid = True
                        for j in range(N):
                            if i == j:
                                continue
                            ddx = nx - x[j]
                            ddy = ny - y[j]
                            dist = np.sqrt(ddx * ddx + ddy * ddy)
                            if dist < new_r + r[j] - 1e-9:
                                valid = False
                                break

                        if valid:
                            x[i] = nx
                            y[i] = ny
                            r[i] = new_r
                            found = True
                            improved = True
                            break
                    if found:
                        break

                if found:
                    break

        if not improved:
            break

    z_new = z.copy()
    z_new[0::3][:N] = x
    z_new[1::3][:N] = y
    z_new[2::3][:N] = r
    return z_new


def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    penalty_weight = 1000.0

    def objective(z):
        return _vectorized_objective(z, penalty_weight)

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        val = float(np.sum(r))
        return True, val

    def get_grid_initialization(rows, cols, w, h):
        """Grid-based cellular decomposition: partition rectangle into rows x cols cells."""
        positions = np.zeros((N, 2))
        radii = np.zeros(N)

        cell_w = w / cols
        cell_h = h / rows

        idx = 0
        for r_row in range(rows):
            for c in range(cols):
                if idx >= N:
                    break
                cx = (c + 0.5) * cell_w
                cy = (r_row + 0.5) * cell_h
                max_r = min(cell_w, cell_h) / 2.0 * 0.90

                remaining_circles = N - idx
                remaining_cells = (rows - r_row) * cols - c

                if remaining_circles > remaining_cells:
                    # This cell needs 2 circles
                    r1 = max_r * 0.55
                    r2 = max_r * 0.35
                    offset = (r1 + r2) * 0.55

                    if cell_w > cell_h:
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
                    positions[idx, 0] = cx
                    positions[idx, 1] = cy
                    radii[idx] = max(max_r, 0.01)
                    idx += 1

        while idx < N:
            positions[idx, 0] = w / 2.0
            positions[idx, 1] = h / 2.0
            radii[idx] = 0.01
            idx += 1

        return positions, radii

    def get_hexagonal_initialization(w, h):
        """Hexagonal lattice packing initial guess."""
        positions = np.zeros((N, 2))
        radii = np.zeros(N)

        # Determine hex grid parameters
        col_spacing = w / 6.0
        row_spacing = col_spacing * np.sqrt(3) / 2.0
        idx = 0
        for row in range(10):
            if idx >= N:
                break
            y_pos = (row + 0.5) * row_spacing
            if y_pos > h - 0.05:
                break
            offset = (col_spacing / 2.0) if (row % 2 == 1) else 0.0
            for col in range(6):
                if idx >= N:
                    break
                x_pos = (col + 0.5) * col_spacing + offset
                if x_pos < 0.05 or x_pos > w - 0.05:
                    continue
                r0 = min(col_spacing, row_spacing) * 0.48
                positions[idx, 0] = x_pos
                positions[idx, 1] = y_pos
                radii[idx] = max(r0, 0.01)
                idx += 1

        while idx < N:
            positions[idx, 0] = w / 2.0
            positions[idx, 1] = h / 2.0
            radii[idx] = 0.01
            idx += 1

        return positions, radii

    def get_initial_guesses():
        """Generate diverse initial guesses using grid and hexagonal decompositions."""
        guesses = []

        # Grid configurations
        grid_configs = [
            (3, 7), (7, 3),
            (4, 6), (6, 4),
            (5, 5),
            (3, 8), (8, 3),
            (4, 5), (5, 4),
            (2, 11), (11, 2),
        ]

        aspect_ratios = [
            1.0, 1.2, 0.8, 1.5, 0.5, 1.3, 0.7, 1.4, 0.6, 1.1, 0.9,
            1.6, 0.4, 1.7, 0.3, 1.8, 0.25
        ]

        for w_frac in aspect_ratios:
            w = w_frac
            h = 2.0 - w
            if w < 0.2 or h < 0.2:
                continue

            for rows, cols in grid_configs:
                positions, radii = get_grid_initialization(rows, cols, w, h)
                z = np.zeros(3 * N + 1)
                z[0::3][:N] = positions[:, 0]
                z[1::3][:N] = positions[:, 1]
                z[2::3][:N] = radii
                z[-1] = w
                guesses.append(z)

        # Hexagonal packing initial guesses
        for w0 in [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6]:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            positions, radii = get_hexagonal_initialization(w0, h0)
            z = np.zeros(3 * N + 1)
            z[0::3][:N] = positions[:, 0]
            z[1::3][:N] = positions[:, 1]
            z[2::3][:N] = radii
            z[-1] = w0
            guesses.append(z)

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.14)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Sort by initial sum of radii (descending)
        guesses.sort(key=lambda z: -np.sum(z[2::3][:N]))

        return guesses

    def polish_result(z0, max_iter=500, ftol=1e-12):
        """Polish a valid packing using SLSQP with smooth objective."""
        bounds = []
        w = z0[-1]
        h = 2.0 - w
        for i in range(N):
            bounds.append((0.001, w - 0.001))
            bounds.append((0.001, h - 0.001))
            bounds.append((0.005, min(w, h) / 2.0 - 0.001))
        bounds.append((0.25, 1.75))

        try:
            result = minimize(
                objective,
                z0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol, 'maxls': 80}
            )

            if result.fun < 0:
                z_opt = result.x
                valid, val = check_validity(z_opt)
                if valid:
                    return z_opt, val
        except Exception:
            pass

        return None, 0.0

    def high_precision_polish(z0, max_iter=800, ftol=1e-14):
        """Final high-precision polish pass with tighter tolerance."""
        w = z0[-1]
        h = 2.0 - w
        x = z0[0::3][:N]
        y = z0[1::3][:N]
        r = z0[2::3][:N]

        bounds = []
        for i in range(N):
            bounds.append((max(0.001, x[i] - 0.05), min(w - 0.001, x[i] + 0.05)))
            bounds.append((max(0.001, y[i] - 0.05), min(h - 0.001, y[i] + 0.05)))
            bounds.append((max(0.005, r[i] * 0.8), min(min(w, h) / 2.0 - 0.001, r[i] * 1.2)))
        bounds.append((max(0.25, w - 0.1), min(1.75, w + 0.1)))

        try:
            result = minimize(
                lambda z: _vectorized_objective(z, penalty_weight=2000.0),
                z0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol, 'maxls': 100}
            )

            if result.fun < 0:
                z_opt = result.x
                valid, val = check_validity(z_opt)
                if valid:
                    return z_opt, val
        except Exception:
            pass

        return None, 0.0

    # Main search loop
    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 20.0:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax(z0, max_iter=200, dt=0.01)

        # SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(150, remaining * 40)))
        result, val = polish_result(z0_relaxed, max_iter=maxiter)
        if result is not None and val > best_value:
            best_value = val
            best_result = result.copy()

            # High-precision polish on the best result
            if time.time() < deadline - 25.0:
                hp_result, hp_val = high_precision_polish(result)
                if hp_result is not None and hp_val > best_value:
                    best_value = hp_val
                    best_result = hp_result.copy()

        # Try expansion relaxation on the best result so far
        if best_result is not None and time.time() < deadline - 10.0:
            expanded = _vectorized_expansion_polish(best_result, deadline - 5.0)
            valid, val_exp = check_validity(expanded)
            if valid and val_exp > best_value:
                best_value = val_exp
                best_result = expanded.copy()

    # Final polish if we have a result
    if best_result is not None and time.time() < deadline - 5.0:
        final_result, final_val = high_precision_polish(best_result)
        if final_result is not None and final_val > best_value:
            best_value = final_val
            best_result = final_result.copy()

    # If we haven't found anything good, return a simple grid solution
    if best_result is None:
        w, h = 1.0, 1.0
        positions, radii = get_grid_initialization(5, 5, w, h)
        best_result = np.zeros(3 * N + 1)
        best_result[0::3][:N] = positions[:, 0]
        best_result[1::3][:N] = positions[:, 1]
        best_result[2::3][:N] = radii
        best_result[-1] = w

    # Convert to (n, 3) array: [x, y, r]
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    return np.column_stack([x, y, r])
