"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
from scipy.optimize import minimize

N = 26


def _objective(params):
    """Return negative sum of radii plus penalty for violations."""
    positions = params[:2 * N]
    radii = params[2 * N:]

    # Penalty for going outside [0, 1]^2
    x = positions[0::2]
    y = positions[1::2]
    boundary_pen = 0.0
    for i in range(N):
        if x[i] - radii[i] < 0:
            boundary_pen += (x[i] - radii[i]) ** 2
        if x[i] + radii[i] > 1:
            boundary_pen += (x[i] + radii[i] - 1) ** 2
        if y[i] - radii[i] < 0:
            boundary_pen += (y[i] - radii[i]) ** 2
        if y[i] + radii[i] > 1:
            boundary_pen += (y[i] + radii[i] - 1) ** 2

    # Penalty for overlapping circles
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                overlap_pen += (min_dist - dist) ** 2

    return -np.sum(radii) + 100.0 * (boundary_pen + overlap_pen)


def _bounds():
    """Create bounds for the optimization."""
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.01, 0.5))  # radius
    return bounds


def construct():
    """Construct 26 circles packed in the unit square, maximizing sum of radii."""
    best_params = None
    best_value = -np.inf

    # Use multiple random restarts with different initial configurations
    for restart in range(50):
        np.random.seed(restart * 12345 + 67890)

        # Random initial positions and radii
        x0 = np.random.uniform(0.1, 0.9, size=N)
        y0 = np.random.uniform(0.1, 0.9, size=N)
        r0 = np.random.uniform(0.05, 0.15, size=N)

        x0[0] = 0.5
        y0[0] = 0.5
        r0[0] = 0.1

        params0 = np.concatenate([np.concatenate([x0, y0]), r0])

        try:
            result = minimize(
                _objective,
                params0,
                method='L-BFGS-B',
                bounds=_bounds(),
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )

            if result.success:
                val = -_objective(result.x)
                # Check if solution is valid (no significant violations)
                positions = result.x[:2 * N]
                radii = result.x[2 * N:]
                x = positions[0::2]
                y = positions[1::2]

                valid = True
                for i in range(N):
                    if x[i] - radii[i] < -1e-6 or x[i] + radii[i] > 1 + 1e-6:
                        valid = False
                        break
                    if y[i] - radii[i] < -1e-6 or y[i] + radii[i] > 1 + 1e-6:
                        valid = False
                        break

                if valid and val > best_value:
                    best_value = val
                    best_params = result.x.copy()
        except Exception:
            continue

    # Fallback to grid if optimization failed
    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_params[0::2]
    y = best_params[1::2]
    r = best_params[2 * N:]

    # Clip to ensure valid positions
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)

    return np.column_stack([x, y, r])
