"""Physics-based circle packing: maximize minimum distance among 200 points in unit square.
Combined approach: hexagonal/grid initializations, L-BFGS-B with smooth penalty,
and a final high-precision polish using a larger target radius."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
SEED = 42
TIME_LIMIT = 115.0  # Safety margin before 150s kill


def _compute_radius(centers):
    """Compute the packing radius: min of half pairwise distances and distance to walls."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0

    # Pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0

    # Distance to walls (square is [-0.5, 0.5]^2)
    dist_to_walls = np.min([
        0.5 - centers[:, 0],
        centers[:, 0] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 1] + 0.5
    ], axis=0)
    min_wall = np.min(dist_to_walls)

    return min(min_pair, min_wall)


def _energy_and_grad(centers, r_target):
    """
    Energy function with smooth penalty for pairs closer than 2*r_target
    and points too close to walls. Returns (energy, gradient).
    """
    centers = centers.reshape(-1, 2)
    n = len(centers)

    # Pairwise distances via cdist
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)

    # Pairwise penalty: max(0, 2*r_target - d)^2
    violations = np.maximum(0, 2.0 * r_target - d)
    pair_energy = np.sum(violations ** 2)

    # Wall distances
    wall_dists = np.min(np.stack([
        0.5 - centers[:, 0],
        0.5 + centers[:, 0],
        0.5 - centers[:, 1],
        0.5 + centers[:, 1]
    ], axis=1), axis=1)
    wall_violations = np.maximum(0, r_target - wall_dists)
    wall_energy = np.sum(wall_violations ** 2)

    total_energy = pair_energy + 10.0 * wall_energy

    # Gradient
    grad = np.zeros_like(centers)

    # Pairwise gradient: d/dx_i [max(0, 2r - d_ij)^2]
    # = -2 * max(0, 2r - d_ij) * (x_i - x_j) / d_ij
    for i in range(n):
        for j in range(i + 1, n):
            dij = d[i, j]
            if dij < 2.0 * r_target and dij > 1e-12:
                v = 2.0 * r_target - dij
                factor = -2.0 * v / dij
                diff = centers[i] - centers[j]
                grad[i] += factor * diff
                grad[j] -= factor * diff

    # Wall gradient
    for axis in range(2):
        # Left wall: 0.5 + centers[:, axis]
        lv = np.maximum(0, r_target - (0.5 + centers[:, axis]))
        grad[:, axis] += 20.0 * lv
        # Right wall: 0.5 - centers[:, axis]
        rv = np.maximum(0, r_target - (0.5 - centers[:, axis]))
        grad[:, axis] -= 20.0 * rv

    return total_energy, grad


def _grid_init():
    """Create a square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _hexagonal_init():
    """Create a hexagonal packing initialization for N points."""
    best_centers = None
    best_count = 0

    # Try various configurations
    for n_rows in range(13, 20):
        for start_offset in [0, 1]:
            for n_cols in range(12, 18):
                # Calculate how many points fit
                total_points = 0
                for row in range(n_rows):
                    if (row + start_offset) % 2 == 0:
                        total_points += n_cols
                    else:
                        total_points += n_cols - 1

                if total_points < N:
                    continue

                # Calculate spacing to fit in unit square
                s_horiz = 1.0 / n_cols
                if n_rows > 1:
                    s_vert = 2.0 / ((n_rows - 1) * np.sqrt(3))
                else:
                    s_vert = 1.0

                s = min(s_horiz, s_vert)

                # Create the grid
                centers = []
                y_extent = (n_rows - 1) * s * np.sqrt(3) / 2
                y_start = -0.5 + (1.0 - y_extent) / 2 + s * np.sqrt(3) / 4

                for row in range(n_rows):
                    y = y_start + row * s * np.sqrt(3) / 2
                    if (row + start_offset) % 2 == 0:
                        x_extent = (n_cols - 1) * s
                        x_start = -0.5 + (1.0 - x_extent) / 2
                        for col in range(n_cols):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centers.append((x, y))
                    else:
                        x_extent = (n_cols - 2) * s
                        x_start = -0.5 + (1.0 - x_extent) / 2 + s / 2
                        for col in range(n_cols - 1):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centers.append((x, y))

                centers = np.array(centers)
                if len(centers) >= N and len(centers) > best_count:
                    best_count = len(centers)
                    best_centers = centers[:N].copy()

    if best_centers is not None:
        return best_centers
    return _grid_init()


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_radius = 0.0
    best_centers = None

    # Initial configurations
    grid_init = _grid_init()
    hex_init = _hexagonal_init()

    # Evaluate initial configurations
    for init in [grid_init, hex_init]:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()

    # Optimization with multiple restarts
    np.random.seed(SEED)

    # Use a base target slightly above current best to encourage improvement
    r_base = max(best_radius * 1.001, 0.035)

    for restart in range(40):
        if time.time() > deadline - 5.0:
            break

        # Alternate between grid and hexagonal bases
        if restart % 2 == 0:
            base = grid_init
        else:
            base = hex_init

        # Perturb the base configuration
        noise_scale = 0.003 + 0.005 * (restart % 5)
        centers = base + np.random.randn(N, 2) * noise_scale
        centers = np.clip(centers, -0.49, 0.49)

        # Use L-BFGS-B to minimize energy
        x0 = centers.flatten()

        def obj(x):
            c = x.reshape(N, 2)
            e, g = _energy_and_grad(c, r_target=r_base)
            return e, g.flatten()

        try:
            result = minimize(obj, x0, method='L-BFGS-B', jac=True,
                             options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-8,
                                      'maxfun': 8000})
        except Exception:
            continue

        if time.time() > deadline - 5.0:
            break

        opt_centers = result.x.reshape(N, 2)

        # Validate
        if not np.all(np.isfinite(opt_centers)):
            continue
        if np.any(opt_centers < -0.5 - 1e-9) or np.any(opt_centers > 0.5 + 1e-9):
            continue

        r_opt = _compute_radius(opt_centers)

        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = opt_centers.copy()

    # Final high-precision polish: use a target slightly above current best
    if time.time() < deadline - 3.0 and best_centers is not None:
        r_target = best_radius * 1.0005
        x0 = best_centers.flatten()

        def obj2(x):
            c = x.reshape(N, 2)
            e, g = _energy_and_grad(c, r_target=r_target)
            return e, g.flatten()

        try:
            result2 = minimize(obj2, x0, method='L-BFGS-B', jac=True,
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                                       'maxfun': 10000})
            opt_centers2 = result2.x.reshape(N, 2)
            if np.all(np.isfinite(opt_centers2)) and \
               np.all(opt_centers2 >= -0.5 - 1e-9) and np.all(opt_centers2 <= 0.5 + 1e-9):
                r_opt2 = _compute_radius(opt_centers2)
                if r_opt2 > best_radius:
                    best_radius = r_opt2
                    best_centers = opt_centers2.copy()
        except Exception:
            pass

    # One more polish pass with even tighter tolerance if time permits
    if time.time() < deadline - 1.0 and best_centers is not None:
        r_target = best_radius * 1.0002
        x0 = best_centers.flatten()

        def obj3(x):
            c = x.reshape(N, 2)
            e, g = _energy_and_grad(c, r_target=r_target)
            return e, g.flatten()

        try:
            result3 = minimize(obj3, x0, method='L-BFGS-B', jac=True,
                              options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11,
                                       'maxfun': 6000})
            opt_centers3 = result3.x.reshape(N, 2)
            if np.all(np.isfinite(opt_centers3)) and \
               np.all(opt_centers3 >= -0.5 - 1e-9) and np.all(opt_centers3 <= 0.5 + 1e-9):
                r_opt3 = _compute_radius(opt_centers3)
                if r_opt3 > best_radius:
                    best_radius = r_opt3
                    best_centers = opt_centers3.copy()
        except Exception:
            pass

    return best_centers


if __name__ == "__main__":
    centers = construct()
    r = _compute_radius(centers)
    print(f"Packing radius: {r:.12f}")
    print(f"Best known:     0.036612798904")
    print(f"Ratio:          {r / 0.036612798904:.6f}")
