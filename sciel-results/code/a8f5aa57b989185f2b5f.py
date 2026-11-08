import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 21
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for 150s limit

def _compute_radii(x, y, W, H):
    """
    Compute the maximum radius for each circle given centers x, y and box W, H.
    r_i = min( dist_to_nearest_neighbor_i, dist_to_boundary_i )
    """
    # Distances to boundary
    db = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))

    # Distances to nearest neighbor (vectorized via KDTree)
    pts = np.column_stack((x, y))
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    dn = dists[:, 1]

    # Radius is min of boundary distance and neighbor distance
    r = np.minimum(db, dn)
    return r

def _sum_radii(x, y, W, H):
    return np.sum(_compute_radii(x, y, W, H))

def _objective(v, W, H):
    x = v[:N]
    y = v[N:2*N]
    return -_sum_radii(x, y, W, H)

def _forces(x, y, W, H):
    """
    Compute repulsive forces on each particle.
    Force magnitude derived from Voronoi inradius gradient:
    For each pair (i,j) with distance d, the contribution to r_i and r_j
    when they are the nearest neighbors is ~ (d - 2r)/d in the direction
    pushing them apart. We use a soft potential: F = k * (d - 2r_target) / d
    where r_target is a fraction of the local inradius.
    """
    fx = np.zeros(N)
    fy = np.zeros(N)

    pts = np.column_stack((x, y))
    tree = cKDTree(pts)
    # Query up to 4 nearest neighbors for force computation
    k = min(5, N)
    dists, idxs = tree.query(pts, k=k)

    r = _compute_radii(x, y, W, H)

    for i in range(N):
        for j_idx in range(1, k):  # skip self (j_idx=0)
            j = idxs[i, j_idx]
            d = dists[i, j_idx]
            if d < 1e-12:
                continue
            # Repulsive force: stronger when circles are close relative to their radii
            # Use a soft potential: F ~ (r_i + r_j - d) if overlapping, else decays
            overlap = r[i] + r[j] - d
            # Force magnitude: proportional to overlap if positive, else small repulsion
            if overlap > 0:
                fmag = 2.0 * overlap
            else:
                fmag = 0.1 * (-overlap) / (1.0 + (-overlap))
            # Direction: from j to i
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            fx[i] += fmag * dx / d
            fy[i] += fmag * dy / d
            fx[j] -= fmag * dx / d
            fy[j] -= fmag * dy / d

    # Boundary forces: push circles away from walls if they'd exceed boundary
    for i in range(N):
        if x[i] - r[i] < 0:
            fx[i] += 5.0 * (r[i] - x[i])
        if x[i] + r[i] > W:
            fx[i] -= 5.0 * (x[i] + r[i] - W)
        if y[i] - r[i] < 0:
            fy[i] += 5.0 * (r[i] - y[i])
        if y[i] + r[i] > H:
            fy[i] -= 5.0 * (y[i] + r[i] - H)

    return fx, fy

def _validate(x, y, W, H, tol=1e-6):
    """
    Check if the configuration is valid.
    """
    # Check boundaries
    if np.any(x < -tol) or np.any(x > W + tol):
        return False
    if np.any(y < -tol) or np.any(y > H + tol):
        return False

    r = _compute_radii(x, y, W, H)

    # Check boundaries with radii
    if np.any(x - r < -tol) or np.any(x + r > W + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > H + tol):
        return False

    # Check pairwise
    pts = np.column_stack((x, y))
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r.max() + tol)
    for i, j in pairs:
        dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
        if dist < r[i] + r[j] - tol:
            return False
    return True

def _initial_grid(W, H):
    """
    Generate a grid-like initial configuration.
    """
    cols = 5
    rows = int(np.ceil(N / cols))

    x = np.zeros(N)
    y = np.zeros(N)
    k = 0
    for i in range(cols):
        for j in range(rows):
            if k >= N:
                break
            x[k] = (i + 0.5) * W / cols
            y[k] = (j + 0.5) * H / rows
            k += 1
    return np.concatenate([x, y])

def _initial_random(W, H, seed=42):
    """
    Generate a random initial configuration.
    """
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, W, N)
    y = rng.uniform(0, H, N)
    return np.concatenate([x, y])

def _force_sa(v0, W, H, max_iter=5000, t0=0.1, t_end=0.001, alpha=0.999, dt=0.01):
    """
    Force-directed simulated annealing.
    Particles experience repulsive forces from Voronoi inradius gradient.
    Temperature controls the stochastic perturbation added to the force step.
    """
    v = v0.copy()
    x = v[:N].copy()
    y = v[N:].copy()
    curr_val = _objective(v, W, H)
    best_val = curr_val
    best_v = v.copy()

    T = t0
    rng = np.random.RandomState(12345)

    for i in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT:
            break

        # Compute forces
        fx, fy = _forces(x, y, W, H)

        # Force step + thermal noise
        noise_scale = T * 0.5
        x_new = x + dt * fx + rng.normal(0, noise_scale, N)
        y_new = y + dt * fy + rng.normal(0, noise_scale, N)

        # Clamp to valid box
        x_new = np.clip(x_new, 0, W)
        y_new = np.clip(y_new, 0, H)

        v_new = np.concatenate([x_new, y_new])
        new_val = _objective(v_new, W, H)
        delta_val = new_val - curr_val

        # Metropolis acceptance
        if delta_val < 0 or rng.rand() < np.exp(-delta_val / max(T, 1e-10)):
            x = x_new
            y = y_new
            v = v_new
            curr_val = new_val
            if curr_val < best_val:
                best_val = curr_val
                best_v = v.copy()

        T *= alpha
        if T < t_end:
            break

    return best_v, best_val

def _polish(v, W, H, max_iter=200):
    """
    Use L-BFGS-B to polish the solution.
    """
    bounds = []
    for i in range(N):
        bounds.append((0, W))
        bounds.append((0, H))

    try:
        res = minimize(_objective, v, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10})
        return res.x, res.fun
    except Exception:
        return v, _objective(v, W, H)

def _optimize_for_aspect(W, H):
    """
    Run force-directed SA + Polish for a specific aspect ratio.
    """
    best_val = np.inf
    best_v = None

    # Try multiple initializations
    inits = []
    inits.append(_initial_grid(W, H))
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        inits.append(_initial_random(W, H, seed))

    for v0 in inits:
        if time.time() - START_TIME > TIME_LIMIT:
            break

        # Force-directed SA
        v_sa, val_sa = _force_sa(v0, W, H, max_iter=4000)

        # Polish with L-BFGS-B
        v_pol, val_pol = _polish(v_sa, W, H, max_iter=150)

        if val_pol < best_val:
            best_val = val_pol
            best_v = v_pol

    return best_v, -best_val

def construct():
    best_val = -1
    best_res = None

    # Focus on aspect ratios near 1:1 (square) since perimeter is fixed
    # W + H = 2, so W in [1, 2] covers all cases by symmetry
    widths = [1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15, 1.2, 0.8]

    for W in widths:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        H = 2.0 - W
        if H <= 0:
            continue

        v, val = _optimize_for_aspect(W, H)

        if v is None:
            continue

        x = v[:N]
        y = v[N:]

        # Validate
        if _validate(x, y, W, H):
            if val > best_val:
                best_val = val
                best_res = np.column_stack((x, y, _compute_radii(x, y, W, H)))

    if best_res is None:
        # Fallback to seed: 1x1 square, grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_res
