"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 150.0 - 10.0  # safety margin before the 150s kill


def _radius(centers):
    """Vectorized max valid radius. Returns 0.0 if invalid."""
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    # boundary distance
    b = 0.5 - np.abs(centers)
    min_b = float(np.min(b))
    if min_b <= 0.0:
        return 0.0
    # pairwise distances via broadcasting
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = float(np.min(dists))
    if min_pair <= 0.0:
        return 0.0
    return min(min_pair / 2.0, min_b)


def _soft_min_radius(centers, beta=100.0):
    """
    Smooth surrogate for the max valid radius using soft-min via log-sum-exp.
    """
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0

    # boundary distances
    b = 0.5 - np.abs(centers)

    # pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)

    # half distances for pairs
    iu = np.triu_indices(N, k=1)
    half_dists = dists[iu] / 2.0

    # Combine all constraints
    all_vals = np.concatenate([half_dists, b])

    # Soft min via log-sum-exp
    min_val = np.min(all_vals)
    if min_val <= 0:
        return 0.0

    scaled = -beta * (all_vals - min_val)
    scaled = np.clip(scaled, -100, 100)

    log_sum = np.log(np.sum(np.exp(scaled)))
    soft_min = min_val - log_sum / beta

    return max(0.0, soft_min)


def _grad_soft_min_radius(centers, beta=100.0):
    """
    Gradient of the soft-min radius w.r.t. centers.
    """
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return np.zeros_like(centers)
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return np.zeros_like(centers)

    b = 0.5 - np.abs(centers)

    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)

    iu = np.triu_indices(N, k=1)
    half_dists = dists[iu] / 2.0

    all_vals = np.concatenate([half_dists, b])
    min_val = np.min(all_vals)
    if min_val <= 0:
        return np.zeros_like(centers)

    scaled = -beta * (all_vals - min_val)
    scaled = np.clip(scaled, -100, 100)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)

    # Weights for soft-min
    weights = exp_scaled / sum_exp

    grad = np.zeros_like(centers)

    # Gradient contribution from pairwise terms
    n_pairs = len(iu)
    for idx in range(n_pairs):
        i, j = iu[0][idx], iu[1][idx]
        d = dists[i, j]
        if d < 1e-10:
            continue
        w = weights[idx]
        vec = (centers[i] - centers[j]) / d
        grad[i] += w * 0.5 * vec
        grad[j] += w * 0.5 * (-vec)

    # Gradient contribution from boundary terms
    n_boundary = N * 3
    for idx in range(n_boundary):
        i = idx // 3
        k = idx % 3
        w = weights[n_pairs + idx]
        sign = np.sign(centers[i, k])
        if sign == 0:
            sign = 1.0
        grad[i, k] += w * (-sign)

    return grad


def _repulsive_energy(centers, r_target=0.14):
    """
    Repulsive potential energy: sum of 1/d^2 for pairs closer than 2*r_target,
    plus wall repulsion.
    """
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 1e10
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 1e10

    # Pairwise repulsion
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)

    iu = np.triu_indices(N, k=1)
    d_pairs = dists[iu]
    # Only count pairs that are too close
    mask = d_pairs < 2 * r_target
    if np.any(mask):
        e_pairs = np.sum(1.0 / (d_pairs[mask] ** 2 + 1e-10))
    else:
        e_pairs = 0.0

    # Wall repulsion
    b = 0.5 - np.abs(centers)
    mask_wall = b < r_target
    if np.any(mask_wall):
        e_wall = np.sum(1.0 / (b[mask_wall] ** 2 + 1e-10))
    else:
        e_wall = 0.0

    return e_pairs + e_wall


def _grad_repulsive_energy(centers, r_target=0.14):
    """
    Gradient of repulsive potential energy.
    """
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return np.zeros_like(centers)
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return np.zeros_like(centers)

    grad = np.zeros_like(centers)

    # Pairwise gradient
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)

    iu = np.triu_indices(N, k=1)
    d_pairs = dists[iu]
    mask = d_pairs < 2 * r_target

    for idx in range(len(iu)):
        if not mask[idx]:
            continue
        i, j = iu[0][idx], iu[1][idx]
        d = dists[i, j]
        if d < 1e-10:
            continue
        # d(1/d^2)/d(centers[i]) = -2 * (centers[i]-centers[j]) / d^4
        vec = (centers[i] - centers[j])
        factor = -2.0 / (d ** 4 + 1e-10)
        grad[i] += factor * vec
        grad[j] += -factor * vec

    # Wall gradient
    b = 0.5 - np.abs(centers)
    mask_wall = b < r_target
    for idx in range(N * 3):
        if not mask_wall[idx]:
            continue
        i = idx // 3
        k = idx % 3
        d = b[idx]
        if d < 1e-10:
            continue
        sign = np.sign(centers[i, k])
        if sign == 0:
            sign = 1.0
        # d(1/b^2)/d(centers[i,k]) = -2/b^3 * sign
        grad[i, k] += -2.0 * sign / (d ** 3 + 1e-10)

    return grad


def _construct_cubic_grid():
    """Simple cubic grid as baseline."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _construct_bcc_lattice():
    """
    BCC (Body-Centered Cubic) lattice perturbation initialization.
    """
    centers = []
    m = 5  # 5x5x2 = 50
    for layer in range(2):
        z = -0.5 + (layer + 0.5) / 2.0
        for i in range(5):
            for j in range(5):
                offset = 0.05 * (1 if (i + j) % 2 == 0 else -1)
                x = -0.5 + (i + 0.5) / 5.0 + offset * 0.1
                y = -0.5 + (j + 0.5) / 5.0 - offset * 0.1
                centers.append([x, y, z])
    return np.array(centers[:N])


def _construct_fcc_like():
    """
    FCC-like layered initialization.
    Layers are offset to maximize local density.
    """
    centers = []
    # 4 layers of 5x4 = 80 positions, pick 50
    # Or 5 layers of 5x3 = 75 positions
    # Let's do 5 layers of 10 spheres (5x2) or similar
    # Simpler: 4 layers. 4x4=16, 4x4=16, 3x3=9, 3x2=6? Total 41. Need 50.
    # Let's use 5 layers.
    # Layer 0: 5x3 = 15
    # Layer 1: 5x3 = 15
    # Layer 2: 5x2 = 10
    # Layer 3: 5x2 = 10
    # Total 50.
    
    layer_configs = [
        (5, 3),
        (5, 3),
        (5, 2),
        (5, 2),
    ]
    
    n_layers = len(layer_configs)
    z_spacing = 1.0 / n_layers
    
    for l in range(n_layers):
        z = -0.5 + (l + 0.5) * z_spacing
        nx, ny = layer_configs[l]
        x_spacing = 1.0 / nx
        y_spacing = 1.0 / ny
        
        # Offset alternating layers to create FCC-like structure
        offset_x = 0.0 if l % 2 == 0 else x_spacing / 2.0
        offset_y = 0.0 if l % 2 == 0 else y_spacing / 2.0
        
        for i in range(nx):
            for j in range(ny):
                x = -0.5 + (i + 0.5) * x_spacing + offset_x
                y = -0.5 + (j + 0.5) * y_spacing + offset_y
                # Clip to stay inside
                x = np.clip(x, -0.49, 0.49)
                y = np.clip(y, -0.49, 0.49)
                centers.append([x, y, z])
                
    return np.array(centers[:N])


def _random_initialization(seed=None):
    """Random initialization inside the cube with some margin."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    margin = 0.1
    centers = rng.uniform(-0.5 + margin, 0.5 - margin, size=(N, 3))
    return centers


def _optimize_smooth(centers_start, time_limit=30.0, beta=50.0):
    """
    Optimize using the smooth soft-min surrogate.
    """
    start_time = time.time()
    params = centers_start.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)

    def obj(p):
        c = p.reshape(N, 3)
        return -_soft_min_radius(c, beta)

    def grad(p):
        c = p.reshape(N, 3)
        return -_grad_soft_min_radius(c, beta).flatten()

    best_params = params.copy()
    best_r = _radius(centers_start)

    try:
        result = minimize(
            obj,
            params,
            jac=grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        if result.fun < -best_r:
            new_centers = result.x.reshape(N, 3)
            new_r = _radius(new_centers)
            if new_r > best_r:
                best_r = new_r
                best_params = result.x.copy()
    except Exception:
        pass

    elapsed = time.time() - start_time
    return best_params.reshape(N, 3), best_r, elapsed


def _optimize_repulsive(centers_start, time_limit=20.0, r_target=0.14):
    """
    Optimize using repulsive potential energy minimization.
    """
    start_time = time.time()
    params = centers_start.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)

    def obj(p):
        c = p.reshape(N, 3)
        return _repulsive_energy(c, r_target)

    def grad(p):
        c = p.reshape(N, 3)
        return _grad_repulsive_energy(c, r_target).flatten()

    best_params = params.copy()
    best_r = _radius(centers_start)

    try:
        result = minimize(
            obj,
            params,
            jac=grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        new_centers = result.x.reshape(N, 3)
        new_r = _radius(new_centers)
        if new_r > best_r:
            best_r = new_r
            best_params = result.x.copy()
    except Exception:
        pass

    elapsed = time.time() - start_time
    return best_params.reshape(N, 3), best_r, elapsed


def _optimize_exact(centers_start, time_limit=30.0):
    """
    Optimize using the exact (nonsmooth) radius function.
    """
    start_time = time.time()
    params = centers_start.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)

    def obj(p):
        c = p.reshape(N, 3)
        return -_radius(c)

    best_params = params.copy()
    best_r = _radius(centers_start)

    try:
        result = minimize(
            obj,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        if result.fun < -best_r:
            new_centers = result.x.reshape(N, 3)
            new_r = _radius(new_centers)
            if new_r > best_r:
                best_r = new_r
                best_params = result.x.copy()
    except Exception:
        pass

    elapsed = time.time() - start_time
    return best_params.reshape(N, 3), best_r, elapsed


def _simulated_annealing(centers_start, time_limit=20.0, seed=None):
    """
    Simulated annealing to escape local minima.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    start_time = time.time()
    centers = centers_start.copy()
    current_r = _radius(centers)
    best_centers = centers.copy()
    best_r = current_r

    T = 0.005
    cooling = 0.998

    while time.time() - start_time < time_limit:
        if T < 1e-7:
            break

        n_move = rng.integers(1, 4)
        idxs = rng.choice(N, size=n_move, replace=False)
        delta_scale = T * 0.5
        for idx in idxs:
            delta = rng.normal(0, delta_scale, size=3)
            centers[idx] += delta
            centers[idx] = np.clip(centers[idx], -0.5, 0.5)

        new_r = _radius(centers)
        delta_r = new_r - current_r

        if delta_r > 0 or rng.random() < np.exp(delta_r / T):
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            centers = best_centers.copy() if current_r < best_r else centers

        T *= cooling

    return best_centers, best_r, time.time() - start_time


def construct():
    """Construct the best packing found within the time budget."""
    global_start = time.time()

    best_centers = None
    best_r = 0.0

    # Define initializations
    inits = [
        ("cubic", _construct_cubic_grid()),
        ("bcc", _construct_bcc_lattice()),
        ("fcc", _construct_fcc_like()),
        ("rand1", _random_initialization(seed=42)),
        ("rand2", _random_initialization(seed=123)),
    ]

    # Sort inits by initial radius to prioritize good starts? 
    # Or just run all.
    
    for name, centers_start in inits:
        if time.time() - global_start > DEADLINE - 30.0:
            break
        
        current_r = _radius(centers_start)
        if current_r > best_r:
            best_r = current_r
            best_centers = centers_start.copy()

        # Phase 1: Repulsive Relaxation
        if time.time() - global_start < DEADLINE - 25.0:
            time_remaining = DEADLINE - (time.time() - global_start) - 10.0
            if time_remaining > 5.0:
                opt_centers, opt_r, elapsed = _optimize_repulsive(
                    centers_start,
                    time_limit=min(10.0, time_remaining),
                    r_target=0.14
                )
                if opt_r > current_r:
                    current_r = opt_r
                    centers_start = opt_centers.copy()

        # Phase 2: Soft-Min Optimization
        if time.time() - global_start < DEADLINE - 20.0:
            time_remaining = DEADLINE - (time.time() - global_start) - 10.0
            if time_remaining > 5.0:
                opt_centers, opt_r, elapsed = _optimize_smooth(
                    centers_start,
                    time_limit=min(15.0, time_remaining),
                    beta=50.0
                )
                if opt_r > current_r:
                    current_r = opt_r
                    centers_start = opt_centers.copy()

        # Phase 3: Exact Polish
        if time.time() - global_start < DEADLINE - 10.0:
            time_remaining = DEADLINE - (time.time() - global_start) - 5.0
            if time_remaining > 2.0:
                opt_centers, opt_r, elapsed = _optimize_exact(
                    centers_start,
                    time_limit=min(10.0, time_remaining)
                )
                if opt_r > current_r:
                    current_r = opt_r
                    centers_start = opt_centers.copy()

        # Update global best
        if current_r > best_r:
            best_r = current_r
            best_centers = centers_start.copy()

    # Final Polish on the absolute best found
    if time.time() - global_start < DEADLINE - 5.0 and best_centers is not None:
        time_remaining = DEADLINE - (time.time() - global_start) - 2.0
        if time_remaining > 2.0:
            opt_centers, opt_r, elapsed = _optimize_exact(
                best_centers,
                time_limit=min(10.0, time_remaining)
            )
            if opt_r > best_r:
                best_r = opt_r
                best_centers = opt_centers.copy()

    # Ensure validity
    if best_centers is None:
        best_centers = _construct_cubic_grid()
        
    final_centers = np.clip(best_centers, -0.5, 0.5)
    final_r = _radius(final_centers)

    if final_r <= 0.0:
        final_centers = _construct_cubic_grid()

    return final_centers
# EVOLVE-BLOCK-END
