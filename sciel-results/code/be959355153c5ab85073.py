"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Known best packing radius for N=50 (Graham et al. 1998)
KNOWN_BEST_R = 0.1258254895304


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    # Distance to boundary
    dists_boundary = 1.0 - np.linalg.norm(centers, axis=1)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists, axis=1)
    return min(np.min(min_pairwise) / 2.0, np.min(dists_boundary))


def _soft_min_radius(centers, beta):
    """Soft-min objective for optimization (maximize this)."""
    n = len(centers)
    dists_boundary = 1.0 - np.linalg.norm(centers, axis=1)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists, axis=1)
    
    # Combine boundary and pairwise constraints
    all_dists = np.concatenate([dists_boundary, min_pairwise / 2.0])
    # Soft min
    log_sum = np.log(np.sum(np.exp(-beta * all_dists)))
    return -log_sum / beta


def _soft_min_radius_grad(centers, beta):
    """Gradient of soft-min radius."""
    n = len(centers)
    dists_boundary = 1.0 - np.linalg.norm(centers, axis=1)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists, axis=1)
    
    all_dists = np.concatenate([dists_boundary, min_pairwise / 2.0])
    weights = np.exp(-beta * all_dists)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(centers)
    
    # Boundary gradient
    for i in range(n):
        norm = np.linalg.norm(centers[i])
        if norm > 1e-10:
            grad[i] -= weights[i] * centers[i] / norm
        else:
            grad[i] -= weights[i] * 0.1 * np.ones(2)
    
    # Pairwise gradient
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                continue
            # This pair contributes to min_pairwise[i] and min_pairwise[j]
            # But we only track the min, so this is approximate
            # We'll use a simpler approach: just the boundary part is exact
    return grad


def _hex_init(n):
    """Hexagonal lattice initialization."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n:
            # Sort by distance from center, take first n
            order = np.argsort(radius[mask], kind="stable")
            return points[order[:n]]
        spacing *= 0.98


def _ring_init(n):
    """Ring-based initialization: center + rings of points."""
    centers = np.zeros((n, 2))
    if n == 1:
        return centers
    
    # Place first at center
    centers[0] = [0.0, 0.0]
    
    # Rings: 6, 12, 18, ... points per ring
    ring_counts = [6, 12, 18, 24, 30, 36, 42, 48, 54]
    idx = 1
    ring_num = 0
    while idx < n and ring_num < len(ring_counts):
        count = min(ring_counts[ring_num], n - idx)
        angle = 2 * np.pi / count
        radius = (ring_num + 1) * 0.12  # approximate spacing
        for i in range(count):
            if idx >= n:
                break
            a = i * angle + ring_num * 0.1  # offset angle
            centers[idx] = [radius * np.cos(a), radius * np.sin(a)]
            idx += 1
        ring_num += 1
    
    return centers


def _random_init(n, seed=0):
    """Random initialization with rejection sampling."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((n, 2))
    centers[0] = [0.0, 0.0]
    
    min_dist = 0.1  # initial minimum distance
    for i in range(1, n):
        for attempt in range(1000):
            r = np.sqrt(rng.uniform(0, 1))
            theta = rng.uniform(0, 2 * np.pi)
            pt = [r * np.cos(theta), r * np.sin(theta)]
            
            # Check distance to all previous points
            dists = np.linalg.norm(centers[:i] - pt, axis=1)
            if np.min(dists) >= min_dist and r <= 0.95:
                centers[i] = pt
                break
        else:
            # If we can't place, just put it at a random position
            r = np.sqrt(rng.uniform(0, 0.9))
            theta = rng.uniform(0, 2 * np.pi)
            centers[i] = [r * np.cos(theta), r * np.sin(theta)]
    
    return centers


def _optimize_from_init(centers, beta_start=50, beta_end=500, max_iter=200, time_limit=30):
    """Optimize from an initial configuration using L-BFGS-B."""
    start_time = time.time()
    x0 = centers.flatten()
    n = len(centers)
    
    def objective(x):
        c = x.reshape(n, 2)
        return -_soft_min_radius(c, beta_start)
    
    def gradient(x):
        c = x.reshape(n, 2)
        # Numerical gradient as fallback
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return grad
    
    result = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return result.x.reshape(n, 2)
    
    # Increase beta and re-optimize
    centers_opt = result.x.reshape(n, 2)
    for beta in [beta_start * 2, beta_start * 4, beta_end]:
        if time.time() - start_time > time_limit:
            break
        
        def objective_b(x, b=beta):
            c = x.reshape(n, 2)
            return -_soft_min_radius(c, b)
        
        def gradient_b(x, b=beta):
            c = x.reshape(n, 2)
            eps = 1e-7
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                grad[i] = (objective_b(x_plus) - objective_b(x_minus)) / (2 * eps)
            return grad
        
        result = minimize(objective_b, centers_opt.flatten(), method='L-BFGS-B',
                          jac=gradient_b, options={'maxiter': 100, 'ftol': 1e-13, 'gtol': 1e-9})
        centers_opt = result.x.reshape(n, 2)
    
    return centers_opt


def _nelder_mead_polish(centers, time_limit=10):
    """Nelder-Mead polishing with high beta."""
    start_time = time.time()
    n = len(centers)
    x0 = centers.flatten()
    
    def objective(x):
        c = x.reshape(n, 2)
        return -_soft_min_radius(c, 5000)
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return centers
    
    return result.x.reshape(n, 2)


def _final_polish(centers, time_limit=5):
    """Final high-precision polish."""
    start_time = time.time()
    n = len(centers)
    x0 = centers.flatten()
    
    def objective(x):
        c = x.reshape(n, 2)
        return -_soft_min_radius(c, 10000)
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return centers
    
    return result.x.reshape(n, 2)


def construct():
    """Construct centers for N equal circles in unit circle."""
    global N
    n = N
    total_deadline = 140.0  # safety margin from 150s limit
    start_time = time.time()
    
    # Generate multiple initial configurations
    inits = []
    
    # Hexagonal lattice
    hex_centers = _hex_init(n)
    inits.append(hex_centers)
    
    # Ring-based
    ring_centers = _ring_init(n)
    inits.append(ring_centers)
    
    # Random inits with different seeds
    for seed in range(5):
        if time.time() - start_time > total_deadline - 60:
            break
        rand_centers = _random_init(n, seed=seed)
        inits.append(rand_centers)
    
    # Optimize each initialization
    best_centers = None
    best_radius = 0.0
    
    for i, init in enumerate(inits):
        if time.time() - start_time > total_deadline - 30:
            break
        
        # Time budget per optimization
        remaining = total_deadline - (time.time() - start_time)
        opt_time = min(20, remaining * 0.5)
        
        if opt_time < 5:
            break
        
        opt_centers = _optimize_from_init(init, beta_start=50, beta_end=500,
                                           max_iter=150, time_limit=opt_time)
        
        # Nelder-Mead polish
        if time.time() - start_time < total_deadline - 15:
            nm_time = min(8, total_deadline - (time.time() - start_time) - 10)
            if nm_time > 3:
                opt_centers = _nelder_mead_polish(opt_centers, time_limit=nm_time)
        
        # Final polish
        if time.time() - start_time < total_deadline - 5:
            fp_time = min(4, total_deadline - (time.time() - start_time) - 3)
            if fp_time > 2:
                opt_centers = _final_polish(opt_centers, time_limit=fp_time)
        
        # Compute radius
        r = _compute_radius(opt_centers)
        if r > best_radius:
            best_radius = r
            best_centers = opt_centers.copy()
    
    # If we haven't beaten the known best, try a few more random restarts
    if best_radius < KNOWN_BEST_R * 0.999 and time.time() - start_time < total_deadline - 20:
        for seed in range(5, 10):
            if time.time() - start_time > total_deadline - 15:
                break
            rand_centers = _random_init(n, seed=seed)
            remaining = total_deadline - (time.time() - start_time)
            opt_time = min(15, remaining * 0.5)
            if opt_time < 5:
                break
            
            opt_centers = _optimize_from_init(rand_centers, beta_start=50, beta_end=500,
                                               max_iter=100, time_limit=opt_time)
            
            if time.time() - start_time < total_deadline - 10:
                nm_time = min(5, total_deadline - (time.time() - start_time) - 8)
                if nm_time > 2:
                    opt_centers = _nelder_mead_polish(opt_centers, time_limit=nm_time)
            
            if time.time() - start_time < total_deadline - 3:
                fp_time = min(3, total_deadline - (time.time() - start_time) - 2)
                if fp_time > 1:
                    opt_centers = _final_polish(opt_centers, time_limit=fp_time)
            
            r = _compute_radius(opt_centers)
            if r > best_radius:
                best_radius = r
                best_centers = opt_centers.copy()
    
    # Ensure validity: clip any points that might be outside
    if best_centers is None:
        best_centers = _hex_init(n)
    
    # Final safety check: ensure all points are inside the unit circle
    norms = np.linalg.norm(best_centers, axis=1)
    for i in range(len(best_centers)):
        if norms[i] > 0.999:
            best_centers[i] *= 0.999 / norms[i]
    
    return best_centers
# EVOLVE-BLOCK-END
