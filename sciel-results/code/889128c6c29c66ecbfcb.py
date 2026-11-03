"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _pairwise_distances(centers):
    """Compute pairwise distances matrix for N points."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    return dists


def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    n = len(centers)
    # Minimum pairwise distance
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Minimum distance to boundary
    # Square is [-0.5, 0.5]^2
    min_boundary = np.min(0.5 - np.abs(centers))
    
    r = min(min_pair_dist / 2.0, min_boundary)
    return r


def _objective(x, use_softmin=False, k=10.0):
    """Objective to minimize: negative of the radius (we maximize radius).
    
    Uses a smooth approximation via log-sum-exp for the minimum.
    """
    centers = x.reshape(-1, 2)
    n = len(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Distance to boundary
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    
    # For the pairwise part, we want min of dists/2
    # Use soft-min approximation: softmin(dists/2) ≈ -1/k * log(sum(exp(-k*dists/2)))
    # But we need to be careful with the diagonal (inf)
    
    # Simple approach: just compute the actual min
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    r = min(min_pair, min_boundary)
    
    return -r  # We minimize negative radius


def _objective_smooth(x, k=50.0):
    """Smooth objective using log-sum-exp for both min operations."""
    centers = x.reshape(-1, 2)
    n = len(centers)
    
    # Pairwise distances squared
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Distances
    dists = np.sqrt(dists_sq)
    
    # Soft-min of pairwise distances / 2
    # softmin(dists/2) ≈ (1/k) * log(sum(exp(-k * dists / 2)))
    # We want to minimize -softmin, which is max of -dists/2... 
    # Actually we want to maximize min(dists/2, min_boundary)
    # So minimize -min(dists/2, min_boundary)
    
    # For smooth approximation of min(a, b):
    # min(a,b) ≈ -1/k * log(exp(-k*a) + exp(-k*b))
    
    # Soft-min of all pairwise dists/2
    half_dists = dists / 2.0
    # Replace inf with a large value
    half_dists_finite = np.where(np.isfinite(half_dists), half_dists, 1e6)
    
    # Soft-min via log-sum-exp
    max_hd = np.max(half_dists_finite)
    exp_terms = np.exp(-k * (half_dists_finite - max_hd))
    log_sum = np.log(np.sum(exp_terms)) + k * max_hd
    softmin_pair = log_sum / k  # This approximates the min
    
    # Soft-min of boundary distances
    boundary_dists = 0.5 - np.abs(centers)
    max_bd = np.max(boundary_dists)
    exp_bd = np.exp(-k * (boundary_dists - max_bd))
    log_sum_bd = np.log(np.sum(exp_bd)) + k * max_bd
    softmin_boundary = log_sum_bd / k
    
    # Soft-min of the two
    a = softmin_pair
    b = softmin_boundary
    max_ab = max(a, b)
    exp_ab = np.exp(-k * (a - max_ab)) + np.exp(-k * (b - max_ab))
    softmin_total = np.log(exp_ab) / k + max_ab
    
    return -softmin_total


def _initial_grid():
    """Create an initial grid arrangement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _initial_jittered():
    """Create initial positions with some jitter."""
    base = _initial_grid()
    jitter = np.random.normal(0, 0.01, base.shape)
    return base + jitter


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_centers = None
    best_radius = -np.inf
    
    # Start with grid
    centers = _initial_grid()
    current_radius = _compute_radius(centers)
    best_centers = centers.copy()
    best_radius = current_radius
    
    # Phase 1: Local optimization from grid using smooth objective
    x0 = centers.flatten()
    
    try:
        result = minimize(
            _objective_smooth,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        elapsed = time.time() - start_time
        if elapsed < deadline - 20:
            new_centers = result.x.reshape(-1, 2)
            new_radius = _compute_radius(new_centers)
            if new_radius > best_radius:
                best_centers = new_centers.copy()
                best_radius = new_radius
    except Exception:
        pass
    
    # Phase 2: Try multiple random restarts with different initializations
    np.random.seed(42)
    n_restarts = 5
    for i in range(n_restarts):
        if time.time() > deadline - 25:
            break
        
        # Different initializations
        if i == 0:
            x_init = _initial_jittered()
        elif i == 1:
            # Hexagonal-like packing
            rows = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
            cols = int(np.ceil(N / rows))
            spacing_x = 1.0 / cols
            spacing_y = spacing_x * np.sqrt(3) / 2
            points = []
            for r in range(rows):
                for c in range(cols):
                    x = (c + 0.5) * spacing_x - 0.5
                    y = (r + 0.5) * spacing_y - 0.5
                    if r % 2 == 1:
                        x += spacing_x / 2
                    points.append((x, y))
                    if len(points) >= N:
                        break
                if len(points) >= N:
                    break
            x_init = np.array(points[:N])
            x_init += np.random.normal(0, 0.005, x_init.shape)
        elif i == 2:
            # Random but constrained
            x_init = np.random.uniform(-0.45, 0.45, (N, 2))
        elif i == 3:
            # Grid with larger jitter
            base = _initial_grid()
            jitter = np.random.normal(0, 0.03, base.shape)
            x_init = base + jitter
        else:
            # Another random
            x_init = np.random.uniform(-0.45, 0.45, (N, 2))
        
        x_flat = x_init.flatten()
        
        try:
            result = minimize(
                _objective_smooth,
                x_flat,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            new_centers = result.x.reshape(-1, 2)
            new_radius = _compute_radius(new_centers)
            if new_radius > best_radius:
                best_centers = new_centers.copy()
                best_radius = new_radius
        except Exception:
            pass
    
    # Phase 3: Final polish with tighter tolerance on the best result
    if time.time() < deadline - 10:
        x_flat = best_centers.flatten()
        try:
            result = minimize(
                _objective_smooth,
                x_flat,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            new_centers = result.x.reshape(-1, 2)
            new_radius = _compute_radius(new_centers)
            if new_radius > best_radius:
                best_centers = new_centers.copy()
                best_radius = new_radius
        except Exception:
            pass
    
    # Phase 4: Additional local refinement using the exact (non-smooth) objective
    if time.time() < deadline - 5:
        x_flat = best_centers.flatten()
        try:
            result = minimize(
                _objective,
                x_flat,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            new_centers = result.x.reshape(-1, 2)
            new_radius = _compute_radius(new_centers)
            if new_radius > best_radius:
                best_centers = new_centers.copy()
                best_radius = new_radius
        except Exception:
            pass
    
    # Validate and return
    if best_centers is None:
        best_centers = _initial_grid()
    
    # Ensure all points are strictly inside the square
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
# EVOLVE-BLOCK-END
