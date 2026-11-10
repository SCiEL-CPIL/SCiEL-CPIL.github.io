import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _build_grid_neighbors(x, cell_size):
    """Build a spatial grid for O(N) neighbor lookups.
    Returns: dict mapping cell index -> list of point indices, and cell_size.
    """
    x = x.reshape(N, 2)
    min_vals = x.min(axis=0)
    max_vals = x.max(axis=0)
    grid_w = int(np.ceil((max_vals[0] - min_vals[0]) / cell_size)) + 2
    grid_h = int(np.ceil((max_vals[1] - min_vals[1]) / cell_size)) + 2
    grid = {}
    for i in range(N):
        ci = int((x[i, 0] - min_vals[0]) / cell_size)
        cj = int((x[i, 1] - min_vals[1]) / cell_size)
        key = (ci, cj)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    return grid, min_vals, cell_size

def _relaxation_pass(x, n_iters=200, cell_size=0.25, dt=0.001):
    """Physics-inspired relaxation: repulsive forces between nearby circles + boundary repulsion.
    Uses grid-based neighbor lookup for efficiency.
    """
    x = x.reshape(N, 2).copy()
    for it in range(n_iters):
        grid, min_vals, cs = _build_grid_neighbors(x, cell_size)
        forces = np.zeros_like(x)
        
        for i in range(N):
            ci = int((x[i, 0] - min_vals[0]) / cs)
            cj = int((x[i, 1] - min_vals[1]) / cs)
            # Check 3x3 neighborhood
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    key = (ci + di, cj + dj)
                    if key in grid:
                        for j in grid[key]:
                            if j == i:
                                continue
                            diff = x[i] - x[j]
                            d = np.sqrt(np.dot(diff, diff))
                            if d < 1e-12:
                                # Random small push for coincident points
                                forces[i] += 0.01 * np.random.randn(2)
                            elif d < 0.35:
                                # Repulsive force: stronger when closer
                                f = 1.0 / (d * d + 1e-6)
                                forces[i] += f * diff / d
        
        # Boundary repulsion: push inward if too close to edge
        radii = np.sqrt(np.sum(x ** 2, axis=1))
        for i in range(N):
            if radii[i] > 0.95:
                forces[i] -= 0.5 * x[i] / max(radii[i], 1e-12)
        
        # Update positions with adaptive step size
        step = dt * (1.0 - 0.5 * it / n_iters)
        x += step * forces
        
        # Clamp to stay inside container
        radii = np.sqrt(np.sum(x ** 2, axis=1))
        mask = radii > 0.999
        if np.any(mask):
            x[mask] *= 0.999 / radii[mask]
    
    return x

def _calc_score(x):
    """Calculate the packing score (max common radius) for centers x."""
    x = np.asarray(x).reshape(N, 2)
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Boundary distances
    radii = np.sqrt(np.sum(x ** 2, axis=1))
    min_bound = np.min(1.0 - radii)
    return min(min_pair, min_bound) / 2.0

def _hex_lattice_init():
    """Generate initial hexagonal lattice points, sorted by radius for better centering."""
    spacing = 0.22
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            # Sort by radius to get a more centered packing
            idx = np.argsort(radius, kind="stable")[:N]
            return points[idx]
        spacing *= 0.97

def _soft_min_energy_and_grad(x, alpha):
    """
    Compute the soft-min energy and gradient.
    We want to maximize the minimum distance, so we minimize the soft-min.
    x: (N, 2) array of centers.
    alpha: temperature parameter for soft-min.
    Returns: (energy, gradient)
    """
    x = x.reshape(N, 2)
    
    # Pairwise distances (upper triangle)
    i_idx, j_idx = np.triu_indices(N, k=1)
    diff_vec = x[i_idx] - x[j_idx]
    d_pairs = np.sqrt(np.sum(diff_vec ** 2, axis=1))
    
    # Boundary distances
    radii = np.sqrt(np.sum(x ** 2, axis=1))
    b_dists = 1.0 - radii
    
    # Combine all distances
    all_dists = np.concatenate([d_pairs, b_dists])
    n_pairs = len(d_pairs)
    n_bounds = N
    
    # Soft-min: E = (1/alpha) * log(sum(exp(-alpha * d_k)))
    # This is approximately min(d_k) when alpha is large.
    exponents = -alpha * all_dists
    max_exp = np.max(exponents)
    exponents_shifted = exponents - max_exp
    weights = np.exp(exponents_shifted)
    sum_weights = np.sum(weights)
    
    if sum_weights <= 0 or not np.isfinite(sum_weights):
        return 0.0, np.zeros_like(x)
    
    energy = (1.0 / alpha) * (np.log(sum_weights) + max_exp)
    
    # Gradient computation
    grad = np.zeros_like(x)
    
    # Pairwise part
    w_pairs = weights[:n_pairs]
    mask = d_pairs > 1e-12
    grad_unit = np.zeros_like(diff_vec)
    grad_unit[mask] = diff_vec[mask] / d_pairs[mask, np.newaxis]
    
    weighted_grad = w_pairs[:, np.newaxis] * grad_unit
    inv_sum = 1.0 / sum_weights
    
    np.add.at(grad, i_idx, -weighted_grad * inv_sum)
    np.add.at(grad, j_idx, weighted_grad * inv_sum)
    
    # Boundary part
    w_bounds = weights[n_pairs:n_pairs + n_bounds]
    radii_safe = np.where(radii < 1e-12, 1e-12, radii)
    grad_bounds = -x / radii_safe[:, np.newaxis]
    
    grad -= w_bounds[:, np.newaxis] * grad_bounds * inv_sum
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin (evaluator kills at 150s)
    
    # Initial guess
    x0 = _hex_lattice_init()
    
    best_x = x0.copy()
    best_score = _calc_score(best_x)
    
    # Phase 1: Grid-based relaxation to quickly find a good region
    remaining = deadline - time.time()
    if remaining > 30.0:
        try:
            x_relaxed = _relaxation_pass(x0, n_iters=300, cell_size=0.25, dt=0.002)
            if np.all(np.isfinite(x_relaxed)) and np.all(np.linalg.norm(x_relaxed, axis=1) < 1.0):
                score = _calc_score(x_relaxed)
                if score > best_score:
                    best_score = score
                    best_x = x_relaxed.copy()
        except Exception:
            pass
    
    current_x = best_x.copy()
    
    # Phase 2: Progressive alpha schedule with soft-min optimization
    alphas = [5.0, 10.0, 20.0, 40.0, 80.0, 150.0, 300.0]
    
    for alpha in alphas:
        remaining = deadline - time.time()
        if remaining < 15.0:  # Leave time for final polish
            break
            
        try:
            # Adjust maxiter based on remaining time
            max_iter = min(500, int(remaining * 2))
            res = minimize(
                _soft_min_energy_and_grad,
                current_x,
                args=(alpha,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 50}
            )
            x_new = res.x.reshape(N, 2)
            
            # Validate: check all finite, inside container
            if not np.all(np.isfinite(x_new)):
                continue
            if np.any(np.linalg.norm(x_new, axis=1) >= 1.0):
                continue
                
            score = _calc_score(x_new)
            if score > best_score:
                best_score = score
                best_x = x_new.copy()
                current_x = x_new.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish with very high alpha
    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            res = minimize(
                _soft_min_energy_and_grad,
                current_x,
                args=(500.0,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 40}
            )
            x_new = res.x.reshape(N, 2)
            if np.all(np.isfinite(x_new)) and np.all(np.linalg.norm(x_new, axis=1) < 1.0):
                score = _calc_score(x_new)
                if score > best_score:
                    best_score = score
                    best_x = x_new.copy()
        except Exception:
            pass
    
    # Phase 4: One more relaxation + polish cycle if time permits
    remaining = deadline - time.time()
    if remaining > 10.0:
        try:
            x_relaxed2 = _relaxation_pass(best_x, n_iters=100, cell_size=0.2, dt=0.0005)
            if np.all(np.isfinite(x_relaxed2)) and np.all(np.linalg.norm(x_relaxed2, axis=1) < 1.0):
                score = _calc_score(x_relaxed2)
                if score > best_score:
                    best_score = score
                    best_x = x_relaxed2.copy()
                    current_x = x_relaxed2.copy()
        except Exception:
            pass
        
        remaining = deadline - time.time()
        if remaining > 5.0:
            try:
                res = minimize(
                    _soft_min_energy_and_grad,
                    current_x,
                    args=(800.0,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 30}
                )
                x_new = res.x.reshape(N, 2)
                if np.all(np.isfinite(x_new)) and np.all(np.linalg.norm(x_new, axis=1) < 1.0):
                    score = _calc_score(x_new)
                    if score > best_score:
                        best_score = score
                        best_x = x_new.copy()
            except Exception:
                pass
    
    # Return the best valid solution
    return best_x
