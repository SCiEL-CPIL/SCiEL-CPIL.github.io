import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _calc_score(x):
    """Calculate the actual packing score (min radius). Vectorized."""
    x = np.asarray(x).reshape(N, 2)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    radii = np.sqrt(np.sum(x * x, axis=1))
    min_bound = np.min(1.0 - radii)
    return min(min_pair, min_bound) / 2.0

def _grid_neighbors(x, cell_size):
    """
    Grid-based neighbor lookup for O(N) expected neighbor queries.
    Returns list of (i, j) pairs where |x_i - x_j| < 2*cell_size.
    """
    n = len(x)
    # Grid cell assignment
    grid = {}
    for i in range(n):
        cx = int(np.floor(x[i, 0] / cell_size))
        cy = int(np.floor(x[i, 1] / cell_size))
        key = (cx, cy)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    
    # Collect neighbor pairs
    pairs = []
    for i in range(n):
        cx = int(np.floor(x[i, 0] / cell_size))
        cy = int(np.floor(x[i, 1] / cell_size))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (cx + dx, cy + dy)
                if key in grid:
                    for j in grid[key]:
                        if j > i:
                            pairs.append((i, j))
    return pairs

def _force_relaxation(x, max_iters=500, step_size=0.001, deadline=None):
    """
    Physics-inspired relaxation using grid-based neighbor lookup.
    Applies repulsive forces between close pairs and attractive forces toward center.
    """
    x = x.copy()
    n = len(x)
    # Cell size ~ expected nearest neighbor distance
    cell_size = 0.25
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        # Get nearby pairs using grid
        pairs = _grid_neighbors(x, cell_size)
        
        # Compute forces
        forces = np.zeros_like(x)
        
        for i, j in pairs:
            diff = x[i] - x[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Random nudge for coincident points
                forces[i] += np.random.randn(2) * 0.01
                forces[j] -= np.random.randn(2) * 0.01
            elif dist < 0.25:  # Repulsion zone
                # Strong repulsion when too close
                force_mag = 1.0 / (dist * dist + 1e-6)
                forces[i] += (diff / dist) * force_mag
                forces[j] -= (diff / dist) * force_mag
        
        # Boundary constraint: push inward if outside
        radii = np.linalg.norm(x, axis=1)
        for i in range(n):
            if radii[i] > 0.95:
                # Push toward center
                force_mag = (radii[i] - 0.95) * 2.0
                forces[i] -= (x[i] / radii[i]) * force_mag
        
        # Update positions with damping
        x += step_size * forces
        
        # Clamp to unit circle
        new_radii = np.linalg.norm(x, axis=1)
        mask = new_radii > 1.0
        if np.any(mask):
            x[mask] /= new_radii[mask, np.newaxis]
            
    return x

def _soft_min_energy_and_grad(x, alpha):
    """
    Compute the soft-min energy and gradient.
    x: (N, 2) array of centers.
    alpha: temperature parameter for soft-min.
    Returns: (energy, gradient)
    """
    x = np.asarray(x).reshape(N, 2)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    radii = np.sqrt(np.sum(x * x, axis=1))
    b_dists = 1.0 - radii
    
    p_dists = dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([p_dists, b_dists])
    
    exponents = -alpha * all_dists
    max_exp = np.max(exponents)
    if max_exp > 0:
        exponents -= max_exp
        shift = max_exp
    else:
        shift = 0.0
        
    weights = np.exp(exponents)
    sum_weights = np.sum(weights)
    
    if sum_weights == 0:
        return 0.0, np.zeros_like(x)
        
    energy = (1.0 / alpha) * (np.log(sum_weights) + shift)
    
    grad = np.zeros_like(x)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    w_pairs = weights[:len(p_dists)]
    d_pairs = p_dists
    
    diff_vec = x[i_idx] - x[j_idx]
    mask = d_pairs > 1e-10
    grad_unit = np.zeros_like(diff_vec)
    grad_unit[mask] = diff_vec[mask] / d_pairs[mask, np.newaxis]
    
    weighted_grad = w_pairs[:, np.newaxis] * grad_unit
    
    np.add.at(grad, i_idx, -weighted_grad / sum_weights)
    np.add.at(grad, j_idx, weighted_grad / sum_weights)
    
    w_bounds = weights[len(p_dists):]
    radii_safe = np.where(radii < 1e-10, 1e-10, radii)
    grad_bounds = -x / radii_safe[:, np.newaxis]
    
    weighted_bounds = w_bounds[:, np.newaxis] * grad_bounds
    grad += -weighted_bounds / sum_weights
    
    return energy, grad

def _ring_init():
    """
    Generate initial configuration based on known ring structure:
    1 center, 5 in ring 1, 13 in ring 2, 31 in ring 3.
    """
    centers = []
    
    # Center
    centers.append([0.0, 0.0])
    
    # Ring 1: 5 points
    r_est = 0.125
    R1 = r_est / np.sin(np.pi / 5)
    for i in range(5):
        angle = 2 * np.pi * i / 5 + np.pi / 5
        centers.append([R1 * np.cos(angle), R1 * np.sin(angle)])
    
    # Ring 2: 13 points
    R2 = R1 + 2 * r_est
    for i in range(13):
        angle = 2 * np.pi * i / 13
        centers.append([R2 * np.cos(angle), R2 * np.sin(angle)])
    
    # Ring 3: 31 points
    R3 = R2 + 2 * r_est
    for i in range(31):
        angle = 2 * np.pi * i / 31 + np.pi / 31
        centers.append([R3 * np.cos(angle), R3 * np.sin(angle)])
    
    return np.array(centers)

def _hex_lattice_init():
    """Generate initial hexagonal lattice points."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _anneal_and_polish(x_start, alphas, deadline, maxiter_base=2000, ftol_base=1e-12, gtol_base=1e-10, use_force_relax=True):
    """
    Run annealing sequence of soft-min optimizations with optional force relaxation between steps.
    Returns best_x, best_score found along the way.
    """
    best_x = x_start.copy()
    best_score = _calc_score(best_x)
    current_x = x_start.copy()
    
    for alpha in alphas:
        if time.time() > deadline:
            break
        try:
            # Scale iterations/tolerance based on alpha for efficiency
            maxiter = maxiter_base
            ftol = ftol_base
            gtol = gtol_base
            
            if alpha > 50:
                ftol = 1e-14
                gtol = 1e-12
                maxiter = int(maxiter_base * 1.5)
                
            res = minimize(
                _soft_min_energy_and_grad,
                current_x,
                args=(alpha,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': int(maxiter), 'ftol': ftol, 'gtol': gtol}
            )
            
            x_new = res.x.reshape(N, 2)
            
            # Apply force relaxation to resolve any remaining local conflicts
            if use_force_relax and time.time() < deadline - 5.0:
                x_new = _force_relaxation(x_new, max_iters=100, step_size=0.0005, deadline=deadline)
            
            score_new = _calc_score(x_new)
            
            if score_new > best_score:
                best_score = score_new
                best_x = x_new.copy()
                current_x = x_new.copy()
        except Exception:
            pass
            
    return best_x, best_score

def construct():
    start_time = time.time()
    # 150s limit, keep 10s margin for safety
    deadline = start_time + 140.0
    
    # Try both initializations and pick the better one
    x_ring = _ring_init()
    x_hex = _hex_lattice_init()
    
    score_ring = _calc_score(x_ring)
    score_hex = _calc_score(x_hex)
    
    if score_ring >= score_hex:
        x0 = x_ring
    else:
        x0 = x_hex
    
    best_x = x0.copy()
    best_score = _calc_score(best_x)
    
    # Strategy 1: Annealing from best initial
    # Coarse to fine alpha
    alphas_main = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    best_x, best_score = _anneal_and_polish(x0, alphas_main, deadline, use_force_relax=True)
    
    # Strategy 2: Random Restarts / Perturbations
    rng = np.random.default_rng(42)
    
    max_restarts = 15
    restarts_done = 0
    
    while time.time() < deadline - 20.0 and restarts_done < max_restarts:
        # Perturb the current best
        perturb_scale = 0.015 + 0.01 * (restarts_done / max_restarts)
        
        x_perturbed = best_x + rng.normal(0, perturb_scale, size=best_x.shape)
        
        # Quick annealing sequence for the perturbed point
        alphas_restart = [20.0, 50.0, 100.0]
        
        x_candidate, score_candidate = _anneal_and_polish(
            x_perturbed, 
            alphas_restart, 
            deadline, 
            maxiter_base=500, 
            ftol_base=1e-11, 
            gtol_base=1e-9,
            use_force_relax=True
        )
        
        if score_candidate > best_score:
            best_score = score_candidate
            best_x = x_candidate.copy()
        
        restarts_done += 1
        
    # Final Polish: Very high alpha, tight tolerances, with force relaxation
    if time.time() < deadline - 5.0:
        alphas_final = [200.0, 500.0]
        best_x, best_score = _anneal_and_polish(
            best_x, 
            alphas_final, 
            deadline, 
            maxiter_base=3000, 
            ftol_base=1e-15, 
            gtol_base=1e-13,
            use_force_relax=True
        )
        
    # Final force relaxation pass
    if time.time() < deadline - 2.0:
        best_x = _force_relaxation(best_x, max_iters=200, step_size=0.0003, deadline=deadline)
        final_score = _calc_score(best_x)
        if final_score > best_score:
            best_score = final_score
    
    return best_x
