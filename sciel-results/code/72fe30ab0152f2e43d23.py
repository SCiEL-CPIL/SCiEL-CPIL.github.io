"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centers):
    """Compute the maximum common radius for the given centers."""
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to walls: square is [-0.5, 0.5]^2
    min_x = 0.5 - np.max(centers[:, 0])
    min_y = 0.5 - np.max(centers[:, 1])
    min_x2 = np.min(centers[:, 0]) + 0.5
    min_y2 = np.min(centers[:, 1]) + 0.5
    min_wall = min(min_x, min_y, min_x2, min_y2)
    
    return min(min_pair, min_wall)


def _smooth_objective(centers_flat, alpha=50.0):
    """
    Smooth surrogate objective: minimize a log-sum-exp of inverse distances.
    This is differentiable and helps gradient-based optimizers.
    """
    centers = centers_flat.reshape(N, 2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Use squared distances for speed, then take sqrt
    # Inverse distances (avoid division by zero)
    inv_dists = 1.0 / np.sqrt(np.maximum(dists_sq, 1e-10))
    
    # Log-sum-exp of inverse distances (soft min of pairwise distances)
    # We want to maximize min distance, so minimize -log(sum(exp(alpha * (-1/d))))
    # Simpler: minimize sum of (1/d)^alpha which approximates min
    penalty = np.sum(np.power(inv_dists, alpha))
    
    # Wall penalties
    wall_pen = 0.0
    for coord in range(2):
        vals = centers[:, coord]
        # Distance to each wall
        d_left = 0.5 + vals       # distance to left wall
        d_right = 0.5 - vals      # distance to right wall
        d_bottom = 0.5 + vals     # same as left for y
        d_top = 0.5 - vals        # same as right for y
        
        # Penalize being too close to walls
        wall_pen += np.sum(np.power(1.0 / np.maximum(d_left, 1e-10), alpha))
        wall_pen += np.sum(np.power(1.0 / np.maximum(d_right, 1e-10), alpha))
    
    return penalty


def _local_search(centers, start_time, deadline, max_iter=200):
    """
    Local search using scipy minimize with smooth objective.
    """
    current = centers.copy()
    current_radius = _compute_radius(current)
    best = current.copy()
    best_radius = current_radius
    
    # Use Nelder-Mead on the smooth objective
    flat = current.flatten()
    
    def objective(x):
        return _smooth_objective(x, alpha=50.0)
    
    def objective_neg_radius(x):
        c = x.reshape(N, 2)
        return -_compute_radius(c)
    
    # Try Nelder-Mead on negative radius (direct approach)
    try:
        result = minimize(
            objective_neg_radius,
            flat,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if result.x is not None:
            candidate = result.x.reshape(N, 2)
            # Check validity
            if np.all(np.isfinite(candidate)):
                cand_radius = _compute_radius(candidate)
                if cand_radius > best_radius:
                    best = candidate.copy()
                    best_radius = cand_radius
    except Exception:
        pass
    
    # Also try COBYLA which handles constraints better
    if time.time() - start_time < deadline - 5:
        try:
            # Constraints: all points inside [-0.5, 0.5]^2
            constraints = []
            for i in range(N):
                for j in range(2):
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 0.5 - x[i*2+j]})
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 0.5 + x[i*2+j]})
            
            result = minimize(
                objective_neg_radius,
                flat,
                method='COBYLA',
                constraints=constraints,
                options={'maxiter': max_iter // 2, 'rhobeg': 0.01}
            )
            if result.x is not None:
                candidate = result.x.reshape(N, 2)
                if np.all(np.isfinite(candidate)):
                    cand_radius = _compute_radius(candidate)
                    if cand_radius > best_radius:
                        best = candidate.copy()
                        best_radius = cand_radius
        except Exception:
            pass
    
    return best, best_radius


def _perturbation_escape(centers, radius, start_time, deadline, rng):
    """
    Perturb 2-5 random points and re-optimize to escape local optima.
    """
    n_perturb = rng.randint(2, 6)  # 2 to 5 points
    indices = rng.choice(N, size=n_perturb, replace=False)
    
    perturbed = centers.copy()
    scale = radius * 0.3  # Perturbation scale
    for idx in indices:
        delta = rng.normal(0, scale, 2)
        perturbed[idx] += delta
        perturbed[idx, 0] = np.clip(perturbed[idx, 0], -0.49, 0.49)
        perturbed[idx, 1] = np.clip(perturbed[idx, 1], -0.49, 0.49)
    
    # Re-optimize from perturbed state
    best, best_radius = _local_search(perturbed, start_time, deadline, max_iter=50)
    
    if best_radius > radius:
        return best, best_radius
    return centers, radius


def _initial_hexagonal_lattice():
    """Generate a hexagonal lattice initial configuration."""
    points = []
    # Use rows with hexagonal offset
    # 50 circles: 7 rows of 7 + 1 extra, or other configurations
    rows = [7, 7, 7, 7, 7, 7, 8]  # 50 total
    
    # Calculate spacing to fit in [-0.5, 0.5]
    max_row = max(rows)
    dx = 0.95 / max_row
    
    # Vertical spacing for hexagonal
    dy = dx * np.sqrt(3) / 2
    
    # Center vertically
    total_height = (len(rows) - 1) * dy
    y_start = -total_height / 2
    
    for i, count in enumerate(rows):
        y = y_start + i * dy
        offset = (dx / 2) if i % 2 == 1 else 0
        row_width = (count - 1) * dx
        x_start = -row_width / 2
        
        for j in range(count):
            x = x_start + j * dx + offset
            points.append([x, y])
    
    return np.array(points[:N])


def _initial_random(rng, n=N):
    """Generate a random initial configuration."""
    centers = rng.uniform(-0.45, 0.45, (n, 2))
    return centers


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Start with hexagonal lattice
    best_centers = _initial_hexagonal_lattice()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Local search from hexagonal start
    best_centers, best_radius = _local_search(best_centers, start_time, deadline, max_iter=300)
    
    # Phase 2: Multi-start with different initializations
    if time.time() - start_time < deadline - 15:
        # Try a few random starts
        for seed in [123, 456, 789, 101112]:
            if time.time() - start_time > deadline - 10:
                break
            rng_local = np.random.RandomState(seed)
            random_centers = _initial_random(rng_local)
            local_best, local_radius = _local_search(random_centers, start_time, deadline, max_iter=100)
            if local_radius > best_radius:
                best_centers = local_best
                best_radius = local_radius
    
    # Phase 3: Perturbation-based escape from current best
    if time.time() - start_time < deadline - 10:
        for _ in range(20):  # Up to 20 perturbation attempts
            if time.time() - start_time > deadline - 5:
                break
            candidate, cand_radius = _perturbation_escape(
                best_centers, best_radius, start_time, deadline, rng
            )
            if cand_radius > best_radius:
                best_centers = candidate
                best_radius = cand_radius
    
    # Phase 4: Final polish with more iterations if time allows
    if time.time() - start_time < deadline - 8:
        final_centers, final_radius = _local_search(best_centers, start_time, deadline, max_iter=500)
        if final_radius > best_radius:
            best_centers = final_centers
            best_radius = final_radius
    
    return best_centers
# EVOLVE-BLOCK-END
