"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _ring_initial(n=50):
    """Create ring-based initial points matching known 50-circle packing structure.
    Known structure: center + rings of 6, 12, 18, 13 (or similar)."""
    # Target structure: 1 center, 6, 12, 18, 13
    # Estimated radii based on r ~ 0.126
    # r1 (6 pts): ~ 0.25
    # r2 (12 pts): ~ 0.50
    # r3 (18 pts): ~ 0.75
    # r4 (13 pts): ~ 0.87
    
    points_list = []
    
    # Center
    points_list.append(np.array([0.0, 0.0]))
    
    # Ring 1: 6 points
    r1 = 0.252
    for k in range(6):
        angle = 2 * np.pi * k / 6
        points_list.append(np.array([r1 * np.cos(angle), r1 * np.sin(angle)]))
        
    # Ring 2: 12 points
    r2 = 0.50
    for k in range(12):
        angle = 2 * np.pi * k / 12
        points_list.append(np.array([r2 * np.cos(angle), r2 * np.sin(angle)]))
        
    # Ring 3: 18 points
    r3 = 0.75
    for k in range(18):
        angle = 2 * np.pi * k / 18
        points_list.append(np.array([r3 * np.cos(angle), r3 * np.sin(angle)]))
        
    # Ring 4: 13 points
    r4 = 0.87
    for k in range(13):
        angle = 2 * np.pi * k / 13
        points_list.append(np.array([r4 * np.cos(angle), r4 * np.sin(angle)]))
        
    pts = np.array(points_list)
    
    # Ensure all points are inside unit circle
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > 0.99
    if np.any(mask):
        pts[mask] = pts[mask] * (0.99 / norms[mask][:, None])
        
    return pts


def _hexagonal_initial(n=50):
    """Create initial hexagonal lattice points inside unit disk."""
    best_points = None
    best_score = -1
    
    for base_spacing in [0.22, 0.23, 0.24, 0.25, 0.26]:
        spacing = base_spacing
        for _ in range(15):
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= n:
                dists = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists)[:n]
                pts = points[idx]
                score = _compute_score(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
            spacing *= 0.97
    
    if best_points is None:
        rng = np.random.default_rng(42)
        best_points = np.zeros((n, 2))
        for i in range(n):
            while True:
                angle = rng.uniform(0, 2*np.pi)
                r = np.sqrt(rng.uniform(0, 1))
                pt = np.array([r*np.cos(angle), r*np.sin(angle)])
                if np.all(np.linalg.norm(pt - best_points[:i], axis=1) > 0.15):
                    best_points[i] = pt
                    break
    return best_points


def _compute_score(points):
    """Compute the packing score: min(min_pair_dist/2, min(1-|c_i|))."""
    n = len(points)
    if n == 0:
        return 0.0
    
    # Min distance from any center to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(norms)
    
    if min_boundary <= 0:
        return 0.0
    
    # Min pairwise distance using cdist for speed
    D = cdist(points, points)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    
    return min(min_pair / 2.0, min_boundary)


def _smooth_objective_with_t(x, beta=100.0):
    """Smooth surrogate with explicit t variable.
    x = [t, x1, y1, x2, y2, ..., xN, yN]
    Minimize: -t + (1/beta) * log(sum(exp(-beta * (d_ij/2 - t)))) + (1/beta) * log(sum(exp(-beta * (1-|c_i| - t))))
    This is the log-sum-exp of the constraint violations, which is 0 when all constraints satisfied with margin."""
    t = x[0]
    points = x[1:].reshape(-1, 2)
    n = len(points)
    
    # Pairwise distances
    D = cdist(points, points)
    np.fill_diagonal(D, np.inf)
    pair_dists = D[D > 0] / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - norms
    
    # Constraint violations: we want d_ij/2 >= t and 1-|c_i| >= t
    # So violation = t - d_ij/2 and t - (1-|c_i|)
    pair_violations = t - pair_dists
    boundary_violations = t - boundary_dists
    
    # Log-sum-exp of violations (should be <= 0 for feasible)
    all_violations = np.concatenate([pair_violations, boundary_violations])
    
    # Soft max of violations
    max_v = np.max(all_violations)
    if max_v > 0:
        log_sum_exp = max_v + np.log(np.sum(np.exp(all_violations - max_v))) / beta
    else:
        log_sum_exp = max_v + np.log(np.sum(np.exp(all_violations - max_v))) / beta
    
    # Objective: -t + penalty for positive violations
    # If all violations <= 0, penalty is small
    return -t + log_sum_exp


def _polish(points, time_budget=20.0):
    """High-precision polish using scipy minimize with smooth objective."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # Use t-variable formulation for better convergence
    t0 = best_score
    x0 = np.concatenate([[t0], points.flatten()])
    
    # Start with lower beta for stability, then increase for precision
    betas = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
    
    for beta in betas:
        if time.time() > deadline - 2:
            break
        
        try:
            result = minimize(
                _smooth_objective_with_t,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            
            # Check if the result is valid and improved
            new_t = result.x[0]
            new_points = result.x[1:].reshape(-1, 2)
            
            # Check validity: points must be inside unit circle
            if np.any(np.linalg.norm(new_points, axis=1) > 1.0):
                continue
            
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
                x0 = np.concatenate([[new_score], new_points.flatten()])
        except Exception:
            continue
    
    return best_points, best_score


def _simulated_annealing(points, time_budget=100.0):
    """Run simulated annealing to improve the packing."""
    n = len(points)
    points = points.copy()
    best_points = points.copy()
    best_score = _compute_score(points)
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Initial temperature
    T = 0.005
    T_min = 1e-12
    cooling_rate = 0.9998
    step_scale = 0.03
    
    rng = np.random.default_rng(12345)
    
    while time.time() < deadline and T > T_min:
        # Propose a random move
        idx = rng.integers(0, n)
        move_type = rng.random()
        
        if move_type < 0.6:
            # Random displacement
            delta = (rng.random(2) - 0.5) * 2 * step_scale * T * 10
            new_points = points.copy()
            new_points[idx] = points[idx] + delta
        elif move_type < 0.8:
            # Swap two points
            jdx = rng.integers(0, n)
            if jdx == idx:
                continue
            new_points = points.copy()
            new_points[idx], new_points[jdx] = points[jdx].copy(), points[idx].copy()
        elif move_type < 0.9:
            # Move along radial direction
            r = np.linalg.norm(points[idx])
            if r > 0.01:
                dir_vec = points[idx] / r
                delta_r = (rng.random() - 0.5) * 0.05
                new_points = points.copy()
                new_points[idx] = points[idx] + dir_vec * delta_r
            else:
                continue
        else:
            # Reflect a point through origin (with noise)
            new_points = points.copy()
            new_points[idx] = -points[idx] + (rng.random(2) - 0.5) * 0.01
        
        # Check validity
        if np.any(np.linalg.norm(new_points, axis=1) > 1.0):
            continue
        
        new_score = _compute_score(new_points)
        
        # Accept or reject
        if new_score >= best_score or rng.random() < np.exp((new_score - best_score) / T):
            points = new_points
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        
        T *= cooling_rate
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Generate initial guess using ring-based structure
    points = _ring_initial(N)
    current_score = _compute_score(points)
    
    # Run simulated annealing for a shorter time, as initial guess is good
    points, current_score = _simulated_annealing(points, time_budget=30.0)
    
    # If time remains, try a few more restarts with different seeds
    elapsed = time.time() - start_time
    if elapsed < deadline - 40:
        for seed in [7, 13, 42]:
            if time.time() > deadline - 45:
                break
            rng = np.random.default_rng(seed)
            # Perturb the best solution
            perturbed = points + (rng.random((N, 2)) - 0.5) * 0.015
            # Ensure inside unit circle
            norms = np.linalg.norm(perturbed, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                perturbed[mask] /= norms[mask][:, None] * 1.005
            p, s = _simulated_annealing(perturbed, time_budget=min(10, deadline - 50 - (time.time() - start_time)))
            if s > current_score:
                points = p
                current_score = s
    
    # Final high-precision polish pass - allocate most remaining time here
    remaining = deadline - 2 - (time.time() - start_time)
    if remaining > 5:
        points, current_score = _polish(points, time_budget=remaining)
    
    return points
# EVOLVE-BLOCK-END
