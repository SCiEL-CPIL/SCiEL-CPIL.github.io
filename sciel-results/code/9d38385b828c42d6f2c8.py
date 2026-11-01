"""Circle packing: 50 equal circles in unit disk.
Strategy: Force-directed relaxation with cooling + Nelder-Mead polish.
Uses a soft repulsive potential to handle overlaps and boundary collisions,
allowing the system to settle into a local minimum of the "energy" landscape,
which corresponds to a high packing density.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the packing score: min of pairwise distances/2 and boundary distances."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points are inside the unit circle
    dists_to_origin = np.linalg.norm(points, axis=1)
    if np.any(dists_to_origin >= 1.0):
        return 0.0
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    # Minimum distance to boundary
    min_boundary = 1.0 - np.max(dists_to_origin)
    return min(min_pairwise, min_boundary)

def _force_energy(points, k_rep=1.0, k_bound=1.0):
    """
    Compute a smooth energy function based on forces.
    Lower energy = better packing (larger minimum distance).
    
    We use a potential that penalizes overlaps and boundary violations.
    To maximize the minimum distance r, we can think of this as minimizing
    a function that is 0 when all constraints are satisfied with margin r,
    and positive otherwise.
    
    A common approach is to sum up penalties for violations of a target radius.
    However, since we don't know the optimal r, we can use a force-directed
    approach where we push points apart until they settle.
    
    Alternative: Minimize the sum of squared violations of a hypothetical radius r.
    But we want to maximize r.
    
    Let's use a different energy:
    E = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, r - (1 - |p_i|))^2
    This requires knowing r.
    
    Instead, let's use a soft-min based energy which is differentiable:
    E = - softmin( [d_ij/2 for i<j], [1 - |p_i| for i], t )
    This is what the current program does.
    
    The force-directed approach usually implies:
    F_i = sum_{j != i} (p_i - p_j)/|p_i - p_j|  (repulsion)
    F_i += -p_i/|p_i| (boundary attraction/repulsion)
    
    Let's implement a simple gradient descent on the soft-min objective but
    using a different initialization and a "cooling" of the temperature t,
    AND adding a small random jitter to escape local minima, which is distinct
    from the pure L-BFGS approach in the seed.
    
    Actually, the directive asks for a "force-directed relaxation".
    Let's implement a specific force model:
    Repulsive force between circles: F_rep = k * (1/d_ij - 1/(2r_target)) if d_ij < 2r_target
    But we don't know r_target.
    
    Let's stick to the soft-min gradient but change the optimization loop to be
    more robust with multiple restarts and a specific cooling schedule, and
    crucially, use a **randomized restart** strategy with **perturbation**
    after each phase, which is a distinct search strategy (Simulated Annealing-like).
    
    Wait, the current program already does annealing of t.
    
    Let's try a different energy function: **Sum of inverse distances**.
    Minimize S = sum_{i<j} 1/d_ij + sum_i 1/(1 - |p_i|)
    This function is smooth everywhere inside the domain and tends to infinity
    as points coincide or hit the boundary. Minimizing this sum tends to
    maximize the minimum distance, although it is not exactly equivalent.
    However, it is a very standard force-directed model.
    
    Let's try minimizing:
    E(x) = sum_{i<j} (1/d_ij)^2 + sum_i (1/(1-|p_i|))^2
    
    Gradient:
    d/dp_i (1/d_ij)^2 = -2 (p_i - p_j) / d_ij^4
    d/dp_i (1/(1-|p_i|))^2 = -2 (p_i/|p_i|) / (1-|p_i|)^3
    
    This is smooth and fast to compute.
    """
    pts = points.reshape(-1, 2)
    n = len(pts)
    
    # Pairwise
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq + 1e-12) # avoid div by 0
    
    # Energy term 1: Sum of inverse squared distances
    # E_pair = sum 1/d_ij^2
    # We compute the gradient analytically.
    
    # Boundary
    r = np.linalg.norm(pts, axis=1)
    r_safe = np.maximum(r, 1e-12)
    
    # To make it a proper optimization, we minimize the sum.
    # This is a differentiable function.
    
    # We will use this as the objective for L-BFGS-B.
    
    # Objective value (for logging/checking)
    e_pair = np.sum(1.0 / (dists_sq + 1e-12))
    e_bound = np.sum(1.0 / ((1.0 - r_safe)**2 + 1e-12))
    obj_val = e_pair + e_bound
    
    # Gradient
    grad = np.zeros_like(pts)
    
    # Pairwise gradient contribution to point i:
    # d E_pair / dp_i = sum_{j!=i} -2 * (p_i - p_j) / d_ij^4
    # Note: d_ij^4 = (dists_sq)^2
    # We can vectorize this.
    # Weight matrix W[i,j] = -2 / dists_sq[i,j]^2 for i!=j
    W = np.zeros_like(dists_sq)
    np.fill_diagonal(W, 0.0)
    W = -2.0 / (dists_sq**2 + 1e-24)
    
    # grad_pair[i] = sum_j W[i,j] * (p_i - p_j)
    # This is equivalent to (W @ diff).sum(axis=1)? No.
    # diff[i,j] = p_i - p_j
    # grad_pair[i] = sum_j W[i,j] * diff[i,j]
    grad_pair = np.sum(W[:, :, np.newaxis] * diff, axis=1)
    
    # Boundary gradient
    # d E_bound / dp_i = -2 * (p_i / r_i) / (1 - r_i)^3
    # Let u_i = p_i / r_i
    # grad_bound[i] = -2 * u_i / (1 - r_i)^3
    denom = (1.0 - r_safe)**3 + 1e-12
    unit_vec = pts / r_safe[:, np.newaxis]
    grad_bound = -2.0 * unit_vec / denom[:, np.newaxis]
    
    total_grad = (grad_pair + grad_bound).flatten()
    
    return obj_val, total_grad

def _force_optimize(points, max_iter=500, deadline=None):
    """
    Optimize using L-BFGS-B on the sum-of-inverse-distances objective.
    """
    x0 = points.flatten()
    
    def obj_and_grad(x):
        return _force_energy(x.reshape(-1, 2))
    
    try:
        res = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-6, 'maxfun': 1000}
        )
        new_pts = res.x.reshape(-1, 2)
        # Check validity
        if np.all(np.linalg.norm(new_pts, axis=1) < 1.0):
            return new_pts
    except Exception:
        pass
    return points

def _nelder_mead_polish(points, max_iter=1000, deadline=None):
    """
    Polish using Nelder-Mead on the raw min-score objective.
    """
    def objective(x):
        pts = x.reshape(-1, 2)
        # Penalty for going outside
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return 10.0 + np.sum(dists[dists > 1.0] - 1.0)
        return -_compute_score(pts)
    
    try:
        res = minimize(
            objective,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        new_points = res.x.reshape(-1, 2)
        # Ensure validity
        if np.all(np.linalg.norm(new_points, axis=1) < 1.0):
            return new_points
    except Exception:
        pass
    return points

def _hex_init(n, seed=0):
    """Hexagonal lattice init."""
    rng = np.random.RandomState(seed)
    best_spacing = None
    best_points = None
    
    # Try different spacings
    for spacing in np.linspace(0.12, 0.25, 15):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        
        if len(points) >= n:
            order = np.argsort(radius[mask], kind="stable")
            selected = points[order[:n]]
            score = _compute_score(selected)
            if best_points is None or score > _compute_score(best_points):
                best_points = selected.copy()
                best_spacing = spacing
    
    if best_points is not None:
        return best_points
    
    return _random_init(n, seed)

def _random_init(n, seed):
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(rng.uniform(0, 1, n))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = -1.0
    
    # Strategy:
    # 1. Generate diverse initializations (Hex + Random)
    # 2. For each, run Force-Directed Optimization (Sum of Inverse Distances)
    #    This is different from the Soft-Min Gradient Ascent.
    # 3. Polish with Nelder-Mead.
    
    inits = []
    # 3 Hex inits
    for s in range(3):
        inits.append(_hex_init(N, seed=s))
    # 4 Random inits
    for s in range(4):
        inits.append(_random_init(N, seed=s))
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
            
        current_pts = init_pts.copy()
        
        # Step 1: Force Optimization
        # This minimizes sum(1/d^2), which spreads points out.
        current_pts = _force_optimize(current_pts, max_iter=300, deadline=deadline)
        
        # Step 2: Nelder-Mead Polish
        if time.time() < deadline - 5.0:
            polished_pts = _nelder_mead_polish(current_pts, max_iter=800, deadline=deadline)
            score = _compute_score(polished_pts)
            if score > best_score:
                best_score = score
                best_points = polished_pts.copy()
        
        # Also check the unpolished point just in case
        score_unpolished = _compute_score(current_pts)
        if score_unpolished > best_score:
            best_score = score_unpolished
            best_points = current_pts.copy()

    # Fallback
    if best_points is None:
        best_points = _hex_init(N, seed=0)
        
    return best_points

# EVOLVE-BLOCK-END
