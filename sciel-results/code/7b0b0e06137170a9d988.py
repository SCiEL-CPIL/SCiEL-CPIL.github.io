"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 145  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _softplus(x, beta=50.0):
    """Smooth approximation of max(x, 0)."""
    x = np.asarray(x)
    return np.where(x > 1.0 / beta, x, np.logaddexp(0, beta * x) / beta)


def _compute_radii(centers):
    """
    Compute radii for given centers.
    Radius of circle i is the minimum of:
    1. Distance to the four boundaries of the unit square.
    2. Half the distance to the nearest other center.
    """
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Distance to boundaries
    dist_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Nearest neighbor distance
    radii = dist_bound.copy()
    for i in range(N):
        min_dist = np.inf
        for j in range(N):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < min_dist:
                min_dist = d
        radii[i] = min(dist_bound[i], min_dist / 2.0)
    
    radii = np.maximum(radii, 0.0)
    return radii


def _objective(params):
    """
    Objective function: Negative sum of radii.
    params: flattened array of shape (2*N,) containing x, y coordinates.
    """
    centers = params.reshape(N, 2)
    radii = _compute_radii(centers)
    return -np.sum(radii)


def _objective_grad(params):
    """
    Analytic gradient of the objective function.
    Determines active constraint for each circle and computes gradient accordingly.
    """
    centers = params.reshape(N, 2)
    x = centers[:, 0]
    y = centers[:, 1]
    
    radii = _compute_radii(centers)
    grad = np.zeros_like(params)
    
    for i in range(N):
        r_i = radii[i]
        
        d_left = x[i]
        d_right = 1.0 - x[i]
        d_bottom = y[i]
        d_top = 1.0 - y[i]
        
        min_dist = np.inf
        nearest_j = -1
        for j in range(N):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < min_dist:
                min_dist = d
                nearest_j = j
        
        d_nn = min_dist / 2.0
        
        # Determine active constraint
        if d_left <= d_right and d_left <= d_bottom and d_left <= d_top and d_left <= d_nn:
            grad[2*i] = -1.0
            grad[2*i+1] = 0.0
        elif d_right <= d_left and d_right <= d_bottom and d_right <= d_top and d_right <= d_nn:
            grad[2*i] = 1.0
            grad[2*i+1] = 0.0
        elif d_bottom <= d_left and d_bottom <= d_right and d_bottom <= d_top and d_bottom <= d_nn:
            grad[2*i] = 0.0
            grad[2*i+1] = -1.0
        elif d_top <= d_left and d_top <= d_right and d_top <= d_bottom and d_top <= d_nn:
            grad[2*i] = 0.0
            grad[2*i+1] = 1.0
        elif d_nn < d_left and d_nn < d_right and d_nn < d_bottom and d_nn < d_top:
            dx = x[i] - x[nearest_j]
            dy = y[i] - y[nearest_j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist > 1e-12:
                factor = -0.5 / dist
                grad[2*i] = factor * dx
                grad[2*i+1] = factor * dy
            else:
                grad[2*i] = -0.1
                grad[2*i+1] = -0.1
        else:
            grad[2*i] = -1.0 if d_left < d_right else 1.0
            grad[2*i+1] = -1.0 if d_bottom < d_top else 1.0
    
    return grad


def _smooth_objective(params, beta=50.0):
    """
    Smooth surrogate objective using soft-min instead of hard min.
    This makes the objective differentiable and allows gradient-based methods
    to converge more reliably.
    """
    centers = params.reshape(N, 2)
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Distance to boundaries
    d_left = x
    d_right = 1.0 - x
    d_bottom = y
    d_top = 1.0 - y
    
    # Nearest neighbor distances
    nn_dists = np.zeros(N)
    for i in range(N):
        min_dist = np.inf
        for j in range(N):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < min_dist:
                min_dist = d
        nn_dists[i] = min_dist / 2.0
    
    # Soft-min over all constraints for each circle
    # Constraints: d_left, d_right, d_bottom, d_top, nn_dists
    # We want to maximize the minimum of these, which is equivalent to
    # minimizing the soft-max of their negatives
    # soft_min(a, b) = -soft_max(-a, -b) = -log(exp(-beta*a) + exp(-beta*b)) / beta
    # For multiple values, soft_min = -log(sum(exp(-beta*v_i))) / beta
    
    # Stack all constraints
    all_constraints = np.column_stack([d_left, d_right, d_bottom, d_top, nn_dists])
    
    # Soft-min: -log(sum(exp(-beta * v))) / beta
    # To avoid overflow, use log-sum-exp trick
    max_val = np.max(all_constraints, axis=1)
    exp_terms = np.exp(-beta * (all_constraints - max_val[:, np.newaxis]))
    log_sum = np.log(np.sum(exp_terms, axis=1))
    soft_mins = max_val - log_sum / beta
    
    # Objective is negative sum of soft-mins
    return -np.sum(soft_mins)


def _smooth_objective_grad(params, beta=50.0):
    """
    Analytic gradient of the smooth surrogate objective.
    """
    centers = params.reshape(N, 2)
    x = centers[:, 0]
    y = centers[:, 1]
    
    d_left = x
    d_right = 1.0 - x
    d_bottom = y
    d_top = 1.0 - y
    
    nn_dists = np.zeros(N)
    nn_indices = np.zeros(N, dtype=int)
    for i in range(N):
        min_dist = np.inf
        nearest_j = -1
        for j in range(N):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < min_dist:
                min_dist = d
                nearest_j = j
        nn_dists[i] = min_dist / 2.0
        nn_indices[i] = nearest_j
    
    all_constraints = np.column_stack([d_left, d_right, d_bottom, d_top, nn_dists])
    
    # Soft-min: -log(sum(exp(-beta * v))) / beta
    # Let f(v) = -log(sum(exp(-beta * v))) / beta
    # df/dv_i = exp(-beta * v_i) / sum(exp(-beta * v_j))
    # This is the softmax weight for constraint i
    
    max_val = np.max(all_constraints, axis=1)
    exp_terms = np.exp(-beta * (all_constraints - max_val[:, np.newaxis]))
    sum_exp = np.sum(exp_terms, axis=1)
    weights = exp_terms / sum_exp[:, np.newaxis]
    
    grad = np.zeros_like(params)
    
    for i in range(N):
        # Weights for each constraint
        w_left = weights[i, 0]
        w_right = weights[i, 1]
        w_bottom = weights[i, 2]
        w_top = weights[i, 3]
        w_nn = weights[i, 4]
        
        # Gradient of soft_min w.r.t. x_i
        # d(soft_min)/dx_i = w_left * 1 + w_right * (-1) + w_nn * d(nn_dist)/dx_i
        # d(nn_dist)/dx_i = 0.5 * (x_i - x_j) / dist(i,j)
        
        dx = x[i] - x[nn_indices[i]]
        dy = y[i] - y[nn_indices[i]]
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist > 1e-12:
            d_nn_dx = 0.5 * dx / dist
            d_nn_dy = 0.5 * dy / dist
        else:
            d_nn_dx = 0.0
            d_nn_dy = 0.0
        
        d_soft_min_dx = w_left - w_right + w_nn * d_nn_dx
        d_soft_min_dy = w_bottom - w_top + w_nn * d_nn_dy
        
        # Objective is -sum(soft_min), so gradient is -d(soft_min)/dx
        grad[2*i] = -d_soft_min_dx
        grad[2*i+1] = -d_soft_min_dy
    
    return grad


def _physics_relax(centers, n_steps=200, dt=0.01):
    """
    Simple physics-based relaxation to spread out circles.
    Uses repulsive forces between circles and attractive forces toward center.
    """
    centers = centers.copy()
    
    for step in range(n_steps):
        if _remaining() < 5:
            break
        
        forces = np.zeros_like(centers)
        
        # Repulsive forces between all pairs
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-6:
                    # Random perturbation for overlapping points
                    forces[i, 0] += 0.01 * (np.random.random() - 0.5)
                    forces[i, 1] += 0.01 * (np.random.random() - 0.5)
                    continue
                # Repulsion proportional to 1/dist^2
                repulsion = 0.01 / (dist * dist)
                forces[i, 0] += repulsion * dx / dist
                forces[i, 1] += repulsion * dy / dist
        
        # Attractive force toward center
        for i in range(N):
            forces[i, 0] += 0.005 * (0.5 - centers[i, 0])
            forces[i, 1] += 0.005 * (0.5 - centers[i, 1])
        
        # Boundary repulsion
        for i in range(N):
            if centers[i, 0] < 0.1:
                forces[i, 0] += 0.05 * (0.1 - centers[i, 0])
            if centers[i, 0] > 0.9:
                forces[i, 0] -= 0.05 * (centers[i, 0] - 0.9)
            if centers[i, 1] < 0.1:
                forces[i, 1] += 0.05 * (0.1 - centers[i, 1])
            if centers[i, 1] > 0.9:
                forces[i, 1] -= 0.05 * (centers[i, 1] - 0.9)
        
        # Update positions
        centers += dt * forces
        # Clip to [0.05, 0.95] to stay away from boundaries
        centers = np.clip(centers, 0.05, 0.95)
    
    return centers


def _polish_analytic(centers, time_budget=10.0, use_smooth=True):
    """
    Refine centers using L-BFGS-B with analytic gradient.
    Optionally uses smooth surrogate objective for better convergence.
    """
    start_time = time.time()
    params = centers.flatten().copy()
    best_params = params.copy()
    
    if use_smooth:
        best_val = _smooth_objective(best_params)
        obj_func = _smooth_objective
        jac_func = _smooth_objective_grad
    else:
        best_val = _objective(best_params)
        obj_func = _objective
        jac_func = _objective_grad
    
    bounds = [(0.01, 0.99) for _ in range(2 * N)]
    
    # First pass with smooth objective
    try:
        result = minimize(
            obj_func, 
            best_params, 
            jac=jac_func, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        # Evaluate with original objective
        val = _objective(result.x)
        if val < _objective(best_params):
            best_params = result.x
    except Exception:
        pass
    
    # Second pass with original objective if time permits
    if time.time() - start_time < time_budget * 0.6:
        try:
            result = minimize(
                _objective, 
                best_params, 
                jac=_objective_grad, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            val = _objective(result.x)
            if val < _objective(best_params):
                best_params = result.x
        except Exception:
            pass
    
    # Third pass with tighter tolerance if time permits
    if time.time() - start_time < time_budget:
        try:
            result = minimize(
                _objective, 
                best_params, 
                jac=_objective_grad, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            val = _objective(result.x)
            if val < _objective(best_params):
                best_params = result.x
        except Exception:
            pass
    
    return best_params.reshape(N, 2)


def _enforce_validity(centers):
    """
    Ensure circles are strictly valid: inside unit square and non-overlapping.
    If any constraint is violated, adjust radii to satisfy constraints.
    """
    x = centers[:, 0].copy()
    y = centers[:, 1].copy()
    
    # Compute initial radii
    radii = _compute_radii(centers)
    
    # Verify and fix any violations
    for i in range(N):
        # Check boundary constraints
        max_r_boundary = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        if radii[i] > max_r_boundary:
            radii[i] = max_r_boundary * 0.999
    
    # Check pairwise constraints
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if radii[i] + radii[j] > dist:
                # Scale down both radii proportionally
                scale = dist / (radii[i] + radii[j]) * 0.999
                radii[i] *= scale
                radii[j] *= scale
    
    # Final check
    for i in range(N):
        radii[i] = min(radii[i], x[i], 1.0 - x[i], y[i], 1.0 - y[i])
    
    return centers, radii


def _initial_hex():
    """Hexagonal packing initial config for centers."""
    cols = 5
    rows = 6
    positions = []
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5 / cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1 or y > 1:
                continue
            positions.append([x, y])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5])
    return np.array(positions[:N])


def _initial_grid():
    """Grid initial config for centers."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5])
    return np.array(positions[:N])


def _initial_random(seed):
    """Random initial config for centers."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.15, 0.85, size=(N, 2))
    return positions


def _initial_voronoi(centers0):
    """
    Generate new initial config by perturbing Voronoi cell centroids.
    """
    from scipy.spatial import Voronoi
    
    try:
        vor = Voronoi(centers0)
        new_centers = np.zeros_like(centers0)
        for i in range(N):
            # Find Voronoi region vertices
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if len(region) > 0 and not -1 in region:
                vertices = vor.vertices[region]
                centroid = np.mean(vertices, axis=0)
                # Clip to valid range
                centroid = np.clip(centroid, 0.1, 0.9)
                new_centers[i] = centroid
            else:
                new_centers[i] = centers0[i]
        return new_centers
    except Exception:
        return centers0


def construct():
    """
    Pack 26 circles in the unit square, maximizing the sum of radii.
    Uses smooth surrogate objective for better gradient convergence,
    followed by high-precision polishing with original objective.
    """
    deadline = _start_time + _TIME_LIMIT
    
    inits = [
        _initial_hex(),
        _initial_grid(),
        _initial_random(42),
        _initial_random(123),
        _initial_random(999),
        _initial_random(777),
        _initial_random(2024),
    ]
    
    best_overall = None
    best_radii_overall = None
    best_score_overall = -np.inf
    
    for i, centers0 in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 8:
            break
        
        # Phase 0: Physics relaxation for better initial config
        if remaining > 15:
            centers0 = _physics_relax(centers0, n_steps=100, dt=0.005)
        
        # Allocate time for polishing
        alloc = min(remaining - 3, max(8, 15 - i * 2))
        
        # Phase 1: Smooth gradient polish
        centers_polished = _polish_analytic(centers0, time_budget=alloc, use_smooth=True)
        
        # Phase 2: Enforce validity and compute final radii
        centers_valid, radii = _enforce_validity(centers_polished)
        score = np.sum(radii)
        
        if score > best_score_overall:
            best_score_overall = score
            best_overall = centers_valid.copy()
            best_radii_overall = radii.copy()
    
    # Final result construction
    if best_overall is not None:
        result = np.column_stack([best_overall, best_radii_overall])
        return result
    else:
        # Fallback: simple grid
        result = np.zeros((N, 3))
        side = 5
        for idx in range(N):
            i = idx % side
            j = idx // side
            result[idx, 0] = (i + 0.5) / side
            result[idx, 1] = (j + 0.5) / side
            result[idx, 2] = 0.5 / side * 0.8
        return result

# EVOLVE-BLOCK-END
