"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

N = 100

# EVOLVE-BLOCK-START

def _pairwise_indices(n):
    """Return arrays of i, j for all i<j pairs."""
    i, j = np.triu_indices(n, k=1)
    return i, j

def _score(c):
    """Compute the radius score for a valid configuration."""
    # Clamp to be safe (shouldn't be needed if constraints work)
    c = np.clip(c, -0.5, 0.5)
    # Pairwise distances
    i, j = _pairwise_indices(len(c))
    d_ij = np.linalg.norm(c[i] - c[j], axis=1)
    min_dist = np.min(d_ij)
    # Boundary distances
    dist_bound = np.min(0.5 - np.abs(c), axis=1)
    min_bound = np.min(dist_bound)
    return 0.5 * min(min_dist, min_bound)

def _objective_and_grad(theta, i, j):
    """
    Objective: maximize r where r <= |c_i - c_j|/2 for all pairs, r <= 0.5 - |c_ik| for all boundaries.
    We minimize -r.
    theta = [x_0..x_{99}, y_0..y_{99}, z_0..z_{99}, r]
    """
    n = len(i) + 1  # approximate, actually N
    N = 100
    coords = theta[:3*N].reshape(N, 3)
    r = theta[3*N]
    
    # Pairwise constraints: |c_i - c_j| >= 2r
    # We want to maximize r subject to constraints.
    # Minimize -r.
    # The "active" constraints are those where |c_i - c_j| is close to 2r or boundary is close to r.
    
    # For gradient, we use a soft approach or exact subgradient.
    # Let's use the exact subgradient of the min function.
    # f = min( min_ij |c_i-c_j|/2, min_i min_k (0.5 - |c_ik|) )
    # We want to maximize f.
    
    d_ij = np.linalg.norm(coords[i] - coords[j], axis=1)
    pair_terms = d_ij / 2.0
    min_pair = np.min(pair_terms)
    
    # Boundary terms: 0.5 - |c_ik|
    # For each point and each axis, term is 0.5 - |c_ik|
    abs_coords = np.abs(coords)
    # shape (N, 3)
    bound_terms = 0.5 - abs_coords
    min_bound = np.min(bound_terms)
    
    f = min(min_pair, min_bound)
    
    # Gradient of f with respect to coords and r is not direct because r is a variable.
    # Actually, the standard formulation is:
    # Maximize r
    # Subject to:
    #   |c_i - c_j| >= 2r  for all i<j
    #   0.5 - |c_ik| >= r  for all i,k
    #   c_ik in [-0.5, 0.5]
    
    # This is a non-linear optimization problem.
    # We can use scipy.optimize.minimize with method 'L-BFGS-B' but it requires bounds and no constraints,
    # or 'SLSQP' with constraints.
    # Given the number of constraints (4950 pairs + 300 boundaries), SLSQP might be slow.
    
    # Alternative: Use a penalty method or trust region.
    # Or, reformulate as minimizing the violation.
    
    # Let's try a different approach:
    # Start with a good initial guess (e.g., from the seed grid or random).
    # Use scipy.optimize.minimize on the vector of centers, minimizing a smooth approximation of the "badness".
    
    # Badness = sum of max(0, 2r - |c_i-c_j|)^2 + sum of max(0, r - (0.5-|c_ik|))^2
    # But r is also a variable.
    
    # Let's stick to the theta = [coords, r] formulation and use a solver that handles constraints.
    # However, implementing the gradient for the min is tricky.
    
    # Let's use a simpler, robust method:
    # 1. Generate a random or grid-based initial configuration.
    # 2. Use a local optimization loop that pushes spheres apart.
    # 3. Use Simulated Annealing or L-BFGS-B on a smooth surrogate.
    
    # For now, let's return the negative of the current radius to minimize.
    # The gradient will be computed numerically or via a smooth surrogate.
    
    # Smooth surrogate for min:
    # soft_min(x) = -logsumexp(-x) / beta
    # This is differentiable.
    
    beta = 50.0
    pair_terms_soft = -np.log(np.mean(np.exp(-pair_terms * beta))) / beta
    bound_terms_flat = bound_terms.flatten()
    bound_terms_soft = -np.log(np.mean(np.exp(-bound_terms_flat * beta))) / beta
    
    obj = -min(pair_terms_soft, bound_terms_soft)
    
    # Gradient
    # d/dx soft_min(x) = - (sum exp(-beta*x_i) * (-beta) * dx_i) / (sum exp(-beta*x_i)) * 1/beta
    # = (sum exp(-beta*x_i) * dx_i) / (sum exp(-beta*x_i))
    
    # For pair terms:
    # x_ij = |c_i - c_j| / 2
    # d x_ij / d c_i = (c_i - c_j) / (2 * |c_i - c_j|)
    # d x_ij / d c_j = (c_j - c_i) / (2 * |c_i - c_j|)
    
    # This is getting complex. Let's use a finite difference gradient for the smooth objective.
    # Or, use a simpler objective: minimize the sum of squared violations for a fixed r, and binary search r.
    
    # Given time constraints, let's implement a robust local optimizer:
    # 1. Start with a configuration.
    # 2. Iteratively find the closest pair and move them apart.
    # 3. Or use L-BFGS-B on a smooth objective.
    
    # Let's try L-BFGS-B on the smooth surrogate.
    
    # We need to return the objective and gradient.
    # Since computing the exact gradient of the soft-min is involved, let's use a finite difference approximation
    # or a simpler objective.
    
    # Simpler objective: minimize the sum of (2r - |c_i-c_j|)^2 for all pairs where 2r > |c_i-c_j|
    # and (r - (0.5-|c_ik|))^2 for boundaries where r > 0.5-|c_ik|.
    # This is differentiable almost everywhere.
    
    # Let's use this penalty objective.
    
    penalty = 0.0
    grad_coords = np.zeros_like(coords)
    
    # Pairwise
    diff = coords[i] - coords[j]
    dist = np.linalg.norm(diff, axis=1)
    # Avoid division by zero
    dist_safe = np.where(dist < 1e-10, 1e-10, dist)
    
    # Violation: 2r - dist
    viol = 2*r - dist
    mask = viol > 0
    if np.any(mask):
        # Gradient of (2r - dist)^2 w.r.t. c_i
        # d/dc_i (2r - dist)^2 = 2(2r - dist) * d(2r - dist)/dc_i
        # d(dist)/dc_i = diff / dist
        # d(2r - dist)/dc_i = -diff / dist
        # So term is 2 * viol * (-diff / dist)
        # Sum over all j for each i
        
        # We need to accumulate gradients
        # For each pair (i, j):
        # grad_c_i += 2 * viol * (-diff / dist)
        # grad_c_j += 2 * viol * (diff / dist)
        
        # Using numpy broadcasting is tricky here, so we'll use a loop or advanced indexing.
        # Given N=100, 4950 pairs, a loop is fast enough.
        
        for k in range(len(i)):
            if mask[k]:
                vi = viol[k]
                di = diff[k] / dist_safe[k]
                grad_coords[i[k]] -= 2 * vi * di
                grad_coords[j[k]] += 2 * vi * di
    
    # Boundary
    # Violation: r - (0.5 - |c_ik|)
    # For each c_ik:
    abs_c = np.abs(coords)
    viol_b = r - (0.5 - abs_c)
    mask_b = viol_b > 0
    if np.any(mask_b):
        # d/dc_ik (r - (0.5 - |c_ik|))^2 = 2 * viol_b * d(0.5 - |c_ik|)/dc_ik
        # d(0.5 - |c_ik|)/dc_ik = -sign(c_ik)
        # So term is 2 * viol_b * sign(c_ik)
        # Note: sign(0) is 0, which is fine.
        grad_coords += 2 * viol_b * np.sign(coords)
    
    # Gradient w.r.t. r
    # d/dr sum (2r - dist)^2 = sum 2(2r - dist) * 2 = 4 * sum viol (for active pairs)
    # d/dr sum (r - (0.5 - |c_ik|))^2 = sum 2 * viol_b * 1 = 2 * sum viol_b (for active boundaries)
    
    grad_r = 0.0
    if np.any(mask):
        grad_r += 4 * np.sum(viol[mask])
    if np.any(mask_b):
        grad_r += 2 * np.sum(viol_b[mask_b])
    
    # Objective value (we are minimizing this)
    obj_val = np.sum(viol[mask]**2) + np.sum(viol_b[mask_b]**2)
    
    # We want to maximize r, so we minimize -r + penalty?
    # No, the standard way is to minimize the penalty for a given r, and increase r.
    # Or, minimize -r + lambda * penalty.
    # Let's minimize -r + 100 * penalty.
    
    total_obj = -r + 100.0 * obj_val
    
    grad = np.concatenate([grad_coords, [grad_r]])
    
    # Wait, the gradient of -r is -1.
    grad[-1] += -1.0
    
    return total_obj, grad

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    # Initial guess: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c0 = grid[:N].copy()
    
    # Add small random perturbation
    np.random.seed(42)
    c0 += np.random.normal(0, 0.01, c0.shape)
    c0 = np.clip(c0, -0.49, 0.49)
    
    # Initial r
    i, j = _pairwise_indices(N)
    d_ij = np.linalg.norm(c0[i] - c0[j], axis=1)
    min_dist = np.min(d_ij)
    min_bound = np.min(0.5 - np.abs(c0), axis=1)
    r0 = 0.5 * min(min_dist, min_bound)
    
    theta0 = np.concatenate([c0.flatten(), [r0]])
    
    best_theta = theta0.copy()
    best_score = _score(c0)
    
    # Optimize using L-BFGS-B
    # We need bounds for the coordinates and r.
    bounds = [(-0.5, 0.5)] * (3 * N) + [(0.0, 0.5)]
    
    try:
        res = minimize(
            lambda t: _objective_and_grad(t, i, j)[0],
            theta0,
            jac=lambda t: _objective_and_grad(t, i, j)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
        )
        
        if res.success or res.fun < -best_score + 100 * 1e-6:
            best_theta = res.x
            c_best = best_theta[:3*N].reshape(N, 3)
            r_best = best_theta[3*N]
            s = _score(c_best)
            if s > best_score:
                best_score = s
    except Exception as e:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_theta[:3*N].reshape(N, 3)
    
    # Try a few random restarts
    for restart in range(5):
        if time.time() > deadline - 20:
            break
        
        c_rand = np.random.uniform(-0.45, 0.45, (N, 3))
        # Simple relaxation: push apart
        for it in range(50):
            # Find closest pair
            i, j = _pairwise_indices(N)
            d_ij = np.linalg.norm(c_rand[i] - c_rand[j], axis=1)
            min_idx = np.argmin(d_ij)
            ci = c_rand[i[min_idx]]
            cj = c_rand[j[min_idx]]
            diff = ci - cj
            norm = np.linalg.norm(diff)
            if norm < 1e-10:
                diff = np.random.normal(0, 1, 3)
                norm = np.linalg.norm(diff)
            # Push apart
            step = 0.01 * diff / norm
            c_rand[i[min_idx]] += step
            c_rand[j[min_idx]] -= step
            # Clamp
            c_rand = np.clip(c_rand, -0.49, 0.49)
        
        c_rand += np.random.normal(0, 0.01, c_rand.shape)
        c_rand = np.clip(c_rand, -0.49, 0.49)
        
        r_rand = _score(c_rand)
        theta_rand = np.concatenate([c_rand.flatten(), [r_rand]])
        
        try:
            res = minimize(
                lambda t: _objective_and_grad(t, i, j)[0],
                theta_rand,
                jac=lambda t: _objective_and_grad(t, i, j)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            
            if res.success or res.fun < -best_score + 100 * 1e-6:
                c_test = res.x[:3*N].reshape(N, 3)
                s = _score(c_test)
                if s > best_score:
                    best_score = s
                    best_theta = res.x
        except Exception as e:
            pass
    
    return best_theta[:3*N].reshape(N, 3)

# EVOLVE-BLOCK-END
