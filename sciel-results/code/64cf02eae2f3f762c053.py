"""Sequential radial expansion packer for 100 equal spheres in a unit cube."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 140.0  # Safety margin

def _score(points):
    """Return the maximal radius for the given set of centres."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = dists.min() / 2.0
    return min(r_pair, r_margin)

def _energy_and_gradient(x, r_target, beta=100.0):
    """
    Compute energy (sum of squared violations) and gradient for a target radius r_target.
    Violations:
    1. Pairwise distance < 2*r_target
    2. Distance to boundary < r_target
    """
    pts = x.reshape(N, 3)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Boundary margins
    margins = 0.5 - np.abs(pts).max(axis=1)
    
    # Violations for pairs
    pair_viol = 2.0 * r_target - dists
    # Violations for boundaries
    bound_viol = r_target - margins
    
    # Soft-min approximation: use exp(beta * viol) for active constraints
    # To avoid overflow, clip viol
    pair_viol_clipped = np.clip(pair_viol, -10.0, 10.0)
    bound_viol_clipped = np.clip(bound_viol, -10.0, 10.0)
    
    pair_weights = np.exp(beta * pair_viol_clipped)
    bound_weights = np.exp(beta * bound_viol_clipped)
    
    # Zero out diagonal for pairs
    np.fill_diagonal(pair_weights, 0.0)
    
    # Energy is sum of weighted squared violations
    # Actually, a common soft-min for max-min is minimizing sum of exp(beta * (target - dist))
    # But for gradient descent, we can just minimize the sum of violations where they are positive?
    # Let's use a standard penalty method: minimize sum(max(0, viol)^2)
    # This is non-smooth at 0.
    # Better: use a smooth approximation like 0.5 * (sqrt(viol^2 + eps) - eps) or just viol if viol>0 else 0.
    # Let's stick to the gradient of the "active" constraints.
    
    # Let's use a simpler objective: Sum of (max(0, 2r - d_ij))^2 + Sum of (max(0, r - margin_i))^2
    # This is differentiable almost everywhere.
    
    active_pair_viol = np.where(pair_viol > 0, pair_viol, 0.0)
    active_bound_viol = np.where(bound_viol > 0, bound_viol, 0.0)
    
    energy = np.sum(active_pair_viol ** 2) + np.sum(active_bound_viol ** 2)
    
    # Gradient
    grad = np.zeros_like(x)
    
    # Pairwise gradient
    # d/dx_i (max(0, 2r - d_ij)^2) = -2 * (2r - d_ij) * (x_i - x_j) / d_ij * 2 (from chain rule on square? No)
    # f(v) = v^2, df/dv = 2v. v = 2r - d. dv/dx = - (x_i - x_j)/d.
    # So grad = 2 * (2r - d) * (x_i - x_j)/d for each pair.
    
    # Vectorized gradient for pairs
    # Only for active violations
    mask = pair_viol > 0
    # Create a matrix of weights
    # w_ij = 2 * (2r - d_ij) / d_ij
    # We need to sum over j for each i.
    
    # To be safe and fast, let's compute gradient only for active pairs
    # This might be slow in Python loops, but N=100 is small.
    # However, vectorization is preferred.
    
    # Let's use the formula:
    # grad_i = sum_j 2 * (2r - d_ij) * (x_i - x_j) / d_ij
    # Note: (x_i - x_j) / d_ij is the unit vector from j to i.
    
    # We can compute this using matrix operations.
    # Let D be the distance matrix.
    # Let U be the unit vector matrix. U_ij = (x_i - x_j) / d_ij.
    # Then grad_i = sum_j 2 * (2r - D_ij) * U_ij
    
    # Compute U
    # Avoid division by zero
    dists_safe = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / dists_safe[:, :, np.newaxis]
    np.fill_diagonal(unit_vecs[:, :, 0], 0.0)
    np.fill_diagonal(unit_vecs[:, :, 1], 0.0)
    np.fill_diagonal(unit_vecs[:, :, 2], 0.0)
    
    # Coefficients
    coeffs = 2.0 * pair_viol
    np.fill_diagonal(coeffs, 0.0)
    # Only apply to active violations?
    # If viol <= 0, the gradient of max(0, viol)^2 is 0.
    # So we multiply by mask.
    coeffs_active = np.where(mask, coeffs, 0.0)
    
    # grad_pair = sum_j coeffs_active_ij * unit_vecs_ij
    grad_pair = np.sum(coeffs_active[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # Boundary gradient
    # d/dx_i (max(0, r - margin_i)^2)
    # margin_i = min_k (0.5 - |x_ik|)
    # This is the minimum over dimensions.
    # The gradient is non-zero only for the dimension k that achieves the minimum.
    # If multiple dimensions tie, it's sub-gradient. We pick one or average.
    
    grad_bound = np.zeros_like(pts)
    for i in range(N):
        if bound_viol[i] > 0:
            # Find the dimension with minimum margin
            min_margin_val = margins[i]
            # Get signs
            signs = np.sign(pts[i])
            # The gradient of -|x_ik| is -sign(x_ik)
            # So grad of margin is -sign(x_ik) for the active dim
            # grad of (r - margin)^2 = 2(r - margin) * (-grad_margin)
            # = 2(bound_viol) * sign(x_ik)
            
            # Find active dims
            active_dims = np.where(np.abs(0.5 - np.abs(pts[i]) - min_margin_val) < 1e-9)[0]
            if len(active_dims) > 0:
                # Average the gradients of the active dimensions to handle non-smoothness
                for k in active_dims:
                    grad_bound[i, k] += 2.0 * bound_viol[i] * signs[k]
                grad_bound[i, k] /= len(active_dims)
    
    grad = grad_pair.flatten() + grad_bound.flatten()
    
    return energy, grad

def _optimize_radius(points, r_target, maxiter=200):
    """Optimize points to satisfy radius r_target."""
    x0 = points.flatten()
    
    def obj(x):
        e, g = _energy_and_gradient(x, r_target)
        return e, g
    
    try:
        res = minimize(obj, x0, jac=True, method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (N * 3),
                       options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-5})
        if res.success or res.fun < 1e-6:
            return res.x.reshape(N, 3)
        else:
            return points
    except Exception:
        return points

def _grid_init(nx, ny, nz, jitter=0.0):
    """Generate a grid-based initialization with optional jitter."""
    # Ensure at least N points
    while nx * ny * nz < N:
        nz += 1
    
    # Generate grid
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    points = []
    for z in zs:
        for y in ys:
            for x in xs:
                points.append([x, y, z])
    
    points = np.array(points[:N])
    
    if jitter > 0:
        points += np.random.normal(0, jitter, points.shape)
        points = np.clip(points, -0.49, 0.49)
    
    return points

def _random_init(scale=0.05):
    """Generate a random initialization within the cube."""
    points = np.random.uniform(-0.5 + scale, 0.5 - scale, (N, 3))
    return points

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    random.seed(42)
    np.random.seed(42)
    
    # Initial radius guess
    # Volume argument: 100 * (4/3) pi r^3 <= 1 => r <= (3/(400 pi))^(1/3) ~ 0.062
    # But packing efficiency is lower. Best known is ~0.111.
    # Start with a small r and increase.
    
    best_points = None
    best_score = 0.0
    
    # Generate a few diverse initializations
    inits = [
        _grid_init(5, 5, 4, jitter=0.02),
        _grid_init(4, 5, 5, jitter=0.02),
        _grid_init(5, 4, 5, jitter=0.02),
        _random_init(scale=0.05),
        _random_init(scale=0.05)
    ]
    
    # Try each initialization
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        current_points = init.copy()
        current_score = _score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
        
        # Start with a radius slightly larger than current score
        r_target = max(0.05, current_score * 1.05)
        
        # Iteratively increase r_target
        # If optimization succeeds (energy close to 0), increase r_target
        # If it fails, decrease r_target or stop
        
        while time.time() < deadline - 5:
            # Check if we can increase r
            # Try to optimize for r_target
            optimized_points = _optimize_radius(current_points, r_target, maxiter=100)
            
            # Check the actual score of the optimized points
            new_score = _score(optimized_points)
            
            if new_score >= r_target - 1e-5:
                # Success! We achieved at least r_target
                current_points = optimized_points
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                
                # Increase target radius
                # Step size can be adaptive
                r_target *= 1.02  # 2% increase
            else:
                # Failed to achieve r_target
                # Maybe r_target is too aggressive
                # Try a smaller increase or break
                if r_target > 0.01:
                    r_target *= 0.98
                else:
                    break
                
                # If we've been stuck for a while, break
                # Simple heuristic: if new_score is not close to r_target, we might be in a local min
                if new_score < r_target * 0.9:
                    break
        
        # If we have a good score, do a final polish with a slightly higher target
        if best_score > 0.1 and time.time() < deadline - 2:
            # Try to push a bit further
            final_target = best_score * 1.01
            final_points = _optimize_radius(best_points, final_target, maxiter=50)
            final_score = _score(final_points)
            if final_score > best_score:
                best_score = final_score
                best_points = final_points
    
    if best_points is None:
        best_points = _grid_init(5, 5, 4, jitter=0.0)
    
    return best_points

# EVOLVE-BLOCK-END
