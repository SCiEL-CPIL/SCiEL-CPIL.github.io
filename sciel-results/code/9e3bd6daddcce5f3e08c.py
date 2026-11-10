"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_initial_hex_points(n_points):
    """Generate initial points using a hexagonal lattice cut to the unit disk."""
    # Adjust spacing to ensure we get at least n_points
    # For N=50, a spacing of ~0.2 is reasonable.
    # Area of unit circle ~ 3.14. Area of hex cell ~ 0.2 * 0.2 * 0.866 ~ 0.034.
    # 3.14 / 0.034 ~ 92 points. So spacing 0.2 should yield enough.
    spacing = 0.2
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Filter points strictly inside
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= n_points:
            # Select the n_points closest to the origin to ensure a dense core
            return points[np.argsort(radius, kind="stable")[:n_points]]
        spacing *= 0.95

def _softmin_objective_and_grad(c, beta):
    """
    Objective: Minimize -softmin(distances)
    Distances are:
    - Pairwise: |c_i - c_j| / 2
    - Boundary: 1 - |c_i|
    """
    c = c.reshape(N, 2)
    
    # 1. Pairwise distances
    # diff[i, j, :] = c[i] - c[j]
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Get distances for i < j
    # We need to be careful with division by zero
    dist_sq_safe = np.where(dist_sq < 1e-12, 1e-12, dist_sq)
    dist = np.sqrt(dist_sq_safe)
    
    # Radius candidates from pairs: dist / 2
    x_pair = dist / 2.0
    
    # 2. Boundary distances
    r = np.linalg.norm(c, axis=1)
    r_safe = np.where(r < 1e-12, 1e-12, r)
    x_bound = 1.0 - r_safe
    
    # Combine all candidates
    # x_pair has shape (N, N) but we only want upper triangle
    # It's easier to flatten the masked array
    x_pairs_flat = x_pair[mask]
    x_all = np.concatenate([x_pairs_flat, x_bound])
    
    # Softmin calculation
    # S = sum(exp(-beta * x))
    # val = -log(S) / beta
    exp_terms = np.exp(-beta * x_all)
    sum_exp = np.sum(exp_terms)
    
    # Objective value: -softmin = log(S) / beta
    obj_val = np.log(sum_exp) / beta
    
    # Gradient calculation
    # grad_val = - (1/S) * sum_k exp(-beta * x_k) * grad_x_k
    # We want to minimize obj_val, so we return gradient of obj_val.
    # d(obj_val)/dc = (1/beta) * (1/S) * d(S)/dc
    # d(S)/dc = sum_k (-beta * exp(-beta * x_k)) * d(x_k)/dc
    # So d(obj_val)/dc = -(1/S) * sum_k exp(-beta * x_k) * d(x_k)/dc
    
    grad = np.zeros_like(c)
    
    # 1. Pairwise Gradients
    # For each pair (i, j):
    # d(x_ij)/dc_i = 0.5 * (c_i - c_j) / |c_i - c_j|
    # d(x_ij)/dc_j = 0.5 * (c_j - c_i) / |c_i - c_j|
    
    # We have exp_terms corresponding to x_all.
    # The first len(x_pairs_flat) terms correspond to pairs.
    weights_pair = exp_terms[:len(x_pairs_flat)]
    
    # dist is (N, N). We need to apply weights to the gradient vectors.
    # grad_dist[i, j] = weights[i, j] * (c_i - c_j) / dist[i, j]
    # Note: weights are only for i < j.
    
    # Create a weight matrix
    weight_mat = np.zeros((N, N))
    weight_mat[mask] = weights_pair
    
    # Gradient contribution from pairs to c_i:
    # Sum_j weight_mat[i, j] * 0.5 * (c_i - c_j) / dist[i, j]
    # Note: weight_mat[i, j] is non-zero only if i < j.
    # But the term for (j, i) would be weight_mat[j, i] * 0.5 * (c_j - c_i) / dist[j, i]
    # Since weight_mat is symmetric in effect (we only have i<j), we can just sum over all j.
    
    # dist_safe is (N, N)
    # unit_vec = diff / dist_safe[..., np.newaxis]
    unit_vec = diff / dist_sq_safe[..., np.newaxis]
    
    # grad_pair[i] = 0.5 * Sum_j weight_mat[i, j] * unit_vec[i, j]
    # Note: weight_mat[i, j] is 0 if i > j.
    # But the physical force is symmetric.
    # If we sum over all j, for a pair i<j:
    # Term in grad[i]: weight_mat[i, j] * 0.5 * unit_vec[i, j]
    # Term in grad[j]: weight_mat[j, i] * 0.5 * unit_vec[j, i] = 0 (since weight_mat[j, i] is 0 for j>i? No, mask is i<j)
    # Wait, mask is i<j. So weight_mat[i, j] is set for i<j.
    # weight_mat[j, i] is 0.
    # So the force on j from this pair is missing in this sum if we just do `weight_mat * unit_vec`.
    # We need to account for the fact that the pair (i, j) contributes to both i and j.
    
    # Correct approach:
    # Force on i from j: w * 0.5 * (c_i - c_j)/dist
    # Force on j from i: w * 0.5 * (c_j - c_i)/dist = - Force on i from j
    
    # So we can compute the matrix of forces F[i, j] = weight_mat[i, j] * 0.5 * unit_vec[i, j]
    # Then grad[i] += Sum_j F[i, j]
    # And grad[j] -= Sum_i F[i, j]  (which is equivalent to Sum_j F[j, i] if we constructed it right)
    
    # Actually, simpler:
    # grad[i] = 0.5 * Sum_{j != i} w_{ij} * (c_i - c_j) / dist_{ij}
    # where w_{ij} is the weight for the pair {i, j}.
    # Since w_{ij} = w_{ji}, we can define a full weight matrix W where W[i, j] = w_{ij} if i<j else w_{ji} if i>j else 0.
    
    W = weight_mat + weight_mat.T
    
    # Now grad_pair[i] = 0.5 * Sum_j W[i, j] * unit_vec[i, j]
    # Note: unit_vec[i, i] is 0/0 -> handled by dist_sq_safe, but W[i, i]=0 so it's 0.
    
    grad_pair = 0.5 * np.sum(W * unit_vec, axis=1)
    
    # 2. Boundary Gradients
    # d(x_i)/dc_i = - c_i / |c_i|
    # weights_bound = exp_terms[len(x_pairs_flat):]
    weights_bound = exp_terms[len(x_pairs_flat):]
    
    # grad_bound[i] = weights_bound[i] * (- c_i / |c_i|)
    # c / r_safe
    unit_r = c / r_safe[:, np.newaxis]
    grad_bound = -weights_bound[:, np.newaxis] * unit_r
    
    # Total Gradient
    grad = grad_pair + grad_bound
    
    # We are minimizing -softmin, which is obj_val.
    # The gradient derived above is for d(softmin)/dc?
    # Let's re-verify signs.
    # softmin = -1/beta * log(S)
    # d(softmin)/dc = -1/beta * 1/S * dS/dc
    # dS/dc = sum (-beta exp(-beta x)) dx/dc = -beta sum exp(-beta x) dx/dc
    # d(softmin)/dc = sum exp(-beta x) dx/dc / S
    # This is the gradient of the function we want to MAXIMIZE.
    # We want to MINIMIZE -softmin.
    # So grad_obj = - grad_softmin = - (grad_pair + grad_bound)
    
    grad_obj = -(grad_pair + grad_bound)
    
    return obj_val, grad_obj.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # 1. Initial Configuration
    c0 = _get_initial_hex_points(N)
    
    # 2. Optimization
    # We will run minimize with a specific beta
    # Start with beta=20
    beta = 20.0
    
    # Pack c0 into flat array
    x0 = c0.flatten()
    
    # Define objective and gradient for scipy
    def func(x):
        return _softmin_objective_and_grad(x, beta)[0]
        
    def jac(x):
        return _softmin_objective_and_grad(x, beta)[1]
        
    # Run optimization
    best_x = x0
    best_val = func(x0)
    
    try:
        # L-BFGS-B is good for large N, but N=50 is small. BFGS or CG also fine.
        # L-BFGS-B handles memory well.
        res = minimize(
            func, 
            x0, 
            jac=jac, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success and res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            
        # Check time
        if time.time() < deadline - 5.0:
            # Try a second pass with higher beta to sharpen
            beta2 = 50.0
            x1 = best_x
            best_val2 = func(x1) # Note: func uses beta from closure, need to update or pass
            # Let's create a new function for beta2
            def func2(x):
                return _softmin_objective_and_grad(x, beta2)[0]
            def jac2(x):
                return _softmin_objective_and_grad(x, beta2)[1]
                
            res2 = minimize(
                func2, 
                x1, 
                jac=jac2, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res2.success and res2.fun < best_val2:
                # Compare actual radius to be sure
                # The objective values are not directly comparable across different betas
                # We should compare the actual softmin value or the raw radius
                pass # This comparison is tricky. Better to just use the result if it converged.
                # Actually, higher beta makes the objective more sensitive to the min.
                # If it converges, it's likely a better packing.
                best_x = res2.x
                
    except Exception:
        # If optimization fails, return initial
        pass
        
    # 3. Finalize
    # Ensure points are within bounds (numerical errors might push them out)
    c_final = best_x.reshape(N, 2)
    
    # Clip points that are slightly outside?
    # Better to project them back if they are outside.
    r = np.linalg.norm(c_final, axis=1)
    mask_out = r > 1.0
    if np.any(mask_out):
        c_final[mask_out] = c_final[mask_out] / r[mask_out, np.newaxis] * 0.99
        
    return c_final
# EVOLVE-BLOCK-END
