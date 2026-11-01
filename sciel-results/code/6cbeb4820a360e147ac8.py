"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _get_min_radius(pos):
    """Calculate the maximum possible radius for a given configuration."""
    # Pairwise distances
    d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(d**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Wall distances
    wall_dists = 0.5 - np.abs(pos)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _init_grid():
    """Generate a grid-based initial configuration."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pos = grid[:N].copy()
    return pos

def _init_random():
    """Generate a random initial configuration."""
    # Random points in the box, slightly inset to avoid immediate boundary issues
    pos = np.random.uniform(-0.45, 0.45, (N, 3))
    return pos

def _softmin_objective(params):
    """
    Objective function for L-BFGS-B.
    We minimize the sum of squared violations of the constraint that 
    all distances must be >= 2r.
    To make this differentiable and stable, we use a penalty method where 
    we optimize for a specific target radius r.
    However, L-BFGS-B works best with a fixed objective. 
    A common trick is to minimize the sum of squared "deficits" relative to a target radius.
    
    Let's define the objective as:
    f(x) = sum_{i<j} max(0, 2r - d_ij)^2 + sum_{i,k} max(0, r - w_ik)^2
    where r is a fixed target for this optimization phase.
    This is not differentiable at the boundary. We can smooth it.
    Soft-plus: max(0, u) approx u/2 + sqrt(u^2 + eps)/2
    Let's use a simpler approach: Minimize sum of 1/d^2 for pairs and walls, 
    but weighted by how close they are to the bottleneck?
    
    Actually, the inspiration suggests: "minimizing max(0, 2r - d_ij)^2".
    Since L-BFGS-B requires a fixed function, we will run a sequence of optimizations 
    with increasing r. If the optimizer can satisfy the constraints for r, we increase r.
    If not, we decrease r or stop.
    
    Alternative: Minimize the "worst-case" violation using a log-sum-exp or similar smooth max.
    f(x) = log( sum_{i<j} exp((2r - d_ij)^2 / tau) + sum_{i,k} exp((r - w_ik)^2 / tau) )
    This is smooth and differentiable.
    """
    pos = params.reshape(N, 3)
    
    # Target radius for this specific optimization call
    # This needs to be passed in or captured. Since minimize doesn't pass extra args easily 
    # without wrapping, we'll use a closure or global. 
    # Better: We will define the function inside construct to capture r.
    
    # This function is a placeholder for the structure. The actual logic is in construct.
    pass

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_pos = None
    best_radius = 0.0
    
    # We will use a multi-start strategy with a direct radius optimization approach.
    # Strategy:
    # 1. Initialize positions.
    # 2. Estimate a starting radius r0.
    # 3. Use L-BFGS-B to minimize the violation of the constraint that all distances >= 2r.
    #    We define the objective as the sum of squared violations: V(x, r) = sum max(0, 2r - d)^2.
    #    To make it smooth, we use: V(x, r) = sum (softplus(2r - d))^2.
    #    softplus(u) = log(1 + exp(u)).
    #    This is smooth and approximates max(0, u) for large |u|.
    # 4. If the minimum violation is small (< tolerance), we increase r.
    # 5. Repeat until time limit or convergence.
    
    def softplus(u):
        # Stable softplus
        # For u > 0, log(1+exp(u)) ~ u
        # For u < 0, log(1+exp(u)) ~ exp(u)
        # np.logaddexp(0, u) is a stable way to compute this
        return np.logaddexp(0, u)
    
    def objective_and_grad(params, r_target):
        pos = params.reshape(N, 3)
        
        # Pairwise distances
        d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(d**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(pos)
        
        # Violations
        # Pair violation: max(0, 2r - dist)
        # Wall violation: max(0, r - wall_dist)
        
        pair_viol = 2.0 * r_target - dist
        wall_viol = r_target - wall_dists
        
        # We only care about upper triangle for pairs to avoid double counting in sum, 
        # but gradient needs to be correct.
        # Let's just sum over all i<j.
        # Mask for i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # Softplus approximation for max(0, x)
        # Let's use a sharp softplus to approximate the kink.
        # s = log(1 + exp(beta * x)) / beta
        beta = 50.0 # Sharpness
        
        sp_pair = softplus(beta * pair_viol) / beta
        sp_wall = softplus(beta * wall_viol) / beta
        
        # Objective: Sum of squared softplus violations
        obj = np.sum((sp_pair[mask]) ** 2) + np.sum((sp_wall) ** 2)
        
        # Gradient
        # d/dx softplus(beta*x)/beta = (beta * exp(beta*x) / (1+exp(beta*x))) / beta = exp(beta*x)/(1+exp(beta*x)) = sigmoid(beta*x)
        # Let sigma_pair = sigmoid(beta * pair_viol)
        # d/dx (sp^2) = 2 * sp * d(sp)/dx
        
        sigma_pair = 1.0 / (1.0 + np.exp(-beta * pair_viol))
        sigma_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))
        
        grad_pair = 2.0 * sp_pair * sigma_pair
        grad_wall = 2.0 * sp_wall * sigma_wall
        
        # Gradient of pair_viol w.r.t pos
        # pair_viol[i,j] = 2r - ||pos[i] - pos[j]||
        # d(pair_viol)/d(pos[i]) = (pos[i] - pos[j]) / ||pos[i] - pos[j]||
        # d(pair_viol)/d(pos[j]) = (pos[j] - pos[i]) / ||pos[i] - pos[j]||
        
        # We need to compute the gradient of the objective w.r.t pos.
        # grad_pos = sum_j grad_pair[i,j] * d(pair_viol[i,j])/d(pos[i])
        
        # Vectorized gradient calculation
        # grad_dist[i,j] = (pos[i] - pos[j]) / dist[i,j]
        # dist is (N, N)
        # d_vec is (N, N, 3)
        
        # Avoid division by zero (diagonal is 1.0)
        inv_dist = 1.0 / dist
        # d_vec = d * inv_dist[:,:,None]
        
        # grad_pair is (N, N). We need to mask i<j.
        grad_pair_masked = np.where(mask, grad_pair, 0.0)
        
        # Force on i from j:
        # F_i = sum_j grad_pair[i,j] * (pos[i] - pos[j]) / dist[i,j]
        # F_j = sum_i grad_pair[i,j] * (pos[j] - pos[i]) / dist[i,j] = - sum_i grad_pair[i,j] * (pos[i] - pos[j]) / dist[i,j]
        
        # Let's compute the contribution to the gradient of pos.
        # grad_pos[i] = sum_j grad_pair[i,j] * (pos[i] - pos[j]) / dist[i,j]
        #               - sum_j grad_pair[j,i] * (pos[i] - pos[j]) / dist[i,j]  (from the j perspective where i is the "j" in the formula)
        
        # Actually, simpler:
        # The term in the sum is S = sum_{i<j} G_{ij} * V_{ij}
        # V_{ij} = 2r - d_{ij}
        # dV_{ij}/dpos_i = (pos_i - pos_j)/d_{ij}
        # dV_{ij}/dpos_j = (pos_j - pos_i)/d_{ij}
        
        # grad_pos_i = sum_{j>i} G_{ij} * (pos_i - pos_j)/d_{ij} + sum_{j<i} G_{ji} * (pos_i - pos_j)/d_{ji}
        
        # Let's use broadcasting.
        # d_pos_i_j = (pos[i] - pos[j]) / dist[i,j]
        # This is d_vec * inv_dist[:,:,None]
        
        d_hat = d * inv_dist[:, :, np.newaxis]
        
        # grad_pair_masked is (N, N)
        # We want to sum over j for each i.
        # Term 1: sum_j grad_pair[i,j] * d_hat[i,j]
        # Term 2: sum_j grad_pair[j,i] * (-d_hat[i,j])  (because d_hat[j,i] = -d_hat[i,j])
        
        # Note: grad_pair_masked is zero for j<=i in the first sum? No, we masked i<j.
        # So for a fixed i, sum_{j>i} grad_pair[i,j] * d_hat[i,j]
        # And for the other term, we need sum_{j<i} grad_pair[j,i] * d_hat[j,i] * (-1) ?
        # Wait, if we sum over all i,j with mask i<j, we cover each pair once.
        # The gradient w.r.t pos_i gets a contribution from pair (i,j) if i<j (as first index) and if j<i (as second index).
        
        # Let's just compute the full matrix gradient contribution.
        # grad_full[i, j] = grad_pair[i, j] * d_hat[i, j]
        # Then grad_pos[i] = sum_j grad_full[i, j] - sum_j grad_full[j, i]
        # Because d_hat[j, i] = -d_hat[i, j].
        # And the term in the objective for pair (i,j) is associated with d_hat[i,j] for pos_i and -d_hat[i,j] for pos_j.
        # If we define G[i,j] = grad_pair[i,j] (for i<j), then:
        # Contribution to grad_pos[i] from pair (i,j) where i<j: G[i,j] * d_hat[i,j]
        # Contribution to grad_pos[i] from pair (j,i) where j<i (i.e., pair (i,j) with j>i, but stored as G[j,i]? No, G is only defined for i<j).
        
        # Let's stick to the mask.
        # grad_pair_masked[i,j] is non-zero only if i<j.
        # So for a fixed i:
        # Sum over j where i<j: grad_pair_masked[i,j] * d_hat[i,j]
        # Sum over j where i>j (i.e., j<i): grad_pair_masked[j,i] * (-d_hat[i,j])  (since d_hat[j,i] = -d_hat[i,j])
        
        # Let A = grad_pair_masked * d_hat
        # A[i,j] is non-zero only if i<j.
        # grad_pos[i] = sum_j A[i,j] - sum_j A[j,i]
        
        A = grad_pair_masked[:, :, np.newaxis] * d_hat
        grad_pos = np.sum(A, axis=1) - np.sum(A, axis=0)
        
        # Wall gradient
        # wall_viol[i,k] = r - (0.5 - |pos[i,k]|) = r - 0.5 + |pos[i,k]|
        # d(wall_viol)/d(pos[i,k]) = sign(pos[i,k])
        # grad_wall is (N, 3)
        # grad_pos_wall = grad_wall * sign(pos)
        
        signs = np.sign(pos)
        # Handle zero
        signs[pos == 0] = 1.0
        
        grad_pos += grad_wall * signs
        
        return obj, grad_pos.flatten()
    
    # Multi-start loop
    n_starts = 5
    start_times = [time.time()]
    
    for s in range(n_starts):
        if time.time() > deadline - 10.0:
            break
            
        # Choose initialization
        if s == 0:
            pos = _init_grid()
            pos += np.random.uniform(-0.01, 0.01, pos.shape)
        else:
            pos = _init_random()
        
        # Initial radius estimate
        r = _get_min_radius(pos)
        if r > best_radius:
            best_radius = r
            best_pos = pos.copy()
        
        # Optimization loop
        # We will try to increase r in small steps.
        # If we can satisfy constraints for r, we keep pos and increase r.
        # If not, we stop this start.
        
        current_r = r
        step_r = 0.0001
        max_iter_per_step = 50
        
        for step in range(100): # Max 100 radius increments
            if time.time() > deadline - 5.0:
                break
            
            target_r = current_r + step_r
            
            # Optimize positions to satisfy target_r
            # We minimize the violation for target_r
            res = minimize(
                objective_and_grad,
                pos.flatten(),
                args=(target_r,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter_per_step, 'gtol': 1e-6}
            )
            
            if res.success or res.status == 1:
                # Check if constraints are satisfied
                final_pos = res.x.reshape(N, 3)
                # Clamp to box just in case
                final_pos = np.clip(final_pos, -0.5, 0.5)
                
                actual_r = _get_min_radius(final_pos)
                
                if actual_r >= target_r - 1e-5:
                    # Success! Update pos and r
                    pos = final_pos
                    current_r = target_r
                    if current_r > best_radius:
                        best_radius = current_r
                        best_pos = pos.copy()
                else:
                    # Failed to reach target_r, try smaller step?
                    # For now, break.
                    break
            else:
                # Optimization failed
                break

    return best_pos

# EVOLVE-BLOCK-END
