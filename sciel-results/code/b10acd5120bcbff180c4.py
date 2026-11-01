"""
Strategy: Penalty-Augmented Lagrangian Method for Non-Convex Quadratic Program (NQP).
Formulates the circle packing as maximizing r^2 subject to ||c_i - c_j||^2 >= 2r^2 and wall constraints.
Uses exact analytic gradients to guide L-BFGS-B efficiently.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    if n == 0:
        return 0.0
    # Wall distances
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles."""
    best_cents = None
    best_r = 0

    # Try a few row configurations
    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]

        max_rc = max(row_counts)
        # Estimate radius based on horizontal fit
        r_horiz = 1.0 / (2.0 * max_rc)
        # Estimate radius based on vertical fit (hexagonal spacing is sqrt(3)*r)
        # Total height = r + (n_rows-1)*sqrt(3)*r + r = 2r + (n_rows-1)*sqrt(3)*r
        # Height must be <= 1.0
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)

        if r <= best_r:
            continue

        centers = []
        # Calculate total height to center vertically
        total_y = (n_rows - 1) * np.sqrt(3.0) * r
        y_start = -0.5 + r + total_y / 2.0

        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * np.sqrt(3.0) * r
            x_span = (rc - 1) * 2.0 * r
            x_start = -0.5 + r + x_span / 2.0
            # Offset every other row
            offset = 0.0
            if row_idx % 2 == 1:
                offset = r
            
            for col in range(rc):
                x = x_start + col * 2.0 * r + offset
                centers.append([x, y])

        centers = np.array(centers[:N])
        r_actual = _score(centers)

        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()

    if best_cents is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)

    return best_cents, best_r

def _nqp_objective_and_grad(x, mu, lam_pairs, lam_walls):
    """
    Objective: Minimize -r^2 + (mu/2) * sum(soft_violations^2)
    Variables: x = [c_0, ..., c_{N-1}, r]
    Constraints:
    Pairs: ||c_i - c_j||^2 - 2r^2 >= 0  -> violation v_ij = 2r^2 - ||c_i - c_j||^2
    Walls: 0.5 - |c_i,k| - r >= 0      -> violation v_ik = r + |c_i,k| - 0.5
    """
    c = x[:2*N].reshape(N, 2)
    r = x[2*N]
    
    # Compute pairwise squared distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 0.0)
    
    # Pairwise violations: v_ij = 2r^2 - d_ij^2 (we want v <= 0)
    # We penalize positive violations
    pair_violations = 2.0 * r**2 - dists_sq
    # Only consider i < j to avoid double counting in sum, but for gradient we can use full matrix / 2
    # Let's sum over i < j
    tri_idx = np.triu_indices(N, k=1)
    pair_v = pair_violations[tri_idx]
    # Soft penalty: sum(max(v, 0)^2)
    # For differentiability, use smooth approximation or just max since we are in penalty method
    # Actually, standard penalty uses (max(v,0))^2. Gradient is 0 where v<0, 2v where v>0.
    # This is non-smooth at v=0. To be safe for L-BFGS, we can use a smooth hinge or just rely on it.
    # Let's use a smooth hinge: 0.5 * (v + sqrt(v^2 + eps))^2 - 0.5*eps ... too complex.
    # Standard penalty methods often work with the kink if the optimizer is robust, but let's use a smooth surrogate for gradient.
    # Surrogate: f(v) = 0.5 * v^2 if v > 0, else 0. 
    # Gradient: v if v > 0, else 0.
    # Let's stick to the simple penalty but compute gradient carefully.
    
    # Wall violations: v_ik = r + |c_i,k| - 0.5
    # c_i,0 and c_i,1
    wall_viol_x = r + np.abs(c[:, 0]) - 0.5
    wall_viol_y = r + np.abs(c[:, 1]) - 0.5
    wall_v = np.concatenate([wall_viol_x, wall_viol_y])
    
    # Penalty term
    # We want to minimize -r^2 + (mu/2) * (sum pair_v_pos^2 + sum wall_v_pos^2)
    # Plus linear terms from Lagrange multipliers? 
    # In augmented Lagrangian: L = f + lambda^T g + (mu/2) ||g||^2
    # Here f = -r^2. g are the violations (we want g <= 0, so we penalize max(g,0)).
    # Actually, standard form is minimize f subject to g(x) <= 0.
    # Penalty: (mu/2) * sum (max(g_i, 0))^2.
    # Gradient w.r.t x: sum (max(g_i, 0) * grad g_i).
    
    # Calculate penalty value
    pair_pos = np.maximum(pair_v, 0.0)
    wall_pos = np.maximum(wall_v, 0.0)
    
    penalty_val = 0.5 * mu * (np.sum(pair_pos**2) + np.sum(wall_pos**2))
    
    # Total objective
    obj = -r**2 + penalty_val
    
    # Calculate gradient
    grad = np.zeros_like(x)
    
    # Gradient of -r^2 w.r.t r is -2r
    grad[2*N] = -2.0 * r
    
    # Gradient of penalty
    # d/dx [ 0.5 * mu * (max(g, 0))^2 ] = mu * max(g, 0) * grad(g)
    
    # Pairwise gradient
    # g_ij = 2r^2 - ||c_i - c_j||^2
    # d(g_ij)/dr = 4r
    # d(g_ij)/dc_i = -2(c_i - c_j)
    # d(g_ij)/dc_j = -2(c_j - c_i) = 2(c_i - c_j)
    
    # Sum over active pairs (where v > 0)
    active_pairs = pair_pos > 0
    if np.any(active_pairs):
        # Get indices of active pairs
        i_idx, j_idx = tri_idx
        active_i = i_idx[active_pairs]
        active_j = j_idx[active_pairs]
        active_v = pair_pos[active_pairs]
        
        # Weight for each active pair
        w = mu * active_v
        
        # Contribution to r gradient
        # sum w * 4r
        grad[2*N] += np.sum(w) * 4.0 * r
        
        # Contribution to c gradients
        # For each active pair (i, j):
        # grad_c_i += w * (-2) * (c_i - c_j)
        # grad_c_j += w * 2 * (c_i - c_j)
        
        # Use scatter_add
        # c is shape (N, 2). Flatten indices for c: i*2, i*2+1
        # We need to accumulate into grad[:2*N]
        
        # Create arrays for indices
        ci_flat = active_i * 2
        cj_flat = active_j * 2
        
        # Differences
        diffs = c[active_i] - c[active_j] # shape (num_active, 2)
        
        # Weights are 1D array of shape (num_active,)
        # We need to multiply each row of diffs by w[:, np.newaxis]
        w_exp = w[:, np.newaxis]
        contrib = w_exp * diffs # shape (num_active, 2)
        
        # grad_c_i gets -2 * contrib
        # grad_c_j gets 2 * contrib
        
        # Flatten contrib to 1D
        contrib_flat = contrib.flatten()
        
        # Scatter add
        np.add.at(grad, ci_flat, -2.0 * contrib_flat)
        np.add.at(grad, cj_flat, 2.0 * contrib_flat)

    # Wall gradient
    # g_ik = r + |c_i,k| - 0.5
    # d(g)/dr = 1
    # d(g)/dc_i,k = sign(c_i,k) (if c_i,k != 0)
    
    active_walls = wall_pos > 0
    if np.any(active_walls):
        w_wall = mu * wall_pos[active_walls]
        
        # Contribution to r
        grad[2*N] += np.sum(w_wall)
        
        # Contribution to c
        # wall_v was concatenated: first N are x, next N are y
        # Indices in wall_v: 0..N-1 for x, N..2N-1 for y
        
        # For x-walls (indices 0..N-1)
        active_x = active_walls[:N]
        if np.any(active_x):
            idx_x = np.where(active_x)[0]
            w_x = w_wall[:N][active_x]
            # sign(c_i, 0)
            signs_x = np.sign(c[idx_x, 0])
            # grad_c_i,0 += w_x * signs_x
            flat_idx_x = idx_x * 2
            np.add.at(grad, flat_idx_x, w_x * signs_x)
            
        # For y-walls (indices N..2N-1)
        active_y = active_walls[N:]
        if np.any(active_y):
            idx_y = np.where(active_y)[0]
            w_y = w_wall[N:][active_y]
            signs_y = np.sign(c[idx_y, 1])
            flat_idx_y = idx_y * 2 + 1
            np.add.at(grad, flat_idx_y, w_y * signs_y)

    return obj, grad

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin

    # Step 1: Hexagonal initial arrangement
    cents, r0 = _construct_hexagonal()
    best_cents = cents.copy()
    best_r = r0
    
    # Initial guess for r is best_r
    r_init = best_r
    
    # Initial Lagrange multipliers
    # Number of pairs
    num_pairs = N * (N - 1) // 2
    # Number of wall constraints
    num_walls = 2 * N
    
    lam_pairs = np.zeros(num_pairs)
    lam_walls = np.zeros(num_walls)
    
    # Penalty parameter
    mu = 10.0
    
    # Variable vector x = [c, r]
    x = np.concatenate([cents.flatten(), [r_init]])
    
    # Bounds for c: (-0.5, 0.5)
    # Bounds for r: (0, 0.5)
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    # Optimization loop
    # We will run a few iterations of the augmented Lagrangian method
    # Each iteration: minimize L, then update multipliers
    
    max_outer_iters = 5
    for outer_iter in range(max_outer_iters):
        if time.time() > deadline - 20:
            break
            
        # Define objective and gradient for this mu and lambdas
        # Note: L-BFGS-B expects (x) -> (f, g)
        def obj_grad(x_inner):
            f, g = _nqp_objective_and_grad(x_inner, mu, lam_pairs, lam_walls)
            return f, g
            
        try:
            result = minimize(
                obj_grad, x,
                method='L-BFGS-B', bounds=bounds,
                jac=True, # We provide gradient
                options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            x_new = result.x
            c_new = x_new[:2*N].reshape(N, 2)
            r_new = x_new[2*N]
            
            # Update best if improved
            r_score = _score(c_new)
            if r_score > best_r:
                best_r = r_score
                best_cents = c_new.copy()
                
            # Update multipliers
            # Get violations at x_new
            # We need to recompute violations to update lambdas
            c_curr = x_new[:2*N].reshape(N, 2)
            r_curr = x_new[2*N]
            
            diff = c_curr[:, np.newaxis, :] - c_curr[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, 0.0)
            
            tri_idx = np.triu_indices(N, k=1)
            pair_v = 2.0 * r_curr**2 - dists_sq[tri_idx]
            
            wall_viol_x = r_curr + np.abs(c_curr[:, 0]) - 0.5
            wall_viol_y = r_curr + np.abs(c_curr[:, 1]) - 0.5
            wall_v = np.concatenate([wall_viol_x, wall_viol_y])
            
            # Update rule: lam_new = max(lam + mu * g, 0)
            # Here g is the violation (positive means constraint broken)
            # We want to penalize g > 0.
            # So if g > 0, increase lambda.
            # lam_pairs = np.maximum(lam_pairs + mu * pair_v, 0.0)
            # lam_walls = np.maximum(lam_walls + mu * wall_v, 0.0)
            
            # To avoid exploding lambdas, we might cap them or use a different update.
            # Standard: lam_{k+1} = max(0, lam_k + mu_k * g(x_{k+1}))
            lam_pairs = np.maximum(lam_pairs + mu * pair_v, 0.0)
            lam_walls = np.maximum(lam_walls + mu * wall_v, 0.0)
            
            # Increase mu
            mu *= 10.0
            
            x = x_new
            
        except Exception as e:
            print(f"Optimization iteration {outer_iter} failed: {e}")
            break
            
    # Final check
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    return best_cents
# EVOLVE-BLOCK-END
