import numpy as np
from scipy.spatial import distance
from scipy.optimize import minimize
import time

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initialization: Jittered Grid
    # Use a grid slightly larger than sqrt(N) to ensure we have enough points, then trim
    m = int(np.ceil(np.sqrt(N)))
    # Create a grid of size m x m
    # We want to place N points. If m*m > N, we take the first N.
    # To break symmetry, add jitter.
    x = np.linspace(-0.45, 0.45, m)
    y = np.linspace(-0.45, 0.45, m)
    xx, yy = np.meshgrid(x, y)
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    centers = centers[:N]
    # Add small random jitter
    np.random.seed(42)
    centers += np.random.normal(0, 0.005, centers.shape)
    # Clip to ensure inside
    centers = np.clip(centers, -0.49, 0.49)

    best_centers = centers.copy()
    best_radius = -1.0

    # Helper to compute radius
    def calc_radius(c):
        # Pairwise distances
        d = distance.pdist(c)
        min_pair_dist = np.min(d) if len(d) > 0 else 1.0
        # Boundary distances
        # min(0.5 - |x|, 0.5 - |y|)
        min_bound = np.min(np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1])))
        return min(min_pair_dist / 2.0, min_bound)

    best_radius = calc_radius(best_centers)

    # 2. Repulsion / Gradient Ascent Phase
    # We will iterate, computing forces and updating positions.
    # Force model: Repel from neighbors closer than 2*r_current * (1+margin)
    # and from boundaries.
    
    current_centers = best_centers.copy()
    current_radius = best_radius
    
    # Learning rate schedule
    base_lr = 0.01
    decay = 0.999
    
    for i in range(20000):
        if time.time() > deadline - 20.0:
            break
            
        # Compute distances
        # For efficiency, we might not need all pairs if N is small, but 200 is small enough for O(N^2)
        # distance.pdist is fast
        d_mat = distance.pdist(current_centers)
        # We need to know which pairs are close.
        # Reconstruct full distance matrix for force calculation? 
        # pdist returns condensed form. 
        # Let's use a simpler loop or cdist for clarity, pdist is fine but indexing is tricky.
        # Let's use cdist for clarity in force calculation, it's fast for N=200.
        dists = distance.cdist(current_centers, current_centers, 'euclidean')
        np.fill_diagonal(dists, np.inf)
        
        r_current = calc_radius(current_centers)
        
        # Determine active cutoff
        # We want to push apart circles that are limiting the radius.
        # Cutoff = 2 * r_current * 1.2 (some margin)
        cutoff = 2.0 * r_current * 1.2
        
        forces = np.zeros_like(current_centers)
        
        # Pairwise repulsion
        # F_i += (P_i - P_j) / dist^2 for dist < cutoff
        # Vectorized:
        # Create mask
        mask = dists < cutoff
        # Avoid division by zero (diagonal is inf, so mask is false there)
        # Force direction: (P_i - P_j)
        # dists^2
        dists_sq = dists ** 2
        # We need to sum over j for each i
        # F_i = sum_j (P_i - P_j) / dists[i,j]^2
        # This is equivalent to:
        # P_i * sum_j (1/dists[i,j]^2) - sum_j (P_j / dists[i,j]^2)
        
        # Let's compute the terms
        inv_dists_sq = np.zeros_like(dists)
        np.divide(1.0, dists_sq, out=inv_dists_sq, where=mask)
        
        sum_inv_dists_sq = np.sum(inv_dists_sq, axis=1)
        
        # Weighted sum of positions
        # weights[i,j] = inv_dists_sq[i,j]
        # sum_j weights[i,j] * P_j
        weighted_pos = inv_dists_sq @ current_centers
        
        forces += current_centers * sum_inv_dists_sq[:, np.newaxis] - weighted_pos
        
        # Boundary repulsion
        # If x_i > 0, and 0.5 - x_i < r_current * 1.2, push left
        # If x_i < 0, and x_i + 0.5 < r_current * 1.2, push right
        # Same for y
        
        bound_cutoff = r_current * 1.2
        
        # X direction
        # Right boundary
        dist_right = 0.5 - current_centers[:, 0]
        mask_right = dist_right < bound_cutoff
        if np.any(mask_right):
            # Force is negative x
            forces[mask_right, 0] -= 1.0 / (dist_right[mask_right] + 1e-6)
            
        # Left boundary
        dist_left = current_centers[:, 0] + 0.5
        mask_left = dist_left < bound_cutoff
        if np.any(mask_left):
            # Force is positive x
            forces[mask_left, 0] += 1.0 / (dist_left[mask_left] + 1e-6)

        # Y direction
        # Top boundary
        dist_top = 0.5 - current_centers[:, 1]
        mask_top = dist_top < bound_cutoff
        if np.any(mask_top):
            forces[mask_top, 1] -= 1.0 / (dist_top[mask_top] + 1e-6)
            
        # Bottom boundary
        dist_bottom = current_centers[:, 1] + 0.5
        mask_bottom = dist_bottom < bound_cutoff
        if np.any(mask_bottom):
            forces[mask_bottom, 1] += 1.0 / (dist_bottom[mask_bottom] + 1e-6)

        # Update positions
        # Normalize forces?
        # Forces can be large. Let's scale them.
        # lr = base_lr * (decay ** i)
        lr = base_lr * (decay ** i)
        
        new_centers = current_centers + lr * forces
        
        # Ensure they stay inside (hard clip)
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        # Check if improved
        new_radius = calc_radius(new_centers)
        
        if new_radius > current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
        else:
            # If not improved, maybe reduce step size or accept with probability (simulated annealing)
            # For now, just keep old centers and decay lr faster
            decay = 0.998
            # Or just don't update
            pass
            
        # Decay base lr slightly more aggressively if stuck?
        if i % 100 == 0:
            base_lr *= 0.95

    # 3. Polish Phase with SLSQP
    # We will use SLSQP to maximize r subject to constraints.
    # Objective: -r(x)
    # This is non-smooth. SLSQP can handle it if we provide gradients or if it approximates them well.
    # Better: Introduce r as a variable.
    # Maximize r
    # Subject to:
    # d_ij(x) - 2r >= 0
    # 0.5 - |x_i| - r >= 0 (approximated by 4 linear constraints per point)
    
    # Since SLSQP requires smooth functions, we will use the active set from the current best.
    # We define the objective as -r, and let SLSQP adjust x and r.
    # However, standard SLSQP minimizes. So we minimize -r.
    
    def objective(v):
        # v = [x_1, y_1, ..., x_N, y_N, r]
        x = v[:2*N].reshape(N, 2)
        r = v[-1]
        return -r

    def gradient(v):
        # Gradient of -r is 0 for x, -1 for r.
        # But r is not a free variable in the standard sense, it's constrained.
        # Actually, we can just let SLSQP handle the constraints.
        # The objective is simply -r.
        g = np.zeros_like(v)
        g[-1] = -1.0
        return g

    constraints = []
    
    # We need to generate constraints. 
    # To keep the number of constraints manageable for SLSQP (N=200 -> ~40k pairwise + 800 boundary),
    # we should only include "active" or "near-active" constraints.
    # Let's identify active constraints based on best_centers and best_radius.
    
    c = best_centers
    r = best_radius
    
    # Pairwise
    dists = distance.cdist(c, c, 'euclidean')
    np.fill_diagonal(dists, np.inf)
    
    # Active if dist < 2*r + tolerance
    tol = 1e-4
    active_pairs = []
    for i in range(N):
        for j in range(i+1, N):
            if dists[i, j] < 2*r + tol:
                active_pairs.append((i, j))
                
    # Boundary
    active_bounds = []
    for i in range(N):
        if 0.5 - c[i, 0] < r + tol: active_bounds.append((i, 'x+'))
        if c[i, 0] + 0.5 < r + tol: active_bounds.append((i, 'x-'))
        if 0.5 - c[i, 1] < r + tol: active_bounds.append((i, 'y+'))
        if c[i, 1] + 0.5 < r + tol: active_bounds.append((i, 'y-'))

    # Define constraints
    # d_ij - 2r >= 0  =>  2r - d_ij <= 0
    for i, j in active_pairs:
        def cons_pair(v, i=i, j=j):
            x = v[:2*N].reshape(N, 2)
            r = v[-1]
            d = np.linalg.norm(x[i] - x[j])
            return 2*r - d
        # Gradient:
        # d/dr (2r - d) = 2
        # d/dx_i (2r - d) = -(x_i - x_j)/d
        # d/dx_j (2r - d) = -(x_j - x_i)/d
        def grad_pair(v, i=i, j=j):
            x = v[:2*N].reshape(N, 2)
            r = v[-1]
            d = np.linalg.norm(x[i] - x[j])
            g = np.zeros_like(v)
            g[-1] = 2.0
            # x_i is at 2*i, 2*i+1
            g[2*i] = -(x[i, 0] - x[j, 0]) / d
            g[2*i+1] = -(x[i, 1] - x[j, 1]) / d
            # x_j is at 2*j, 2*j+1
            g[2*j] = -(x[j, 0] - x[i, 0]) / d
            g[2*j+1] = -(x[j, 1] - x[i, 1]) / d
            return g
        constraints.append({'type': 'ineq', 'fun': cons_pair, 'jac': grad_pair})

    # Boundary constraints:
    # 0.5 - |x| - r >= 0
    # Linear approximations:
    # x+ : 0.5 - x - r >= 0 => x + r - 0.5 <= 0
    # x- : 0.5 + x - r >= 0 => -x + r - 0.5 <= 0
    # y+ : 0.5 - y - r >= 0 => y + r - 0.5 <= 0
    # y- : 0.5 + y - r >= 0 => -y + r - 0.5 <= 0
    
    for i, typ in active_bounds:
        if typ == 'x+':
            def cons_b(v, i=i):
                x = v[:2*N].reshape(N, 2)
                r = v[-1]
                return x[i, 0] + r - 0.5
            def grad_b(v, i=i):
                g = np.zeros_like(v)
                g[2*i] = 1.0
                g[-1] = 1.0
                return g
            constraints.append({'type': 'ineq', 'fun': cons_b, 'jac': grad_b})
        elif typ == 'x-':
            def cons_b(v, i=i):
                x = v[:2*N].reshape(N, 2)
                r = v[-1]
                return -x[i, 0] + r - 0.5
            def grad_b(v, i=i):
                g = np.zeros_like(v)
                g[2*i] = -1.0
                g[-1] = 1.0
                return g
            constraints.append({'type': 'ineq', 'fun': cons_b, 'jac': grad_b})
        elif typ == 'y+':
            def cons_b(v, i=i):
                x = v[:2*N].reshape(N, 2)
                r = v[-1]
                return x[i, 1] + r - 0.5
            def grad_b(v, i=i):
                g = np.zeros_like(v)
                g[2*i+1] = 1.0
                g[-1] = 1.0
                return g
            constraints.append({'type': 'ineq', 'fun': cons_b, 'jac': grad_b})
        elif typ == 'y-':
            def cons_b(v, i=i):
                x = v[:2*N].reshape(N, 2)
                r = v[-1]
                return -x[i, 1] + r - 0.5
            def grad_b(v, i=i):
                g = np.zeros_like(v)
                g[2*i+1] = -1.0
                g[-1] = 1.0
                return g
            constraints.append({'type': 'ineq', 'fun': cons_b, 'jac': grad_b})

    # Initial guess for v
    v0 = np.concatenate([best_centers.ravel(), [best_radius]])
    
    # Bounds for x, y in [-0.5, 0.5], r >= 0
    bounds = [(-0.5, 0.5)] * (2*N) + [(0, 0.5)]
    
    # Run SLSQP
    try:
        res = minimize(objective, v0, method='SLSQP', jac=gradient, 
                       constraints=constraints, bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12})
        
        if res.success:
            x_opt = res.x[:2*N].reshape(N, 2)
            r_opt = res.x[-1]
            
            # Verify
            r_check = calc_radius(x_opt)
            if r_check > best_radius:
                best_centers = x_opt
                best_radius = r_check
    except Exception as e:
        pass

    # Final validity check
    if not np.all(np.isfinite(best_centers)):
        return np.zeros((N, 2))
    
    return best_centers

# EVOLVE-BLOCK-END
