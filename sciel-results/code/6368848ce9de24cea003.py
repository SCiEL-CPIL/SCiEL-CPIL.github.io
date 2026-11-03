import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _energy_and_grad(x, w, h, r_init):
    """
    Calculate the energy and gradient for the packing problem.
    x: array of shape (2*N,). x[2*i] = cx, x[2*i+1] = cy.
    Radii are fixed at r_init for the purpose of finding a valid packing.
    If a valid packing exists for these radii, we can then increase radii?
    No, the standard trick is: 
    Minimize E(x, r) = sum of constraint violations.
    But we want to maximize sum(r).
    
    Alternative formulation:
    Fix the rectangle. Initialize radii to a large value R_max.
    Minimize E = sum_{constraints} (violation)^2.
    The optimizer will reduce the "effective" overlap by moving centers.
    However, if we fix r, we can't change r.
    
    Correct formulation for "Maximize sum of radii":
    We treat radii as variables too? Or we do a binary search on a common scale?
    
    Actually, the most robust way is:
    1. Start with a guess for radii.
    2. Minimize the "overlap energy" where energy is the sum of squared overlaps and wall violations.
    3. If energy is ~0, the packing is valid.
    4. To maximize the sum, we can start with large radii and let the optimizer move the centers to fit them.
       If they don't fit, the energy will be high.
       We want to find the largest set of radii that fits.
       
    A better strategy:
    Since the problem is "maximize sum of radii", and the constraints are convex (mostly), we can use the following:
    Let r_i be the variables.
    Objective: Maximize sum(r_i).
    Constraints: 
      1. r_i + r_j <= dist(i,j)
      2. r_i <= x_i, r_i <= w-x_i, etc.
      
    This is a non-convex optimization problem because of the dist(i,j) term (hyperbolic constraint).
    However, for a fixed topology (which circles touch which), it is convex.
    
    The "Soft Body" approach:
    Define F(x, r) = sum_i [ max(0, r_i - x_i)^2 + ... ] + sum_{i<j} [ max(0, r_i + r_j - d_{ij})^2 ]
    We want to minimize F.
    If we start with large r, the gradient will push centers apart and shrink radii if they are also variables.
    
    Let's make r variables too.
    State vector: [x_1, y_1, r_1, ..., x_N, y_N, r_N]
    
    Energy:
    E = sum_i (penalty for walls) + sum_{i<j} (penalty for overlap)
    
    We want to MINIMIZE E.
    But we also want to MAXIMIZE sum(r).
    These are conflicting if we just minimize E (E=0 is achieved by r=0).
    
    So we use a Lagrangian or a two-step process?
    Actually, the standard trick is:
    Minimize E(x, r) subject to sum(r) >= target? No.
    
    Let's use the fact that for a valid packing, E=0.
    We can perform a "pressure" test.
    Start with r_i = R0.
    Minimize E(x) for fixed r.
    If min E(x) > 0, the radii are too big.
    We can scale down all radii and retry?
    
    Better: 
    Use an objective function: 
    J(x, r) = -sum(r) + lambda * E(x, r)
    We want to find the maximum sum(r) such that E(x, r) = 0.
    
    This is effectively a constrained optimization.
    We can use `scipy.optimize.minimize` with method='SLSQP' or 'BFGS' with penalties.
    
    Let's try a simpler heuristic that works well:
    1. Initialize positions randomly.
    2. Initialize radii to a reasonable guess (e.g., based on area).
    3. Use Gradient Descent to minimize the "overlap energy" ONLY.
       Wait, if we only minimize overlap energy, radii will shrink to 0.
       
    Correct approach:
    We want to maximize sum(r).
    Let's fix the aspect ratio w.
    We perform a binary search on a "scale factor" S?
    No, radii can be different.
    
    Let's use the following robust method:
    1. Randomize centers.
    2. Set all radii to a large value, e.g., 0.5.
    3. Minimize the "penalty" function P(x, r) = sum of squared violations.
       The gradient will adjust x and r to reduce violations.
       Since we want to maximize r, we should add a term to the objective that encourages large r?
       
    Actually, the "Soft Body" packing algorithm usually works like this:
    - Start with small radii.
    - Iteratively increase radii.
    - Relax positions to minimize overlap.
    
    Let's implement this iterative relaxation:
    1. Start with small r (e.g., 0.01).
    2. While time remains:
       a. Increase all r by a small delta (or scale them up).
       b. Run a few steps of Gradient Descent on positions x only, to minimize overlap with the current r.
       c. If convergence is good, continue.
    
    This is effectively "Maximize sum(r) subject to non-overlap".
    
    Let's refine the GD step.
    Energy E(x) = sum_{i<j} max(0, r_i + r_j - d_{ij})^2 + sum_i (wall violations)^2.
    Gradient w.r.t x_i:
    dE/dx_i = -2 * sum_{j} max(0, r_i + r_j - d_{ij}) * (x_i - x_j) / d_{ij}  (for overlap term)
             + wall gradients.
             
    This is a standard repulsive force model.
    """
    n = N
    x = x[:2*n]
    r = x[2*n:]
    
    # Compute distances
    # Vectorized distance calculation
    # x is (2n,). centers = x.reshape(n, 2)
    centers = x[:2*n].reshape(n, 2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1e-10) # Avoid div by zero
    
    # Overlap matrix
    # overlap_{ij} = r_i + r_j - dist_{ij}
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    overlap = r_sum - dist
    
    # Only consider positive overlaps
    pos_overlap = np.maximum(overlap, 0)
    
    # Energy from overlaps
    E_overlap = np.sum(pos_overlap**2)
    
    # Gradient of overlap energy w.r.t centers
    # d/dx_i sum_{j} (r_i + r_j - d_{ij})^2
    # = 2 * sum_{j} (r_i + r_j - d_{ij}) * (-(x_i - x_j)/d_{ij})
    # Note: we only sum over j where overlap > 0
    
    grad_x = np.zeros_like(centers)
    
    # For each pair i, j with overlap
    # We can vectorize this using the overlap matrix
    # Force on i from j: F_{ij} = 2 * pos_overlap_{ij} * (x_i - x_j) / dist_{ij}
    # Wait, derivative of (r_i + r_j - d) is -d(d)/dx_i = -(x_i - x_j)/d.
    # So term is 2 * (r_i + r_j - d) * (-(x_i - x_j)/d)
    # We want to MINIMIZE energy, so we move AGAINST the gradient.
    # Gradient points in direction of increasing energy.
    # If overlap > 0, energy increases as they get closer.
    # So gradient should point towards each other? No, away.
    
    # Let's compute the force (negative gradient) directly for clarity in GD.
    # Force on i due to j: 
    # If overlapping, they repel.
    # Force vector on i: 2 * pos_overlap_{ij} * (x_i - x_j) / dist_{ij}
    
    # Vectorized:
    # pos_overlap is (n, n)
    # diff is (n, n, 2)
    # dist is (n, n)
    
    # We need to sum forces from all j on i.
    # force_i = sum_j 2 * pos_overlap_{ij} * diff_{ij} / dist_{ij}
    
    # Handle dist=0
    safe_dist = np.where(dist > 1e-8, dist, 1e-8)
    force_factor = 2 * pos_overlap / safe_dist
    # Set diagonal to 0
    np.fill_diagonal(force_factor, 0)
    
    # force = force_factor @ diff ? No.
    # force_i = sum_j force_factor_{ij} * diff_{ij}
    # diff_{ij} = x_i - x_j
    # force = np.sum(force_factor[:, :, np.newaxis] * diff, axis=1)
    
    force_overlap = np.sum(force_factor[:, :, np.newaxis] * diff, axis=1)
    
    # Wall constraints
    # E_wall = sum_i max(0, r_i - x_i)^2 + max(0, r_i - (w-x_i))^2 + ...
    # Gradient w.r.t x_i:
    # d/dx_i max(0, r_i - x_i)^2 = 2(r_i - x_i) * (-1) = 2(x_i - r_i) if x_i < r_i else 0
    # d/dx_i max(0, r_i - (w-x_i))^2 = 2(r_i - (w-x_i)) * (1) = 2(r_i - w + x_i) if x_i > w-r_i else 0
    
    grad_wall = np.zeros_like(centers)
    
    # Left wall
    viol_l = np.maximum(r - centers[:, 0], 0)
    grad_wall[:, 0] += -2 * viol_l # Force pushes right (positive x) if violating left? 
    # Wait. Energy increases as x goes left (towards 0). 
    # dE/dx = -2(r-x). Gradient points left. 
    # We want to move in direction of -Gradient. So -(-2(r-x)) = 2(r-x) -> pushes right. Correct.
    
    # Right wall
    viol_r = np.maximum(r - (w - centers[:, 0]), 0)
    grad_wall[:, 0] += 2 * viol_r # dE/dx = 2(r - (w-x)) = 2(r - w + x). 
    # If x > w-r, viol > 0. Gradient is positive (pushes right). 
    # We want to move left. So force = -grad = -2(viol). 
    # My previous line added +2*viol to grad_wall. 
    # Let's stick to calculating Gradient, then doing x = x - step * Gradient.
    
    # Recalculate Gradients carefully.
    # E = sum (viol)^2
    # dE/dx_i = 2 * viol * d(viol)/dx_i
    
    # Left: viol = max(0, r - x). d(viol)/dx = -1 if x < r.
    # Grad_x += 2 * viol * (-1) = -2 * viol.
    
    # Right: viol = max(0, r - (w-x)) = max(0, r - w + x). d(viol)/dx = 1 if x > w-r.
    # Grad_x += 2 * viol * (1) = 2 * viol.
    
    # Bottom: viol = max(0, r - y). Grad_y += -2 * viol.
    # Top: viol = max(0, r - (h-y)). Grad_y += 2 * viol.
    
    grad_wall[:, 0] = -2 * np.maximum(r - centers[:, 0], 0) + 2 * np.maximum(r - (w - centers[:, 0]), 0)
    grad_wall[:, 1] = -2 * np.maximum(r - centers[:, 1], 0) + 2 * np.maximum(r - (h - centers[:, 1]), 0)
    
    # Overlap Gradient
    # E_overlap = sum_{i<j} (r_i + r_j - d_{ij})^2
    # dE/dx_i = sum_{j != i} 2(r_i + r_j - d_{ij}) * (-(x_i - x_j)/d_{ij})
    # = -2 * sum_j pos_overlap_{ij} * (x_i - x_j) / d_{ij}
    # Note: pos_overlap is 0 if no overlap.
    
    # My previous force calculation was for the FORCE (negative gradient).
    # force_overlap = sum_j 2 * pos_overlap_{ij} * (x_i - x_j) / d_{ij}
    # So Gradient = -force_overlap.
    
    total_grad = grad_wall - force_overlap
    
    # Total Energy
    E_wall = np.sum(np.maximum(r - centers[:, 0], 0)**2) + \
             np.sum(np.maximum(r - (w - centers[:, 0]), 0)**2) + \
             np.sum(np.maximum(r - centers[:, 1], 0)**2) + \
             np.sum(np.maximum(r - (h - centers[:, 1]), 0)**2)
             
    E = E_overlap + E_wall
    
    # We also need gradient w.r.t r?
    # If we treat r as fixed in the inner loop, we don't need it.
    # We will update r separately.
    
    return E, total_grad

def _pack_single(w, h, time_limit, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    n = N
    # Initial positions: random
    x = np.random.uniform(0.1, w - 0.1, n)
    y = np.random.uniform(0.1, h - 0.1, n)
    centers = np.column_stack([x, y])
    
    # Initial radii: small
    r = np.full(n, 0.05)
    
    best_score = 0.0
    best_state = None
    
    start_time = time.time()
    
    # Parameters
    step_size = 0.05
    grow_rate = 0.001
    max_iter = 10000
    
    current_energy = 1e10
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Combine state
        state = np.concatenate([centers.flatten(), r])
        
        # Compute energy and gradient (w.r.t centers only)
        E, grad = _energy_and_grad(state, w, h, r)
        
        if E < 1e-6:
            # Valid packing
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_state = state.copy()
                
            # Try to grow radii
            # Grow all radii slightly
            r += grow_rate
            # Reset step size?
            step_size = 0.05
            continue
            
        # Update centers
        # Adaptive step size
        # If energy increased, reduce step. If decreased, increase.
        new_centers = centers - step_size * grad
        
        # Clip to bounds to help convergence
        new_centers[:, 0] = np.clip(new_centers[:, 0], 0, w)
        new_centers[:, 1] = np.clip(new_centers[:, 1], 0, h)
        
        # Check if move improved energy
        state_new = np.concatenate([new_centers.flatten(), r])
        E_new, _ = _energy_and_grad(state_new, w, h, r)
        
        if E_new < current_energy:
            centers = new_centers
            current_energy = E_new
            step_size = min(step_size * 1.05, 0.5)
        else:
            step_size *= 0.5
            if step_size < 1e-6:
                # Stuck? Try growing radii? No, if stuck with high energy, radii are too big.
                # Shrink radii?
                # In this formulation, we are trying to fit the CURRENT r.
                # If we can't fit, we are stuck.
                # We should have shrunk r if we couldn't fit.
                # But we only grow r when E < 1e-6.
                # So if we are stuck with E > 1e-6, it means current r is too big for this configuration.
                # We should backtrack? Or just accept that this r is not feasible and stop?
                # Actually, if we start small and grow, we should always be feasible.
                # If we get stuck, it's a local minimum in the "configuration space" for fixed r.
                # We can try a small random jitter to escape.
                centers += np.random.uniform(-0.01, 0.01, centers.shape)
                step_size = 0.01

    if best_state is not None:
        # Reconstruct result
        best_centers = best_state[:2*n].reshape(n, 2)
        best_r = best_state[2*n:]
        return np.column_stack([best_centers, best_r]), np.sum(best_r)
    else:
        # Return the last valid state if we had one, or a fallback
        # If we never got E < 1e-6, the packing is invalid.
        # We should return a valid packing with small radii.
        # Let's just return the initial small radii packing if it was valid.
        # The initial state was random with r=0.05. It might be valid.
        state_init = np.concatenate([centers.flatten(), r])
        E_init, _ = _energy_and_grad(state_init, w, h, r)
        if E_init < 1e-6:
             return np.column_stack([centers, r]), np.sum(r)
        return None, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140 # Safety margin for 150s limit
    
    best_result = None
    best_score = 0.0
    
    # Aspect ratios to try
    # Perimeter 4 => w + h = 2 => h = 2 - w
    # Optimal aspect ratio for circle packing is often close to 1, but can vary.
    # We will sweep w from 0.8 to 1.2.
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.85, 1.15]
    
    # Sort by likelihood? 1.0 is usually good.
    # We'll just iterate.
    
    for w in aspect_ratios:
        if time.time() > deadline:
            break
            
        h = 2.0 - w
        
        # Time remaining
        time_remaining = deadline - time.time()
        if time_remaining < 10:
            continue
            
        # Allocate time for this aspect ratio
        # We'll run multiple seeds for the most promising ones (1.0, 0.9, 1.1)
        num_seeds = 3 if w in [1.0, 0.9, 1.1] else 1
        time_per_seed = (time_remaining - 5) / num_seeds
        
        for s in range(num_seeds):
            if time.time() > deadline:
                break
                
            seed = int(time.time() * 1000) + s + int(w * 100)
            try:
                result, score = _pack_single(w, h, time_per_seed, seed)
                if result is not None and score > best_score:
                    best_score = score
                    best_result = result
            except Exception:
                pass

    if best_result is None:
        # Fallback: Simple grid
        w, h = 1.0, 1.0
        positions = []
        radii = []
        cols = 5
        rows = 5
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pos = [(i + 0.5) * w / cols, (j + 0.5) * h / rows]
                r = 0.5 * min(w/cols, h/rows) * 0.9
                positions.append(pos)
                radii.append(r)
                count += 1
        positions = np.array(positions)
        radii = np.array(radii)
        best_result = np.column_stack([positions, radii])
        
    return best_result
