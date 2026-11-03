"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if len(centers) < 2:
        return float(np.min(0.5 - np.abs(centers)))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair_dist = np.sqrt(np.min(sq_dists))
    
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _energy_and_grad(x, R, mu=100.0):
    """
    Compute energy and gradient for a fixed target radius R.
    Energy = sum of squared overlaps with neighbors and walls.
    Overlap pair = max(0, 2R - d_ij)
    Overlap wall = max(0, R - w_ik)
    We use a smooth approximation: 0.5 * (max(0, gap))^2 is C1 differentiable.
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # We only care about pairs where dist < 2R
    # To avoid singularities at dist=0, we assume distinct centers or add small epsilon
    # Actually, if dist is very small, gradient is huge, which is good for pushing apart.
    
    # Smooth overlap: f(d) = 0.5 * max(0, 2R - d)^2
    # grad_f wrt d = -(2R - d) if d < 2R else 0
    # grad_f wrt centers_i = grad_f_d * (centers_i - centers_j) / d
    
    # Wall distances
    walls = 0.5 - np.abs(centers)
    # Smooth overlap: g(w) = 0.5 * max(0, R - w)^2
    # grad_g wrt w = -(R - w) if w < R else 0
    # grad_g wrt centers_ik = grad_g_w * (-sign(centers_ik))
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise contributions
    # Mask for active pairs
    mask = dists < 2 * R
    # Exclude diagonal
    mask[np.diag_indices(N)] = False
    
    if np.any(mask):
        # We need to sum over all pairs i<j
        # To be efficient, we can iterate or use vectorization.
        # Vectorization over N^2 is fine for N=100.
        
        # dE/dcenters_i = sum_{j != i, active} -(2R - d_ij) * (centers_i - centers_j) / d_ij
        # Note: The term is symmetric.
        
        # Let's compute the force vector for each i
        # F_i = sum_{j} (2R - d_ij) * (centers_i - centers_j) / d_ij  (for d_ij < 2R)
        # This is the gradient of 0.5 * (2R - d)^2?
        # d/dx 0.5(2R - ||x-y||)^2 = -(2R - ||x-y||) * (x-y)/||x-y||
        # So gradient is negative of the "repulsive force" direction?
        # Wait, minimize energy. Energy is positive when overlapping.
        # If we move x away from y, d increases, (2R-d) decreases, energy decreases.
        # So gradient should point in direction of increasing d.
        # (x-y)/||x-y|| points from y to x. Moving x in this direction increases d.
        # So grad = sum (2R - d_ij) * (centers_i - centers_j) / d_ij
        
        # Let's verify signs.
        # E = 0.5 * (2R - d)^2. dE/dd = -(2R-d). 
        # d = ||x-y||. dd/dx = (x-y)/d.
        # dE/dx = -(2R-d) * (x-y)/d.
        # Since 2R > d, (2R-d) is positive. (x-y)/d is unit vector from y to x.
        # So dE/dx is in direction from y to x? No, negative sign.
        # If x is to the right of y, x-y is positive. dE/dx is negative.
        # This means increasing x (moving right) decreases energy?
        # If x moves right, d increases. (2R-d) decreases. E decreases. Correct.
        # So gradient points towards decreasing x?
        # L-BFGS-B minimizes. It moves x in direction -grad.
        # If grad is negative (points left), -grad is positive (points right).
        # Moving right increases d, reduces overlap. Correct.
        
        # So we compute:
        # grad_i = sum_j [ (2R - d_ij) * (centers_i - centers_j) / d_ij ] for d_ij < 2R
        
        # Vectorized approach:
        # We have dists (N, N).
        # We want to compute forces.
        # Force on i from j: (2R - d_ij) * (c_i - c_j) / d_ij
        # Let's create a weight matrix W where W_ij = (2R - d_ij) / d_ij if d_ij < 2R else 0.
        # Then grad_i = sum_j W_ij * (c_i - c_j) = c_i * sum_j W_ij - sum_j W_ij * c_j
        
        W = np.zeros((N, N))
        # Avoid division by zero
        safe_dists = np.where(dists > 1e-12, dists, 1.0)
        W[mask] = (2 * R - dists[mask]) / safe_dists[mask]
        
        sum_W = np.sum(W, axis=1)
        weighted_centers = W @ centers
        
        grad_pair = centers * sum_W[:, np.newaxis] - weighted_centers
        
        # Energy
        overlaps = np.where(mask, 2 * R - dists, 0.0)
        energy_pair = 0.5 * np.sum(overlaps**2)
        
    else:
        grad_pair = np.zeros_like(centers)
        energy_pair = 0.0

    # Wall contributions
    # E_wall = sum_i,k 0.5 * max(0, R - w_ik)^2
    # w_ik = 0.5 - |c_ik|
    # dE/dw = -(R - w) if w < R
    # dw/dc_ik = -sign(c_ik)
    # dE/dc_ik = -(R - w) * (-sign(c_ik)) = (R - w) * sign(c_ik)
    
    mask_wall = walls < R
    if np.any(mask_wall):
        # Overlap amount
        overlap_wall = np.where(mask_wall, R - walls, 0.0)
        # Gradient
        signs = np.sign(centers)
        signs[centers == 0] = 1.0
        grad_wall = overlap_wall * signs
        
        energy_wall = 0.5 * np.sum(overlap_wall**2)
    else:
        grad_wall = np.zeros_like(centers)
        energy_wall = 0.0
        
    total_grad = grad_pair + grad_wall
    total_energy = energy_pair + energy_wall
    
    return total_energy, total_grad.flatten()

def _optimize_for_radius(centers, R, time_limit=10.0):
    """
    Minimize energy for a fixed radius R starting from centers.
    Returns (success, optimized_centers, energy).
    """
    start_time = time.time()
    x0 = centers.flatten()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        val, grad = _energy_and_grad(x, R)
        return val, grad
        
    try:
        result = minimize(
            obj, x0,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        
        # Check if we actually reduced energy significantly
        final_energy, _ = _energy_and_grad(result.x, R)
        
        # Success if energy is very low (spheres fit)
        # Threshold depends on N and R. 1e-6 is a good start.
        success = final_energy < 1e-6
        
        return success, result.x.reshape(N, 3), final_energy
    except Exception:
        return False, centers.copy(), float('inf')

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _bcc_init(n):
    """Generate body-centered cubic lattice initial configuration."""
    m = int(np.ceil((n / 2) ** (1.0/3.0)))
    spacing = 1.0 / m
    centers = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                x = (ix + 0.5) * spacing - 0.5
                y = (iy + 0.5) * spacing - 0.5
                z = (iz + 0.5) * spacing - 0.5
                centers.append([x, y, z])
                if ix < m-1 and iy < m-1 and iz < m-1:
                    bx = (ix + 1.0) * spacing - 0.5
                    by = (iy + 1.0) * spacing - 0.5
                    bz = (iz + 1.0) * spacing - 0.5
                    centers.append([bx, by, bz])
    
    centers = np.array(centers)
    if len(centers) > n:
        centers = centers[:n]
    elif len(centers) < n:
        grid = _grid_init(n)
        centers = np.vstack([centers, grid[len(centers):n]])
    
    return centers

def _random_init(n, seed=None):
    """Generate random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def _polish_critical_pairs(centers, time_limit=5.0):
    """Final high-precision polish focusing only on critical (closest) pairs."""
    start_time = time.time()
    centers = centers.copy()
    
    for iteration in range(300):
        if time.time() - start_time > time_limit:
            break
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        min_sq_dist = np.min(sq_dists)
        min_pair_dist = np.sqrt(min_sq_dist)
        min_pair_idx = np.unravel_index(np.argmin(sq_dists), sq_dists.shape)
        
        min_wall_dist = np.min(0.5 - np.abs(centers))
        min_wall_idx = np.unravel_index(np.argmin(0.5 - np.abs(centers)), (N, 3))
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if min_pair_dist / 2.0 < min_wall_dist:
            i, j = min_pair_idx
            if i > j:
                i, j = j, i
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.array([1e-6, 0, 0])
                dist = 1e-6
            push = (min_pair_dist / 2.0 - r) * d / dist * 0.5
            if push > 0:
                centers[i] += push
                centers[j] -= push
        else:
            i, k = min_wall_idx
            wall = 0.5 - np.abs(centers[i, k])
            if wall < r:
                sign = np.sign(centers[i, k])
                if sign == 0:
                    sign = 1.0
                centers[i, k] += (r - wall) * sign
        
        centers = np.clip(centers, -0.5, 0.5)
    
    return centers

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # 1. Generate initial configurations
    inits = []
    inits.append(_grid_init(N))
    inits.append(_bcc_init(N))
    for seed in range(5):
        inits.append(_random_init(N, seed=seed))
    
    best_centers = None
    best_radius = 0.0
    
    # 2. Evaluate inits
    for init in inits:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()
    
    # 3. Iterative Radius Increase
    # Start from a radius slightly smaller than the best known or current best
    # The best known is ~0.11138. Grid is ~0.1.
    # We will try to push R up.
    
    current_radius = best_radius * 0.95
    step_size = 0.001
    
    # We keep a history of successful centers to allow backtracking if needed?
    # For now, simple greedy: if R fails, we stop or reduce step?
    # Better: Try to find max R for each init, then pick best.
    
    # Given time constraints, we focus on the best init so far.
    # We can restart from best_centers if we get stuck.
    
    active_centers = best_centers.copy()
    
    while time.time() < deadline - 20.0:
        # Try to increase radius
        target_R = current_radius + step_size
        
        # Check if target_R is physically possible (volume bound)
        # Volume of 100 spheres = 100 * 4/3 pi R^3 <= 1
        # R^3 <= 3 / (400 pi) ~ 0.00238 -> R ~ 0.133
        # So 0.111 is well within volume bound.
        
        # Optimize for this R
        success, new_centers, energy = _optimize_for_radius(active_centers, target_R, time_limit=15.0)
        
        if success:
            # Check actual radius
            actual_r = _compute_radius(new_centers)
            if actual_r >= target_R - 1e-5:
                # Accept
                active_centers = new_centers
                current_radius = target_R
                best_radius = max(best_radius, actual_r)
                best_centers = new_centers.copy()
                
                # Try larger steps if we are doing well?
                # Keep step size constant for stability
            else:
                # Energy was low but radius didn't match? Maybe numerical issue.
                # Try smaller step
                step_size = max(step_size * 0.5, 1e-4)
        else:
            # Failed to find a packing for this R
            # Try smaller step
            step_size = max(step_size * 0.5, 1e-4)
            
            # If step size is too small, we are stuck
            if step_size < 1e-4:
                # Try a different initialization or break
                # Let's try restarting from best_centers with a fresh random jitter
                # But we are limited in time.
                
                # Try one more time with a slightly smaller R
                target_R = current_radius + 0.5 * step_size
                success, new_centers, energy = _optimize_for_radius(active_centers, target_R, time_limit=10.0)
                if success:
                    actual_r = _compute_radius(new_centers)
                    if actual_r >= target_R - 1e-5:
                        active_centers = new_centers
                        current_radius = target_R
                        best_radius = max(best_radius, actual_r)
                        best_centers = new_centers.copy()
                        step_size = max(step_size, 1e-4) # Reset step size
                    else:
                        break
                else:
                    break
            
            # If we just reduced step size, continue loop to try again
            # But if we reduced step size multiple times, we might be in a local minimum
            # A simple strategy: if we fail, try to "shake" the configuration
            # Add small noise to active_centers
            if np.random.rand() < 0.1:
                noise = np.random.normal(0, 0.01, size=active_centers.shape)
                active_centers = np.clip(active_centers + noise, -0.5, 0.5)
    
    # 4. Final Polish
    if time.time() < deadline - 5.0:
        polished = _polish_critical_pairs(best_centers, time_limit=4.0)
        r_polished = _compute_radius(polished)
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = polished
    
    # Final validation and clipping
    if best_centers is None:
        best_centers = _grid_init(N)
    
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    return best_centers
# EVOLVE-BLOCK-END
