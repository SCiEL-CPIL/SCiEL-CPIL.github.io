import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5
TOTAL_TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    # min_i min_k (0.5 - |c_ik|)
    abs_centers = np.abs(centers)
    min_coord_dist = HALF - np.max(abs_centers, axis=1)
    min_wall_dist = np.min(min_coord_dist)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    # Using pdist or manual calculation. For N=100, manual is fine, but let's be efficient.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return float(r)

def _violation_energy_and_grad(x, R):
    """
    Calculates the sum of squared violations of the packing constraints for a target radius R.
    Constraints:
    1. Pairwise: ||c_i - c_j|| >= 2R
    2. Wall: 0.5 - |c_ik| >= R  =>  |c_ik| <= 0.5 - R
    
    Energy E = sum_{i<j, d_ij < 2R} (2R - d_ij)^2 + sum_{i,k, |c_ik| > 0.5-R} (|c_ik| - (0.5-R))^2
    """
    centers = x.reshape(N, 3)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq + 1e-15) # Avoid 0 division
    np.fill_diagonal(dists, np.inf)
    
    # We only care about pairs where distance < 2R
    mask = dists < 2.0 * R
    if np.any(mask):
        i_idx, j_idx = np.where(mask)
        # Ensure i < j to avoid double counting
        valid = i_idx < j_idx
        i_idx = i_idx[valid]
        j_idx = j_idx[valid]
        
        d_ij = dists[i_idx, j_idx]
        violations = 2.0 * R - d_ij
        # Gradient of (2R - d)^2 is -2(2R-d) * d(d)/d(x)
        # d(d)/d(x_i) = (x_i - x_j) / d
        # So grad_x_i += -2 * violations * (x_i - x_j) / d
        # grad_x_j += -2 * violations * (x_j - x_i) / d
        
        factors = 2.0 * violations / d_ij
        vecs = centers[i_idx] - centers[j_idx]
        forces = factors[:, np.newaxis] * vecs
        
        np.add.at(grad, i_idx, -forces)
        np.add.at(grad, j_idx, forces)
        
        energy += np.sum(violations**2)

    # 2. Wall terms
    # Constraint: |c_ik| <= HALF - R
    limit = HALF - R
    if limit < 0:
        # R is too large to fit even at center, infinite penalty
        return 1e10, grad.flatten()
        
    abs_centers = np.abs(centers)
    # Find coordinates that exceed the limit
    mask_wall = abs_centers > limit
    if np.any(mask_wall):
        i_idx, k_idx = np.where(mask_wall)
        c_vals = centers[i_idx, k_idx]
        abs_vals = abs_centers[i_idx, k_idx]
        violations = abs_vals - limit
        
        # Gradient of (|c| - L)^2 is 2(|c|-L) * sign(c)
        signs = np.sign(c_vals)
        # If c is 0, sign is 0, but violation implies c != 0, so safe.
        grad_vals = 2.0 * violations * signs
        
        np.add.at(grad, (i_idx, k_idx), grad_vals)
        
        energy += np.sum(violations**2)
        
    return energy, grad.flatten()

def _minimize_violation(centers_init, target_R, time_limit=5.0):
    """
    Minimizes the violation energy for a given target radius R.
    Returns the optimized centers and the final energy.
    """
    x0 = centers_init.flatten()
    bounds = [(-HALF, HALF)] * (N * 3)
    
    def objective_and_grad(x):
        val, g = _violation_energy_and_grad(x, target_R)
        return val, g

    start_t = time.time()
    stop_flag = [False]
    
    def callback(xk):
        if time.time() > start_t + time_limit:
            stop_flag[0] = True
        return stop_flag[0]
        
    try:
        res = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 500,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'maxls': 50,
                'callback': callback,
                'disp': False
            }
        )
        best_x = res.x
        best_x = np.clip(best_x, -HALF, HALF)
        final_energy, _ = _violation_energy_and_grad(best_x, target_R)
        return best_x.reshape(N, 3), final_energy
    except Exception:
        return centers_init, np.inf

def _generate_initial_positions():
    """
    Generate initial positions using an FCC lattice structure.
    FCC is denser than simple cubic.
    """
    inits = []
    
    # FCC lattice generation
    # Unit cell side length 'a'.
    # In a unit cube [-0.5, 0.5], we can fit a lattice.
    # Let's try a 4x4x4 grid of FCC unit cells? Or just generate points and select best N.
    # Simpler: Generate a dense grid of FCC points, then select the N that are most central or randomly.
    
    # FCC points in a unit cell [0,1]:
    # (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    
    # Let's create a supercell of 4x4x4 unit cells.
    # Total points: 64 * 4 = 256.
    # We need 100.
    
    # To center the lattice in [-0.5, 0.5], we shift the supercell.
    # Supercell size: 4.
    # Shift: -2.0 to center at 0?
    # If supercell is [0, 4], center is 2.
    # We want center at 0. So shift by -2.
    # Points will be in [-2, 2].
    # We need points in [-0.5, 0.5].
    # So we should scale the lattice or choose a smaller supercell.
    
    # Let's use a 3x3x3 supercell. 27 * 4 = 108 points.
    # Supercell size 3.
    # Shift by -1.5. Points in [-1.5, 1.5].
    # We want to fit in [-0.5, 0.5].
    # We can scale the coordinates by 1/3.
    # Then points are in [-0.5, 0.5].
    # Distance between nearest neighbors in FCC with unit cell 'a' is a/sqrt(2).
    # If a=1, d = 0.707.
    # If we scale by 1/3, a=1/3, d = 0.235.
    # Radius r = d/2 = 0.117.
    # This is close to the optimum 0.111.
    
    # Let's generate the 3x3x3 FCC lattice.
    lattice_points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Base corner
                x_base = i
                y_base = j
                z_base = k
                
                # FCC offsets
                offsets = [
                    (0, 0, 0),
                    (0.5, 0.5, 0),
                    (0.5, 0, 0.5),
                    (0, 0.5, 0.5)
                ]
                
                for ox, oy, oz in offsets:
                    px = x_base + ox
                    py = y_base + oy
                    pz = z_base + oz
                    lattice_points.append([px, py, pz])
                    
    lattice_points = np.array(lattice_points)
    # Scale to fit in [-0.5, 0.5]
    # Current range: [0, 3]. Center 1.5.
    # Shift to center at 0: [-1.5, 1.5]
    lattice_points -= 1.5
    # Scale by 1/3: [-0.5, 0.5]
    lattice_points /= 3.0
    
    # We have 108 points. We need 100.
    # The points on the boundary might be less optimal or cause issues.
    # Let's just take the first 100.
    init_fcc = lattice_points[:N].copy()
    inits.append(init_fcc)
    
    # Add a jittered version
    rng = np.random.RandomState(42)
    jitter = rng.normal(0, 0.01, (N, 3))
    init_jittered = np.clip(init_fcc + jitter, -0.499, 0.499)
    inits.append(init_jittered)
    
    # Add a random uniform version as fallback
    init_rand = rng.uniform(-0.45, 0.45, (N, 3))
    inits.append(init_rand)
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + TOTAL_TIME_LIMIT
    
    best_radius = 0.0
    best_centers = None
    
    inits = _generate_initial_positions()
    
    # Phase 1: Establish a baseline
    # Use the best init to find a feasible R
    # Start with a small R to ensure we find a valid config
    # Theoretical max is ~0.111.
    # Let's start with R = 0.05
    
    # We will use a binary search on R.
    # Low: 0.0
    # High: 0.112 (slightly above known best)
    
    R_low = 0.0
    R_high = 0.112
    
    # First, ensure we have a valid center for R_low
    # Actually, R_low=0 is always valid.
    # Let's pick the init with the highest initial radius as the starting point for optimization.
    # Or just use the FCC one.
    
    current_centers = inits[0].copy()
    current_R = _compute_radius(current_centers)
    if current_R > best_radius:
        best_radius = current_R
        best_centers = current_centers.copy()
        
    # Update R_low to the best found so far
    R_low = max(R_low, best_radius)
    
    # Binary Search Loop
    # We have ~120s.
    # Each minimize takes ~2-5s.
    # We can do ~20-30 iterations.
    
    iterations = 0
    max_iterations = 25
    
    while time.time() < deadline - 10.0 and iterations < max_iterations:
        iterations += 1
        
        R_mid = (R_low + R_high) / 2.0
        
        # If the range is small enough, stop
        if R_high - R_low < 1e-6:
            break
            
        # Try to find a valid configuration for R_mid
        # Use the best known centers as the initial guess for the new R
        # This is a "continuation" method, which is very effective.
        
        time_left = deadline - time.time() - 5.0
        if time_left < 2.0:
            break
            
        t_limit = min(4.0, time_left * 0.9)
        
        opt_centers, energy = _minimize_violation(best_centers, R_mid, time_limit=t_limit)
        
        # Check if the result is valid for R_mid
        # Energy should be close to 0.
        # Also compute the actual radius to be sure.
        r_actual = _compute_radius(opt_centers)
        
        if r_actual >= R_mid - 1e-5: # Tolerance for floating point
            # Success! We found a config that fits radius R_mid
            R_low = R_mid
            if r_actual > best_radius:
                best_radius = r_actual
                best_centers = opt_centers.copy()
        else:
            # Failed to fit R_mid
            R_high = R_mid
            # Keep best_centers as is (it's valid for R_low)
            
    # Phase 2: Final Polish
    # We have a good R_low.
    # Let's try to squeeze a bit more by running minimize with R_low + epsilon
    # Or just run a high-precision minimize on R_low.
    
    if time.time() < deadline - 2.0 and best_centers is not None:
        # Try to improve the current best radius slightly
        # Use a slightly higher target to push boundaries
        R_target = best_radius + 1e-5
        time_left = deadline - time.time() - 1.0
        if time_left > 1.0:
            polished_centers, _ = _minimize_violation(best_centers, R_target, time_limit=time_left)
            r_polished = _compute_radius(polished_centers)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished_centers
    
    if best_centers is None:
        # Fallback
        best_centers = inits[0]
        best_radius = _compute_radius(best_centers)
        
    return best_centers
