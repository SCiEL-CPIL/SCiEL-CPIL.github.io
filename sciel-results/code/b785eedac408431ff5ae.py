"""
Improved packing algorithm using iterative radius increase and violation minimization.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        return float(np.min(0.5 - np.abs(centers)))
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _energy_and_gradient(centers, r, stiffness=100.0):
    """
    Compute energy and gradient for a fixed radius r.
    Energy is the sum of squared violations of constraints:
    1. Pairwise distance >= 2r
    2. Distance to wall >= r
    """
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise violations
    # We only need to check pairs, i < j
    for i in range(n):
        for j in range(i + 1, n):
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 2.0 * r:
                # Violation amount
                viol = 2.0 * r - dist
                # Energy contribution: 0.5 * (stiffness * viol)^2 ? 
                # Let's use E = 0.5 * k * viol^2
                # dE/ddist = -k * viol
                # d(dist)/d(centers[i]) = d / dist
                # dE/d(centers[i]) = -k * viol * (d / dist)
                # dE/d(centers[j]) = -k * viol * (-d / dist) = k * viol * (d / dist)
                
                # To avoid division by zero if dist is extremely small, add epsilon
                if dist < 1e-12:
                    # If centers are identical, push them apart arbitrarily? 
                    # Gradient is undefined. We can add a small random kick or just return high energy.
                    # For stability, let's assume a tiny perturbation direction or just high energy.
                    energy += 1e10
                    continue
                
                k = stiffness
                force_mag = k * viol
                force_vec = force_mag * d / dist
                
                grad[i] -= force_vec
                grad[j] += force_vec
                energy += 0.5 * k * viol * viol
                
    # 2. Wall violations
    # Constraint: 0.5 - |c_ik| >= r  => |c_ik| <= 0.5 - r
    # Let max_coord = 0.5 - r
    max_coord = 0.5 - r
    
    for i in range(n):
        for k in range(3):
            c = centers[i, k]
            abs_c = np.abs(c)
            if abs_c > max_coord:
                viol = abs_c - max_coord
                # E = 0.5 * k * viol^2
                # dE/d(abs_c) = k * viol
                # d(abs_c)/d(c) = sign(c)
                # dE/d(c) = k * viol * sign(c)
                
                sign = np.sign(c)
                if sign == 0:
                    sign = 1.0
                grad[i, k] += stiffness * viol * sign
                energy += 0.5 * stiffness * viol * viol
                
    return energy, grad

def _optimize_for_radius(centers, r, stiffness=100.0, max_iter=2000, time_limit=5.0):
    """
    Minimize energy for a fixed radius r.
    Returns the optimized centers and the final energy.
    """
    start_time = time.time()
    x0 = centers.flatten()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            # Return a huge value to stop optimization
            return 1e10, np.zeros_like(x)
        c = x.reshape(N, 3)
        e, g = _energy_and_gradient(c, r, stiffness)
        return e, g.flatten()
    
    try:
        result = minimize(
            obj, x0,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50, 'maxfun': 5000}
        )
        final_centers = result.x.reshape(N, 3)
        final_energy, _ = _energy_and_gradient(final_centers, r, stiffness)
        return final_centers, final_energy
    except Exception:
        return centers, 1e10

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _jitter_init(n, seed=None, jitter=0.01):
    """Generate jittered grid initial configuration."""
    grid = _grid_init(n)
    rng = np.random.RandomState(seed)
    noise = rng.uniform(-jitter, jitter, size=grid.shape)
    return np.clip(grid + noise, -0.49, 0.49)

def _random_init(n, seed=None):
    """Generate random initial configuration."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def construct():
    """
    Construct the optimal packing of N spheres in the unit cube.
    Strategy: Iterative radius increase.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # 1. Initialization
    # Start with a jittered grid to break symmetry and ensure initial validity
    init_centers = _jitter_init(N, seed=42, jitter=0.02)
    current_centers = init_centers.copy()
    current_r = _compute_radius(current_centers)
    
    best_centers = current_centers.copy()
    best_r = current_r
    
    # 2. Iterative Radius Increase
    # Start with a small step, increase it if successful, decrease if failed
    step = 0.001 
    min_step = 1e-6
    max_steps = 5000
    stall_counter = 0
    max_stalls = 50
    
    # Pre-allocate for speed? No, N is small.
    
    while time.time() < deadline - 10.0 and stall_counter < max_stalls:
        
        # Determine time budget for this step
        time_budget = 2.0
        if time.time() > deadline - 30.0:
            time_budget = 1.0
        if time.time() > deadline - 10.0:
            break
            
        target_r = current_r + step
        
        # Optimization
        opt_centers, energy = _optimize_for_radius(
            current_centers, target_r, 
            stiffness=50.0, 
            max_iter=1000, 
            time_limit=time_budget
        )
        
        # Check if the new configuration is valid for target_r
        # We check the actual radius to be safe
        new_r = _compute_radius(opt_centers)
        
        if new_r >= target_r - 1e-9:
            # Success! Update state
            current_centers = opt_centers
            current_r = new_r
            best_centers = opt_centers.copy()
            best_r = new_r
            
            # Increase step size slightly to accelerate
            step *= 1.1
            stall_counter = 0
        else:
            # Failure
            # Reduce step size
            step *= 0.8
            if step < min_step:
                break
            stall_counter += 1
            
            # If we've been failing for a while, try to "kick" the configuration out of local minima
            if stall_counter % 10 == 0:
                # Add small noise
                rng = np.random.RandomState(int(time.time()))
                noise = rng.uniform(-0.001, 0.001, size=(N, 3))
                perturbed = np.clip(current_centers + noise, -0.49, 0.49)
                
                # Try to optimize from perturbed state with current_r (not target_r) to relax
                # This helps escape shallow minima
                relaxed_centers, _ = _optimize_for_radius(
                    perturbed, current_r,
                    stiffness=20.0,
                    max_iter=500,
                    time_limit=1.0
                )
                relaxed_r = _compute_radius(relaxed_centers)
                if relaxed_r >= current_r - 1e-9:
                    current_centers = relaxed_centers
                    current_r = relaxed_r
                    best_centers = relaxed_centers.copy()
                    best_r = relaxed_r
                    # Reset stall counter slightly? No, keep it but maybe increase step back to normal
                    step = 0.001 

    # 3. Final Polishing
    # Now that we are stuck at a local optimum for radius, try to squeeze more out.
    # Use a very high stiffness and small step to polish the surface.
    if time.time() < deadline - 2.0:
        # Try a few more aggressive steps with smaller increments
        for _ in range(5):
            if time.time() > deadline - 1.0:
                break
            target_r = current_r + 1e-7
            opt_centers, energy = _optimize_for_radius(
                current_centers, target_r,
                stiffness=1000.0,
                max_iter=2000,
                time_limit=1.5
            )
            new_r = _compute_radius(opt_centers)
            if new_r > current_r:
                current_centers = opt_centers
                current_r = new_r
                best_centers = opt_centers.copy()
                best_r = new_r

    # Final validation and clipping
    if best_centers is None:
        best_centers = _grid_init(N)
    
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    return best_centers
# EVOLVE-BLOCK-END
