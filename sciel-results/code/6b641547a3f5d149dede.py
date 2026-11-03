"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        return float(min_dist_to_wall)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair_dist = np.sqrt(np.min(sq_dists))
    
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _smooth_energy_and_grad(centers, target_r):
    """
    Compute smooth potential energy and its gradient.
    Energy: Sum of squared violations for pairs and walls.
    V(d) = (d_min - d)^2 if d < d_min, else 0.
    This is differentiable almost everywhere (gradient is 0 at boundary).
    """
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise interactions
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    
    d_min_pair = 2.0 * target_r
    # Mask for pairs closer than d_min_pair
    mask = dists < d_min_pair
    
    if np.any(mask):
        # Violation amount: v = d_min - d
        # Gradient of v^2 w.r.t centers[i] is -2*v * (centers[i]-centers[j])/d
        # We accumulate forces.
        # Force on i from j is proportional to (centers[i] - centers[j]) / d * (d_min - d)
        
        # Vectorized calculation of forces
        # d_inv = 1/dists
        # v = d_min - dists
        # force_vec = diff * (v * d_inv)[:, :, np.newaxis]
        # grad[i] = - sum_{j != i} force_vec[i,j]
        
        d_inv = 1.0 / dists
        v = d_min_pair - dists
        v = np.where(mask, v, 0.0)
        
        # Gradient of (d_min - d)^2 is -2(d_min - d) * (x_i - x_j) / d
        # So grad[i] += -2 * v[i,j] * diff[i,j] / dists[i,j]
        # Note: diff[i,j] = c[i] - c[j]
        
        coeff = 2.0 * v * d_inv
        coeff = np.where(mask, coeff, 0.0)
        
        # Sum over j for each i
        # grad[i] = - sum_j coeff[i,j] * diff[i,j]
        # Using einsum: grad[i] = - sum_j coeff[i,j] * (c[i] - c[j])
        # This is equivalent to:
        # grad[i] = - sum_j coeff[i,j]*c[i] + sum_j coeff[i,j]*c[j]
        # sum_j coeff[i,j]*c[i] = c[i] * sum_j coeff[i,j]
        
        sum_coeff = np.sum(coeff, axis=1)
        sum_coeff_c = np.sum(coeff[:, :, np.newaxis] * centers[np.newaxis, :, :], axis=1)
        
        grad -= (sum_coeff[:, np.newaxis] * centers) - sum_coeff_c
        
    # Wall interactions
    wall_dists = 0.5 - np.abs(centers)
    d_min_wall = target_r
    wall_mask = wall_dists < d_min_wall
    
    if np.any(wall_mask):
        v_wall = d_min_wall - wall_dists
        v_wall = np.where(wall_mask, v_wall, 0.0)
        
        # Gradient of (d_min - wall_dist)^2
        # wall_dist = 0.5 - |x|
        # d(wall_dist)/dx = -sign(x)
        # d/dx (d_min - wall_dist)^2 = -2(d_min - wall_dist) * d(wall_dist)/dx = 2(d_min - wall_dist) * sign(x)
        
        signs = np.sign(centers)
        # Handle zeros in sign (center on axis)
        signs[signs == 0] = 1.0
        
        grad += 2.0 * v_wall * signs
        
        energy += np.sum(v_wall**2)
    
    energy += np.sum(v**2) # Note: v is 0 where mask is False
    
    return energy, grad

def _sa_optimize(centers, target_r, time_limit=10.0, T_init=0.05, T_final=0.001, seed=42):
    """
    Simulated Annealing using smooth energy.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    c = centers.copy()
    current_energy, _ = _smooth_energy_and_grad(c, target_r)
    best_energy = current_energy
    best_c = c.copy()
    
    step_size = 0.02
    cooling_rate = (T_final / T_init) ** (1.0 / 10000)
    
    step = 0
    while time.time() - start_time < time_limit:
        T = T_init * (cooling_rate ** step)
        
        # Random move
        idx = rng.randint(N)
        delta = rng.normal(0, step_size, size=3)
        new_c = c.copy()
        new_c[idx] = np.clip(new_c[idx] + delta, -0.4999, 0.4999)
        
        new_energy, _ = _smooth_energy_and_grad(new_c, target_r)
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-10)):
            c = new_c
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_c = c.copy()
                # Reduce step size slightly on improvement
                step_size *= 0.999
        else:
            # If rejected, maybe try a larger move occasionally
            if step % 50 == 0:
                step_size *= 1.05
                step_size = min(step_size, 0.05)
        
        step += 1
        
    return best_c, best_energy

def _polish_lbfgs(centers, target_r, time_limit=10.0):
    """
    Polish using L-BFGS-B on smooth energy.
    """
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        c = x.reshape(N, 3)
        e, g = _smooth_energy_and_grad(c, target_r)
        return e, g.flatten()
    
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        result = minimize(
            obj_and_grad, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        return result.x.reshape(N, 3)
    except Exception:
        return centers

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

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_centers = None
    best_radius = 0.0
    
    # Initial grid
    grid = _grid_init(N)
    r_grid = _compute_radius(grid)
    best_radius = r_grid
    best_centers = grid.copy()
    
    # Strategy: 
    # 1. SA with moderate target to explore
    # 2. Polish
    # 3. Repeat with tighter targets
    
    # Target radii to try
    # Known best is ~0.11138
    # Grid radius for 100 spheres (5x5x4 grid approx) is roughly 0.1
    targets = [0.10, 0.105, 0.108, 0.110, 0.111, 0.1113, 0.11138, 0.1114]
    
    # Filter targets that are not too far above current best
    # Start with a few moderate targets
    for t_idx, target_r in enumerate(targets):
        if time.time() > deadline - 20.0:
            break
        
        # Skip if target is too low
        if target_r < best_radius * 0.95:
            continue
            
        # Run SA for a short time
        sa_time = min(5.0, deadline - time.time() - 20.0)
        if sa_time < 1.0:
            break
            
        try:
            # Use best_centers as start for SA to refine
            # Add small jitter to avoid exact local minima
            init_c = best_centers + np.random.normal(0, 0.005, size=(N, 3))
            init_c = np.clip(init_c, -0.4999, 0.4999)
            
            sa_c, sa_e = _sa_optimize(init_c, target_r, time_limit=sa_time, seed=100 + t_idx)
            
            # Polish
            polish_time = min(5.0, deadline - time.time() - 10.0)
            if polish_time > 1.0:
                polished_c = _polish_lbfgs(sa_c, target_r, time_limit=polish_time)
            else:
                polished_c = sa_c
            
            r_new = _compute_radius(polished_c)
            if r_new > best_radius:
                best_radius = r_new
                best_centers = polished_c.copy()
                
        except Exception:
            continue
    
    # Final polish with current best radius as target
    if time.time() < deadline - 5.0:
        try:
            final_target = best_radius * 1.001
            polish_time = deadline - time.time() - 2.0
            if polish_time > 1.0:
                polished_c = _polish_lbfgs(best_centers, final_target, time_limit=polish_time)
                r_new = _compute_radius(polished_c)
                if r_new > best_radius:
                    best_radius = r_new
                    best_centers = polished_c.copy()
        except Exception:
            pass
    
    # Final validation
    if best_centers is None:
        best_centers = _grid_init(N)
        
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    return best_centers
# EVOLVE-BLOCK-END
