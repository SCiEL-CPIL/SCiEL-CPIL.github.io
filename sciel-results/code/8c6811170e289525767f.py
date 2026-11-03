"""Combined program: iterative radius increase with energy minimization and smooth soft-penalty."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

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
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return float(r)

def _energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Uses a smooth soft-penalty: quadratic when violated, zero otherwise.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms - smooth penalty for d_ij < 2R
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(sq_dists + 1e-15)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(sq_dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    
    # Only process pairs that are close enough to potentially violate
    close_mask = d_ij < 2.5 * R  # slightly wider margin for smooth transition
    if np.any(close_mask):
        i_close = i_idx[close_mask]
        j_close = j_idx[close_mask]
        d_close = d_ij[close_mask]
        
        # Smooth penalty: (max(0, 2R - d))^2
        violation = np.maximum(0.0, 2.0 * R - d_close)
        energy += np.sum(violation**2)
        
        # Gradient: d/dx_i sum (max(0, 2R - d_ij))^2
        # For pairs where 2R - d_ij > 0:
        # dE/dx_i = -2 * (2R - d_ij) * (x_i - x_j) / d_ij
        active = violation > 0
        if np.any(active):
            i_a = i_close[active]
            j_a = j_close[active]
            d_a = d_close[active]
            v_a = violation[active]
            
            factors = 2.0 * v_a / d_a
            vecs = centers[i_a] - centers[j_a]
            forces = factors[:, np.newaxis] * vecs
            
            np.add.at(grad, i_a, -forces)
            np.add.at(grad, j_a, forces)

    # 2. Wall terms - smooth penalty
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    wall_violation = np.maximum(0.0, R - wall_dists)
    energy += np.sum(wall_violation**2)
    
    active_wall = wall_violation > 0
    if np.any(active_wall):
        for idx in np.where(active_wall)[0]:
            v = wall_violation[idx]
            max_k = np.argmax(abs_centers[idx])
            sign_val = np.sign(centers[idx, max_k])
            if sign_val == 0:
                sign_val = 1.0
            grad[idx, max_k] += 2.0 * v * sign_val
            
    return energy, grad.flatten()

def _minimize_energy(init_centers, target_R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Minimizes the energy for a given target radius.
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        final_energy, _ = _energy_and_gradient(new_centers, target_R)
        return new_centers, final_energy, res.success
    except Exception:
        return init_centers, np.inf, False

def _generate_bcc_points(n, rng):
    """Generate BCC lattice points in [-0.5, 0.5]^3."""
    # BCC: simple cubic + body center
    # For n=100, use a 4x4x4 grid = 64 points + 36 body centers
    # Or better: use a denser arrangement
    
    # Try a 5x5x4 grid = 100 points, but that's too dense
    # Use 4x5x5 = 100 points
    nx, ny, nz = 4, 5, 5
    spacing_x = 1.0 / nx
    spacing_y = 1.0 / ny
    spacing_z = 1.0 / nz
    
    points = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                x = -0.5 + (ix + 0.5) * spacing_x
                y = -0.5 + (iy + 0.5) * spacing_y
                z = -0.5 + (iz + 0.5) * spacing_z
                points.append([x, y, z])
    
    points = np.array(points[:n])
    
    # Add small random perturbation to break symmetry
    points = points + rng.normal(0, 0.005, size=points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # 1. Initialization: BCC lattice with symmetry breaking
    best_pos = _generate_bcc_points(N, rng)
    
    # Initial optimization to get a good baseline
    # Start with a conservative radius
    initial_r = 0.1
    best_pos, energy, _ = _minimize_energy(best_pos, initial_r, max_iter=300)
    best_r = _compute_radius(best_pos)
    
    if best_r <= 0:
         best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
         best_r = _compute_radius(best_pos)

    # 2. Iterative Radius Increase
    current_r = best_r
    step_size = 0.002  # Larger steps for faster progress
    
    consecutive_failures = 0
    max_failures = 3
    
    rng_search = np.random.default_rng(123)
    
    while time.time() < deadline - 20.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.5
            consecutive_failures = 0
            if step_size < 1e-8:
                break
        
        # Generate diverse candidates
        candidates = []
        candidates.append(best_pos.copy())
        candidates.append(best_pos + rng_search.normal(0, 0.002, size=(N, 3)))
        candidates.append(best_pos + rng_search.normal(0, 0.008, size=(N, 3)))
        candidates.append(best_pos + rng_search.normal(0, 0.02, size=(N, 3)))
        # Add a completely random restart occasionally
        if rng_search.random() < 0.1:
            candidates.append(rng_search.uniform(-0.45, 0.45, size=(N, 3)))
        
        success = False
        for start_pos in candidates:
            start_pos = np.clip(start_pos, -0.499, 0.499)
            
            # Adaptive max_iter: more iterations when close to optimum
            if current_r < 0.105:
                max_iter = 500
            elif current_r < 0.11:
                max_iter = 700
            else:
                max_iter = 1000
            
            new_pos, energy, opt_success = _minimize_energy(start_pos, target_r, max_iter=max_iter)
            
            if energy < 1e-10:
                actual_r = _compute_radius(new_pos)
                if actual_r >= target_r - 1e-6:
                    best_pos = new_pos
                    best_r = actual_r
                    current_r = actual_r
                    success = True
                    consecutive_failures = 0
                    break
            else:
                # Even if energy is not perfectly zero, if we improved the radius, we can take it
                actual_r = _compute_radius(new_pos)
                if actual_r > current_r:
                    best_pos = new_pos
                    best_r = actual_r
                    current_r = actual_r
                    # Don't mark as full success for step size increase, but accept the position
                    pass
        
        if not success:
            consecutive_failures += 1
            
            # If we failed, try to polish the current best position
            if time.time() < deadline - 25.0 and consecutive_failures % 2 == 0:
                polished, _, _ = _minimize_energy(best_pos, current_r, max_iter=500)
                polished_r = _compute_radius(polished)
                if polished_r > best_r:
                    best_pos = polished
                    best_r = polished_r
                    current_r = best_r

    # 3. Final High-Precision Polish
    if time.time() < deadline - 5.0:
        # Try to push slightly beyond current best with very tight tolerances
        for attempt in range(5):
            if time.time() > deadline - 3.0:
                break
            target_r = best_r + 1e-5
            polished, energy, _ = _minimize_energy(best_pos, target_r, max_iter=1500, ftol=1e-16, gtol=1e-15)
            polished_r = _compute_radius(polished)
            if polished_r > best_r:
                best_pos = polished
                best_r = polished_r
            else:
                # Try a slightly different perturbation
                perturbed = best_pos + rng_search.normal(0, 0.001, size=(N, 3))
                perturbed = np.clip(perturbed, -0.499, 0.499)
                polished, energy, _ = _minimize_energy(perturbed, target_r, max_iter=1000, ftol=1e-16, gtol=1e-15)
                polished_r = _compute_radius(polished)
                if polished_r > best_r:
                    best_pos = polished
                    best_r = polished_r
    
    # Final validation and clipping
    best_pos = np.clip(best_pos, -0.499999, 0.499999)
    
    return best_pos
