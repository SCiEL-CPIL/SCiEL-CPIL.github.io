import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 50

def _energy_and_grad(x, R):
    """
    Calculate energy (sum of squared overlaps) and gradient for a fixed radius R.
    x is a flat array of shape (N*3,).
    """
    centers = x.reshape(N, 3)
    energy = 0.0
    grad = np.zeros_like(x)
    
    # Pairwise overlaps
    # d_{ij} = |c_i - c_j|
    # violation = max(0, 2R - d_{ij})
    # We compute this efficiently.
    
    # To save time, we only compute this for pairs that are "close" or just all pairs since N=50 is small (1225 pairs).
    # 1225 pairs is trivial for numpy.
    
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # We only care about i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dist[i_idx, j_idx]
    
    # Violations
    target = 2.0 * R
    viol = np.maximum(0.0, target - d_ij)
    
    # Energy contribution
    energy += np.sum(viol ** 2)
    
    # Gradient contribution
    # d/d(c_i) [ (2R - d_ij)^2 ] = 2 * (2R - d_ij) * (-1) * d(d_ij)/d(c_i)
    # d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
    # So grad_i += -2 * viol * (c_i - c_j) / d_ij
    # Note: if viol > 0, then d_ij < 2R, so d_ij is not zero.
    
    # We need to accumulate gradients for each center.
    # grad_centers is (N, 3)
    grad_centers = np.zeros_like(centers)
    
    mask = viol > 0
    if np.any(mask):
        # Get the relevant distances and differences
        d_masked = d_ij[mask]
        viol_masked = viol[mask]
        i_masked = i_idx[mask]
        j_masked = j_idx[mask]
        
        # Vectorized accumulation is tricky with scatter_add in pure numpy without cupy, 
        # but N is small enough for a loop or bincount tricks.
        # Let's use a simple loop for clarity and correctness, or vectorize with advanced indexing if possible.
        # Actually, a loop over 1225 items is fast in Python? No, 1225 iterations in Python is ~0.1ms, acceptable.
        # But we can do better.
        
        # Let's construct the gradient vectors for each pair
        # vec_ij = centers[i] - centers[j]
        # grad_i += -2 * viol * vec_ij / d_ij
        # grad_j +=  2 * viol * vec_ij / d_ij
        
        vecs = centers[i_masked] - centers[j_masked] # (num_masked, 3)
        coeff = -2.0 * viol_masked / d_masked # (num_masked,)
        
        # Scatter add to grad_centers
        # np.add.at is the way to go for unbuffered addition
        np.add.at(grad_centers, i_masked, coeff[:, np.newaxis] * vecs)
        np.add.at(grad_centers, j_masked, -coeff[:, np.newaxis] * vecs)
        
    # Wall overlaps
    # Violation if 0.5 - |c_ik| < R  =>  |c_ik| > 0.5 - R
    # Or if c_ik < -(0.5 - R) or c_ik > 0.5 - R
    # Let's define limits: L = 0.5 - R
    L = 0.5 - R
    
    # Distance to walls:
    # dist_left = c - (-L) = c + L (if c < -L)
    # dist_right = L - c (if c > L)
    # Actually, simpler:
    # violation = max(0, |c| - L)
    # But we need to be careful with signs for gradient.
    
    # Let's handle each coordinate separately.
    abs_c = np.abs(centers)
    wall_viol = np.maximum(0.0, abs_c - L)
    
    energy += np.sum(wall_viol ** 2)
    
    # Gradient:
    # d/d(c) (|c| - L)^2 = 2 * (|c| - L) * sign(c)
    # Note: if wall_viol > 0, then |c| > L, so c != 0 (unless L<0 which is impossible for R<0.5)
    grad_centers += 2.0 * wall_viol * np.sign(centers)
    
    grad = grad_centers.ravel()
    
    return energy, grad

def _try_fit_radius(centers, R, max_iter=100, tol=1e-8):
    """
    Try to fit spheres of radius R starting from centers.
    Returns (success, new_centers, min_energy)
    """
    x0 = centers.ravel()
    bounds = [(-0.5 + R, 0.5 - R)] * (N * 3)
    
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            args=(R,),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6, 'maxls': 20}
        )
        min_energy = res.fun
        if min_energy < 1e-5: # Threshold for "valid"
            return True, res.x.reshape(N, 3), min_energy
        else:
            return False, res.x.reshape(N, 3), min_energy
    except Exception:
        return False, centers, 1.0

def _generate_initial_guess(seed=0):
    """
    Generate a diverse set of initial guesses.
    """
    rng = np.random.default_rng(seed)
    centers = np.zeros((N, 3))
    
    strategy = seed % 4
    
    if strategy == 0:
        # Simple cubic grid + jitter
        # 3x4x4 = 48, add 2 random? Or 4x3x4=48?
        # Let's do a 3x4x4 grid.
        xs = np.linspace(-0.4, 0.4, 3)
        ys = np.linspace(-0.4, 0.4, 4)
        zs = np.linspace(-0.4, 0.4, 4)
        grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3)
        # Shuffle and take 50? No, just use the grid points and add jitter.
        # We have 48 points. Add 2 random points?
        # Better: use a denser grid and pick best?
        # For now, just use the 48 points and 2 random ones.
        centers[:48] = grid
        centers[48:] = rng.uniform(-0.3, 0.3, (2, 3))
        
    elif strategy == 1:
        # BCC-like lattice
        # Body centered cubic.
        # Points at (0,0,0) and (0.5,0.5,0.5) scaled.
        # In a unit cube, BCC has 2 points per cell.
        # Let's create a 3x3x3 BCC lattice.
        # Corners and body centers.
        pts = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Scale to fit in [-0.4, 0.4] roughly
                    x = (i - 1) * 0.4
                    y = (j - 1) * 0.4
                    z = (k - 1) * 0.4
                    pts.append([x, y, z])
                    # Body center of this cell
                    if i < 2 and j < 2 and k < 2:
                         pts.append([x + 0.2, y + 0.2, z + 0.2])
        # We have 27 corners + 8 centers = 35 points.
        # Add 15 random points?
        centers[:35] = np.array(pts)
        centers[35:] = rng.uniform(-0.35, 0.35, (15, 3))
        
    elif strategy == 2:
        # Random uniform
        centers = rng.uniform(-0.4, 0.4, (N, 3))
        
    else:
        # Helical / Spiral
        t = np.linspace(0, 10 * np.pi, N)
        r = 0.3 * t / (10 * np.pi)
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = (t / (10 * np.pi) - 0.5) * 0.8
        centers = np.column_stack((x, y, z))
        centers += rng.normal(0, 0.05, (N, 3))

    # Clip to valid region
    centers = np.clip(centers, -0.45, 0.45)
    return centers

def construct():
    """
    Construct optimal sphere packing.
    """
    start_time = time.time()
    time_limit = 140.0 # 10s margin
    
    best_radius = 0.0
    best_centers = None
    
    # We will try to find the best radius by running multiple independent searches.
    # Each search tries to incrementally increase the radius.
    
    num_restarts = 10
    base_radius = 0.1
    radius_step = 0.001
    
    for restart_idx in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Generate initial guess
        centers = _generate_initial_guess(seed=restart_idx)
        
        # Start with a small radius that is definitely feasible
        # Estimate max possible radius: Volume argument.
        # 50 spheres of radius R in volume 1.
        # 50 * (4/3) pi R^3 <= 1 => R^3 <= 1 / (50 * 4.18) => R <= 0.107
        # Wait, packing efficiency is ~0.74.
        # 50 * (4/3) pi R^3 * 0.74 <= 1 => R^3 <= 1 / (50 * 4.18 * 0.74) => R <= 0.127
        # Best known is 0.135. So start at 0.1.
        
        current_R = base_radius
        
        # Try to increase R
        # We will do a loop:
        # 1. Try to fit current_R.
        # 2. If success, current_R += radius_step.
        # 3. If fail, current_R -= radius_step / 2 and maybe perturb.
        # 4. Stop if we can't make progress or time is up.
        
        local_best_R = 0.0
        local_centers = centers
        
        # First, ensure we can fit base_radius
        success, new_centers, _ = _try_fit_radius(centers, current_R, max_iter=50)
        if success:
            local_centers = new_centers
            local_best_R = current_R
        else:
            # If we can't even fit the base radius, try a smaller one or different init
            # For now, skip this restart or try smaller
            current_R = 0.05
            success, new_centers, _ = _try_fit_radius(centers, current_R, max_iter=50)
            if success:
                local_centers = new_centers
                local_best_R = current_R
            else:
                continue

        # Now increment
        stall_count = 0
        max_stalls = 5
        
        while time.time() - start_time < time_limit - 5.0:
            # Check if we beat global best
            if local_best_R > best_radius:
                best_radius = local_best_R
                best_centers = local_centers.copy()
                
            # Try to increase
            next_R = local_best_R + radius_step
            
            # Use the current best centers as starting point
            success, new_centers, energy = _try_fit_radius(local_centers, next_R, max_iter=100)
            
            if success:
                local_best_R = next_R
                local_centers = new_centers
                stall_count = 0
            else:
                # Failed to fit next_R
                # Try to reduce R slightly and see if we can recover
                # Or just perturb and try same R?
                # Let's try to fit a slightly smaller R to "relax" the configuration
                # This helps escape local minima where the optimizer got stuck in a bad shape
                relax_R = local_best_R - radius_step * 0.5
                if relax_R > 0.05:
                    success, new_centers, _ = _try_fit_radius(local_centers, relax_R, max_iter=50)
                    if success:
                        local_centers = new_centers
                        # Now try the next_R again from this relaxed state?
                        # No, just let the loop try next_R again?
                        # If we relax, we might be able to push further.
                        # But we already failed next_R.
                        # Let's try next_R again from the relaxed state.
                        success2, new_centers2, _ = _try_fit_radius(new_centers, next_R, max_iter=50)
                        if success2:
                            local_best_R = next_R
                            local_centers = new_centers2
                            stall_count = 0
                        else:
                            stall_count += 1
                    else:
                        stall_count += 1
                else:
                    stall_count += 1
                    
            if stall_count > max_stalls:
                break
                
            # If we've been at the same R for a while, maybe add noise
            # The relaxation step above acts as a noise/perturbation.
            
    # Final check
    if best_centers is None:
        # Fallback
        centers = _generate_initial_guess(seed=99)
        return centers
        
    return best_centers
