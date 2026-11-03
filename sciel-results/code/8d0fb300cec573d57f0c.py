import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _score(points):
    """
    Calculate the exact score (maximum common radius) for a given configuration.
    points: (N, 3) array of centers.
    Returns: max radius r such that spheres of radius r are disjoint and inside [-0.5, 0.5]^3.
    """
    # Check validity
    if not np.all(np.isfinite(points)):
        return 0.0
    if np.any(points < -0.5) or np.any(points > 0.5):
        return 0.0
        
    # Wall clearance
    clearance = (0.5 - np.abs(points)).min()
    
    # Pairwise distances
    # For N=100, we can compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Get upper triangle indices (i < j)
    iu = np.triu_indices(N, k=1)
    min_pair_dist = dists[iu].min()
    
    # Radius is limited by half the minimum pairwise distance and the wall clearance
    return min(min_pair_dist * 0.5, clearance)

def _energy_and_grad(x, target_r):
    """
    Energy function for BFGS optimization.
    x: flat array of length 3N.
    target_r: desired radius.
    Returns: (energy, gradient).
    Energy is the sum of squared violations of the constraint that all spheres
    of radius target_r are disjoint and inside the cube.
    """
    c = x.reshape(N, 3)
    
    # --- Pairwise Part ---
    # diff[i, j] = c[i] - c[j]
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    # dists[i, j] = ||c[i] - c[j]||
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Active pairs: those closer than 2 * target_r
    # We use a small margin to ensure smoothness, but strict inequality is fine for BFGS
    # Actually, for the gradient to be zero when satisfied, we only care about violations.
    # violation_ij = max(0, 2*target_r - dists_ij)
    
    # To make it differentiable, we can use a smooth approximation or just the hinge loss.
    # Hinge loss is non-smooth at 0, but BFGS with line search usually handles it.
    # However, a quadratic penalty on violations is smoother: 0.5 * (max(0, 2r - d))^2
    # Let's use the quadratic penalty on violations.
    
    # violations = 2*target_r - dists
    # active = violations > 0
    # penalty = 0.5 * sum(violations[active]^2)
    
    violations = 2.0 * target_r - dists
    # Mask out diagonal and inactive pairs
    # We need to be careful with the gradient.
    # Let v_ij = max(0, 2r - d_ij)
    # E = 0.5 * sum v_ij^2
    # dE/dx = sum (v_ij * dv_ij/dx) for v_ij > 0
    
    # For numerical stability and to avoid issues with the non-differentiability at 0,
    # we can use a soft hinge: v_soft = 0.5 * (v + sqrt(v^2 + eps))
    # But standard BFGS often works fine with the hard hinge if we initialize well.
    # Let's stick to the hard hinge for simplicity and speed, as the inspiration did.
    
    active_mask = violations > 1e-10
    # Ensure diagonal is not active
    active_mask[np.diag_indices_from(active_mask)] = False
    
    # Gradient contribution from active pairs
    # d/dx_i (0.5 * (2r - ||x_i - x_j||)^2) = (2r - d) * (x_i - x_j) / d
    # Note: the 0.5 cancels with the 2 from the derivative of the square.
    
    # unit_vecs[i, j] = (x_i - x_j) / d_ij
    # We need to handle d=0 carefully, but we filled diagonal with 1.0.
    # For active pairs, d < 2r, so d is not 0 unless points coincide.
    
    # Compute gradients
    grad = np.zeros_like(c)
    
    # Only compute for active pairs to save time?
    # For N=100, 4950 pairs. It's fast enough to compute all.
    
    # weights[i, j] = violations[i, j] if active else 0
    weights = np.where(active_mask, violations, 0.0)
    
    # d_ij is dists
    # term = weights / dists * diff
    # grad_i = sum_j term[i, j]
    # grad_j = -sum_i term[i, j]
    
    # To avoid division by zero for inactive pairs (where weights=0), we can just divide by dists.
    # dists on diagonal is 1.0, weights is 0, so term is 0.
    # For inactive pairs, weights is 0, so term is 0.
    
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # term[i, j] = weights[i, j] * unit_vecs[i, j]
    terms = weights[:, :, np.newaxis] * unit_vecs
    
    # grad[i] = sum_j terms[i, j]
    # grad[j] = -sum_i terms[i, j] = -sum_i terms[i, j]
    # Note: terms is antisymmetric? No, weights is symmetric, unit_vecs is antisymmetric.
    # So terms[i, j] = - terms[j, i].
    # Therefore, sum_j terms[i, j] is the correct gradient for x_i.
    
    grad_pair = np.sum(terms, axis=1)
    
    # --- Wall Part ---
    # Wall constraints: |c_ik| + r <= 0.5
    # Violation: r - (0.5 - |c_ik|) = r - 0.5 + |c_ik|
    # Let w_ik = r - 0.5 + |c_ik|
    # Active if w_ik > 0
    # Penalty: 0.5 * sum w_ik^2
    
    wall_violations = target_r - (0.5 - np.abs(c))
    wall_active = wall_violations > 1e-10
    
    # Gradient of |c_ik| wrt c_ik is sign(c_ik)
    # d/dc (0.5 * w^2) = w * sign(c_ik)
    
    signs = np.sign(c)
    # Handle c=0: sign(0)=0. If c is very close to 0, the gradient is undefined, 
    # but the penalty is minimized at c=0 if r < 0.5. If r > 0.5 - |c|, it pushes away.
    # For robustness, we can use a smoothed sign if needed, but sign() is usually ok for BFGS.
    
    grad_wall = np.where(wall_active, wall_violations * signs, 0.0)
    
    total_grad = (grad_pair + grad_wall).flatten()
    
    # Energy
    energy_pair = 0.5 * np.sum(np.where(active_mask, violations**2, 0.0))
    energy_wall = 0.5 * np.sum(np.where(wall_active, wall_violations**2, 0.0))
    total_energy = energy_pair + energy_wall
    
    return total_energy, total_grad

def _optimize_bfgs(init_pts, target_r, max_iter=2000, time_limit=50.0):
    """
    Run BFGS optimization with a time limit.
    """
    t_start = time.time()
    
    def objective(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _energy_and_grad(x, target_r)
    
    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            method='BFGS',
            jac=True,
            options={'maxiter': max_iter, 'gtol': 1e-9, 'disp': False}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts, result.success
    except TimeoutError:
        return init_pts, False
    except Exception:
        return init_pts, False

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit
    
    # 1. Initialization: 5x5x4 grid
    # This provides a better starting point than a simple 5x5x5 truncation
    xs = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    ys = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    zs = np.array([-0.375, -0.125, 0.125, 0.375])
    
    centers = []
    for z in zs:
        for y in ys:
            for x in xs:
                centers.append([x, y, z])
    centers = np.array(centers)
    
    if len(centers) != N:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        centers = grid[:N]

    best_pts = centers.copy()
    best_r = _score(best_pts)
    
    rng = np.random.RandomState(42)
    
    # Strategy: Increase target radius gradually
    # The grid radius is roughly 0.1 (min spacing 0.2 -> r=0.1, wall dist 0.1 or 0.125)
    # Let's start slightly above the current best.
    
    # We will try a sequence of targets.
    # If a target is feasible (energy ~ 0), we can increase it.
    # If not, we keep the best feasible configuration.
    
    # Initial radius from grid
    current_r = best_r
    
    # Define a range of target radii to try
    # Best known is ~0.11138
    # We will try to push towards 0.112 or higher.
    
    targets = []
    # Start from current_r + small delta, up to 0.115
    # Step size 0.002
    t = current_r + 0.001
    while t < 0.115:
        targets.append(t)
        t += 0.002
    
    current_pts = best_pts.copy()
    
    for target_r in targets:
        if time.time() > deadline - 15.0:
            break
            
        # Add small noise to break symmetry and help escape local minima
        # Noise magnitude should be small
        noise = rng.normal(0, 0.005, size=(N, 3))
        init_pts = np.clip(current_pts + noise, -0.49, 0.49)
        
        # Time limit for this optimization step
        remaining = deadline - (time.time() - start_time) - 5.0
        if remaining < 5.0:
            break
            
        opt_pts, success = _optimize_bfgs(
            init_pts, 
            target_r, 
            max_iter=2000, 
            time_limit=remaining
        )
        
        if np.all(np.isfinite(opt_pts)):
            # Clip to ensure validity
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _score(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
                current_pts = opt_pts.copy()
            else:
                # If we couldn't improve the radius, we might still want to keep the 
                # optimized points if they are "closer" to being feasible for the target,
                # but usually if r_opt < best_r, the configuration is worse for our goal.
                # However, if the optimizer converged to a local minimum where r_opt < target_r
                # but r_opt > best_r, we would have updated.
                # If r_opt <= best_r, we stick to best_pts for the next iteration's noise base.
                pass

    # Random restarts with larger noise if time permits
    if time.time() < deadline - 20.0:
        for i in range(10):
            if time.time() > deadline - 5.0:
                break
            
            # Try to jump out of local minima
            noise = rng.normal(0, 0.02, size=(N, 3))
            init_pts = np.clip(best_pts + noise, -0.49, 0.49)
            
            # Target slightly above best_r
            target_r = best_r * 1.01
            
            remaining = deadline - (time.time() - start_time) - 5.0
            if remaining < 5.0:
                break
                
            opt_pts, success = _optimize_bfgs(
                init_pts, 
                target_r, 
                max_iter=1000, 
                time_limit=remaining
            )
            
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _score(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()

    # Final Polish: High precision optimization on the best found radius
    # This can squeeze out a tiny bit more if the previous runs stopped early
    if time.time() < deadline - 5.0 and best_r > 0.1:
        noise = rng.normal(0, 0.001, size=(N, 3))
        init_pts = np.clip(best_pts + noise, -0.49, 0.49)
        
        # Use a target equal to best_r to polish the configuration
        # This ensures the final configuration is as close to optimal as possible for that radius
        remaining = deadline - (time.time() - start_time) - 2.0
        if remaining > 2.0:
            opt_pts, success = _optimize_bfgs(
                init_pts, 
                best_r, 
                max_iter=500, 
                time_limit=remaining
            )
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _score(opt_pts)
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()

    return best_pts

# EVOLVE-BLOCK-END
