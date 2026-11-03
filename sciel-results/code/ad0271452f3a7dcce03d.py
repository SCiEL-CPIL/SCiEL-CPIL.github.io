"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(points):
    """Largest common radius: min of half pairwise distances and wall clearance."""
    # points is (N, 3)
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    # Get unique pairs
    iu = np.triu_indices(N, k=1)
    min_pair = dists[iu].min()
    clearance = (0.5 - np.abs(points)).min()
    return min(min_pair * 0.5, clearance)

def _energy_and_grad(x, target_r):
    """
    Energy function for BFGS optimization.
    x is a flat array of length 3N.
    target_r is the desired radius.
    Returns (energy, gradient).
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
    active = dists < 2.0 * target_r
    # Ensure diagonal is not active
    active[np.diag_indices_from(active)] = False
    
    # Violations: (2r - d) for active pairs, 0 otherwise
    violations = np.where(active, 2.0 * target_r - dists, 0.0)
    
    # Energy is sum of squared violations (0.5 factor for clean gradient)
    energy_pair = 0.5 * np.sum(violations**2)
    
    # Gradient calculation for pairs
    # d/dx_i [ 0.5 * (2r - ||x_i - x_j||)^2 ]
    # = (2r - d) * d/dx_i (2r - d)
    # = (2r - d) * (-(x_i - x_j) / d)
    # = - (2r - d) / d * (x_i - x_j)
    # Note: violations[i,j] = 2r - d. So we need - violations[i,j] / d * (x_i - x_j)
    # To avoid division by zero, we use safe_dists
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # weights[i,j] = violations[i,j]
    # grad_i = - sum_j weights[i,j] * unit_vecs[i,j]
    grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # --- Wall Part ---
    # Distance to each wall
    # Walls are at -0.5 and 0.5 for each dimension
    # dist_wall_k = 0.5 - |c_k|
    wall_dists = 0.5 - np.abs(c)
    
    # Violations: target_r - wall_dist if positive
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    
    energy_wall = np.sum(wall_violations**2)
    
    # Gradient for walls
    # d/dc [ (r - (0.5 - |c|))^2 ]
    # = 2 * (r - (0.5 - |c|)) * d/dc (0.5 - |c|)
    # = 2 * wall_violation * (-sign(c))
    signs = np.sign(c)
    # Handle c=0 case: sign(0)=0, which is fine, gradient is 0 at center
    grad_wall = 2.0 * wall_violations * signs
    
    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall).flatten()
    
    return total_energy, total_grad

def _optimize_bfgs(init_pts, target_r, max_iter=1000, gtol=1e-9, time_limit=50.0):
    """
    Run BFGS optimization with a time limit.
    Returns (optimized_pts, success).
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
            options={'maxiter': max_iter, 'gtol': gtol, 'disp': False}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts, result.success
    except TimeoutError:
        # Return current best if timeout? minimize doesn't easily return intermediate state.
        # We rely on the outer loop checking time.
        return init_pts, False
    except Exception:
        return init_pts, False

def _slda_optimize(init_pts, target_r, time_limit=10.0, n_steps=None):
    """
    Stochastic Gradient Langevin Dynamics optimization.
    Uses the energy gradient from _energy_and_grad.
    """
    t_start = time.time()
    x = init_pts.flatten().copy()
    best_x = x.copy()
    best_r = _score(x.reshape(N, 3))
    
    if n_steps is None:
        n_steps = 20000
    
    # Adaptive step size and temperature
    step_size = 0.001
    temp = 0.01
    step_decay = 0.9999
    temp_decay = 0.9999
    
    for i in range(n_steps):
        if time.time() - t_start > time_limit:
            break
        
        energy, grad = _energy_and_grad(x, target_r)
        
        # Langevin update: x_{t+1} = x_t - eta * grad + sqrt(2 * eta * T) * noise
        noise = np.random.randn(len(x))
        x = x - step_size * grad + np.sqrt(2 * step_size * temp) * noise
        
        # Clip to stay within bounds (with margin)
        x = np.clip(x, -0.49, 0.49)
        
        # Occasionally check score and update best
        if i % 100 == 0 or i == n_steps - 1:
            pts = x.reshape(N, 3)
            if np.all(np.isfinite(pts)):
                r = _score(pts)
                if r > best_r:
                    best_r = r
                    best_x = x.copy()
        
        # Decay step size and temperature
        step_size *= step_decay
        temp *= temp_decay
        
        # Early stopping if gradient is very small
        if np.linalg.norm(grad) < 1e-10:
            break
    
    return best_x.reshape(N, 3)

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin under 120s

    # ---- Seed: Simple cubic 5x5x5 grid ----
    # 5^3 = 125 points. We take the first 100.
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:N].copy()

    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    rng = np.random.RandomState(42)

    # Strategy 1: Run BFGS with increasing target radii
    targets = [0.105, 0.110, 0.112, 0.114, 0.116, 0.118]
    
    current_pts = best_pts.copy()
    
    for target_r in targets:
        if time.time() - t0 > DEADLINE - 30.0:
            break
            
        # Add small noise to break symmetry and help escape local minima
        noise = rng.normal(0, 0.005, size=(N, 3))
        init_pts = np.clip(current_pts + noise, -0.49, 0.49)
        
        opt_pts, success = _optimize_bfgs(
            init_pts, 
            target_r, 
            max_iter=2000, 
            gtol=1e-10,
            time_limit=DEADLINE - (time.time() - t0) - 5.0
        )
        
        if np.all(np.isfinite(opt_pts)):
            # Clip to ensure validity
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _score(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
                current_pts = opt_pts.copy()

    # Strategy 2: SGLD for escaping local minima
    # Run several SGLD runs with different initial conditions and targets
    if time.time() - t0 < DEADLINE - 20.0:
        slda_targets = [0.112, 0.114, 0.116]
        
        for target_r in slda_targets:
            if time.time() - t0 > DEADLINE - 10.0:
                break
            
            # Multiple SGLD restarts
            for restart in range(3):
                if time.time() - t0 > DEADLINE - 5.0:
                    break
                
                # Different initialization strategies
                if restart == 0:
                    # Start from best found
                    noise = rng.normal(0, 0.01, size=(N, 3))
                    init_pts = np.clip(best_pts + noise, -0.49, 0.49)
                elif restart == 1:
                    # Start from grid with larger noise
                    noise = rng.normal(0, 0.03, size=(N, 3))
                    init_pts = np.clip(pts + noise, -0.49, 0.49)
                else:
                    # Start from random points
                    init_pts = rng.uniform(-0.45, 0.45, size=(N, 3))
                
                # Run SGLD
                slda_time = min(15.0, DEADLINE - (time.time() - t0) - 5.0)
                opt_pts = _slda_optimize(init_pts, target_r, time_limit=slda_time)
                
                if np.all(np.isfinite(opt_pts)):
                    opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                    r_opt = _score(opt_pts)
                    
                    if r_opt > best_r:
                        best_r = r_opt
                        best_pts = opt_pts.copy()
                        print(f"SGLD improved to {r_opt:.6f} at target {target_r}")

    # Strategy 3: Final BFGS polish on the best found configuration
    if time.time() - t0 < DEADLINE - 5.0 and best_r > 0.1:
        # Try to optimize exactly for the best_r found, with very tight tolerance
        noise = rng.normal(0, 0.001, size=(N, 3))
        init_pts = np.clip(best_pts + noise, -0.49, 0.49)
        
        opt_pts, success = _optimize_bfgs(
            init_pts, 
            best_r, 
            max_iter=5000, 
            gtol=1e-12,
            time_limit=DEADLINE - (time.time() - t0) - 2.0
        )
        
        if np.all(np.isfinite(opt_pts)):
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _score(opt_pts)
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()

    # Final safety check
    best_pts = np.clip(best_pts, -0.5 + 1e-9, 0.5 - 1e-9)
    if not np.all(np.isfinite(best_pts)):
        best_pts = grid[:N].copy()
        
    return best_pts
# EVOLVE-BLOCK-END
