"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

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
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
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
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # --- Wall Part ---
    wall_dists = 0.5 - np.abs(c)
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    
    energy_wall = np.sum(wall_violations**2)
    
    signs = np.sign(c)
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
    
    step_size = 0.001
    temp = 0.01
    step_decay = 0.9999
    temp_decay = 0.9999
    
    for i in range(n_steps):
        if time.time() - t_start > time_limit:
            break
        
        energy, grad = _energy_and_grad(x, target_r)
        
        noise = np.random.randn(len(x))
        x = x - step_size * grad + np.sqrt(2 * step_size * temp) * noise
        
        x = np.clip(x, -0.49, 0.49)
        
        if i % 100 == 0 or i == n_steps - 1:
            pts = x.reshape(N, 3)
            if np.all(np.isfinite(pts)):
                r = _score(pts)
                if r > best_r:
                    best_r = r
                    best_x = x.copy()
        
        step_size *= step_decay
        temp *= temp_decay
        
        if np.linalg.norm(grad) < 1e-10:
            break
    
    return best_x.reshape(N, 3)

def _generate_bcc_points(n_max=4):
    """Generate a set of BCC lattice points centered at the origin."""
    points = []
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                points.append([i, j, k])
    for i in range(-n_max, n_max):
        for j in range(-n_max, n_max):
            for k in range(-n_max, n_max):
                points.append([i + 0.5, j + 0.5, k + 0.5])
    
    points = np.array(points)
    center = points.mean(axis=0)
    points = points - center
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:N]]

def _bcc_objective(params, V):
    """
    Objective function for BCC lattice optimization.
    params: [scale, tx, ty, tz, rx, ry, rz]
    V: base BCC points (N, 3)
    Returns negative score.
    """
    s, tx, ty, tz, rx, ry, rz = params
    
    # Rotation matrices
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    C = s * (V @ R.T) + np.array([tx, ty, tz])
    
    # Check boundaries
    dist_to_bound = 0.5 - np.abs(C).max(axis=1)
    r_bound = np.min(dist_to_bound)
    
    if r_bound < 0:
        return -r_bound + 10.0
    
    D = cdist(C, C)
    np.fill_diagonal(D, np.inf)
    min_dist = np.min(D)
    r_pair = min_dist / 2.0
    
    r = min(r_bound, r_pair)
    return -r

def _lattice_optimize(V, time_limit=20.0):
    """
    Optimize BCC lattice parameters using differential evolution.
    """
    t_start = time.time()
    
    # Bounds: scale, tx, ty, tz, rx, ry, rz
    bounds = [(0.05, 0.2), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5),
              (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
    
    def objective(params):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _bcc_objective(params, V)
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            args=(V,),
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=True
        )
        return result.x
    except TimeoutError:
        return None
    except Exception:
        return None

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin under 120s

    # ---- Seed: Simple cubic 5x5x5 grid ----
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:N].copy()

    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    rng = np.random.RandomState(42)

    # Strategy 1: BCC Lattice Optimization with Differential Evolution
    if time.time() - t0 < DEADLINE - 40.0:
        V = _generate_bcc_points(4)
        # Initial guess for DE
        s_init = 1.0 / (0.866 + V.max() - V.min())
        x0 = [s_init, 0, 0, 0, 0, 0, 0]
        
        # Run DE
        de_time = min(30.0, DEADLINE - (time.time() - t0) - 30.0)
        params = _lattice_optimize(V, time_limit=de_time)
        
        if params is not None:
            s, tx, ty, tz, rx, ry, rz = params
            cx, sx = np.cos(rx), np.sin(rx)
            cy, sy = np.cos(ry), np.sin(ry)
            cz, sz = np.cos(rz), np.sin(rz)
            
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            
            pts_bcc = s * (V @ R.T) + np.array([tx, ty, tz])
            pts_bcc = np.clip(pts_bcc, -0.5 + 1e-9, 0.5 - 1e-9)
            
            if np.all(np.isfinite(pts_bcc)):
                r_bcc = _score(pts_bcc)
                if r_bcc > best_r:
                    best_r = r_bcc
                    best_pts = pts_bcc.copy()

    # Strategy 2: Run BFGS with increasing target radii
    targets = [0.105, 0.110, 0.112, 0.114, 0.116, 0.118]
    
    current_pts = best_pts.copy()
    
    for target_r in targets:
        if time.time() - t0 > DEADLINE - 30.0:
            break
            
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
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _score(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
                current_pts = opt_pts.copy()

    # Strategy 3: SGLD for escaping local minima
    if time.time() - t0 < DEADLINE - 20.0:
        slda_targets = [0.112, 0.114, 0.116]
        
        for target_r in slda_targets:
            if time.time() - t0 > DEADLINE - 10.0:
                break
            
            for restart in range(3):
                if time.time() - t0 > DEADLINE - 5.0:
                    break
                
                if restart == 0:
                    noise = rng.normal(0, 0.01, size=(N, 3))
                    init_pts = np.clip(best_pts + noise, -0.49, 0.49)
                elif restart == 1:
                    noise = rng.normal(0, 0.03, size=(N, 3))
                    init_pts = np.clip(pts + noise, -0.49, 0.49)
                else:
                    init_pts = rng.uniform(-0.45, 0.45, size=(N, 3))
                
                slda_time = min(15.0, DEADLINE - (time.time() - t0) - 5.0)
                opt_pts = _slda_optimize(init_pts, target_r, time_limit=slda_time)
                
                if np.all(np.isfinite(opt_pts)):
                    opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                    r_opt = _score(opt_pts)
                    
                    if r_opt > best_r:
                        best_r = r_opt
                        best_pts = opt_pts.copy()

    # Strategy 4: Final BFGS polish
    if time.time() - t0 < DEADLINE - 5.0 and best_r > 0.1:
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
