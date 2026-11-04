"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

def _score(points):
    """Largest common radius: min of half pairwise distances and wall clearance."""
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = dists[iu].min()
    clearance = (0.5 - np.abs(points)).min()
    return min(min_pair * 0.5, clearance)

def _soft_min_value_and_grad(values, alpha=50.0):
    """
    Soft-min of values using log-sum-exp (exponential).
    Returns (soft_min_value, gradient w.r.t. values).
    """
    v = np.clip(values, -100, 100)
    v_max = np.max(v)
    shifted = v - v_max
    exp_vals = np.exp(alpha * shifted)
    sum_exp = np.sum(exp_vals)
    soft_min = v_max - np.log(sum_exp) / alpha
    grad = -exp_vals / sum_exp
    return soft_min, grad

def _energy_and_grad(x, target_r):
    """
    Energy function for BFGS optimization.
    x is a flat array of length 3N.
    target_r is the desired radius.
    Returns (energy, gradient).
    """
    c = x.reshape(N, 3)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0)
    active = dists < 2.0 * target_r
    active[np.diag_indices_from(active)] = False
    violations = np.where(active, 2.0 * target_r - dists, 0.0)
    energy_pair = 0.5 * np.sum(violations**2)
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)
    wall_dists = 0.5 - np.abs(c)
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    energy_wall = np.sum(wall_violations**2)
    signs = np.sign(c)
    grad_wall = 2.0 * wall_violations * signs
    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall).flatten()
    return total_energy, total_grad

def _soft_min_objective_and_grad(x, alpha=50.0):
    """
    Smooth surrogate: maximize soft-min of {d_ij/2 for i<j, wall_clearance_i}.
    Returns (negative_soft_min, gradient) for minimization.
    Vectorized gradient computation.
    """
    c = x.reshape(N, 3)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(N, k=1)
    pair_vals = dists[iu] / 2.0
    wall_vals = (0.5 - np.abs(c)).flatten()
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    soft_min, grad_vals = _soft_min_value_and_grad(all_vals, alpha)
    
    # Vectorized gradient propagation
    grad_c = np.zeros((N, 3))
    
    # Pairwise: for each pair (i,j), grad of d_ij/2 w.r.t. c_i = (c_i - c_j)/(2*d_ij)
    # We need: grad_c[i] += pair_grad[i,j] * (c_i - c_j) / (2 * d_ij)
    #          grad_c[j] -= pair_grad[i,j] * (c_i - c_j) / (2 * d_ij)
    # Using scatter operations
    pair_grads = grad_vals[:len(pair_vals)]
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    
    # Build contribution matrix: contrib[i,j] = pair_grad[i,j] * (c_i - c_j) / (2 * d_ij)
    # Only upper triangle matters
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    contrib = np.zeros_like(diff)
    contrib[iu[0], iu[1]] = pair_grads[:, np.newaxis] * unit_vecs[iu[0], iu[1]] * 0.5
    
    # grad_c[i] = sum_j contrib[i,j]  (but we need sign handling)
    # Actually: for pair (i,j) with i<j:
    #   grad_c[i] += contrib[i,j]
    #   grad_c[j] -= contrib[i,j]
    # So grad_c = sum_j contrib[i,j] - sum_j contrib[j,i]
    # But contrib is only filled in upper triangle, so:
    #   grad_c[i] += sum_{j>i} contrib[i,j] - sum_{j<i} contrib[j,i]
    
    # More efficiently: grad_c = row_sum(contrib) - col_sum(contrib) for upper triangle
    # Since contrib[j,i] = 0 for j>i in our setup:
    # grad_c[i] = sum_j contrib[i,j] - sum_j contrib[j,i]
    # But contrib[j,i] is only non-zero when j<i, i.e., contrib[i,j] for j>i in upper triangle
    
    # Let's just do it directly with the structure:
    # For each i: grad_c[i] = sum_{j>i} contrib[i,j] - sum_{j<i} contrib[j,i]
    # Since contrib is zero in lower triangle:
    # grad_c[i] = sum_{j>i} contrib[i,j] + sum_{j<i} (-contrib[j,i])
    
    # Simpler: use the fact that the full gradient is antisymmetric
    # grad_c[i] = sum_j [contrib[i,j] - contrib[j,i]]
    # But since we only have upper triangle, let's compute properly
    
    # Actually the cleanest vectorized way:
    # The pairwise part of grad_c is: for each i, sum over j != i of:
    #   if i < j: + pair_grad[i,j] * (c_i - c_j)/(2*d_ij)
    #   if i > j: - pair_grad[j,i] * (c_j - c_i)/(2*d_ij) = + pair_grad[j,i] * (c_i - c_j)/(2*d_ij)
    # So in both cases: grad_c[i] += sum_{j != i} pair_grad[min(i,j),max(i,j)] * (c_i - c_j)/(2*d_ij)
    
    # This is: grad_c[i] = sum_j G[i,j] * (c_i - c_j) where G is symmetric with G[i,j] = pair_grad[min,max]/(2*d)
    
    # Build symmetric weight matrix
    W = np.zeros((N, N))
    W[iu[0], iu[1]] = pair_grads / (2.0 * dists[iu])
    W[iu[1], iu[0]] = W[iu[0], iu[1]]
    
    # grad_c[i] = sum_j W[i,j] * (c_i - c_j) = c_i * sum_j W[i,j] - sum_j W[i,j] * c_j
    row_sum = W.sum(axis=1, keepdims=True)  # (N, 1)
    weighted_sum = W @ c  # (N, 3)
    grad_c += c * row_sum - weighted_sum
    
    # Wall gradient contribution
    wall_grads = grad_vals[len(pair_vals):].reshape(N, 3)
    signs = np.sign(c)
    grad_c += wall_grads * (-signs)
    
    objective = -soft_min
    gradient = (-grad_c).flatten()
    return objective, gradient

def _optimize_bfgs(init_pts, target_r, max_iter=1000, gtol=1e-9, time_limit=50.0):
    t_start = time.time()
    def objective(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _energy_and_grad(x, target_r)
    try:
        result = minimize(objective, init_pts.flatten(), method='BFGS', jac=True,
                          options={'maxiter': max_iter, 'gtol': gtol, 'disp': False})
        return result.x.reshape(N, 3), result.success
    except TimeoutError:
        return init_pts, False
    except Exception:
        return init_pts, False

def _soft_min_optimize(init_pts, alpha=50.0, max_iter=2000, gtol=1e-10, time_limit=30.0):
    t_start = time.time()
    def objective(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _soft_min_objective_and_grad(x, alpha)
    try:
        result = minimize(objective, init_pts.flatten(), method='BFGS', jac=True,
                          options={'maxiter': max_iter, 'gtol': gtol, 'disp': False})
        return result.x.reshape(N, 3)
    except TimeoutError:
        return init_pts
    except Exception:
        return init_pts

def _slda_optimize(init_pts, target_r, time_limit=10.0, n_steps=None):
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

def _generate_fcc_points(n_max=4):
    """Generate FCC lattice points centered at origin, select N closest."""
    points = []
    # FCC: simple cubic + face centers
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                points.append([i, j, k])
    # Face centers
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max):
                points.append([i, j, k + 0.5])
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max):
            for k in range(-n_max, n_max + 1):
                points.append([i, j + 0.5, k])
    for i in range(-n_max, n_max):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                points.append([i + 0.5, j, k])
    
    points = np.array(points)
    center = points.mean(axis=0)
    points = points - center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:N]]

def _generate_bcc_points(n_max=4):
    """Generate BCC lattice points centered at origin, select N closest."""
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

def _lattice_objective(params, V):
    """Objective for lattice optimization. params: [scale, tx, ty, tz, rx, ry, rz]"""
    s, tx, ty, tz, rx, ry, rz = params
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    C = s * (V @ R.T) + np.array([tx, ty, tz])
    dist_to_bound = 0.5 - np.abs(C).max(axis=1)
    r_bound = np.min(dist_to_bound)
    if r_bound < 0:
        return -r_bound + 10.0
    D = cdist(C, C)
    np.fill_diagonal(D, np.inf)
    min_dist = np.min(D)
    r = min(r_bound, min_dist / 2.0)
    return -r

def _lattice_optimize(V, time_limit=20.0):
    t_start = time.time()
    bounds = [(0.05, 0.2), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5),
              (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
    def objective(params):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _lattice_objective(params, V)
    try:
        result = differential_evolution(objective, bounds, args=(V,),
                                        maxiter=50, popsize=15, tol=1e-6, seed=42, polish=True)
        return result.x
    except (TimeoutError, Exception):
        return None

def construct():
    t0 = time.time()
    DEADLINE = 115.0

    rng = np.random.RandomState(42)

    # ---- Initial configurations ----
    # Simple cubic grid
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts_grid = grid[:N].copy()

    # FCC lattice
    V_fcc = _generate_fcc_points(4)
    # Scale FCC to fit in cube
    fcc_scale = 0.45 / np.abs(V_fcc).max()
    pts_fcc = V_fcc * fcc_scale
    pts_fcc = np.clip(pts_fcc, -0.5 + 1e-9, 0.5 - 1e-9)

    # BCC lattice
    V_bcc = _generate_bcc_points(4)
    bcc_scale = 0.45 / np.abs(V_bcc).max()
    pts_bcc_init = V_bcc * bcc_scale
    pts_bcc_init = np.clip(pts_bcc_init, -0.5 + 1e-9, 0.5 - 1e-9)

    # Pick best initial
    candidates = [pts_grid, pts_fcc, pts_bcc_init]
    best_pts = candidates[0]
    best_r = _score(best_pts)
    for cand in candidates[1:]:
        r = _score(cand)
        if r > best_r:
            best_r = r
            best_pts = cand.copy()

    # Strategy 1: Lattice optimization (shorter time)
    if time.time() - t0 < DEADLINE - 50.0:
        for V, name in [(V_fcc, 'fcc'), (V_bcc, 'bcc')]:
            if time.time() - t0 > DEADLINE - 40.0:
                break
            de_time = min(15.0, DEADLINE - (time.time() - t0) - 25.0)
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
                pts_lat = s * (V @ R.T) + np.array([tx, ty, tz])
                pts_lat = np.clip(pts_lat, -0.5 + 1e-9, 0.5 - 1e-9)
                if np.all(np.isfinite(pts_lat)):
                    r_lat = _score(pts_lat)
                    if r_lat > best_r:
                        best_r = r_lat
                        best_pts = pts_lat.copy()

    # Strategy 2: BFGS with violation energy (increasing targets)
    targets = [0.108, 0.110, 0.112, 0.114, 0.116]
    current_pts = best_pts.copy()
    for target_r in targets:
        if time.time() - t0 > DEADLINE - 45.0:
            break
        noise = rng.normal(0, 0.004, size=(N, 3))
        init_pts = np.clip(current_pts + noise, -0.49, 0.49)
        opt_pts, success = _optimize_bfgs(init_pts, target_r, max_iter=1500, gtol=1e-10,
                                           time_limit=min(12.0, DEADLINE - (time.time() - t0) - 5.0))
        if np.all(np.isfinite(opt_pts)):
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _score(opt_pts)
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
                current_pts = opt_pts.copy()

    # Strategy 3: Multiple soft-min restarts (main workhorse)
    alphas = [30.0, 50.0, 70.0, 100.0]
    n_restarts = 4
    for alpha in alphas:
        if time.time() - t0 > DEADLINE - 20.0:
            break
        for restart in range(n_restarts):
            if time.time() - t0 > DEADLINE - 12.0:
                break
            if restart == 0:
                noise = rng.normal(0, 0.002, size=(N, 3))
                init_pts = np.clip(best_pts + noise, -0.49, 0.49)
            elif restart == 1:
                noise = rng.normal(0, 0.008, size=(N, 3))
                init_pts = np.clip(best_pts + noise, -0.49, 0.49)
            elif restart == 2:
                # Mix with FCC
                mix = 0.3
                init_pts = (1 - mix) * best_pts + mix * pts_fcc
                noise = rng.normal(0, 0.005, size=(N, 3))
                init_pts = np.clip(init_pts + noise, -0.49, 0.49)
            else:
                # Random restart
                init_pts = rng.uniform(-0.45, 0.45, size=(N, 3))
            
            sm_time = min(12.0, DEADLINE - (time.time() - t0) - 5.0)
            opt_pts = _soft_min_optimize(init_pts, alpha=alpha, max_iter=3000, gtol=1e-11, time_limit=sm_time)
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _score(opt_pts)
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()

    # Strategy 4: SGLD for escaping local minima
    if time.time() - t0 < DEADLINE - 15.0:
        slda_targets = [0.113, 0.115]
        for target_r in slda_targets:
            if time.time() - t0 > DEADLINE - 8.0:
                break
            for restart in range(2):
                if time.time() - t0 > DEADLINE - 5.0:
                    break
                if restart == 0:
                    noise = rng.normal(0, 0.01, size=(N, 3))
                    init_pts = np.clip(best_pts + noise, -0.49, 0.49)
                else:
                    init_pts = 0.7 * best_pts + 0.3 * pts_fcc
                    noise = rng.normal(0, 0.005, size=(N, 3))
                    init_pts = np.clip(init_pts + noise, -0.49, 0.49)
                slda_time = min(8.0, DEADLINE - (time.time() - t0) - 5.0)
                opt_pts = _slda_optimize(init_pts, target_r, time_limit=slda_time)
                if np.all(np.isfinite(opt_pts)):
                    opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                    r_opt = _score(opt_pts)
                    if r_opt > best_r:
                        best_r = r_opt
                        best_pts = opt_pts.copy()

    # Strategy 5: Final high-precision polish
    if time.time() - t0 < DEADLINE - 3.0 and best_r > 0.105:
        polish_time = min(6.0, DEADLINE - (time.time() - t0) - 1.0)
        opt_pts = _soft_min_optimize(best_pts.copy(), alpha=150.0, max_iter=5000, gtol=1e-13, time_limit=polish_time)
        if np.all(np.isfinite(opt_pts)):
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _score(opt_pts)
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()

    # Final safety check
    best_pts = np.clip(best_pts, -0.5 + 1e-9, 0.5 - 1e-9)
    if not np.all(np.isfinite(best_pts)):
        best_pts = pts_grid.copy()
    return best_pts
# EVOLVE-BLOCK-END
