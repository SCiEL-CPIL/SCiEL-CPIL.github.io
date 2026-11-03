import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 50
DEADLINE = 140.0  # safety margin before 150s limit


def _score(centers):
    """Compute the maximum radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_wall)


def _soft_min_obj(x, beta):
    """
    Soft-min objective: -log(sum(exp(-beta * d_i))) / beta
    where d_i are the clearance values (pair distances / 2 and wall distances).
    As beta -> inf, this approaches -min(d_i).
    We minimize this (i.e., maximize the soft-min of clearances).
    """
    centers = x.reshape(N, 3)
    
    # Wall clearances: 0.5 - |c_ik| for each center and axis
    wall_vals = 0.5 - np.abs(centers)  # shape (N, 3)
    
    # Pair clearances: |c_i - c_j| / 2 for i < j
    ii, jj = np.triu_indices(N, k=1)
    diff = centers[ii] - centers[jj]
    pair_vals = np.sqrt(np.sum(diff ** 2, axis=1)) / 2.0
    
    all_vals = np.concatenate([wall_vals.ravel(), pair_vals])
    
    # Soft min: -log(sum(exp(-beta * v))) / beta
    # For numerical stability, subtract max
    v_min = np.min(all_vals)
    shifted = all_vals - v_min  # >= 0
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = -(v_min + log_sum / beta)
    
    return -soft_min_val  # We want to minimize this (maximize soft_min)


def _soft_min_grad(x, beta):
    """
    Gradient of the soft-min objective with respect to x.
    """
    centers = x.reshape(N, 3)
    
    # Wall values
    wall_vals = 0.5 - np.abs(centers)  # (N, 3)
    
    # Pair values
    ii, jj = np.triu_indices(N, k=1)
    diff = centers[ii] - centers[jj]
    pair_vals = np.sqrt(np.sum(diff ** 2, axis=1)) / 2.0
    
    all_vals = np.concatenate([wall_vals.ravel(), pair_vals])
    
    v_min = np.min(all_vals)
    shifted = all_vals - v_min
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    
    # Soft min = -(v_min + log(sum_exp) / beta)
    # d(soft_min)/d(v_k) = -1/beta * exp(-beta*(v_k - v_min)) / sum_exp
    # = -exp_vals[k] / (beta * sum_exp)
    
    weights = -exp_vals / (beta * sum_exp)  # d(soft_min)/d(v_k)
    
    # Now compute d(x)/d(v_k) for each clearance value
    grad = np.zeros_like(x)
    
    # Wall clearances: v = 0.5 - |c_ik|
    # d(v)/d(c_ik) = -sign(c_ik)
    # d(soft_min)/d(c_ik) = weights[k] * (-sign(c_ik))
    wall_weights = weights[:N * 3].reshape(N, 3)
    signs = np.sign(centers)
    grad += -wall_weights * signs
    
    # Pair clearances: v = |c_i - c_j| / 2
    # d(v)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|) = diff / (2 * |diff|)
    # d(v)/d(c_j) = -(c_i - c_j) / (2 * |c_i - c_j|)
    pair_weights = weights[N * 3:]  # shape (num_pairs,)
    norms = np.sqrt(np.sum(diff ** 2, axis=1))
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    unit_vecs = diff / safe_norms[:, np.newaxis]  # (num_pairs, 3)
    
    # d(soft_min)/d(c_i) for pair (i,j) = pair_weights[p] * unit_vecs[p]
    # d(soft_min)/d(c_j) for pair (i,j) = -pair_weights[p] * unit_vecs[p]
    
    # Accumulate using np.add.at
    for p in range(len(pair_weights)):
        grad[ii[p]] += pair_weights[p] * unit_vecs[p]
        grad[jj[p]] -= pair_weights[p] * unit_vecs[p]
    
    return grad.ravel()


def _soft_min_obj_grad(x, beta):
    """Combined objective and gradient for L-BFGS-B."""
    obj = _soft_min_obj(x, beta)
    grad = _soft_min_grad(x, beta)
    return obj, grad


def _init_grid():
    """Structured grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _init_bcc():
    """BCC-inspired initialization."""
    n_cell = 4
    cell_size = 1.0 / n_cell
    bcc_points = []
    for iz in range(n_cell):
        for iy in range(n_cell):
            for ix in range(n_cell):
                cx = -0.5 + cell_size * (ix + 0.5)
                cy = -0.5 + cell_size * (iy + 0.5)
                cz = -0.5 + cell_size * (iz + 0.5)
                bcc_points.append([cx, cy, cz])
                bcc_points.append([cx + cell_size/2, cy + cell_size/2, cz + cell_size/2])
    bcc_arr = np.array(bcc_points)
    mask = np.all(np.abs(bcc_arr) <= 0.5, axis=1)
    bcc_arr = bcc_arr[mask]
    if len(bcc_arr) >= N:
        return bcc_arr[:N].copy()
    else:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(bcc_arr), 3))
        return np.vstack([bcc_arr, extra])


def _init_rotated(seed_val):
    """Rotated random initialization."""
    np.random.seed(seed_val)
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)
    Rx = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])
    Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                  [0, 1, 0],
                  [-np.sin(phi), 0, np.cos(phi)]])
    R = Ry @ Rx
    base = np.random.uniform(-0.3, 0.3, size=(N, 3))
    rotated = base @ R.T
    max_abs = np.max(np.abs(rotated))
    if max_abs > 0:
        rotated = rotated * (0.45 / max_abs)
    return rotated


def _optimize_stage(x0, beta, maxiter, start_time, time_budget):
    """Run L-BFGS-B for one beta stage."""
    def fun_and_grad(x):
        if time.time() - start_time > time_budget:
            return _soft_min_obj(x, beta), np.zeros_like(x)
        return _soft_min_obj_grad(x, beta)
    
    bounds = [(-0.49, 0.49)] * (N * 3)
    
    try:
        result = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter, 'maxls': 50})
        return result.x
    except Exception:
        return x0


def construct():
    start_time = time.time()
    
    # Generate multiple initializations
    inits = [
        _init_grid(),
        _init_bcc(),
        _init_rotated(42),
        _init_rotated(123),
        _init_rotated(456),
    ]
    
    best_x = None
    best_score = -1.0
    
    # Try each initialization with multi-stage beta annealing
    for init_idx, centers0 in enumerate(inits):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        x = centers0.ravel()
        x = np.clip(x, -0.49, 0.49)
        
        # Multi-stage beta annealing
        beta_schedule = [50, 100, 200, 500, 1000, 2000, 5000]
        maxiter_schedule = [80, 100, 120, 150, 180, 200, 250]
        
        for i, (beta, maxiter) in enumerate(zip(beta_schedule, maxiter_schedule)):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            time_budget = min(DEADLINE - 5 - (time.time() - start_time), 25.0)
            if time_budget < 5:
                break
            
            x = _optimize_stage(x, beta, maxiter, start_time, time_budget)
            x = np.clip(x, -0.49, 0.49)
            
            current_score = _score(x.reshape(N, 3))
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
        
        # Early stopping: if we've found a good solution, stop trying more inits
        if best_score > 0.135:
            break
    
    # Perturbation restarts on the best solution found
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Perturb best solution
        x_perturbed = best_x + np.random.randn(N * 3) * 0.008
        x_perturbed = np.clip(x_perturbed, -0.49, 0.49)
        
        # Run a few stages
        for beta, maxiter in [(100, 100), (500, 150), (2000, 200)]:
            if time.time() - start_time > DEADLINE - 5:
                break
            time_budget = min(DEADLINE - 5 - (time.time() - start_time), 15.0)
            if time_budget < 3:
                break
            x_perturbed = _optimize_stage(x_perturbed, beta, maxiter, start_time, time_budget)
            x_perturbed = np.clip(x_perturbed, -0.49, 0.49)
        
        current_score = _score(x_perturbed.reshape(N, 3))
        if current_score > best_score:
            best_score = current_score
            best_x = x_perturbed.copy()
    
    # Final polish with high beta
    if time.time() - start_time < DEADLINE - 10:
        for beta in [5000, 10000, 20000]:
            if time.time() - start_time > DEADLINE - 5:
                break
            time_budget = min(DEADLINE - 5 - (time.time() - start_time), 15.0)
            if time_budget < 3:
                break
            best_x = _optimize_stage(best_x, beta, 300, start_time, time_budget)
            best_x = np.clip(best_x, -0.49, 0.49)
        
        final_score = _score(best_x.reshape(N, 3))
        if final_score > best_score:
            best_score = final_score
    
    return best_x.reshape(N, 3)
