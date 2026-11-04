"""Combined program: soft-min surrogate with Adam + iterative radius increase."""
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

def _softmin_value_and_grad(centers, beta):
    """
    Differentiable soft-min of (pairwise_dists/2, wall_dists).
    Returns (softmin_value, gradient w.r.t. centers).
    Uses log-sum-exp: softmin(x) = -logsumexp(-x * beta) / beta
    """
    c = centers
    n = len(c)
    
    # Pairwise distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(sq_dists + 1e-30)
    np.fill_diagonal(dists, np.inf)
    
    # Collect all constraint values: pairwise/2 and wall distances
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_vals = dists[i_idx, j_idx] / 2.0
    
    # Wall distances
    abs_c = np.abs(c)
    max_abs = np.max(abs_c, axis=1)
    wall_vals = 0.5 - max_abs
    
    # All values for soft-min
    all_vals = np.concatenate([pair_vals, wall_vals])
    n_pairs = len(pair_vals)
    n_walls = len(wall_vals)
    
    # Soft-min via log-sum-exp: softmin(x) = -logsumexp(-beta * x) / beta
    neg_beta_x = -beta * all_vals
    # Numerical stability
    m = np.max(neg_beta_x)
    logsumexp = m + np.log(np.sum(np.exp(neg_beta_x - m)))
    softmin_val = -logsumexp / beta
    
    # Gradient: d(softmin)/dx_i = -sum_j (exp(-beta*x_j) / sum_k exp(-beta*x_k)) * d(x_j)/dx_i
    # weights_j = exp(-beta * x_j) / sum_k exp(-beta * x_k)
    weights = np.exp(neg_beta_x - m) / np.sum(np.exp(neg_beta_x - m))
    
    grad = np.zeros_like(c)
    
    # Pairwise gradient: d(pair_val_ij)/dx_i = -(c_i - c_j) / (2 * dist_ij)
    # Note: pair_vals[i] corresponds to pair (i_idx[i], j_idx[i])
    # d(pair_vals[i])/dx_{i_idx[i]} = -(c_i - c_j) / (2 * dist_ij) = (c_j - c_i) / (2 * dist_ij)
    # d(pair_vals[i])/dx_{j_idx[i]} = (c_i - c_j) / (2 * dist_ij)
    pair_weights = weights[:n_pairs]
    d_pairs = dists[i_idx, j_idx]
    
    # Factor: weight / (2 * dist)
    factors = pair_weights / (2.0 * d_pairs)
    
    # For pair (i,j): val = dist/2
    # d(val)/dx_i = (x_j - x_i) / (2*dist)  [since dist = |x_i - x_j|, d(dist)/dx_i = (x_i-x_j)/dist]
    # Wait: d(dist)/dx_i = (x_i - x_j) / dist, so d(dist/2)/dx_i = (x_i - x_j) / (2*dist)
    # So grad contribution to x_i from pair (i,j) with weight w: w * (x_i - x_j) / (2*dist)
    # But softmin gradient is -sum w_j * d(x_j)/dx, so:
    # d(softmin)/dx_i = -sum_j w_j * d(x_j)/dx_i
    # For pair (i,j): d(pair_val)/dx_i = (x_i - x_j)/(2*dist), d(pair_val)/dx_j = (x_j - x_i)/(2*dist)
    # So: d(softmin)/dx_i += -w * (x_i - x_j)/(2*dist)
    #     d(softmin)/dx_j += -w * (x_j - x_i)/(2*dist) = w * (x_i - x_j)/(2*dist)
    
    vecs = c[i_idx] - c[j_idx]
    forces = factors[:, np.newaxis] * vecs
    
    np.add.at(grad, i_idx, -forces)
    np.add.at(grad, j_idx, forces)
    
    # Wall gradient: d(wall_val_i)/dx_i = -sign(c_i[k_max]) for the dominant axis
    wall_weights = weights[n_pairs:]
    for idx in range(n):
        if wall_weights[idx] > 1e-15:
            max_k = np.argmax(abs_c[idx])
            sign_val = np.sign(c[idx, max_k])
            if sign_val == 0:
                sign_val = 1.0
            # d(softmin)/dx_i[k_max] += -w * d(wall_val)/dx_i[k_max] = -w * (-sign) = w * sign
            grad[idx, max_k] += wall_weights[idx] * sign_val
    
    return softmin_val, grad

def _softmax_optimize(centers_init, max_iter=3000, lr=0.001, beta_start=50.0, beta_end=500.0, deadline=None):
    """
    Maximize the soft-min of (pairwise_dists/2, wall_dists) using Adam optimizer.
    This is a smooth surrogate for the non-differentiable max-min problem.
    """
    c = centers_init.copy()
    n = len(c)
    
    # Adam state
    m = np.zeros_like(c)
    v = np.zeros_like(c)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for t in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Anneal beta: start low (smoother), increase for sharper min
        progress = t / max_iter
        beta = beta_start + (beta_end - beta_start) * progress**2
        
        val, grad = _softmin_value_and_grad(c, beta)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**(t + 1))
        v_hat = v / (1 - beta2**(t + 1))
        
        # We want to MAXIMIZE softmin, so gradient ascent
        c -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Project back into cube with small margin
        c = np.clip(c, -0.4999, 0.4999)
        
        # Track best
        if t % 50 == 0 or t == max_iter - 1:
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    return best_c, best_r

def _minimize_energy(init_centers, target_R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Minimizes the energy for a given target radius using L-BFGS-B.
    Uses a smooth penalty for constraint violations.
    """
    def objective_and_grad(x):
        centers = x.reshape(-1, 3)
        n = len(centers)
        energy = 0.0
        grad = np.zeros_like(centers)
        
        # Pairwise penalty
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        dists = np.sqrt(sq_dists + 1e-15)
        np.fill_diagonal(dists, np.inf)
        
        i_idx, j_idx = np.triu_indices(n, k=1)
        d_ij = dists[i_idx, j_idx]
        
        close_mask = d_ij < 2.5 * target_R
        if np.any(close_mask):
            i_close = i_idx[close_mask]
            j_close = j_idx[close_mask]
            d_close = d_ij[close_mask]
            
            violation = np.maximum(0.0, 2.0 * target_R - d_close)
            energy += np.sum(violation**2)
            
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

        # Wall penalty
        abs_centers = np.abs(centers)
        max_abs = np.max(abs_centers, axis=1)
        wall_dists = 0.5 - max_abs
        
        wall_violation = np.maximum(0.0, target_R - wall_dists)
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
        # Compute actual energy at result
        e, _ = objective_and_grad(new_centers.flatten())
        return new_centers, float(e), res.success
    except Exception:
        return init_centers, np.inf, False

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    best_pos = None
    best_r = 0.0
    
    # Phase 1: Grid initialization + soft-max Adam optimization
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    # Try multiple initializations with soft-max Adam
    inits = [
        grid[:N].copy() + rng.normal(0, 0.01, size=(N, 3)),
        grid[:N].copy() + rng.normal(0, 0.02, size=(N, 3)),
        rng.uniform(-0.4, 0.4, size=(N, 3)),
    ]
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        init = np.clip(init, -0.49, 0.49)
        # Adam soft-max optimization
        new_pos, new_r = _softmax_optimize(init, max_iter=2000, lr=0.002, beta_start=30.0, beta_end=300.0, deadline=deadline - 25)
        if new_r > best_r:
            best_r = new_r
            best_pos = new_pos.copy()
    
    if best_pos is None or best_r <= 0:
        best_pos = grid[:N].copy()
        best_r = _compute_radius(best_pos)
    
    # Phase 2: Iterative radius increase with L-BFGS-B
    current_r = best_r
    step_size = 0.0005
    consecutive_failures = 0
    max_failures = 5
    
    rng_search = np.random.default_rng(123)
    
    while time.time() < deadline - 10.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.7
            consecutive_failures = 0
            if step_size < 1e-8:
                break
        
        # Try a few perturbations
        n_tries = 3
        success = False
        for trial in range(n_tries):
            if time.time() > deadline - 8.0:
                break
            
            if trial == 0:
                start_pos = best_pos.copy()
            else:
                jitter = 0.005 * trial
                start_pos = best_pos + rng_search.normal(0, jitter, size=(N, 3))
                start_pos = np.clip(start_pos, -0.499, 0.499)
            
            max_iter = 500 if current_r < 0.11 else 700
            new_pos, energy, opt_success = _minimize_energy(start_pos, target_r, max_iter=max_iter)
            
            actual_r = _compute_radius(new_pos)
            if actual_r >= target_r - 1e-7:
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
                success = True
                consecutive_failures = 0
                break
            elif actual_r > current_r:
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
        
        if not success:
            consecutive_failures += 1
            # Polish current best
            if time.time() < deadline - 15.0 and consecutive_failures % 2 == 0:
                polished, _, _ = _minimize_energy(best_pos, current_r, max_iter=400)
                polished_r = _compute_radius(polished)
                if polished_r > best_r:
                    best_pos = polished
                    best_r = polished_r
                    current_r = best_r
    
    # Phase 3: Final Adam polish with high beta for precision
    if time.time() < deadline - 3.0:
        polished, polished_r = _softmax_optimize(best_pos, max_iter=3000, lr=0.0005, beta_start=200.0, beta_end=1000.0, deadline=deadline - 1.0)
        if polished_r > best_r:
            best_pos = polished
            best_r = polished_r
    
    # Final L-BFGS-B micro-polish
    if time.time() < deadline - 1.0:
        target_r = best_r + 1e-7
        polished, energy, _ = _minimize_energy(best_pos, target_r, max_iter=500, ftol=1e-15, gtol=1e-14)
        polished_r = _compute_radius(polished)
        if polished_r > best_r:
            best_pos = polished
            best_r = polished_r
    
    # Final validation and clipping
    best_pos = np.clip(best_pos, -0.499999, 0.499999)
    
    return best_pos
