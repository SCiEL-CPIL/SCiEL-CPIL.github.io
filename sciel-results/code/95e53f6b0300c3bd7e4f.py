import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 50
HALF = 0.5
IU = np.triu_indices(N, k=1)

def _compute_r(pts):
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    d_walls = np.minimum(
        np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
        np.minimum(HALF + pts[:, 0], HALF + pts[:, 1])
    )
    r_wall = np.min(d_walls)
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    return float(min(r_wall, r_pair))

def _soft_min_obj_and_grad(x_flat, beta):
    pts = x_flat.reshape(N, 2)
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1e15)
    
    d_left = HALF + pts[:, 0]
    d_right = HALF - pts[:, 0]
    d_bottom = HALF + pts[:, 1]
    d_top = HALF - pts[:, 1]
    
    pair_dists = dists[IU]
    wall_dists = np.concatenate([d_left, d_right, d_bottom, d_top])
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    max_d = np.max(all_dists)
    exp_terms = np.exp(-beta * (all_dists - max_d))
    S = np.sum(exp_terms)
    weights = exp_terms / S
    
    soft_min_val = -1.0 / beta * (np.log(S) - beta * max_d)
    obj = -soft_min_val
    
    grad = np.zeros_like(pts)
    
    pair_weights = weights[:len(pair_dists)]
    pair_diffs = diff[IU[0], IU[1], :]
    pair_dists_safe = np.maximum(pair_dists, 1e-12)
    
    i_idx = IU[0]
    j_idx = IU[1]
    factors = pair_weights / pair_dists_safe
    force = factors[:, None] * pair_diffs
    
    np.add.at(grad, i_idx, force)
    np.add.at(grad, j_idx, -force)
    
    w_left = weights[len(pair_dists):len(pair_dists)+N]
    w_right = weights[len(pair_dists)+N:len(pair_dists)+2*N]
    w_bottom = weights[len(pair_dists)+2*N:len(pair_dists)+3*N]
    w_top = weights[len(pair_dists)+3*N:]
    
    grad[:, 0] += w_left
    grad[:, 0] -= w_right
    grad[:, 1] += w_bottom
    grad[:, 1] -= w_top
    
    return obj, grad.flatten()

def _violation_obj_and_grad(x_flat, r_target):
    pts = x_flat.reshape(N, 2)
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1e15)
    
    pair_dists = dists[IU]
    pair_diffs = diff[IU[0], IU[1], :]
    
    pair_viol = np.maximum(0.0, 2.0 * r_target - pair_dists)
    obj = np.sum(pair_viol ** 2)
    grad = np.zeros_like(pts)
    
    active = pair_viol > 0
    if np.any(active):
        v_active = pair_viol[active]
        d_active = pair_dists[active]
        diff_active = pair_diffs[active]
        factor = -2.0 * v_active / d_active
        i_idx = IU[0][active]
        j_idx = IU[1][active]
        force = factor[:, None] * diff_active
        np.add.at(grad, i_idx, force)
        np.add.at(grad, j_idx, -force)

    d_left = HALF + pts[:, 0]
    d_right = HALF - pts[:, 0]
    d_bottom = HALF + pts[:, 1]
    d_top = HALF - pts[:, 1]
    
    v_left = np.maximum(0.0, r_target - d_left)
    v_right = np.maximum(0.0, r_target - d_right)
    v_bottom = np.maximum(0.0, r_target - d_bottom)
    v_top = np.maximum(0.0, r_target - d_top)
    
    obj += np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)
    
    grad[:, 0] += -2.0 * v_left
    grad[:, 0] += 2.0 * v_right
    grad[:, 1] += -2.0 * v_bottom
    grad[:, 1] += 2.0 * v_top
    
    return obj, grad.flatten()

def _hex_init():
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    return np.array(pts[:N])

def _grid_init():
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _polish_soft_min(pts, beta, deadline, max_iter=500):
    x0 = pts.flatten()
    margin = 1e-9
    bounds = [(-HALF + margin, HALF - margin)] * (2 * N)
    try:
        res = minimize(
            lambda x: _soft_min_obj_and_grad(x, beta)[0],
            x0,
            jac=lambda x: _soft_min_obj_and_grad(x, beta)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return pts

def _polish_violation(pts, r_target, deadline, max_iter=500):
    x0 = pts.flatten()
    margin = 1e-9
    bounds = [(-HALF + margin, HALF - margin)] * (2 * N)
    try:
        res = minimize(
            lambda x: _violation_obj_and_grad(x, r_target)[0],
            x0,
            jac=lambda x: _violation_obj_and_grad(x, r_target)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    # Initialize with multiple strategies and pick the best
    inits = [_hex_init(), _grid_init()]
    # Add a random init
    inits.append(rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2)))
    
    best_pts = None
    best_r = 0.0
    for init in inits:
        r = _compute_r(init)
        if r > best_r:
            best_r = r
            best_pts = init
    
    # Progressive Beta Optimization
    # Start with low beta, increase gradually
    beta = 10.0
    beta_max = 2000.0
    
    # Main loop: Optimize soft-min objective
    while time.time() < deadline - 20.0:
        # Polish with current beta
        polished_pts = _polish_soft_min(best_pts, beta, deadline, max_iter=200)
        r_new = _compute_r(polished_pts)
        
        if r_new > best_r:
            best_r = r_new
            best_pts = polished_pts
        
        # Increase beta
        beta *= 1.1
        if beta > beta_max:
            beta = beta_max
    
    # Final high-precision polish using violation minimization
    # Use the best radius found as the target
    if time.time() < deadline - 5.0:
        final_target = best_r
        # Run multiple polishes to ensure convergence
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            polished_pts = _polish_violation(best_pts, final_target, deadline, max_iter=500)
            r_final = _compute_r(polished_pts)
            if r_final > best_r:
                best_r = r_final
                best_pts = polished_pts
                final_target = r_final
    
    return best_pts
