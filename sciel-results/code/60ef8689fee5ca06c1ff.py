import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def _compute_radius(c):
    """Exact max radius: min(min_pair_dist/2, min wall clearance)."""
    if c.shape[0] < 2:
        return 0.5 - np.max(np.abs(c))
    wall = 0.5 - np.max(np.abs(c))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return min(wall, min_pair)

def _soft_min_objective_and_grad(x, beta):
    """Negative soft-min of all constraints (wall clearances + pair half-distances)."""
    c = x.reshape(N, 3)

    # Wall clearances per coordinate
    wall = 0.5 - np.abs(c)              # (N, 3)
    wall_flat = wall.ravel()            # (3N,)

    # Pairwise half-distances (upper triangle)
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    d_half = d / 2.0
    i, j = np.triu_indices(N, k=1)
    d_half_upper = d_half[i, j]         # (n_pairs,)

    vals = np.concatenate([wall_flat, d_half_upper])
    n_vals = len(vals)

    # Numerically stable log-sum-exp
    m = np.min(vals)
    shifted = vals - m
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = np.log(sum_exp)
    soft_min_val = m - log_sum_exp / beta

    weights = exp_vals / sum_exp        # (n_vals,)

    grad_x = np.zeros_like(c)           # (N, 3)

    # Wall contribution: d(0.5-|c_ik|)/dc_ik = -sign(c_ik)
    ww = weights[:3 * N].reshape(N, 3)
    sign_c = np.sign(c)
    grad_x -= ww * sign_c

    # Pair contribution: d(||ci-cj||/2)/dci = (ci-cj)/(2||ci-cj||)
    pw = weights[3 * N:]                # (n_pairs,)
    ci = c[i]
    cj = c[j]
    diff = ci - cj
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    grad_ci = diff / (2.0 * norms)
    grad_cj = -grad_ci

    np.add.at(grad_x, i, pw[:, np.newaxis] * grad_ci)
    np.add.at(grad_x, j, pw[:, np.newaxis] * grad_cj)

    f = -soft_min_val
    g = -grad_x.ravel()
    return f, g

def _generate_bcc_lattice(n_points, seed=123):
    """Points on a BCC lattice scaled to [-0.5,0.5]^3, then greedily pick well-spread ones."""
    n_div = 5
    pts = []
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                pts.append([i / n_div, j / n_div, k / n_div])
    for i in range(n_div):
        for j in range(n_div):
            for k in range(n_div):
                pts.append([(i + 0.5) / n_div, (j + 0.5) / n_div, (k + 0.5) / n_div])
    pts = np.array(pts) - 0.5

    rng = np.random.default_rng(seed)
    idx = np.arange(len(pts))
    rng.shuffle(idx)
    pts = pts[idx]

    selected = [pts[0]]
    for p in pts[1:]:
        if len(selected) >= n_points:
            break
        dmin = np.min(np.linalg.norm(np.array(selected) - p, axis=1))
        if dmin > 0.08:
            selected.append(p)
    while len(selected) < n_points:
        selected.append(rng.uniform(-0.45, 0.45, size=3))
    return np.array(selected[:n_points])

def _grid_initial(n_points):
    m = int(np.ceil(n_points ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n_points]

def _optimize_from_initial(c_init, deadline):
    """Beta-continuation then high-precision polish from one initial config."""
    best_c = np.clip(c_init, -0.5, 0.5).copy()
    best_val = _compute_radius(best_c)

    betas = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
    for beta in betas:
        if time.time() > deadline - 25:
            break
        try:
            res = minimize(_soft_min_objective_and_grad, best_c.ravel(),
                           args=(beta,), method='L-BFGS-B', jac=True,
                           bounds=[(-0.5, 0.5)] * (N * 3),
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            new_c = np.clip(res.x.reshape(N, 3), -0.4995, 0.4995)
            new_val = _compute_radius(new_c)
            if new_val > best_val:
                best_val = new_val
                best_c = new_c.copy()
        except Exception:
            pass

    # High-precision polish
    for beta_p in [5000, 10000]:
        if time.time() > deadline - 12:
            break
        try:
            res = minimize(_soft_min_objective_and_grad, best_c.ravel(),
                           args=(beta_p,), method='L-BFGS-B', jac=True,
                           bounds=[(-0.5, 0.5)] * (N * 3),
                           options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30})
            new_c = np.clip(res.x.reshape(N, 3), -0.49999, 0.49999)
            new_val = _compute_radius(new_c)
            if new_val > best_val:
                best_val = new_val
                best_c = new_c.copy()
        except Exception:
            pass

    return best_c, best_val

def _active_set_polish(c, r, deadline):
    """
    Refine the solution by solving a constrained optimization problem
    on the active set of constraints (pairs and walls close to the limit).
    We maximize r subject to d_ij >= 2r and wall_i >= r.
    """
    # Identify active constraints
    # Pairs
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    d_half = d / 2.0
    i, j = np.triu_indices(N, k=1)
    d_pairs = d_half[i, j]
    
    # Walls
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()
    
    # Threshold for "active"
    # We want constraints that are close to the current minimum
    # Current min is r
    # Let's take constraints with value < r + 0.01
    active_pairs_idx = np.where(d_pairs < r + 0.01)[0]
    active_walls_idx = np.where(wall_flat < r + 0.01)[0]
    
    # If no active constraints, we are far from the boundary, no need to polish
    if len(active_pairs_idx) == 0 and len(active_walls_idx) == 0:
        return c, r

    # Construct the SLSQP problem
    # Variables: c_flat (3N) + r (1)
    # Objective: -r (minimize -r)
    # Constraints:
    # 1. d_pairs[active] - 2r >= 0
    # 2. wall_flat[active] - r >= 0
    # Bounds: c in [-0.5, 0.5], r in [0, 0.5]

    def objective(x):
        return -x[-1] # Maximize r

    def objective_grad(x):
        grad = np.zeros_like(x)
        grad[-1] = -1.0
        return grad

    constraints = []
    
    # Pair constraints
    if len(active_pairs_idx) > 0:
        a_i = i[active_pairs_idx]
        a_j = j[active_pairs_idx]
        
        def pair_constraint(x):
            c_x = x[:3*N].reshape(N, 3)
            r_x = x[-1]
            diffs = c_x[a_i] - c_x[a_j]
            dists = np.linalg.norm(diffs, axis=1)
            return dists - 2 * r_x

        def pair_constraint_grad(x):
            c_x = x[:3*N].reshape(N, 3)
            r_x = x[-1]
            diffs = c_x[a_i] - c_x[a_j]
            norms = np.linalg.norm(diffs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            grads = np.zeros((len(a_i), 3*N + 1))
            
            # Gradient w.r.t c
            # d/dc_i (||c_i - c_j||) = (c_i - c_j) / ||c_i - c_j||
            # d/dc_j (||c_i - c_j||) = -(c_i - c_j) / ||c_i - c_j||
            g_i = diffs / norms
            g_j = -g_i
            
            # Scatter back to flat array
            # Note: SLSQP expects gradient w.r.t x
            # x[:3N] is c_flat
            for k in range(len(a_i)):
                grads[k, a_i[k]*3 : a_i[k]*3+3] += g_i[k]
                grads[k, a_j[k]*3 : a_j[k]*3+3] += g_j[k]
            
            # Gradient w.r.t r
            # d/d r (d - 2r) = -2
            grads[:, -1] = -2.0
            
            return grads.ravel()

        constraints.append({'type': 'ineq', 'fun': pair_constraint, 'jac': pair_constraint_grad})

    # Wall constraints
    if len(active_walls_idx) > 0:
        # Map flat indices back to (sphere, coord)
        w_spheres = active_walls_idx // 3
        w_coords = active_walls_idx % 3
        
        def wall_constraint(x):
            c_x = x[:3*N].reshape(N, 3)
            r_x = x[-1]
            clearances = 0.5 - np.abs(c_x)
            return clearances.ravel()[active_walls_idx] - r_x

        def wall_constraint_grad(x):
            c_x = x[:3*N].reshape(N, 3)
            r_x = x[-1]
            grads = np.zeros((len(active_walls_idx), 3*N + 1))
            
            # Gradient of (0.5 - |c_ik|) w.r.t c_ik is -sign(c_ik)
            signs = np.sign(c_x[w_spheres, w_coords])
            # If c_ik is 0, sign is 0, but subgradient can be -1, 0, or 1.
            # Let's use -sign(c_ik) but handle 0 as -1 or 1 arbitrarily? 
            # Actually, if c_ik=0, the wall is at max distance, so it's unlikely to be active unless r is very small.
            # If it is active, we can push it away. Let's use -sign(c_ik) with 0 mapped to 1.
            signs[signs == 0] = 1.0
            grads[:, w_spheres*3 + w_coords] = -signs
            
            # Gradient w.r.t r
            grads[:, -1] = -1.0
            
            return grads.ravel()

        constraints.append({'type': 'ineq', 'fun': wall_constraint, 'jac': wall_constraint_grad})

    # Initial guess
    x0 = np.concatenate([c.ravel(), [r]])
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (3*N) + [(0, 0.5)]
    
    try:
        res = minimize(objective, x0, method='SLSQP', jac=objective_grad, 
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-12})
        if res.success:
            new_c = res.x[:3*N].reshape(N, 3)
            new_r = res.x[-1]
            # Verify
            new_c = np.clip(new_c, -0.5, 0.5)
            actual_r = _compute_radius(new_c)
            if actual_r > r:
                return new_c, actual_r
    except Exception:
        pass
        
    return c, r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before the 150s kill

    rng = np.random.default_rng(42)

    # Diverse initial configurations
    c_bcc = _generate_bcc_lattice(N)
    c_grid = _grid_initial(N)

    inits = [c_bcc.copy(), c_grid.copy()]
    for seed in [1, 2, 3]:
        inits.append(np.clip(c_bcc + rng.normal(0, 0.015, size=(N, 3)), -0.499, 0.499))
    inits.append(np.clip(c_grid + rng.normal(0, 0.015, size=(N, 3)), -0.499, 0.499))
    inits.append(rng.uniform(-0.4, 0.4, size=(N, 3)))

    best_radius = 0.0
    best_centers = None

    for c0 in inits:
        if time.time() > deadline - 30:
            break
        c, val = _optimize_from_initial(c0, deadline)
        if val > best_radius:
            best_radius = val
            best_centers = c.copy()

    # Final exact local polish on the global best
    if best_centers is not None and time.time() < deadline - 5:
        # Use the new active set polish
        c, val = _active_set_polish(best_centers, best_radius, deadline)
        if val > best_radius:
            best_radius = val
            best_centers = c.copy()

    return best_centers
# EVOLVE-BLOCK-END
