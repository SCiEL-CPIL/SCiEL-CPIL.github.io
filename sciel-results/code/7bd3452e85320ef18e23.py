"""
Pack 200 equal circles in a unit square using soft-constrained gradient descent (L-BFGS-B).
Strategy: Maximize a smooth surrogate objective based on soft-min of constraints.
This avoids the explicit constraint handling of SLSQP and allows for more stable
convergence on the non-convex landscape by treating the packing problem as an
unconstrained optimization with a penalty/soft-min objective.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 200
HALF = 0.5
TIME_LIMIT = 140.0  # Safety margin for 150s limit

def _compute_radius(c):
    """Compute the valid radius for a configuration. Returns 0 if invalid."""
    c = np.asarray(c)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check inside square
    if np.any(np.abs(c) > HALF + 1e-12):
        return 0.0
    
    # Compute min pairwise distance / 2
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    # Compute min wall margin
    # margin_i = HALF - max(|x_i|, |y_i|)
    # Actually, distance to boundary is min(HALF - x_i, HALF + x_i, HALF - y_i, HALF + y_i)
    # Which is HALF - max(|x_i|, |y_i|)
    margins = HALF - np.max(np.abs(c), axis=1)
    min_margin = np.min(margins)
    
    return min(min_pair, min_margin)

def _soft_min_neg(vals, beta):
    """
    Compute -soft_min(vals) = logsumexp(-beta * vals) / beta
    This is a smooth approximation of -min(vals).
    We want to maximize min(vals), so we minimize -soft_min(vals).
    """
    # To avoid overflow in logsumexp, shift by min
    v = -beta * vals
    v_max = np.max(v)
    # If v_max is very large positive, exp(v - v_max) is safe.
    # If v_max is very negative, exp is small.
    # logsumexp(v) = v_max + log(sum(exp(v - v_max)))
    if np.isfinite(v_max):
        s = np.sum(np.exp(v - v_max))
        if s > 0:
            return (v_max + np.log(s)) / beta
    return 0.0

def _objective_and_grad(x, beta):
    """
    x: flattened array of shape (2N). Centers.
    Objective: -soft_min(min_pair_dists, wall_margins)
    We want to maximize the min distance/margin, so we minimize the negative.
    
    Let d_ij = ||c_i - c_j|| / 2. We want to maximize min d_ij.
    Let m_i = HALF - max(|x_i|, |y_i|). We want to maximize min m_i.
    
    Combined constraint values: V = [d_ij for i<j, m_i for i]
    We want to maximize min(V).
    Objective f(x) = -soft_min(V, beta) = logsumexp(-beta * V) / beta
    
    Gradient:
    f(x) = (1/beta) * log(sum_k exp(-beta * V_k))
    df/dx = (1/beta) * (1/sum_k exp(-beta * V_k)) * sum_k (-beta * exp(-beta * V_k) * dV_k/dx)
    df/dx = - (sum_k exp(-beta * V_k) * dV_k/dx) / sum_k exp(-beta * V_k)
    df/dx = - sum_k (w_k * dV_k/dx)
    where w_k = exp(-beta * V_k) / sum_j exp(-beta * V_j)
    """
    c = x.reshape(N, 2)
    
    # 1. Compute pairwise distances (half)
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    d_half = d / 2.0
    
    # 2. Compute wall margins
    # margin_i = HALF - max(|x_i|, |y_i|)
    # Derivative of max(|x|, |y|) is sign(x) if |x|>|y| else sign(y)
    abs_c = np.abs(c)
    max_abs = np.max(abs_c, axis=1)
    margins = HALF - max_abs
    
    # Gradient of margins w.r.t c
    # d(margin_i)/dx_i = -sign(x_i) if |x_i| > |y_i| else 0
    # d(margin_i)/dy_i = -sign(y_i) if |y_i| >= |x_i| else 0
    # Note: at equality, subgradient. We can pick either or average.
    # Let's use: if |x| >= |y|, grad_x = -sign(x), grad_y = 0
    #            else grad_x = 0, grad_y = -sign(y)
    
    grad_margins = np.zeros_like(c)
    mask_x = np.abs(c[:, 0]) >= np.abs(c[:, 1])
    grad_margins[mask_x, 0] = -np.sign(c[mask_x, 0])
    grad_margins[~mask_x, 1] = -np.sign(c[~mask_x, 1])
    
    # 3. Combine values V
    # We need to compute weights w_k and then sum w_k * dV_k/dx
    # V consists of d_half (upper triangle) and margins
    
    # To save memory and time, we can compute the gradient in chunks or directly.
    # N=200, pairs ~ 20k. This is manageable.
    
    iu = np.triu_indices(N, k=1)
    d_vals = d_half[iu] # shape (n_pairs,)
    
    # Weights for distances
    # w_d_k = exp(-beta * d_vals_k) / Z
    # Weights for margins
    # w_m_i = exp(-beta * margins_i) / Z
    # Z = sum(exp(-beta * d_vals)) + sum(exp(-beta * margins))
    
    # Calculate exponentials
    exp_d = np.exp(-beta * d_vals)
    exp_m = np.exp(-beta * margins)
    
    Z = np.sum(exp_d) + np.sum(exp_m)
    
    if Z < 1e-15:
        # If Z is tiny, all values are huge, gradient is ill-defined. 
        # This shouldn't happen if we start with reasonable config.
        return 0.0, np.zeros_like(x)
    
    obj = np.log(Z) / beta
    
    # Gradient
    grad = np.zeros_like(c)
    
    # Contribution from margins
    # sum_i w_m_i * d(margin_i)/dc_i
    w_m = exp_m / Z
    # grad_margins has shape (N, 2)
    # w_m has shape (N,)
    # Add to grad: grad += w_m[:, None] * grad_margins
    grad += w_m[:, None] * grad_margins
    
    # Contribution from distances
    # sum_k w_d_k * d(d_k)/dc
    # d_k = ||c_i - c_j|| / 2
    # d(d_k)/dc_i = (c_i - c_j) / (2 * ||c_i - c_j||) = (c_i - c_j) / (2 * 2 * d_k) = (c_i - c_j) / (4 * d_k)
    # Wait, d_k is the distance value in V. V_k = ||c_i - c_j|| / 2.
    # Let D_k = ||c_i - c_j||. V_k = D_k / 2.
    # dV_k/dc_i = (1/2) * (c_i - c_j) / D_k = (c_i - c_j) / (2 * D_k)
    # Since D_k = 2 * V_k, dV_k/dc_i = (c_i - c_j) / (4 * V_k)
    
    w_d = exp_d / Z
    
    i_idx, j_idx = iu
    # For each pair k (i, j):
    # grad[c_i] += w_d_k * (c_i - c_j) / (4 * V_k)
    # grad[c_j] += w_d_k * (c_j - c_i) / (4 * V_k)
    
    # Vectorized update
    # diff_ij = c[i] - c[j]
    # factor = w_d / (4 * V_k)
    # grad[i] += factor * diff_ij
    # grad[j] -= factor * diff_ij
    
    # Handle V_k == 0 (coincident points)
    # If V_k is very small, gradient is huge. 
    # We can clamp V_k to a small epsilon.
    V_safe = np.maximum(d_vals, 1e-8)
    
    factors = w_d / (4.0 * V_safe)
    
    diff_ij = c[i_idx] - c[j_idx]
    
    # Scatter add
    # np.add.at(grad, i_idx, factors[:, None] * diff_ij)
    # np.add.at(grad, j_idx, -factors[:, None] * diff_ij)
    
    # This might be slow for 20k pairs. Let's try.
    # Alternative: use sparse matrix or bincount? 
    # N=200 is small enough that np.add.at might be okay, but let's check.
    
    np.add.at(grad, i_idx, factors[:, None] * diff_ij)
    np.add.at(grad, j_idx, -factors[:, None] * diff_ij)
    
    # The objective is -soft_min, so we want to MINIMIZE it.
    # The gradient computed above is the gradient of soft_min? 
    # Let's re-verify.
    # f = -soft_min = - (1/beta) log sum exp(-beta V)
    # df/dx = - (1/beta) * (1/Sum) * sum (-beta exp(-beta V_k) dV_k/dx)
    # df/dx = sum (exp(-beta V_k)/Sum) dV_k/dx
    # df/dx = sum w_k dV_k/dx
    # My derivation above:
    # df/dx = - sum w_k dV_k/dx ?
    # Let's re-do.
    # f = (1/beta) log(sum exp(-beta V))  <-- This is -soft_min?
    # soft_min(V) = -1/beta log sum exp(-beta V)
    # So -soft_min(V) = 1/beta log sum exp(-beta V)
    # f = 1/beta log Z
    # df/dV_k = (1/beta) * (1/Z) * (-beta exp(-beta V_k)) = - exp(-beta V_k)/Z = -w_k
    # df/dx = sum_k (df/dV_k) * (dV_k/dx) = - sum_k w_k dV_k/dx
    # So my gradient calculation `grad` above is `sum w_k dV_k/dx`.
    # So the gradient of the objective `f` is `-grad`.
    
    return obj, -grad.ravel()

def _initial_hex_grid(s, ox, oy):
    """Generate hex grid points, select N, return array (N, 2)."""
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    # Generate a large enough grid
    cols = int(np.ceil(1.0 / dx)) + 4
    rows = int(np.ceil(1.0 / dy)) + 4
    xs = np.arange(-2, cols) * dx + ox
    ys = np.arange(-2, rows) * dy + oy
    
    pts = []
    for j, x in enumerate(xs):
        yoff = (dy / 2.0) if (j % 2 == 1) else 0.0
        for i, y in enumerate(ys):
            pts.append((x, y + yoff))
    arr = np.array(pts)
    
    # Filter to square
    keep = np.all(np.abs(arr) <= HALF, axis=1)
    arr = arr[keep]
    
    if arr.shape[0] > N:
        selected = []
        min_dists = np.full(arr.shape[0], np.inf)
        idxs = np.arange(arr.shape[0])
        rng = np.random.default_rng(42)
        start = rng.choice(idxs)
        selected.append(start)
        
        for _ in range(N - 1):
            last = arr[selected[-1]]
            d = np.linalg.norm(arr - last, axis=1)
            min_dists = np.minimum(min_dists, d)
            for s_idx in selected:
                min_dists[s_idx] = -1
            next_idx = np.argmax(min_dists)
            if min_dists[next_idx] < 0:
                break
            selected.append(next_idx)
        arr = arr[selected]
        
    return arr

def construct():
    t0 = time.time()
    DEADLINE = TIME_LIMIT - 5.0
    
    rng = np.random.default_rng(2024)
    
    best_c = None
    best_r = 0.0
    
    # Initial radius estimate
    r_init = 0.035
    
    # Generate initial configurations
    configs = []
    
    # 1. Hex grid centered
    s_hex = 2.0 * r_init * 1.01
    c_hex = _initial_hex_grid(s_hex, 0.0, 0.0)
    if c_hex.shape[0] == N:
        configs.append(c_hex)
        
    # 2. Hex grid shifted
    c_hex2 = _initial_hex_grid(s_hex, 0.01, 0.01)
    if c_hex2.shape[0] == N:
        configs.append(c_hex2)
        
    # 3. Rectangular grid
    nx, ny = 15, 14
    xs = np.linspace(-HALF + r_init, HALF - r_init, nx)
    ys = np.linspace(-HALF + r_init, HALF - r_init, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])
    if grid_pts.shape[0] >= N:
        selected = []
        min_dists = np.full(grid_pts.shape[0], np.inf)
        start = rng.choice(grid_pts.shape[0])
        selected.append(start)
        for _ in range(N - 1):
            last = grid_pts[selected[-1]]
            d = np.linalg.norm(grid_pts - last, axis=1)
            min_dists = np.minimum(min_dists, d)
            for s_idx in selected:
                min_dists[s_idx] = -1
            next_idx = np.argmax(min_dists)
            if min_dists[next_idx] < 0:
                break
            selected.append(next_idx)
        c_rect = grid_pts[selected]
        if c_rect.shape[0] == N:
            configs.append(c_rect)
            
    # 4. Random perturbation of hex
    if len(configs) > 0:
        c_rand = configs[0] + rng.normal(0, 0.005, (N, 2))
        c_rand = np.clip(c_rand, -HALF + r_init*0.5, HALF - r_init*0.5)
        configs.append(c_rand)
        
    if not configs:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        c_fallback = grid[:N]
        c_fallback = np.clip(c_fallback, -HALF + 0.03, HALF - 0.03)
        configs.append(c_fallback)

    # Run L-BFGS-B on each config
    # We will use a schedule for beta (temperature of soft-min)
    # Start with low beta (smooth), increase to high beta (sharp)
    
    betas = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    for c_init in configs:
        if time.time() > DEADLINE:
            break
            
        # Verify initial config
        r_current = _compute_radius(c_init)
        if r_current <= 0:
            c_init = np.clip(c_init, -HALF + 0.03, HALF - 0.03)
            r_current = _compute_radius(c_init)
            if r_current <= 0:
                continue
                
        x0 = c_init.ravel()
        
        # Bounds for L-BFGS-B
        bounds = [(-HALF, HALF)] * (2*N)
        
        for beta in betas:
            if time.time() > DEADLINE:
                break
                
            try:
                res = minimize(
                    _objective_and_grad,
                    x0,
                    args=(beta,),
                    jac=True,
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                
                c_opt = res.x.reshape(N, 2)
                r_opt = _compute_radius(c_opt)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_c = c_opt.copy()
                
                # Use this result as start for next beta
                x0 = res.x
                
            except Exception as e:
                # If it fails, try to continue with current x0
                pass

    # If no valid configuration found, fallback
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
        best_c = np.clip(best_c, -HALF + 0.03, HALF - 0.03)
    else:
        # Final clamp to ensure validity
        best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
        
    return best_c

if __name__ == "__main__":
    construct()
