import numpy as np
import time
from scipy.optimize import minimize

N = 200
HALF = 0.5
TIME_LIMIT = 150.0

def _score(c):
    """Return the common radius for a configuration (higher is better)."""
    c = np.asarray(c)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # inside square
    m = np.max(np.abs(c), axis=1)
    if np.any(m > HALF + 1e-12):
        return 0.0
    # pairwise min distance
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    return min(min_pair / 2.0, HALF - np.max(m))

def _soft_min_surrogate(x, beta=50.0):
    """
    Smooth surrogate for maximizing the minimum radius.
    Returns negative soft-min of constraints so we can minimize it.
    Constraints: pairwise dist/2 and wall margins.
    """
    x = x.reshape(N, 2)
    
    # Pairwise distances
    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, 1e9)
    pd = d / 2.0
    
    # Wall margins
    margin = HALF - np.max(np.abs(x), axis=1)
    
    # Collect all constraint values
    iu = np.triu_indices(N, k=1)
    pd_vals = pd[iu]
    all_c = np.concatenate([pd_vals, margin])
    
    # Soft min: -1/beta * log(sum(exp(-beta * c)))
    # To avoid overflow, subtract min
    m = np.min(all_c)
    exp_vals = np.exp(-beta * (all_c - m))
    soft_min_val = -(m + np.log(np.sum(exp_vals))) / beta
    
    return -soft_min_val

def _soft_min_grad(x, beta=50.0):
    """Gradient of the soft-min surrogate."""
    x = x.reshape(N, 2)
    
    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, 1e9)
    pd = d / 2.0
    
    margin = HALF - np.max(np.abs(x), axis=1)
    
    iu = np.triu_indices(N, k=1)
    pd_vals = pd[iu]
    all_c = np.concatenate([pd_vals, margin])
    
    m = np.min(all_c)
    exp_vals = np.exp(-beta * (all_c - m))
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    grad_x = np.zeros((N, 2))
    
    # Gradient of soft_min w.r.t c is -weights
    # We want to minimize -soft_min, so gradient is weights * dc/dx
    # dc/dx for pairwise: (x_i - x_j) / (4 * pd_ij)
    # Note: pd_ij = d_ij / 2. d(pd_ij)/dx_i = (x_i - x_j) / (2 * d_ij) = (x_i - x_j) / (4 * pd_ij)
    
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        w = weights[idx]
        if pd[i, j] < 1e-12:
            continue
        diff_ij = x[i] - x[j]
        grad_x[i] += w * diff_ij / (4.0 * pd[i, j])
        grad_x[j] -= w * diff_ij / (4.0 * pd[i, j])
        
    # Gradient for margin: dc/dx_i = -sign(x_i) for the max component
    for idx in range(N):
        w = weights[N + idx]
        x_i = x[idx]
        abs_x = np.abs(x_i)
        max_abs = np.max(abs_x)
        if max_abs < 1e-12:
            continue
        k = np.argmax(abs_x)
        grad_x[idx] -= w * np.sign(x_i[k]) * np.eye(2)[k]
        
    return grad_x.ravel()

def _generate_hex_grid(n_circles, radius_approx):
    """Generate a hexagonal grid of points."""
    # Spacing for hex grid is 2 * radius
    s = 2.0 * radius_approx
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    
    # Generate a grid slightly larger than the square
    cols = int(np.ceil(1.0 / dx)) + 4
    rows = int(np.ceil(1.0 / dy)) + 4
    
    points = []
    for j in range(-2, cols):
        x = j * dx
        y_offset = (dy / 2.0) if (j % 2 != 0) else 0.0
        for i in range(-2, rows):
            y = i * dy + y_offset
            points.append([x, y])
            
    points = np.array(points)
    
    # Keep points inside the square (with a small margin to allow movement)
    margin = 0.01
    keep = np.all(np.abs(points) <= HALF - margin, axis=1)
    points = points[keep]
    
    # If we have too many, randomly select N
    if len(points) > n_circles:
        idx = np.random.choice(len(points), n_circles, replace=False)
        points = points[idx]
        
    # If we have too few, add random points (shouldn't happen with good radius_approx)
    while len(points) < n_circles:
        p = np.random.uniform(-HALF + margin, HALF - margin, 2)
        points = np.vstack([points, p])
        
    return points[:n_circles]

def _simulated_annealing(c0, time_budget):
    """Run Simulated Annealing to improve the packing."""
    c = c0.copy()
    best_c = c.copy()
    best_score = _score(c)
    
    # Initial temperature
    T = 0.01
    # Cooling rate
    alpha = 0.9995
    T_min = 1e-6
    
    rng = np.random.default_rng(42)
    
    t0 = time.time()
    steps = 0
    
    while time.time() - t0 < time_budget and T > T_min:
        # Move: pick a random point and displace it
        i = rng.integers(N)
        # Displacement magnitude scales with T
        scale = 0.01 * np.sqrt(T)
        delta = rng.normal(0, scale, 2)
        
        c_new = c.copy()
        c_new[i] += delta
        c_new = np.clip(c_new, -HALF, HALF)
        
        # Calculate score
        score_new = _score(c_new)
        score_old = _score(c)
        
        # Delta energy (we want to maximize score, so minimize -score)
        delta_E = -score_new - (-score_old)
        
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            c = c_new
            if score_new > best_score:
                best_score = score_new
                best_c = c.copy()
                
        T *= alpha
        steps += 1
        
        # Periodic restarts or cooling adjustments could be added, but keep simple
        
    return best_c, best_score

def _polish_gradient(c, time_budget):
    """Polish using L-BFGS-B with soft-min surrogate."""
    t0 = time.time()
    x0 = c.ravel()
    beta = 50.0
    
    def objective(x_flat):
        return _soft_min_surrogate(x_flat, beta)
        
    def gradient(x_flat):
        return _soft_min_grad(x_flat, beta)
        
    bounds = [(-HALF, HALF)] * (2 * N)
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        c_opt = result.x.reshape(N, 2)
    except Exception:
        c_opt = c
        
    c_opt = np.clip(c_opt, -HALF, HALF)
    v = _score(c_opt)
    
    if v < _score(c):
        c_opt = c
        v = _score(c)
        
    return c_opt, v

def construct():
    t0 = time.time()
    DEADLINE = TIME_LIMIT - 10.0  # Safety margin
    
    # Estimate initial radius
    # Area of square is 1. Area of N circles is N * pi * r^2.
    # r ~ sqrt(1 / (N * pi)) * packing_density_factor
    # Packing density for hex is ~0.9069
    r_approx = np.sqrt(1.0 / (N * np.pi)) * np.sqrt(0.9069)
    
    # Generate initial configuration
    c_init = _generate_hex_grid(N, r_approx)
    
    best_c = c_init.copy()
    best_v = _score(c_init)
    
    # Phase 1: Simulated Annealing
    # Allocate 70% of time to SA
    sa_time = (DEADLINE - time.time()) * 0.7
    if sa_time > 5.0:
        c_sa, v_sa = _simulated_annealing(best_c, sa_time)
        if v_sa > best_v:
            best_c = c_sa
            best_v = v_sa
            
    # Phase 2: Gradient Polish
    # Allocate remaining time to Polish
    polish_time = DEADLINE - time.time() - 2.0
    if polish_time > 2.0:
        c_polished, v_polished = _polish_gradient(best_c, polish_time)
        if v_polished > best_v:
            best_c = c_polished
            best_v = v_polished
            
    # Final check
    best_c = np.clip(best_c, -HALF, HALF)
    return best_c
