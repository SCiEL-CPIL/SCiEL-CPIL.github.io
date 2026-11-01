import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
DEADLINE = 140.0  # Safety margin before 150s limit

def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary: min distance from each point to the square edge
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r

def _soft_min_obj(x, beta, deadline):
    """
    Objective: -soft_min of all pairwise distances and wall distances.
    soft_min(z) = -1/beta * log(sum(exp(-beta*z)))
    We want to maximize the minimum distance, so minimize -soft_min.
    """
    c = x.reshape(N, 2)
    
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1e12)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu]
    
    # Wall distances
    d_left = c[:, 0] + HALF
    d_right = HALF - c[:, 0]
    d_bottom = c[:, 1] + HALF
    d_top = HALF - c[:, 1]
    
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    
    # Combine all distances
    all_dist = np.concatenate([pair_dist, wall_dist])
    
    # Soft minimum: -1/beta * log(sum(exp(-beta * d)))
    # For numerical stability, subtract max
    max_d = np.max(all_dist)
    exp_terms = np.exp(-beta * (all_dist - max_d))
    soft_min_val = max_d - np.log(np.sum(exp_terms)) / beta
    
    obj = -soft_min_val
    return float(obj)

def _soft_min_grad(x, beta):
    """
    Gradient of -soft_min objective.
    d(-soft_min)/dx_i = -d(soft_min)/dx_i
    soft_min = max_d - (1/beta)*log(sum_j exp(-beta*d_j))
    d(soft_min)/dx_i = sum_j (d(soft_min)/d(d_j)) * (d(d_j)/dx_i)
    d(soft_min)/d(d_j) = -exp(-beta*d_j) / sum_k exp(-beta*d_k)
    So d(-soft_min)/dx_i = sum_j [exp(-beta*d_j)/sum_k exp(-beta*d_k)] * (d(d_j)/dx_i)
    """
    c = x.reshape(N, 2)
    
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1e12)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu]
    
    # Wall distances
    d_left = c[:, 0] + HALF
    d_right = HALF - c[:, 0]
    d_bottom = c[:, 1] + HALF
    d_top = HALF - c[:, 1]
    
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    
    # Combine all distances
    all_dist = np.concatenate([pair_dist, wall_dist])
    n_pairs = len(pair_dist)
    n_walls = N
    
    # Numerically stable weights
    max_d = np.max(all_dist)
    exp_terms = np.exp(-beta * (all_dist - max_d))
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros_like(x)
    
    # Pair contributions
    for k in range(n_pairs):
        i, j = iu[0][k], iu[1][k]
        d = pair_dist[k]
        if d < 1e-12:
            continue
        vec = (c[i] - c[j]) / d
        w = weights[k]
        # d(pair_dist)/d(c_i) = -vec, d(pair_dist)/d(c_j) = vec
        grad_c_i = -w * vec
        grad_c_j = w * vec
        grad[2*i:2*i+2] += grad_c_i
        grad[2*j:2*j+2] += grad_c_j
    
    # Wall contributions
    for i in range(N):
        w = weights[n_pairs + i]
        dl = d_left[i]
        dr = d_right[i]
        db = d_bottom[i]
        dt = d_top[i]
        wmin = wall_dist[i]
        
        # d(wall_dist)/d(c_i) depends on which wall is closest
        if abs(dl - wmin) < 1e-10:
            # wall_dist = c_i_x + 0.5, d/dx = 1
            grad[2*i] += w * 1.0
        if abs(dr - wmin) < 1e-10:
            # wall_dist = 0.5 - c_i_x, d/dx = -1
            grad[2*i] += w * (-1.0)
        if abs(db - wmin) < 1e-10:
            # wall_dist = c_i_y + 0.5, d/dy = 1
            grad[2*i+1] += w * 1.0
        if abs(dt - wmin) < 1e-10:
            # wall_dist = 0.5 - c_i_y, d/dy = -1
            grad[2*i+1] += w * (-1.0)
    
    return grad

def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice initialization."""
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
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)

def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF, HALF)

def _random_init(rng):
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))

def _lbfgs_optimize(x0, beta, deadline):
    """Run L-BFGS-B optimization with the soft-min objective."""
    start = time.time()
    
    def obj_func(x):
        return _soft_min_obj(x, beta, deadline)
    
    def jac_func(x):
        return _soft_min_grad(x, beta)
    
    try:
        res = minimize(
            obj_func,
            x0,
            jac=jac_func,
            method='L-BFGS-B',
            bounds=[(-HALF, HALF)] * (2 * N),
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
        )
        return res.x
    except Exception:
        return x0

def _simulated_annealing(x0, deadline, rng, beta_start=20.0, beta_end=200.0, 
                          temp_start=0.01, temp_end=0.0001, steps_per_beta=50):
    """
    Simulated Annealing with adaptive beta (annealing the soft-min parameter).
    """
    x = x0.copy()
    current_obj = _soft_min_obj(x, beta_start, deadline)
    best_x = x.copy()
    best_obj = current_obj
    
    n_steps = 0
    while time.time() < deadline - 5:
        # Interpolate beta and temperature
        progress = min(1.0, n_steps / (steps_per_beta * 10))
        beta = beta_start * (beta_end / beta_start) ** progress
        temp = temp_start * (temp_end / temp_start) ** progress
        
        # Propose a move: random perturbation
        step_size = 0.01 * (1.0 - 0.5 * progress)
        delta = rng.normal(0, step_size, size=x.shape)
        x_new = x + delta
        x_new = np.clip(x_new, -HALF, HALF)
        
        obj_new = _soft_min_obj(x_new, beta, deadline)
        
        # Accept if better or with probability exp(-delta_obj/temp)
        delta_obj = obj_new - current_obj
        if delta_obj < 0 or rng.random() < np.exp(-delta_obj / max(temp, 1e-12)):
            x = x_new
            current_obj = obj_new
            if current_obj < best_obj:
                best_obj = current_obj
                best_x = x.copy()
        
        n_steps += 1
        
        # Occasionally re-temperature
        if n_steps % 100 == 0 and time.time() > deadline - 10:
            break
    
    return best_x

def _polish_with_lbfgs(x, deadline):
    """Final L-BFGS polish with high beta for precision."""
    # Try a few beta values for polishing
    for beta in [100, 200, 500]:
        if time.time() > deadline - 2:
            break
        x = _lbfgs_optimize(x, beta, deadline)
    return x

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_r = 0.0
    
    # Multiple initializations
    inits = [
        _hex_init(rng, jitter=0.0),
        _hex_init(rng, jitter=0.005),
        _hex_init(rng, jitter=0.01),
        _grid_init(rng, jitter=0.005),
        _random_init(rng),
    ]
    
    # For each init, run SA + L-BFGS polish
    for init in inits:
        if time.time() > deadline - 20:
            break
        
        # Phase 1: Simulated Annealing
        x_sa = _simulated_annealing(init.ravel(), deadline, rng,
                                     beta_start=10.0, beta_end=100.0,
                                     temp_start=0.02, temp_end=0.0005,
                                     steps_per_beta=30)
        c_sa = x_sa.reshape(N, 2)
        r_sa = _compute_r(c_sa)
        if r_sa > best_r:
            best_r = r_sa
            best_pts = c_sa
        
        # Phase 2: L-BFGS polish
        if time.time() > deadline - 5:
            break
        x_polished = _polish_with_lbfgs(x_sa, deadline)
        c_polished = x_polished.reshape(N, 2)
        c_polished = np.clip(c_polished, -HALF, HALF)
        r_polished = _compute_r(c_polished)
        if r_polished > best_r:
            best_r = r_polished
            best_pts = c_polished
    
    # Additional restarts if time permits
    n_extra = 0
    while time.time() < deadline - 15 and n_extra < 5:
        init = _random_init(rng)
        x_sa = _simulated_annealing(init.ravel(), deadline, rng,
                                     beta_start=10.0, beta_end=100.0,
                                     temp_start=0.02, temp_end=0.0005,
                                     steps_per_beta=30)
        c_sa = x_sa.reshape(N, 2)
        r_sa = _compute_r(c_sa)
        if r_sa > best_r:
            best_r = r_sa
            best_pts = c_sa
        
        if time.time() > deadline - 5:
            break
        x_polished = _polish_with_lbfgs(x_sa, deadline)
        c_polished = x_polished.reshape(N, 2)
        c_polished = np.clip(c_polished, -HALF, HALF)
        r_polished = _compute_r(c_polished)
        if r_polished > best_r:
            best_r = r_polished
            best_pts = c_polished
        n_extra += 1
    
    if best_pts is None:
        best_pts = _hex_init(rng)
    
    return best_pts
