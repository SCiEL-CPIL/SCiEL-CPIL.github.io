import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _score(cents):
    """Compute the packing radius."""
    n = len(cents)
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    if n > 1:
        tree = cKDTree(cents)
        pairs = tree.query_pairs(1.0)
        if len(pairs) > 0:
            d = np.array([np.linalg.norm(cents[i] - cents[j]) for i, j in pairs])
            pair_min = np.min(d) / 2.0
        else:
            pair_min = 0.5
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _soft_min_obj(cents, beta=50.0):
    """
    Soft-min surrogate for the packing radius.
    r = min(wall_min, pair_min)
    We use a soft-min over all constraints:
    For each circle i, wall constraints: 0.5 - |x_i|, 0.5 - |y_i|
    For each pair (i,j), pair constraint: |c_i - c_j| / 2
    
    soft_min = -1/beta * log(sum(exp(-beta * d_k)))
    where d_k are all the "distances" (wall and pair).
    """
    n = len(cents)
    # Wall distances for all circles
    wx = 0.5 - np.abs(cents[:, 0])
    wy = 0.5 - np.abs(cents[:, 1])
    wall_dists = np.concatenate([wx, wy])
    
    # Pair distances
    tree = cKDTree(cents)
    pairs = tree.query_pairs(1.0)
    if len(pairs) > 0:
        pair_dists = np.array([np.linalg.norm(cents[i] - cents[j]) / 2.0 for i, j in pairs])
    else:
        pair_dists = np.array([0.5])
    
    all_dists = np.concatenate([wall_dists, pair_dists])
    
    # Soft min
    min_d = np.min(all_dists)
    # Shift for numerical stability
    shifted = all_dists - min_d
    exp_vals = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = min_d - np.log(np.sum(exp_vals)) / beta
    
    return -soft_min_val  # We minimize this (so maximize soft_min)

def _soft_min_grad(cents, beta=50.0):
    """Gradient of the soft-min objective."""
    n = len(cents)
    grad = np.zeros_like(cents)
    
    # Wall distances
    wx = 0.5 - np.abs(cents[:, 0])
    wy = 0.5 - np.abs(cents[:, 1])
    wall_dists = np.concatenate([wx, wy])
    
    # Pair distances
    tree = cKDTree(cents)
    pairs = tree.query_pairs(1.0)
    if len(pairs) > 0:
        pair_dists = np.array([np.linalg.norm(cents[i] - cents[j]) / 2.0 for i, j in pairs])
    else:
        pair_dists = np.array([0.5])
    
    all_dists = np.concatenate([wall_dists, pair_dists])
    
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-beta * shifted)
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient contributions from wall distances
    # d(wx)/dx_i = -sign(x_i)
    # d(wy)/dy_i = -sign(y_i)
    for i in range(n):
        w_xi = weights[i]
        w_yi = weights[n + i]
        grad[i, 0] -= w_xi * (-np.sign(cents[i, 0]))
        grad[i, 1] -= w_yi * (-np.sign(cents[i, 1]))
    
    # Gradient contributions from pair distances
    # d(|c_i - c_j|/2)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    for k, (i, j) in enumerate(pairs):
        if k >= len(pair_dists):
            break
        w_k = weights[2*n + k]
        diff = cents[i] - cents[j]
        dist = np.linalg.norm(diff)
        if dist > 1e-12:
            grad[i] -= w_k * diff / (2.0 * dist)
            grad[j] += w_k * diff / (2.0 * dist)
    
    return grad

def _initial_grid():
    """Create a grid initial packing."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _hex_init():
    """Create a hexagonal initial packing."""
    best_cents = None
    best_r = 0
    
    for n_rows in range(5, 9):
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]
        
        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)
        
        if r <= best_r:
            continue
            
        centers = []
        total_y = (n_rows - 1) * np.sqrt(3.0) * r
        y_start = -0.5 + r + total_y / 2.0
        
        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * np.sqrt(3.0) * r
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = np.sqrt(3.0) * r * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])
                
        centers = np.array(centers[:N])
        r_actual = _score(centers)
        
        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()
            
    if best_cents is None:
        best_cents = _initial_grid()
        best_r = _score(best_cents)
        
    return best_cents, best_r

def _simulated_annealing(cents, time_limit=30.0, beta=50.0):
    """Simulated annealing to find a good packing."""
    start_time = time.time()
    rng = np.random.default_rng(12345)
    
    best_cents = cents.copy()
    best_r = _score(best_cents)
    current_cents = cents.copy()
    current_r = best_r
    
    # Temperature schedule
    t_start = 0.01
    t_end = 0.0001
    n_iter = 5000
    
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Temperature
        frac = it / n_iter
        T = t_start * (t_end / t_start) ** frac
        
        # Random perturbation
        step_size = T * 0.5
        delta = rng.normal(0, step_size, size=current_cents.shape)
        new_cents = current_cents + delta
        new_cents[:, 0] = np.clip(new_cents[:, 0], -0.5, 0.5)
        new_cents[:, 1] = np.clip(new_cents[:, 1], -0.5, 0.5)
        
        new_r = _score(new_cents)
        
        if new_r >= current_r or rng.random() < np.exp((new_r - current_r) / T):
            current_cents = new_cents
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_cents = current_cents.copy()
    
    return best_cents, best_r

def _lbfgs_polish(cents, beta=50.0, time_limit=10.0):
    """L-BFGS polish using soft-min objective."""
    start_time = time.time()
    
    def obj_and_grad(x):
        c = x.reshape(-1, 2)
        o = _soft_min_obj(c, beta)
        g = _soft_min_grad(c, beta).flatten()
        return o, g
    
    x0 = cents.flatten()
    
    try:
        res = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
        )
        if time.time() - start_time > time_limit:
            return cents
        new_cents = res.x.reshape(-1, 2)
        new_r = _score(new_cents)
        old_r = _score(cents)
        if new_r >= old_r:
            return new_cents
    except Exception:
        pass
    return cents

def _refine_with_soft_min(cents, time_limit=15.0):
    """Iterative refinement: SA + L-BFGS polish."""
    start_time = time.time()
    current = cents.copy()
    best = cents.copy()
    best_r = _score(best)
    
    while time.time() - start_time < time_limit - 5:
        # SA phase
        remaining_sa = max(1.0, time_limit - (time.time() - start_time) - 5)
        sa_cents, sa_r = _simulated_annealing(current, time_limit=remaining_sa * 0.6)
        
        # L-BFGS polish
        remaining_lbfgs = max(0.5, time_limit - (time.time() - start_time) - 2)
        polished = _lbfgs_polish(sa_cents, time_limit=remaining_lbfgs)
        polish_r = _score(polished)
        
        if polish_r > best_r:
            best_r = polish_r
            best = polished.copy()
            current = polished.copy()
        else:
            # Try from best
            polished2 = _lbfgs_polish(best, time_limit=remaining_lbfgs * 0.5)
            polish_r2 = _score(polished2)
            if polish_r2 > best_r:
                best_r = polish_r2
                best = polished2.copy()
                current = polished2.copy()
        
        if time.time() - start_time > time_limit - 5:
            break
            
    return best, best_r

def _high_precision_polish(cents, time_limit=5.0):
    """Final high-precision polish with tighter tolerances."""
    start_time = time.time()
    current = cents.copy()
    best = cents.copy()
    best_r = _score(best)
    
    for _ in range(3):
        if time.time() - start_time > time_limit:
            break
        polished = _lbfgs_polish(current, beta=100.0, time_limit=min(2.0, time_limit - (time.time() - start_time)))
        r = _score(polished)
        if r > best_r:
            best_r = r
            best = polished.copy()
            current = polished.copy()
    
    return best, best_r

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # 1. Multiple initializations
    init_hex, r_hex = _hex_init()
    init_grid = _initial_grid()
    r_grid = _score(init_grid)
    
    rng = np.random.default_rng(42)
    init_rand = rng.uniform(-0.45, 0.45, size=(N, 2))
    r_rand = _score(init_rand)
    
    # Pick best initialization
    inits = [(init_hex, r_hex), (init_grid, r_grid), (init_rand, r_rand)]
    inits.sort(key=lambda x: x[1], reverse=True)
    best_init = inits[0][0]
    best_r = inits[0][1]
    
    # 2. Main refinement loop with multiple starts
    best_cents = best_init.copy()
    
    remaining = deadline - (time.time() - start_time)
    
    # Try each initialization
    for init_cents, init_r in inits:
        if time.time() - start_time > deadline - 20:
            break
        
        remaining = deadline - (time.time() - start_time)
        if remaining < 15:
            break
            
        refined, refined_r = _refine_with_soft_min(init_cents, time_limit=min(remaining - 10, 40))
        
        if refined_r > best_r:
            best_r = refined_r
            best_cents = refined.copy()
    
    # 3. High-precision polish on the best
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        polished, polished_r = _high_precision_polish(best_cents, time_limit=min(remaining - 5, 15))
        if polished_r > best_r:
            best_r = polished_r
            best_cents = polished
    
    # 4. Final safety check
    remaining = deadline - (time.time() - start_time)
    if remaining > 5:
        final = _lbfgs_polish(best_cents, beta=100.0, time_limit=min(remaining - 2, 5))
        final_r = _score(final)
        if final_r > best_r:
            best_cents = final
    
    return best_cents

# EVOLVE-BLOCK-END
