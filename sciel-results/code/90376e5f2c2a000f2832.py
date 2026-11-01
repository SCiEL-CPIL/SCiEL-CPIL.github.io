import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _compute_r(cents):
    """Compute the maximum feasible radius for given centers."""
    n = len(cents)
    # Wall distances
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _soft_min_obj(params, n, beta, r_target):
    """
    Objective: sum of soft violations of constraints.
    Variables: 2N positions + 1 radius r.
    We want to maximize r subject to:
      |c_i - c_j| >= 2r  for all i<j
      -0.5 + r <= c_ix <= 0.5 - r
      -0.5 + r <= c_iy <= 0.5 - r
    
    We define violations as:
      For pair (i,j): max(0, 2r - |c_i - c_j|)
      For wall: max(0, r - (0.5 - |c_ix|)), max(0, r - (0.5 - |c_iy|))
    
    Using soft-min (smooth approximation of max) for differentiability.
    Actually, let's use a penalty approach:
    obj = -r + penalty * sum of violations
    
    But better: we want to find if a given r is feasible, and maximize r.
    Let's use: minimize -r + lambda * sum of soft_violations
    
    Actually, the inspiration uses soft-min. Let's define:
    For each constraint g_k(x) >= 0, we add max(0, -g_k(x))^2 or similar.
    
    Let's use: obj = -r + beta * sum_k (max(0, -g_k))^2
    where g_k are the constraint functions (should be >= 0).
    
    g_pair(i,j) = |c_i - c_j|^2 - (2r)^2
    g_wall_left(i) = c_ix + 0.5 - r
    g_wall_right(i) = 0.5 - r - c_ix
    g_wall_bottom(i) = c_iy + 0.5 - r
    g_wall_top(i) = 0.5 - r - c_iy
    
    We use a smooth approximation of max(0, -g)^2.
    max(0, -g) can be approximated by softplus(-g) but that's for max(0, x).
    Actually, let's just use the hard penalty with a large beta and rely on the optimizer.
    Or use the smooth min: soft_min = -1/beta * log(sum(exp(-beta * g_k)))
    But we want to penalize violations, so we want to push g_k >= 0.
    
    Let's use: obj = -r + (1/beta) * sum_k log(1 + exp(-beta * g_k))
    This is a smooth approximation of -r + sum_k max(0, -g_k).
    """
    cents = params[:2*n].reshape(n, 2)
    r = params[2*n]
    
    # Objective: -r + penalty for violations
    obj = -r
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i+1, n):
            ci = cents[i]
            cj = cents[j]
            d2 = np.dot(ci - cj, ci - cj)
            g = d2 - 4*r*r
            if g < 0:
                obj += (1.0/beta) * np.log1p(np.exp(-beta * g))
            # else: no penalty
    
    # Wall constraints
    for i in range(n):
        x, y = cents[i]
        g_left = x + 0.5 - r
        g_right = 0.5 - r - x
        g_bot = y + 0.5 - r
        g_top = 0.5 - r - y
        
        for g in [g_left, g_right, g_bot, g_top]:
            if g < 0:
                obj += (1.0/beta) * np.log1p(np.exp(-beta * g))
    
    return obj

def _soft_min_grad(params, n, beta, r_target):
    """Gradient of the soft-min objective."""
    cents = params[:2*n].reshape(n, 2)
    r = params[2*n]
    grad = np.zeros_like(params)
    grad[2*n] = -1.0  # d(-r)/dr
    
    # Pairwise
    for i in range(n):
        for j in range(i+1, n):
            ci = cents[i]
            cj = cents[j]
            diff = ci - cj
            d2 = np.dot(diff, diff)
            g = d2 - 4*r*r
            if g < 0:
                # d/dparams of (1/beta)*log(1+exp(-beta*g))
                # = -exp(-beta*g) / (1+exp(-beta*g)) * dg/dparams
                # = -sigmoid(-beta*g) * dg/dparams
                sig = 1.0 / (1.0 + np.exp(beta * g))  # sigmoid(-beta*g)
                # dg/dci = 2*diff, dg/dcj = -2*diff, dg/dr = -8r
                grad[2*i:2*i+2] += -sig * 2 * diff
                grad[2*j:2*j+2] += -sig * (-2 * diff)
                grad[2*n] += -sig * (-8 * r)
    
    # Wall
    for i in range(n):
        x, y = cents[i]
        idx_x = 2*i
        idx_y = 2*i+1
        idx_r = 2*n
        
        g_left = x + 0.5 - r
        g_right = 0.5 - r - x
        g_bot = y + 0.5 - r
        g_top = 0.5 - r - y
        
        for g, ix, iy, ir in [(g_left, 1, 0, -1), (g_right, -1, 0, -1), 
                               (g_bot, 0, 1, -1), (g_top, 0, -1, -1)]:
            if g < 0:
                sig = 1.0 / (1.0 + np.exp(beta * g))
                grad[ix] += -sig * ix
                grad[iy] += -sig * iy
                grad[ir] += -sig * ir
    
    return grad

def _violation(cents, r):
    """Check if (cents, r) is feasible. Returns max violation (0 if feasible)."""
    n = len(cents)
    max_viol = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(cents[i] - cents[j])
            if d < 2*r:
                max_viol = max(max_viol, 2*r - d)
    
    for i in range(n):
        x, y = cents[i]
        for val in [x + 0.5, 0.5 - x, y + 0.5, 0.5 - y]:
            if val < r:
                max_viol = max(max_viol, r - val)
    
    return max_viol

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
        r_actual = _compute_r(centers)
        
        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()
            
    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _compute_r(best_cents)
        
    return best_cents, best_r

def _grid_init():
    """Create a grid initial packing."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    cents = grid[:N]
    r = _compute_r(cents)
    return cents, r

def _random_init(rng):
    """Create a random initial packing."""
    cents = rng.uniform(-0.4, 0.4, size=(N, 2))
    r = _compute_r(cents)
    return cents, r

def _lbfgs_polish(cents, r, time_limit=10.0, beta=500):
    """Polish the packing using L-BFGS with soft constraints."""
    start_time = time.time()
    n = len(cents)
    
    x0 = np.zeros(2*n + 1)
    x0[:2*n] = cents.flatten()
    x0[2*n] = r
    
    bounds = [(-0.5, 0.5)] * (2*n) + [(0.01, 0.5)]
    
    try:
        res = minimize(
            lambda p: _soft_min_obj(p, n, beta, r),
            x0,
            jac=lambda p: _soft_min_grad(p, n, beta, r),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 10000, 'disp': False}
        )
        
        new_cents = res.x[:2*n].reshape(n, 2)
        new_r = res.x[2*n]
        actual_r = _compute_r(new_cents)
        
        # Ensure feasibility
        if actual_r >= r * 0.99:
            return new_cents, actual_r
    except Exception:
        pass
    
    return cents, r

def _simulated_annealing(cents, r, time_limit=30.0, rng=None):
    """Simulated annealing to improve the packing."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    start_time = time.time()
    n = len(cents)
    best_cents = cents.copy()
    best_r = r
    current_cents = cents.copy()
    current_r = r
    
    T = 0.01
    T_min = 1e-6
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        # Perturb
        mask = rng.random(n) < 0.1
        if not np.any(mask):
            mask[rng.integers(n)] = True
        current_cents[mask] += rng.normal(0, T * 0.5, size=(np.sum(mask), 2))
        current_cents[:, 0] = np.clip(current_cents[:, 0], -0.45, 0.45)
        current_cents[:, 1] = np.clip(current_cents[:, 1], -0.45, 0.45)
        
        new_r = _compute_r(current_cents)
        
        if new_r > current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_cents = current_cents.copy()
        else:
            delta = new_r - current_r
            if rng.random() < np.exp(delta / T):
                current_r = new_r
                if new_r > best_r:
                    best_r = new_r
                    best_cents = current_cents.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_cents, best_r

def _binary_search_radius(cents, r_low, r_high, tol=1e-6):
    """Binary search for the maximum feasible radius."""
    while r_high - r_low > tol:
        r_mid = (r_low + r_high) / 2
        if _violation(cents, r_mid) < 1e-9:
            r_low = r_mid
        else:
            r_high = r_mid
    return r_low

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # 1. Initial packings
    hex_cents, hex_r = _hex_init()
    grid_cents, grid_r = _grid_init()
    
    best_cents = hex_cents.copy() if hex_r >= grid_r else grid_cents.copy()
    best_r = max(hex_r, grid_r)
    
    rng = np.random.default_rng(42)
    
    # 2. Try multiple initializations and polish
    initializations = [(hex_cents, hex_r), (grid_cents, grid_r)]
    
    for _ in range(3):
        rc, rr = _random_init(rng)
        initializations.append((rc, rr))
    
    for cents, r in initializations:
        remaining = deadline - (time.time() - start_time)
        if remaining < 15:
            break
        
        # SA phase
        sa_cents, sa_r = _simulated_annealing(cents, r, time_limit=min(remaining - 10, 20.0), rng=rng)
        if sa_r > best_r:
            best_r = sa_r
            best_cents = sa_cents.copy()
        
        # L-BFGS polish
        remaining = deadline - (time.time() - start_time)
        if remaining > 10:
            pol_cents, pol_r = _lbfgs_polish(sa_cents, sa_r, time_limit=min(remaining - 5, 10.0), beta=500)
            if pol_r > best_r:
                best_r = pol_r
                best_cents = pol_cents.copy()
    
    # 3. High-precision polish on the best solution
    remaining = deadline - (time.time() - start_time)
    if remaining > 15:
        # Multiple rounds of polish with increasing beta
        for beta in [1000, 5000, 8000]:
            remaining = deadline - (time.time() - start_time)
            if remaining < 10:
                break
            pol_cents, pol_r = _lbfgs_polish(best_cents, best_r, time_limit=min(remaining - 5, 10.0), beta=beta)
            if pol_r > best_r:
                best_r = pol_r
                best_cents = pol_cents.copy()
    
    # 4. Final SA + polish cycle if time permits
    remaining = deadline - (time.time() - start_time)
    if remaining > 20:
        sa_cents, sa_r = _simulated_annealing(best_cents, best_r, time_limit=min(remaining - 10, 15.0), rng=rng)
        if sa_r > best_r:
            best_r = sa_r
            best_cents = sa_cents.copy()
        
        remaining = deadline - (time.time() - start_time)
        if remaining > 10:
            pol_cents, pol_r = _lbfgs_polish(best_cents, best_r, time_limit=min(remaining - 5, 10.0), beta=8000)
            if pol_r > best_r:
                best_r = pol_r
                best_cents = pol_cents.copy()
    
    # Ensure the returned solution is valid
    final_r = _compute_r(best_cents)
    if final_r <= 0:
        # Fallback to hexagonal
        best_cents, best_r = _hex_init()
    
    return best_cents

# EVOLVE-BLOCK-END
