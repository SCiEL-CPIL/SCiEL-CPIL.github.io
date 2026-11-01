"""Combined strategy: SA search + Hexagonal init + SLSQP radius maximization."""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
_DEADLINE = 0.0

def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    # Pairwise distances via KDTree
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)

def _compute_radius_fast(centers):
    """Fast radius computation for SA inner loop."""
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _construct_hexagonal():
    """Create a good initial hexagonal packing."""
    best_cents = None
    best_r = 0
    
    # Try a few row configurations
    for n_rows in range(5, 9):
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]
        
        max_rc = max(row_counts)
        # Estimate r
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
        r_actual = _compute_radius(centers)
        
        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()
            
    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _compute_radius(best_cents)
        
    return best_cents, best_r

def _sa_search(centers, rng, budget_time, T_start=0.025, T_end=1e-8):
    """Simulated annealing to improve packing."""
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius_fast(current)
    best = current.copy()
    best_r = current_r
    
    max_steps = 2000000
    steps_done = 0
    
    for step in range(max_steps):
        if time.time() - start_time > budget_time:
            break
        steps_done = step + 1
        
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Adaptive step size
        step_size = 0.012 * (1.0 + 0.5 * np.cos(np.pi * step / max_steps))
        
        idx = rng.integers(0, N)
        delta = rng.normal(0, step_size, 2)
        
        new = current.copy()
        new[idx] += delta
        new = np.clip(new, -0.5, 0.5)
        
        new_r = _compute_radius_fast(new)
        
        if new_r >= current_r:
            current = new
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            delta_r = new_r - current_r
            if delta_r > -10 * T:
                prob = np.exp(delta_r / T)
                if rng.random() < prob:
                    current = new
                    current_r = new_r
    
    return best, best_r

def _slsqp_polish(centers, time_budget=10.0):
    """
    SLSQP optimization to maximize radius r.
    Variables: x, y for each circle (2N), and r (1). Total 2N+1.
    Constraints: 
      1. |c_i - c_j|^2 - (2r)^2 >= 0
      2. -0.5 + r <= c_ix <= 0.5 - r
      3. -0.5 + r <= c_iy <= 0.5 - r
    """
    start_time = time.time()
    x0 = centers.flatten().copy()
    r0 = _compute_radius(centers)
    x0 = np.append(x0, r0)
    
    # Bounds for positions and r
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    cons = []
    
    # Pairwise constraints: |ci - cj|^2 - 4r^2 >= 0
    for i in range(N):
        for j in range(i + 1, N):
            def constraint_func(x, i=i, j=j):
                ci = x[2*i:2*i+2]
                cj = x[2*j:2*j+2]
                r_val = x[-1]
                return np.dot(ci - cj, ci - cj) - 4 * r_val * r_val
            
            def constraint_jac(x, i=i, j=j):
                g = np.zeros_like(x)
                ci = x[2*i:2*i+2]
                cj = x[2*j:2*j+2]
                r_val = x[-1]
                diff = ci - cj
                g[2*i:2*i+2] = 2 * diff
                g[2*j:2*j+2] = -2 * diff
                g[-1] = -8 * r_val
                return g
            
            cons.append({'type': 'ineq', 'fun': constraint_func, 'jac': constraint_jac})
            
    # Wall constraints
    for i in range(N):
        ix = 2 * i
        iy = 2 * i + 1
        
        # Left: x_i >= -0.5 + r  => x_i - r + 0.5 >= 0
        def c_left(x, ix=ix):
            return x[ix] - x[-1] + 0.5
        def g_left(x, ix=ix):
            g = np.zeros_like(x)
            g[ix] = 1.0
            g[-1] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c_left, 'jac': g_left})
        
        # Right: x_i <= 0.5 - r  => 0.5 - r - x_i >= 0
        def c_right(x, ix=ix):
            return 0.5 - x[-1] - x[ix]
        def g_right(x, ix=ix):
            g = np.zeros_like(x)
            g[ix] = -1.0
            g[-1] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c_right, 'jac': g_right})
        
        # Bottom: y_i >= -0.5 + r  => y_i - r + 0.5 >= 0
        def c_bot(x, iy=iy):
            return x[iy] - x[-1] + 0.5
        def g_bot(x, iy=iy):
            g = np.zeros_like(x)
            g[iy] = 1.0
            g[-1] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c_bot, 'jac': g_bot})
        
        # Top: y_i <= 0.5 - r  => 0.5 - r - y_i >= 0
        def c_top(x, iy=iy):
            return 0.5 - x[-1] - x[iy]
        def g_top(x, iy=iy):
            g = np.zeros_like(x)
            g[iy] = -1.0
            g[-1] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c_top, 'jac': g_top})
        
    def objective(x):
        return -x[-1]
        
    def objective_grad(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g
        
    try:
        result = opt.minimize(
            objective, x0, jac=objective_grad, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': 100, 'ftol': 1e-12, 'disp': False}
        )
        
        if result.success:
            new_centers = result.x[:2*N].reshape(-1, 2)
            new_r = result.x[-1]
            
            # Verify result
            new_centers = np.clip(new_centers, -0.5, 0.5)
            verified_r = _compute_radius(new_centers)
            
            return new_centers, verified_r
    except Exception:
        pass
        
    return centers, _compute_radius(centers)

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 145.0
    
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_radius = 0.0
    
    # 1. Hexagonal Initialization
    hex_centers, hex_r = _construct_hexagonal()
    if hex_r > best_radius:
        best_radius = hex_r
        best_centers = hex_centers.copy()
        
    # 2. Grid-based initializations
    m = int(np.ceil(np.sqrt(N)))  # m=8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    # Try several different starting points
    n_starts = 3
    for s in range(n_starts):
        if time.time() > _DEADLINE - 30:
            break
        
        if s == 0:
            centers = grid[:N].copy()
            centers += rng.normal(0, 0.008, centers.shape)
        elif s == 1:
            # Slightly different grid offset
            ticks2 = (np.arange(m) + 0.3) / m - 0.5
            grid2 = np.array([(x, y) for y in ticks2 for x in ticks2])
            centers = grid2[:N].copy()
            centers += rng.normal(0, 0.005, centers.shape)
        else:
            # Random perturbation of grid
            centers = grid[:N].copy()
            centers += rng.normal(0, 0.02, centers.shape)
            
        centers = np.clip(centers, -0.49, 0.49)
        
        # SA Search
        remaining = _DEADLINE - time.time()
        sa_budget = min(20.0, remaining * 0.3)
        if sa_budget > 3:
            centers, r = _sa_search(centers, rng, sa_budget)
        else:
            r = _compute_radius_fast(centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    # 3. SLSQP Polish on the best configuration
    remaining = _DEADLINE - time.time()
    if remaining > 15 and best_centers is not None:
        polished_centers, polished_r = _slsqp_polish(best_centers, time_budget=min(10.0, remaining - 5))
        if polished_r > best_radius:
            best_radius = polished_r
            best_centers = polished_centers
            
    # 4. Final SLSQP polish if time permits (maybe on a slightly perturbed version?)
    remaining = _DEADLINE - time.time()
    if remaining > 10 and best_centers is not None:
        # Try polishing a slightly perturbed version to escape local minima
        perturbed = best_centers + rng.normal(0, 0.001, best_centers.shape)
        perturbed = np.clip(perturbed, -0.49, 0.49)
        polished2_centers, polished2_r = _slsqp_polish(perturbed, time_budget=min(5.0, remaining - 2))
        if polished2_r > best_radius:
            best_radius = polished2_r
            best_centers = polished2_centers

    # Final verification
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
