# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_area(x, a, y, w, L_base):
    """
    Compute the area of the union of triangles.
    x: (64,) array of base positions
    a: (64,) array of slopes (a_i)
    y: (20000,) array of heights
    w: (20000,) array of widths at height y
    L_base: (64, 20000) array where L_base[i, m] = a[i] * y[m]
    """
    # L[i, m] = x[i] + a[i] * y[m]
    L = L_base + x[:, np.newaxis]
    
    # Sort L along axis 0 (for each height m, sort the 64 left edges)
    L_sorted = np.sort(L, axis=0)
    
    # Differences between consecutive sorted edges
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # Union length at each height:
    # union_len[m] = w[m] + sum_i min(d[i, m], w[m])
    mins = np.minimum(d, w[np.newaxis, :])
    
    # Sum over i (axis 0)
    sum_mins = np.sum(mins, axis=0)
    
    # Total union length vector
    union_lens = w + sum_mins
    
    # Area is mean of union lengths
    return np.mean(union_lens)

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Returns 64 finite floats x_0..x_63.
    """
    n = 64
    k = int(np.log2(n))
    
    # Precompute static parts
    a = np.arange(1, n + 1, dtype=np.float64) / n
    num_heights = 20000
    y = (np.arange(num_heights, dtype=np.float64) + 0.5) / num_heights
    w = (1.0 - y) / n
    
    # L_base[i, m] = a[i] * y[m]
    L_base = np.outer(a, y)
    
    # Initial guess: Keich construction
    j = np.arange(1, k + 1)
    x_keich = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x_keich[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    start_time = time.time()
    deadline = start_time + 600.0  # 10 minutes, leave 30s margin
    
    best_x = x_keich.copy()
    best_area = _compute_area(best_x, a, y, w, L_base)
    
    # Simulated Annealing
    # We use a simple SA approach with random restarts to explore the landscape.
    # The objective is non-smooth due to the min() and sort(), so gradient methods
    # get stuck. SA can jump over barriers.
    
    # Parameters
    T_start = 0.1
    T_end = 1e-6
    alpha = 0.9999  # Cooling rate
    max_iters = 5000  # Max iterations per restart
    num_restarts = 20
    
    # We need to be careful with the time budget.
    # Each evaluation takes ~5-10ms. 5000 iters * 20 restarts = 100k evals = ~1000s.
    # Too slow. We need to reduce iterations or restarts.
    # Let's estimate time per eval first.
    
    t0 = time.time()
    _ = _compute_area(x_keich, a, y, w, L_base)
    t1 = time.time()
    eval_time = t1 - t0
    
    # Budget for search: 550 seconds
    search_budget = 550.0
    # Estimated evals possible:
    max_total_evals = int(search_budget / eval_time)
    
    # Distribute evals among restarts
    # Let's do fewer restarts with more iterations, or adaptive.
    # Let's try 10 restarts.
    num_restarts = min(10, max(1, max_total_evals // 1000))
    iters_per_restart = max(100, max_total_evals // num_restarts)
    
    # To make SA effective, we need to move in a way that changes the area.
    # Randomly perturbing all 64 coordinates is inefficient.
    # Better: perturb a subset of coordinates, or use a structured move.
    # However, the Keich solution has a specific structure.
    # Let's try: x_new = x_old + noise * delta, where delta is a random vector.
    
    # We will use a "kick" strategy: occasionally make a large random jump to escape local minima.
    
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        # Initialize current state
        # First restart starts from Keich, subsequent from random or best?
        # Starting from best is good for exploitation, random for exploration.
        if restart == 0:
            x_current = best_x.copy()
        else:
            # Start from best + noise, or purely random?
            # Purely random might be too far. Best + noise is safer.
            x_current = best_x + rng.normal(0, 0.5, n)
            # Clip to reasonable range
            x_current = np.clip(x_current, -1.0, 2.0)
        
        area_current = _compute_area(x_current, a, y, w, L_base)
        
        T = T_start
        for i in range(iters_per_restart):
            if time.time() > deadline - 30:
                break
                
            # Generate neighbor
            # Perturb a random subset of coordinates
            num_perturb = rng.integers(1, 8)
            idx = rng.choice(n, num_perturb, replace=False)
            
            x_neighbor = x_current.copy()
            # Step size decreases with T
            step_size = T * 10.0
            x_neighbor[idx] += rng.normal(0, step_size, num_perturb)
            x_neighbor = np.clip(x_neighbor, -1.0, 2.0)
            
            area_neighbor = _compute_area(x_neighbor, a, y, w, L_base)
            
            # Acceptance criterion
            delta_area = area_neighbor - area_current
            if delta_area < 0:
                x_current = x_neighbor
                area_current = area_neighbor
            else:
                # Accept with probability exp(-delta/T)
                if rng.random() < np.exp(-delta_area / max(T, 1e-10)):
                    x_current = x_neighbor
                    area_current = area_neighbor
            
            # Update best
            if area_current < best_area:
                best_area = area_current
                best_x = x_current.copy()
            
            # Cool down
            T *= alpha
    
    # Final polish: use L-BFGS-B on the best found so far to squeeze out local improvements
    # This combines global search (SA) with local refinement (Gradient)
    from scipy.optimize import minimize
    
    if time.time() < deadline - 60:
        try:
            res = minimize(
                lambda x: _compute_area(x, a, y, w, L_base),
                best_x,
                method='L-BFGS-B',
                bounds=[(-1.0, 2.0)] * n,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success:
                area_res = _compute_area(res.x, a, y, w, L_base)
                if area_res < best_area:
                    best_x = res.x
                    best_area = area_res
        except Exception:
            pass

    if not np.all(np.isfinite(best_x)):
        return x_keich
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
