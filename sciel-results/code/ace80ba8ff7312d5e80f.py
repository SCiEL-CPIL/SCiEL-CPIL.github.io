"""
Strategy: Direct smooth optimization of a repulsive force objective.
Instead of binary searching for a feasible radius r and minimizing violations (which has flat zero-loss regions),
we directly optimize a smooth objective function that penalizes close pairs and wall violations.
Objective: minimize sum_{i<j} (1 / max(d_ij^2, eps)^2) + sum_i (1 / max(wall_i^2, eps)^2)
This is differentiable everywhere (with eps) and guides the optimizer to increase the minimum distance.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    if n == 0:
        return 0.0
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

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles."""
    best_cents = None
    best_r = 0

    # Try a few row configurations
    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]

        max_rc = max(row_counts)
        # Estimate radius based on horizontal fit
        r_horiz = 1.0 / (2.0 * max_rc)
        # Estimate radius based on vertical fit (hexagonal spacing is sqrt(3)*r)
        # Total height = r + (n_rows-1)*sqrt(3)*r + r = 2r + (n_rows-1)*sqrt(3)*r
        # Height must be <= 1.0
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)

        if r <= best_r:
            continue

        centers = []
        # Calculate total height to center vertically
        total_y = (n_rows - 1) * np.sqrt(3.0) * r
        y_start = -0.5 + r + total_y / 2.0

        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * np.sqrt(3.0) * r
            x_span = (rc - 1) * 2.0 * r
            x_start = -0.5 + r + x_span / 2.0
            # Offset every other row
            offset = 0.0
            if row_idx % 2 == 1:
                offset = r
            
            for col in range(rc):
                x = x_start + col * 2.0 * r + offset
                centers.append([x, y])

        centers = np.array(centers[:N])
        r_actual = _score(centers)

        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()

    if best_cents is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)

    return best_cents, best_r

def _repulsive_loss(x, eps=1e-6):
    """
    Loss function to minimize.
    x is shape (2N,).
    We want to maximize the minimum distance. 
    Minimizing sum of 1/d^4 is a strong proxy for maximizing min d.
    Also penalize going outside the box.
    """
    c = x.reshape(N, 2)
    
    # Wall penalty: If a point is too close to the wall, penalize heavily.
    # Distance to wall in x: 0.5 - |x|. We want this to be > r.
    # But we don't know r. We just want to maximize min distance.
    # Let's define "effective distance" to wall as 0.5 - |coord|.
    # We want to maximize the minimum of these.
    # So we minimize sum(1 / (0.5 - |coord|)^4) for coords where 0.5 - |coord| > eps.
    
    dx = 0.5 - np.abs(c[:, 0])
    dy = 0.5 - np.abs(c[:, 1])
    
    # Clip to avoid division by zero or negative (if out of bounds)
    # If out of bounds, dx < 0. We should penalize that heavily.
    # Let's use a barrier function.
    # If dx <= 0, return large penalty.
    # Actually, L-BFGS-B handles bounds. Let's use bounds (-0.5, 0.5).
    # Inside the box, dx > 0.
    # We want to maximize min(dx, dy, d_ij/2).
    
    wall_terms_x = 1.0 / np.maximum(dx, eps)**4
    wall_terms_y = 1.0 / np.maximum(dy, eps)**4
    wall_loss = np.sum(wall_terms_x) + np.sum(wall_terms_y)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Minimize sum(1/d^4)
    # Use dists_sq to avoid sqrt
    pair_terms = 1.0 / np.maximum(dists_sq, eps)**2
    # Sum over unique pairs (i<j) to avoid double counting, or just sum all and divide by 2.
    # Summing all is fine, factor of 2 doesn't matter for minimization.
    pair_loss = np.sum(pair_terms)
    
    return wall_loss + pair_loss

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Step 1: Hexagonal initial arrangement
    cents, r0 = _construct_hexagonal()
    best_cents = cents.copy()
    best_r = r0

    # Step 2: Direct optimization
    # We will run multiple restarts with different initializations if time permits.
    # But first, refine the hexagonal one.
    
    x0 = cents.reshape(-1).copy()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Use L-BFGS-B
    # maxfun is the max number of function evaluations.
    # 5000 evaluations * ~1ms per eval (vectorized) = ~5s.
    # Let's try 20000 to be safe, but check time.
    
    try:
        result = minimize(
            _repulsive_loss, x0,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        new_cents = result.x.reshape(N, 2)
        new_r = _score(new_cents)
        
        if new_r > best_r:
            best_r = new_r
            best_cents = new_cents
    except Exception as e:
        print(f"Optimization failed: {e}")
        
    elapsed = time.time() - start_time
    
    # Step 3: If time remains, try a different initialization or restart
    # For example, a grid initialization or a random perturbation of the best found.
    if elapsed < deadline - 30:
        # Try a grid start
        m = 7
        ticks = (np.arange(m) + 0.5) / m - 0.5
        # Create a 7x7 grid, take 50 points
        # We need to select 50 points from 49? No, 7x7=49. 
        # Let's use 8x8 grid and remove corners?
        # Or just use the hexagonal one again with random jitter?
        
        # Let's try a random restart from the current best with jitter
        np.random.seed(42)
        jitter = np.random.normal(0, 0.01, size=(N, 2))
        x_init = (best_cents + jitter).reshape(-1)
        # Clip to bounds
        x_init = np.clip(x_init, -0.49, 0.49)
        
        try:
            result2 = minimize(
                _repulsive_loss, x_init,
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            new_cents2 = result2.x.reshape(N, 2)
            new_r2 = _score(new_cents2)
            
            if new_r2 > best_r:
                best_r = new_r2
                best_cents = new_cents2
        except Exception as e:
            print(f"Second optimization failed: {e}")
            
    # Final safety check
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    return best_cents
# EVOLVE-BLOCK-END
