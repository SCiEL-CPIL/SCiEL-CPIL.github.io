"""Packing 50 circles in a unit square: Binary search on radius with violation minimization."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    if n == 0:
        return 0.0
    # Wall distances: min distance from any center to any wall
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
    
    # Try different row configurations
    for n_rows in range(4, 11):
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
        total_y = (n_rows - 1) * 2 * r * np.sqrt(3.0) / 2.0
        y_start = -0.5 + r + total_y / 2.0
        
        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * 2 * r * np.sqrt(3.0) / 2.0
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = (2 * r * np.sqrt(3.0) / 2.0) * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])
        
        centers = np.array(centers[:N])
        r_actual = _score(centers)
        
        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()
    
    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)
    
    return best_cents, best_r

def _violation_loss(params, r_target, n):
    """
    Compute violation loss for a given radius target.
    params: flattened array of 2n coordinates
    Returns: sum of squared violations + gradient
    """
    cents = params.reshape(n, 2)
    
    # Wall constraints: 0.5 - |x| >= r and 0.5 - |y| >= r
    # Violation: max(0, r - (0.5 - |x|))
    wall_x = np.maximum(0, r_target - (0.5 - np.abs(cents[:, 0])))
    wall_y = np.maximum(0, r_target - (0.5 - np.abs(cents[:, 1])))
    
    # Pairwise constraints: |c_i - c_j| >= 2r
    # Violation: max(0, 2r - |c_i - c_j|)
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_viol = np.maximum(0, 2 * r_target - dists)
    
    # Loss = sum of squared violations
    loss = np.sum(wall_x**2) + np.sum(wall_y**2) + np.sum(pair_viol**2)
    
    # Gradient computation
    grad = np.zeros_like(cents)
    
    # Wall gradients
    # d/dx max(0, r - (0.5 - |x|)) = -sign(x) if violated
    mask_x = wall_x > 0
    mask_y = wall_y > 0
    grad[mask_x, 0] += -2 * wall_x[mask_x] * np.sign(cents[mask_x, 0])
    grad[mask_y, 1] += -2 * wall_y[mask_y] * np.sign(cents[mask_y, 1])
    
    # Pairwise gradients
    # d/dx_i max(0, 2r - |c_i - c_j|) = (c_i - c_j) / |c_i - c_j| * 2 * violation
    for i in range(n):
        for j in range(i + 1, n):
            v = pair_viol[i, j]
            if v > 0:
                d = diff[i, j]  # c_i - c_j
                dist = dists[i, j]
                if dist > 1e-12:
                    factor = 2 * v / dist
                    grad[i] += factor * d
                    grad[j] -= factor * d
    
    return loss, grad.flatten()

def _try_radius(cents_init, r_target, n, maxiter=200, time_limit=5.0):
    """
    Try to find a valid configuration for radius r_target.
    Returns (success, cents) or (False, None)
    """
    start_time = time.time()
    params_init = cents_init.flatten()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(params)
        loss, grad = _violation_loss(params, r_target, n)
        return loss, grad
    
    # Bounds: centers must be in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * n)
    
    try:
        result = minimize(
            objective,
            params_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
    except Exception:
        return False, None
    
    if time.time() - start_time > time_limit:
        return False, None
    
    # Check if we achieved a valid configuration
    final_cents = result.x.reshape(n, 2)
    final_loss = _violation_loss(result.x, r_target, n)[0]
    
    # Consider it successful if violation is very small
    if final_loss < 1e-10:
        # Verify with actual score
        actual_r = _score(final_cents)
        if actual_r >= r_target - 1e-10:
            return True, final_cents
    
    return False, None

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Step 1: Create hexagonal initial arrangement
    cents, r0 = _construct_hexagonal()
    best_cents = cents.copy()
    best_r = r0
    
    # Step 2: Binary search on radius
    # Upper bound: theoretical max for 50 circles
    # Rough estimate: ~0.072
    r_low = best_r  # We know this is feasible
    r_high = 0.075  # Upper bound
    
    # First, try to improve from the initial configuration
    # Do a few iterations of binary search
    n = N
    
    # Try to push the radius up using binary search
    for iteration in range(20):
        if time.time() - start_time > deadline - 10:
            break
        
        r_mid = (r_low + r_high) / 2
        
        # Try to find a feasible configuration for r_mid
        success, new_cents = _try_radius(best_cents, r_mid, n, maxiter=300, time_limit=4.0)
        
        if success:
            r_low = r_mid
            best_cents = new_cents
            best_r = _score(best_cents)
        else:
            r_high = r_mid
        
        # If we're converging, stop early
        if r_high - r_low < 1e-8:
            break
    
    # Step 3: Final local optimization from the best found configuration
    remaining = deadline - (time.time() - start_time)
    if remaining > 5:
        # Do a final L-BFGS optimization to maximize the minimum distance
        # Using a smooth surrogate: minimize -log(min_dist) or similar
        # Actually, let's just do a final perturbation-based local search
        rng = np.random.default_rng(42)
        step_size = 0.002
        final_cents = best_cents.copy()
        final_r = best_r
        
        opt_start = time.time()
        while time.time() - opt_start < min(remaining - 2, 30):
            perturbation = rng.normal(0, step_size, size=final_cents.shape)
            new_cents = final_cents + perturbation
            
            # Check bounds
            if np.any(np.abs(new_cents) > 0.5 - 1e-9):
                step_size *= 0.9
                continue
            
            new_r = _score(new_cents)
            if new_r > final_r:
                final_r = new_r
                final_cents = new_cents.copy()
                step_size = max(step_size * 0.95, 0.0005)
            else:
                step_size = max(step_size * 0.95, 0.0005)
        
        if final_r > best_r:
            best_r = final_r
            best_cents = final_cents
    
    return best_cents
# EVOLVE-BLOCK-END
