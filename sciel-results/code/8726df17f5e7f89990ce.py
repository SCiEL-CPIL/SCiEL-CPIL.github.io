import numpy as np
from scipy.optimize import minimize
import time

N = 21
PERIMETER = 4.0
TIME_LIMIT = 145.0  # Safety margin before 150s limit

def _soft_min_sum(centers, W, H, beta):
    """
    Compute the sum of soft-min approximations of the inradii.
    This is a smooth surrogate for sum(min(distances)).
    
    soft_min(d) = min(d) - (1/beta) * log(sum(exp(-beta * (d - min(d)))))
    """
    n = len(centers)
    
    # 1. Distances to walls
    # Left: x, Right: W-x, Bottom: y, Top: H-y
    d_left = centers[:, 0]
    d_right = W - centers[:, 0]
    d_bottom = centers[:, 1]
    d_top = H - centers[:, 1]
    
    # Stack wall distances: shape (n, 4)
    wall_dists = np.column_stack([d_left, d_right, d_bottom, d_top])
    
    # 2. Distances to other circles (half-distance)
    # diff[i, j] = centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2)) / 2.0
    # Set diagonal to infinity so it doesn't affect the min
    np.fill_diagonal(dists, np.inf)
    
    # Combine all distances: shape (n, 4 + n-1)
    all_dists = np.concatenate([wall_dists, dists], axis=1)
    
    # 3. Soft-min calculation
    # m = min(all_dists, axis=1)
    mins = np.min(all_dists, axis=1)
    
    # To avoid overflow in exp, subtract the min
    # exp_terms = exp(-beta * (all_dists - mins))
    # Note: all_dists - mins >= 0, so -beta * (...) <= 0, exp is stable <= 1
    
    # Handle case where mins might be 0 or very small
    shifted = all_dists - mins[:, np.newaxis]
    
    # exp(-beta * shifted)
    exp_terms = np.exp(-beta * shifted)
    
    # sum(exp_terms)
    sums = np.sum(exp_terms, axis=1)
    
    # log(sum)
    # If sums is 0 (unlikely due to min term being exp(0)=1), log(0) is -inf
    # But min term ensures sum >= 1, so log(sum) >= 0
    log_sums = np.log(sums)
    
    # soft_min = mins - (1/beta) * log_sums
    soft_mins = mins - (1.0 / beta) * log_sums
    
    # We want to MAXIMIZE sum(soft_mins), so we minimize -sum(soft_mins)
    return -np.sum(soft_mins)

def _objective(params, W, H, beta):
    """
    Objective function for L-BFGS-B.
    params: flat array [x1, y1, ..., xN, yN]
    """
    centers = params.reshape(N, 2)
    return _soft_min_sum(centers, W, H, beta)

def _exact_value(centers, W, H):
    """
    Calculate the exact sum of radii using hard min.
    This is used for final evaluation and checking.
    """
    n = len(centers)
    
    d_left = centers[:, 0]
    d_right = W - centers[:, 0]
    d_bottom = centers[:, 1]
    d_top = H - centers[:, 1]
    
    wall_dists = np.column_stack([d_left, d_right, d_bottom, d_top])
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    all_dists = np.concatenate([wall_dists, dists], axis=1)
    radii = np.min(all_dists, axis=1)
    
    return np.sum(radii), radii

def _seed_grid(W, H, seed_val=0):
    """
    Generate a grid seed with slight perturbation.
    """
    rng = np.random.default_rng(seed_val)
    
    # Try to fit N circles in a grid
    # 21 circles -> 5x4=20 (need 1 extra), 7x3=21
    best_val = -1
    best_pos = None
    
    for cols in range(1, N + 1):
        rows = (N + cols - 1) // cols
        if rows * cols < N:
            continue
            
        sx = W / cols
        sy = H / rows
        
        pos = []
        for i in range(cols):
            for j in range(rows):
                if len(pos) >= N:
                    break
                cx = (i + 0.5) * sx
                cy = (j + 0.5) * sy
                pos.append([cx, cy])
        
        pos = np.array(pos[:N]).flatten()
        
        # Add small random perturbation to break symmetry
        pos_perturbed = pos + rng.normal(0, 0.01, pos.shape)
        pos_perturbed = np.clip(pos_perturbed, 0, np.array([W, H] * N).flatten())
        
        val, _ = _exact_value(pos_perturbed.reshape(N, 2), W, H)
        if val > best_val:
            best_val = val
            best_pos = pos_perturbed.copy()
            
    if best_pos is None:
        # Fallback: random
        best_pos = rng.uniform(0, 1, N * 2) * np.array([W, H] * N).flatten()
        
    return best_pos

def _optimize(W, H, initial_pos, deadline):
    """
    Multi-stage optimization with increasing beta.
    """
    current_pos = initial_pos.copy()
    best_pos = current_pos.copy()
    best_val, _ = _exact_value(current_pos.reshape(N, 2), W, H)
    
    bounds = [(0, W)] * N + [(0, H)] * N
    
    # Beta schedule: start loose, get tight
    # Low beta = smooth landscape, easy to move
    # High beta = sharp landscape, close to exact min
    beta_schedule = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    for beta in beta_schedule:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(
                _objective, current_pos,
                args=(W, H, beta),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            current_pos = res.x
        except Exception:
            break
        
        # Evaluate exact value
        val, _ = _exact_value(current_pos.reshape(N, 2), W, H)
        if val > best_val:
            best_val = val
            best_pos = current_pos.copy()
            
    return best_pos

def _construct_circles(pos, W, H):
    """
    Construct the final circle array from optimized positions.
    Ensures validity by clipping radii if necessary.
    """
    centers = pos.reshape(N, 2)
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Calculate exact radii
    _, radii = _exact_value(centers, W, H)
    
    circles = np.zeros((N, 3))
    circles[:, 0] = x
    circles[:, 1] = y
    circles[:, 2] = radii
    
    # Final validity check and repair
    # If any circle violates constraints due to numerical errors, reduce its radius
    for i in range(N):
        # Boundary check
        max_r = min(x[i], W - x[i], y[i], H - y[i])
        if circles[i, 2] > max_r:
            circles[i, 2] = max_r
            
        # Overlap check
        for j in range(N):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            req_r = (dist - 1e-9) / 2.0
            
            # If current radius sum exceeds distance, we need to reduce
            # The inradius calculation should have handled this, but let's be safe.
            # If r_i + r_j > dist, we can reduce r_i to (dist - r_j)
            # However, this might violate other constraints.
            # The _exact_value function takes the MIN of all constraints, so it should be valid.
            # This loop is just a final safety net for floating point issues.
            if circles[i, 2] + circles[j, 2] > dist + 1e-9:
                # Reduce the larger one or split the difference?
                # Simple approach: reduce i to fit
                new_r = max(0.0, dist - circles[j, 2] - 1e-9)
                if new_r < circles[i, 2]:
                    circles[i, 2] = new_r
                    
    # Ensure radii are positive
    circles[:, 2] = np.maximum(circles[:, 2], 1e-9)
    
    return circles

def _is_valid(circles, tol=1e-9):
    """Check if the configuration is valid."""
    if len(circles) == 0:
        return False
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    if np.any(r <= 0):
        return False
    
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    W = x_max - x_min
    H = y_max - y_min
    
    if 2 * (W + H) > PERIMETER + tol:
        return False
    
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            sum_r = r[i] + r[j]
            if dist_sq < (sum_r - tol) ** 2:
                return False
    return True

def construct():
    t_start = time.time()
    best_circles = None
    best_value = 0.0
    
    # Aspect ratios to try
    # W + H = 2
    # Square: 1x1
    # Rect: 1.5x0.5, 1.2x0.8, etc.
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.6, 1.8, 0.5]
    
    # Shuffle to ensure diversity in case time runs out
    rng = np.random.default_rng(42)
    rng.shuffle(aspect_ratios)
    
    for ar in aspect_ratios:
        elapsed = time.time() - t_start
        if elapsed > TIME_LIMIT - 15.0:
            break
            
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 - W
        
        # Seed 1: Grid
        pos1 = _seed_grid(W, H, seed_val=1)
        
        time_remaining = TIME_LIMIT - (time.time() - t_start) - 5.0
        if time_remaining > 10.0:
            opt_pos = _optimize(W, H, pos1, time.time() + time_remaining)
            circles = _construct_circles(opt_pos, W, H)
            
            if _is_valid(circles):
                val = np.sum(circles[:, 2])
                if val > best_value:
                    best_value = val
                    best_circles = circles.copy()
                    
        # Seed 2: Random perturbation of grid
        if time.time() - t_start > TIME_LIMIT - 20.0:
            break
            
        pos2 = pos1.copy()
        pos2 += rng.normal(0, 0.1, pos2.shape)
        pos2 = np.clip(pos2, 0, np.array([W, H] * N).flatten())
        
        time_remaining = TIME_LIMIT - (time.time() - t_start) - 5.0
        if time_remaining > 10.0:
            opt_pos = _optimize(W, H, pos2, time.time() + time_remaining)
            circles = _construct_circles(opt_pos, W, H)
            
            if _is_valid(circles):
                val = np.sum(circles[:, 2])
                if val > best_value:
                    best_value = val
                    best_circles = circles.copy()
                    
        # Seed 3: Fully random
        if time.time() - t_start > TIME_LIMIT - 25.0:
            break
            
        pos3 = rng.uniform(0, 1, N * 2) * np.array([W, H] * N).flatten()
        
        time_remaining = TIME_LIMIT - (time.time() - t_start) - 5.0
        if time_remaining > 10.0:
            opt_pos = _optimize(W, H, pos3, time.time() + time_remaining)
            circles = _construct_circles(opt_pos, W, H)
            
            if _is_valid(circles):
                val = np.sum(circles[:, 2])
                if val > best_value:
                    best_value = val
                    best_circles = circles.copy()

    if best_circles is None:
        # Fallback
        W, H = 1.0, 1.0
        pos = _seed_grid(W, H, seed_val=0)
        best_circles = _construct_circles(pos, W, H)
        if not _is_valid(best_circles):
            # Last resort: tiny circles
            best_circles = np.zeros((N, 3))
            for i in range(N):
                best_circles[i, 0] = 0.1 + (i % 5) * 0.2
                best_circles[i, 1] = 0.1 + (i // 5) * 0.2
                best_circles[i, 2] = 0.05

    return best_circles

if __name__ == "__main__":
    # Simple test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Value: {np.sum(res[:, 2]):.6f}")
    print(f"Valid: {_is_valid(res)}")
    # Print first few circles
    print(res[:5])
