"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    wall = np.min(np.min(0.5 - np.abs(c), axis=1), axis=0)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(wall, min_dist / 2.0)

def _true_min_gradient(c):
    """
    Compute the true gradient of the minimum of {wall distances} U {pair distances/2}.
    Returns (value, gradient) where gradient points in the direction that increases the min.
    """
    # Wall distances: for each sphere, min over 3 coords of (0.5 - |c_i,k|)
    wall_per_coord = 0.5 - np.abs(c)  # (N, 3)
    wall_min_idx = np.argmin(wall_per_coord, axis=1)  # (N,)
    wall_min_val = wall_per_coord[np.arange(N), wall_min_idx]  # (N,)
    
    # Pair distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]  # (N, N, 3)
    dist = np.sqrt(np.sum(diff**2, axis=2))  # (N, N)
    np.fill_diagonal(dist, np.inf)
    half_dist = dist / 2.0  # (N, N)
    
    # Find global minimum
    wall_flat = wall_min_val
    pair_flat = half_dist.ravel()
    all_vals = np.concatenate([wall_flat, pair_flat])
    min_val = np.min(all_vals)
    
    # Build gradient: only contributions from constraints at the minimum matter
    grad = np.zeros_like(c)
    
    # Wall constraints at minimum
    wall_at_min = np.where(np.abs(wall_min_val - min_val) < 1e-10)[0]
    for i in wall_at_min:
        k = wall_min_idx[i]
        # d(0.5 - |c_i,k|)/dc_i,k = -sign(c_i,k)
        sign = np.sign(c[i, k])
        if sign == 0:
            sign = 1.0
        grad[i, k] += -sign  # we want to move away from wall, so +sign direction
    
    # Pair constraints at minimum
    pair_min_mask = np.abs(half_dist - min_val) < 1e-10
    np.fill_diagonal(pair_min_mask, False)
    for i in range(N):
        for j in range(N):
            if pair_min_mask[i, j]:
                # d(|c_i - c_j|/2)/dc_i = (c_i - c_j) / |c_i - c_j|
                dij = dist[i, j]
                if dij > 1e-12:
                    grad[i] += diff[i, j] / dij
                    grad[j] += -diff[i, j] / dij
    
    return min_val, grad

def _ascent_step(c, target_r, max_iter=500, lr_init=0.01, time_limit=None):
    """
    Steepest ascent on true min, with target radius constraint.
    We maximize the min distance, but also try to push all distances above target_r.
    """
    c = c.copy()
    start = time.time()
    lr = lr_init
    
    for it in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
        
        val, grad = _true_min_gradient(c)
        
        # If current min is already above target, we're good for this phase
        if val >= target_r:
            break
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-14:
            break
        grad_norm = grad / gnorm
        
        # Take step
        new_c = c + lr * grad_norm
        new_c = np.clip(new_c, -0.4999, 0.4999)
        
        new_val = _score(new_c)
        if new_val > val - 1e-12:
            c = new_c
            # Increase lr slightly if improving
            lr = min(lr * 1.05, 0.05)
        else:
            # Decrease lr
            lr *= 0.7
            if lr < 1e-8:
                break
    
    return c

def _perturb(c, magnitude=0.02):
    """Random perturbation to escape local minima."""
    c_new = c + np.random.normal(0, magnitude, size=c.shape)
    return np.clip(c_new, -0.4999, 0.4999)

def _init_grid():
    """Simple cubic grid as starting point."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _init_bcc():
    """Body-centered cubic inspired initialization."""
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = []
    for z in ticks:
        for y in ticks:
            for x in ticks:
                pts.append([x, y, z])
                # Add BCC-like offset
                if x < 0.3 and y < 0.3 and z < 0.3:
                    pts.append([x + 0.1, y + 0.1, z + 0.1])
    pts = np.array(pts[:N])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Multiple initializations
    inits = [_init_grid(), _init_bcc()]
    for _ in range(3):
        inits.append(np.random.uniform(-0.5, 0.5, size=(N, 3)))
    
    best_c = inits[0].copy()
    best_score = _score(best_c)
    
    # Phase 1: For each init, do ascent with increasing target radius
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        
        c = init.copy()
        current_r = _score(c)
        
        # Binary search on target radius: try to achieve higher and higher r
        # Start from current, try to push up
        target = current_r + 0.001
        
        while time.time() < deadline - 25.0:
            # Try to reach target
            c = _ascent_step(c, target, max_iter=200, lr_init=0.008, time_limit=deadline - 25.0)
            new_r = _score(c)
            
            if new_r > best_score:
                best_score = new_r
                best_c = c.copy()
            
            # If we reached or exceeded target, increase it
            if new_r >= target - 1e-8:
                target = new_r + 0.0005
            else:
                # Stuck, try perturbation
                c = _perturb(c, magnitude=0.01)
                target = max(_score(c) + 0.0005, current_r + 0.0005)
            
            # If target is too optimistic, back off
            if target > 0.12:
                target = 0.12
        
        # Local polish at the end
        if time.time() < deadline - 10.0:
            c = _ascent_step(c, _score(c) + 0.0001, max_iter=100, lr_init=0.005, time_limit=deadline - 10.0)
            if _score(c) > best_score:
                best_score = _score(c)
                best_c = c.copy()
    
    # Final polish
    if time.time() < deadline - 2.0:
        best_c = _ascent_step(best_c, _score(best_c) + 0.0001, max_iter=200, lr_init=0.003, time_limit=deadline - 1.0)
    
    return best_c
# EVOLVE-BLOCK-END
