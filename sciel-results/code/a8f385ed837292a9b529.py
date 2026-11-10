import numpy as np
import time
import sys

N = 200
TIME_LIMIT = 110.0  # Safety margin from 120s
START_TIME = time.time()

def _compute_radius(centres):
    """Compute packing radius. Returns r if valid, 0.0 if invalid."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    dx = np.abs(centres[:, 0])
    dy = np.abs(centres[:, 1])
    wall_dists = np.min(np.stack([0.5 - dx, 0.5 - dy], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall <= 0:
        return 0.0
    from scipy.spatial.distance import cdist
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_wall)

def _generate_hex_init(n_rows, spacing, jitter=0.0):
    """Generate hexagonal packing configuration."""
    points = []
    sqrt3 = np.sqrt(3)
    max_j = int(1.0 / (spacing * sqrt3 / 2)) + 3
    for j in range(-max_j, max_j + 1):
        y = j * spacing * sqrt3 / 2
        if abs(y) > 0.48:
            continue
        offset = (j % 2) * spacing / 2
        max_i = int(1.0 / spacing) + 3
        for i in range(-max_i, max_i + 1):
            x = i * spacing + offset
            if -0.48 <= x <= 0.48:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        return None
    # Sort by distance from center, take N closest
    dists_from_center = np.sum(points**2, axis=1)
    idx = np.argsort(dists_from_center)[:N]
    pts = points[idx].copy()
    if jitter > 0:
        pts += np.random.randn(N, 2) * jitter
        pts = np.clip(pts, -0.48, 0.48)
    return pts

def _softmin(values, beta):
    """Soft minimum using log-sum-exp trick."""
    v = np.asarray(values)
    m = np.min(v)
    return m - np.log(np.sum(np.exp(-(v - m) * beta))) / beta

def _objective_and_grad(centres, beta):
    """Compute soft objective and gradient. Returns (obj, grad, min_dist, min_wall)."""
    # Wall constraints
    wx = 0.5 - np.abs(centres[:, 0])
    wy = 0.5 - np.abs(centres[:, 1])
    wall = np.minimum(wx, wy)
    
    # Pairwise distances
    from scipy.spatial.distance import cdist
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    
    min_dist = np.min(dists)
    min_wall = np.min(wall)
    
    # Soft objective
    soft_d = _softmin(dists, beta) / 2.0
    soft_w = _softmin(wall, beta)
    obj = -min(soft_d, soft_w)
    
    # Gradient with respect to centres
    grad = np.zeros_like(centres)
    
    # Wall gradient
    # softmax weights for wall constraints
    exp_w = np.exp(-(wall - np.min(wall)) * beta)
    sum_w = np.sum(exp_w)
    if sum_w > 0:
        sig_w = exp_w / sum_w
        # grad_wall_i = -sig_w_i * sign(c_i)
        grad[:, 0] -= sig_w * np.sign(centres[:, 0])
        grad[:, 1] -= sig_w * np.sign(centres[:, 1])
    
    # Pairwise gradient (Vectorized O(N^2))
    # sig_d[i,j] = exp(-(d_ij - min_d) * beta) / sum
    min_d = np.min(dists)
    exp_d = np.exp(-(dists - min_d) * beta)
    np.fill_diagonal(exp_d, 0)
    sum_d = np.sum(exp_d)
    
    if sum_d > 0:
        sig_d = exp_d / sum_d
        
        # diff[i, j] = c_i - c_j
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        
        # unit[i, j] = diff[i, j] / d_ij
        # Avoid division by zero by masking diagonal (d=inf there, so 0/inf=0 is fine, but let's be safe)
        # dists has inf on diagonal, so diff/dists will be 0 on diagonal.
        unit = diff / dists[:, :, np.newaxis]
        
        # grad_i = sum_j sig_d[i,j] * unit[i,j]
        # We need to sum over j for each i.
        # sig_d is (N, N), unit is (N, N, 2)
        # grad[i] = sum_j sig_d[i, j] * unit[i, j, :]
        grad += np.sum(sig_d[:, :, np.newaxis] * unit, axis=1)
    
    return obj, grad, min_dist, min_wall

def _optimize(centres, time_budget):
    """Gradient-based optimization with adaptive step size."""
    deadline = time.time() + time_budget
    best_centres = centres.copy()
    best_r = _compute_radius(centres)
    
    beta = 50.0
    lr = 0.002
    min_lr = 1e-6
    momentum = np.zeros_like(centres)
    decay = 0.9
    check_interval = 10 # Check exact radius less frequently
    
    step_count = 0
    while time.time() < deadline:
        obj, grad, min_d, min_w = _objective_and_grad(centres, beta)
        
        # Update momentum
        momentum = decay * momentum - lr * grad
        new_centres = centres + momentum
        new_centres = np.clip(new_centres, -0.499, 0.499)
        
        centres = new_centres
        step_count += 1
        
        # Only compute exact radius periodically or if soft objective suggests improvement
        # This saves significant time as cdist is expensive
        if step_count % check_interval == 0 or obj < -0.036: # heuristic: if soft obj is good
            r = min(min_d / 2.0, min_w)
            if r > best_r:
                best_r = r
                best_centres = centres.copy()
                lr = max(min_lr, lr * 0.99) # Slowly increase lr if improving
            else:
                lr = max(min_lr, lr * 0.95)
                momentum *= 0.5
        
        if lr < min_lr:
            # If learning rate is very small, try a few restarts
            if np.random.rand() < 0.05:
                centres = best_centres.copy() + np.random.randn(N, 2) * 0.002
                centres = np.clip(centres, -0.499, 0.499)
                lr = 0.001
                momentum = np.zeros_like(centres)
            else:
                break
    
    # Final check
    r = _compute_radius(centres)
    if r > best_r:
        best_r = r
        best_centres = centres.copy()
        
    return best_centres, best_r

def _polish(centres, time_budget):
    """Final high-precision polish pass."""
    deadline = time.time() + time_budget
    best_centres = centres.copy()
    best_r = _compute_radius(centres)
    
    for it in range(200):
        if time.time() > deadline - 2:
            break
        
        # Find bottleneck
        from scipy.spatial.distance import cdist
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        min_i, min_j = np.unravel_index(np.argmin(dists), dists.shape)
        
        wx = 0.5 - np.abs(best_centres[:, 0])
        wy = 0.5 - np.abs(best_centres[:, 1])
        wall = np.minimum(wx, wy)
        min_w = np.min(wall)
        min_k = np.argmin(wall)
        
        current_r = min(min_d / 2.0, min_w)
        
        if current_r >= best_r:
            best_r = current_r
            best_centres = centres.copy()
        
        # Move to increase bottleneck
        step = 0.001
        improved = False
        
        if min_d / 2.0 <= min_w:
            # Distance bottleneck: move i and j apart
            diff = best_centres[min_i] - best_centres[min_j]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                direction = diff / norm
                new_c = best_centres.copy()
                new_c[min_i] += direction * step
                new_c[min_j] -= direction * step
                new_c = np.clip(new_c, -0.499, 0.499)
                new_r = _compute_radius(new_c)
                if new_r > best_r:
                    best_r = new_r
                    best_centres = new_c
                    improved = True
        
        if not improved and min_w < min_d / 2.0:
            # Wall bottleneck: move k away from wall
            k = min_k
            new_c = best_centres.copy()
            # Move away from nearest wall
            if abs(best_centres[k, 0]) >= abs(best_centres[k, 1]):
                new_c[k, 0] -= np.sign(best_centres[k, 0]) * step
            else:
                new_c[k, 1] -= np.sign(best_centres[k, 1]) * step
            new_c = np.clip(new_c, -0.499, 0.499)
            new_r = _compute_radius(new_c)
            if new_r > best_r:
                best_r = new_r
                best_centres = new_c
                improved = True
        
        if not improved:
            step *= 0.7
            if step < 1e-6:
                break
    
    return best_centres, best_r

def construct():
    start_time = time.time()
    # Hard deadline: 110 seconds from start to ensure we return well before 120s
    deadline = start_time + 110.0
    
    # Generate multiple hexagonal initializations and pick best
    best_centres = None
    best_r = 0.0
    
    spacings = [0.055, 0.056, 0.057, 0.058, 0.059, 0.060]
    
    for s in spacings:
        if time.time() > deadline - 80: # Keep 80s for optimization
            break
        for jitter in [0.0, 0.002, 0.005]:
            pts = _generate_hex_init(20, s, jitter)
            if pts is None:
                continue
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best_centres = pts.copy()
    
    # If no valid config, use grid fallback
    if best_centres is None:
        m = 14
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centres = grid[:N]
        best_r = _compute_radius(best_centres)
    
    # Gradient optimization
    remaining = deadline - time.time()
    if remaining > 20:
        # Leave 5s for polish
        best_centres, best_r = _optimize(best_centres, remaining - 5)
    
    # Final polish
    remaining = deadline - time.time()
    if remaining > 3:
        best_centres, best_r = _polish(best_centres, remaining - 1)
    
    # Final validation
    final_r = _compute_radius(best_centres)
    if final_r <= 0:
        m = 14
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centres
