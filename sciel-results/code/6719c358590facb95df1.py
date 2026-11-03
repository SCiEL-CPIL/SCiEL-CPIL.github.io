import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 145.0  # Safety margin before 150s limit

def _score(centers):
    """Compute the radius for a given configuration."""
    boundary = (0.5 - np.abs(centers)).min()
    if boundary <= 0:
        return 0.0
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    
    r = min(boundary, min_dist / 2.0)
    return r

def _soft_min(values, k=50.0):
    """Soft minimum: -1/k * log(sum(exp(-k*x)))"""
    return -np.log(np.sum(np.exp(-k * values))) / k

def _soft_radius(centers, k=50.0):
    """Differentiable soft approximation of the true radius."""
    boundary = 0.5 - np.abs(centers)
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2) + 1e-10)
    np.fill_diagonal(dists, np.inf)
    
    # Soft min of all boundary distances and half-distances
    all_vals = np.concatenate([boundary, dists / 2.0])
    return _soft_min(all_vals, k)

def _negative_soft_radius(centers, k=50.0):
    return -_soft_radius(centers, k)

def _neg_soft_radius_grad(centers, k=50.0):
    """Gradient of negative soft radius."""
    boundary = 0.5 - np.abs(centers)
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2) + 1e-10
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(dists_sq, np.inf)
    
    all_vals = np.concatenate([boundary, dists / 2.0])
    
    # Soft min gradient
    exp_neg = np.exp(-k * all_vals)
    weights = exp_neg / np.sum(exp_neg)
    
    grad = np.zeros_like(centers)
    
    # Boundary part
    for i in range(N):
        for d in range(2):
            grad[i, d] += weights[i, d] * (-np.sign(centers[i, d]))
    
    # Pairwise part
    n_pairs = N * (N - 1) // 2
    pair_idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            w = weights[N + pair_idx]
            d = dists[i, j]
            if d > 1e-10:
                grad[i] += w * (centers[i] - centers[j]) / (2 * d)
                grad[j] += w * (centers[j] - centers[i]) / (2 * d)
            pair_idx += 1
    
    return -k * grad

def _init_hexagonal():
    """Create a hexagonal-like initial configuration."""
    # Try to fit hexagonal packing
    # For N=50, try various hexagonal arrangements
    
    best_pts = None
    best_r = -1
    
    # Hexagonal lattice with different spacings and offsets
    for trial in range(20):
        # Random spacing and offset
        spacing = 0.12 + 0.03 * np.random.random()
        offset_x = (np.random.random() - 0.5) * 0.1
        offset_y = (np.random.random() - 0.5) * 0.1
        
        pts = []
        row = 0
        while len(pts) < N + 10:
            y = row * spacing * np.sqrt(3) / 2 + offset_y
            if y > 0.5:
                break
            col = 0
            x_offset = (row % 2) * spacing / 2 + offset_x
            while True:
                x = col * spacing + x_offset
                if x > 0.5:
                    break
                if abs(x) <= 0.5 and abs(y) <= 0.5:
                    pts.append([x, y])
                col += 1
            row += 1
        
        pts = np.array(pts)
        if len(pts) < N:
            continue
        
        # Select N points with maximum minimum distance (greedy)
        selected = []
        remaining = pts.copy()
        for _ in range(N):
            if len(selected) == 0:
                # Start with point closest to center
                dists_c = np.sqrt((remaining ** 2).sum(axis=1))
                idx = np.argmin(dists_c)
            else:
                sel = np.array(selected)
                dists = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
                idx = np.argmax(dists)
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
        
        if len(selected) == N:
            selected = np.array(selected)
            r = _score(selected)
            if r > best_r:
                best_r = r
                best_pts = selected.copy()
    
    if best_pts is None:
        # Fallback to grid
        best_pts = _init_grid()
    
    return best_pts

def _init_grid():
    """Create a grid initial configuration."""
    # 5x10 grid
    xs = (np.arange(10) + 0.5) / 10 - 0.5
    ys = (np.arange(5) + 0.5) / 5 - 0.5
    pts = np.array([(x, y) for y in ys for x in xs])
    return pts

def _force_relax(centers, n_steps=100, lr_init=0.01, lr_decay=0.99):
    """Force-directed relaxation."""
    centers = centers.copy()
    lr = lr_init
    
    for step in range(n_steps):
        forces = np.zeros_like(centers)
        
        # Boundary forces
        for k in range(2):
            mask_r = centers[:, k] > 0.4
            forces[mask_r, k] -= (centers[mask_r, k] - 0.4) * 5
            mask_l = centers[:, k] < -0.4
            forces[mask_l, k] += ((-0.4) - centers[mask_l, k]) * 5
        
        # Pairwise repulsion
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        
        threshold = 0.15
        mask = dists < threshold
        np.fill_diagonal(mask, False)
        
        force_mag = np.zeros_like(dists)
        force_mag[mask] = (threshold - dists[mask]) / (dists[mask] + 1e-10) ** 2
        force_mag = np.minimum(force_mag, 5)
        
        forces += (force_mag[:, :, np.newaxis] * diffs).sum(axis=1)
        
        centers += forces * lr
        centers = np.clip(centers, -0.5, 0.5)
        
        lr *= lr_decay
    
    return centers

def _optimize_lbfgsb(centers, time_budget=30.0):
    """Optimize using L-BFGS-B on soft radius surrogate."""
    start_time = time.time()
    
    # Flatten centers
    x0 = centers.flatten()
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def objective(x):
        c = x.reshape(N, 2)
        return _negative_soft_radius(c, k=30.0)
    
    def gradient(x):
        c = x.reshape(N, 2)
        return _neg_soft_radius_grad(c, k=30.0).flatten()
    
    best_x = x0.copy()
    best_obj = objective(x0)
    
    while time.time() - start_time < time_budget:
        result = minimize(objective, x0, method='L-BFGS-B', jac=gradient, 
                         bounds=bounds, options={'maxiter': 100, 'gtol': 1e-6})
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x.copy()
        
        # Check if we're making progress
        if result.success and result.fun >= best_obj - 1e-8:
            break
        
        # Restart from best found
        x0 = best_x + np.random.randn(len(best_x)) * 0.001
        x0 = np.clip(x0, -0.5, 0.5)
    
    best_centers = best_x.reshape(N, 2)
    return best_centers

def _optimize_sa(centers, time_budget=30.0):
    """Simulated Annealing."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    current_centers = centers.copy()
    current_score = _score(current_centers)
    
    T = 0.002
    T_min = 1e-7
    alpha = 0.9995
    
    while time.time() - start_time < time_budget:
        idx = np.random.randint(N)
        move = np.random.randn(2) * 0.005
        
        new_centers = current_centers.copy()
        new_centers[idx] += move
        
        if np.any(np.abs(new_centers) > 0.5):
            T *= alpha
            continue
        
        new_score = _score(new_centers)
        
        delta = new_score - current_score
        if delta > 0 or (T > T_min and np.random.random() < np.exp(delta / T)):
            current_centers = new_centers
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_centers = current_centers.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
        
        if np.random.random() < 0.005:
            T = max(T, 0.001)
    
    return best_centers, best_score

def _local_refine(centers, time_budget=10.0):
    """Local refinement."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    current_centers = centers.copy()
    current_score = _score(current_centers)
    
    step_size = 0.001
    
    while time.time() - start_time < time_budget:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
            
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_c = current_centers.copy()
                    new_c[i, 0] += dx
                    new_c[i, 1] += dy
                    if np.any(np.abs(new_c) > 0.5):
                        continue
                    s = _score(new_c)
                    if s > current_score + 1e-10:
                        current_centers = new_c
                        current_score = s
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_centers = current_centers.copy()
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    return best_centers, best_score

def construct():
    """Main entry point."""
    start_time = time.time()
    
    np.random.seed(42)
    
    # 1. Initialization - try hexagonal first
    centers = _init_hexagonal()
    current_score = _score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    # Also try grid
    grid_centers = _init_grid()
    grid_score = _score(grid_centers)
    if grid_score > best_score:
        best_score = grid_score
        best_centers = grid_centers.copy()
        centers = grid_centers
    
    # 2. Force-directed relaxation
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 50:
        centers = _force_relax(centers, n_steps=80, lr_init=0.008, lr_decay=0.99)
        score = _score(centers)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    # 3. L-BFGS-B optimization on soft surrogate
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 20:
        t_lbfgs = min(remaining * 0.4, 30.0)
        centers_lbfgs = _optimize_lbfgsb(centers, time_budget=t_lbfgs)
        score_lbfgs = _score(centers_lbfgs)
        if score_lbfgs > best_score:
            best_score = score_lbfgs
            best_centers = centers_lbfgs.copy()
    
    # 4. Simulated Annealing
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 15:
        t_sa = min(remaining * 0.5, 30.0)
        centers_sa, score_sa = _optimize_sa(best_centers, time_budget=t_sa)
        if score_sa > best_score:
            best_score = score_sa
            best_centers = centers_sa.copy()
    
    # 5. Another L-BFGS-B pass from SA result
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10:
        t_lbfgs2 = min(remaining * 0.4, 15.0)
        centers_lbfgs2 = _optimize_lbfgsb(best_centers, time_budget=t_lbfgs2)
        score_lbfgs2 = _score(centers_lbfgs2)
        if score_lbfgs2 > best_score:
            best_score = score_lbfgs2
            best_centers = centers_lbfgs2.copy()
    
    # 6. Local Refinement
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 5:
        t_lr = min(remaining * 0.6, 10.0)
        centers_lr, score_lr = _local_refine(best_centers, time_budget=t_lr)
        if score_lr > best_score:
            best_score = score_lr
            best_centers = centers_lr.copy()
    
    # 7. Final validation and clipping
    best_centers = np.clip(best_centers, -0.5, 0.5)
    final_score = _score(best_centers)
    
    # Ensure validity
    if final_score <= 0 or not np.all(np.isfinite(best_centers)):
        best_centers = _init_grid()
    
    return best_centers

if __name__ == "__main__":
    t0 = time.time()
    c = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Score: {_score(c)}")
    print(f"Shape: {c.shape}")
