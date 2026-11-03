"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def _score(centers):
    """Compute the packing radius for given centers."""
    # Boundary constraint
    if centers.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(centers)):
        return -1.0
    # Check inside square
    if np.any(centers[:, 0] < -0.5 + 1e-12) or np.any(centers[:, 0] > 0.5 - 1e-12):
        pass  # We'll handle via min
    if np.any(centers[:, 1] < -0.5 + 1e-12) or np.any(centers[:, 1] > 0.5 - 1e-12):
        pass
    
    # Pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    
    # Boundary distances
    boundary = np.min([
        np.min(centers[:, 0] + 0.5),
        np.min(0.5 - centers[:, 0]),
        np.min(centers[:, 1] + 0.5),
        np.min(0.5 - centers[:, 1])
    ])
    
    r = 0.5 * min(min_pair, boundary)
    return r

def _project(centers, r_min):
    """Project centers to satisfy minimum distance r_min (approximate)."""
    centers = centers.copy()
    # Boundary projection
    centers[:, 0] = np.clip(centers[:, 0], -0.5 + r_min, 0.5 - r_min)
    centers[:, 1] = np.clip(centers[:, 1], -0.5 + r_min, 0.5 - r_min)
    
    # Pairwise separation: iterative push-apart
    for _ in range(50):
        d = cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        # Find pairs too close
        too_close = d < 2 * r_min - 1e-10
        if not np.any(too_close):
            break
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                if too_close[i, j]:
                    diff = centers[j] - centers[i]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-12:
                        diff = np.array([1.0, 0.0])
                        dist = 1.0
                    else:
                        diff = diff / dist
                    # Push apart to exactly 2*r_min
                    target = 2 * r_min
                    if dist < target:
                        push = (target - dist) / 2.0
                        centers[i] -= push * diff
                        centers[j] += push * diff
                        moved = True
        # Re-project boundaries
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + r_min, 0.5 - r_min)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + r_min, 0.5 - r_min)
        if not moved:
            break
    return centers

def _energy_grad(centers):
    """Gradient of sum of squared pairwise distances (negative gradient for maximization).
    We want to maximize sum_{i<j} ||c_i - c_j||^2, so gradient is 2*sum_{j!=i}(c_i - c_j) for each i.
    For minimization of negative energy, gradient is -2*sum_{j!=i}(c_i - c_j) = 2*sum_{j!=i}(c_j - c_i)."""
    # grad[i] = 2 * sum_{j != i} (c_j - c_i) = 2 * (sum_all - N*c_i)
    sum_all = np.sum(centers, axis=0)
    grad = 2.0 * (sum_all - N * centers)
    return grad

def _pgd_solve(centers0, r_min, max_iters, lr, momentum, time_limit):
    """Projected gradient descent to maximize sum of squared distances subject to min distance >= 2*r_min."""
    centers = centers0.copy()
    velocity = np.zeros_like(centers)
    best_centers = centers.copy()
    best_score = _score(centers)
    start = time.time()
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Gradient of negative energy (we want to minimize -energy)
        grad = _energy_grad(centers)
        
        # Momentum update
        velocity = momentum * velocity - lr * grad
        centers_new = centers + velocity
        
        # Project to feasible set
        centers_new = _project(centers_new, r_min)
        
        # Check improvement
        score = _score(centers_new)
        if score > best_score:
            best_score = score
            best_centers = centers_new.copy()
        
        centers = centers_new
    
    return best_centers, best_score

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Initial configurations to try
    configs = []
    
    # 1. Grid with small perturbation
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    configs.append(grid + 0.01 * np.random.randn(N, 2))
    
    # 2. Hexagonal-like grid
    hex_configs = []
    rows = int(np.ceil(N / 14))  # ~14 per row
    y_spacing = 1.0 / (rows + 1)
    for row in range(rows):
        y = -0.5 + (row + 1) * y_spacing
        x_offset = 0.0 if row % 2 == 0 else y_spacing * 0.5
        x_spacing = 1.0 / (14 + 1)
        for col in range(14):
            if len(hex_configs) < N:
                x = -0.5 + (col + 1) * x_spacing + x_offset
                x = np.clip(x, -0.49, 0.49)
                hex_configs.append([x, y])
    hex_configs = np.array(hex_configs[:N])
    configs.append(hex_configs + 0.01 * np.random.randn(N, 2))
    
    # 3. Random uniform
    configs.append(np.random.uniform(-0.45, 0.45, size=(N, 2)))
    
    # 4. Concentric rings
    ring_configs = []
    num_rings = 14
    for ring in range(num_rings):
        r = (ring + 1) / (num_rings + 1) * 0.45
        n_points = max(1, int(2 * np.pi * r / 0.15))
        for k in range(n_points):
            if len(ring_configs) < N:
                angle = 2 * np.pi * k / n_points
                ring_configs.append([r * np.cos(angle), r * np.sin(angle)])
    ring_configs = np.array(ring_configs[:N])
    configs.append(ring_configs + 0.01 * np.random.randn(N, 2))
    
    best_centers = None
    best_score = -1.0
    
    # Estimate initial radius from best config
    for cfg in configs:
        cfg = np.clip(cfg, -0.49, 0.49)
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_centers = cfg.copy()
    
    if best_centers is None:
        return configs[0]
    
    # Use best initial radius as starting point for constraint
    r_start = max(best_score, 0.03)
    
    # Try different r_min values, increasing from r_start
    r_values = [r_start]
    for i in range(1, 10):
        r_values.append(r_start * (1.0 + 0.02 * i))
    
    for r_min in r_values:
        if time.time() - start_time > time_limit:
            break
        
        for cfg in configs[:2]:  # Use top 2 initial configs
            if time.time() - start_time > time_limit:
                break
            
            cfg = np.clip(cfg, -0.5 + r_min, 0.5 - r_min)
            cfg = _project(cfg, r_min)
            
            centers, score = _pgd_solve(
                cfg, r_min,
                max_iters=500,
                lr=0.001,
                momentum=0.9,
                time_limit=time_limit - (time.time() - start_time)
            )
            
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
    
    # Final polish: local refinement around best
    if best_centers is not None and time.time() - start_time < time_limit - 5:
        r_best = best_score
        # Try to improve with smaller steps
        for _ in range(3):
            if time.time() - start_time > time_limit - 2:
                break
            cfg = best_centers + 0.005 * np.random.randn(N, 2)
            cfg = np.clip(cfg, -0.5 + r_best * 0.99, 0.5 - r_best * 0.99)
            cfg = _project(cfg, r_best * 0.99)
            centers, score = _pgd_solve(
                cfg, r_best * 0.99,
                max_iters=300,
                lr=0.0005,
                momentum=0.95,
                time_limit=time_limit - (time.time() - start_time)
            )
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
    
    # Final validation and clipping
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END

