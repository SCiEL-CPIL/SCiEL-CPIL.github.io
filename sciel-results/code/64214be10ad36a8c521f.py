"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_lattice_init(N):
    """Generate a hexagonal lattice configuration for N points."""
    r_est = np.sqrt(0.9 / N)
    spacing = 2 * r_est
    
    k = int(1.0 / spacing) + 3
    
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append((x, y))
            
    points = np.array(points)
    radii = np.hypot(points[:, 0], points[:, 1])
    points = points[radii <= 1.0]
    
    if len(points) > N:
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii, kind='stable')
        points = points[indices[:N]]
    elif len(points) < N:
        # Fallback: random points in unit disk
        # This is unlikely to happen with the above logic for N=300
        theta = np.random.uniform(0, 2 * np.pi, N)
        r = np.sqrt(np.random.uniform(0, 1, N))
        points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        
    return points

def _compute_gradients(points, beta):
    N = len(points)
    eps = 1e-9
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    dist_center = np.linalg.norm(points, axis=1)
    boundary_dist = 1.0 - dist_center
    
    safe_dist = np.where(dist > eps, dist, 1.0)
    safe_center_dist = np.where(dist_center > eps, dist_center, 1.0)
    
    # Compute weights
    # To avoid underflow issues or overflow, we can subtract the max log weight?
    # log_w = -beta * s
    # max_log_w = max(log_w)
    # w = exp(log_w - max_log_w)
    # This is safer.
    
    # Pair weights
    log_w_pair = -beta * dist
    # Boundary weights
    log_w_bound = -beta * boundary_dist
    
    # Combine?
    # We need sum of exps.
    # Let's just use exp directly for now, beta is moderate.
    
    exp_term = np.exp(-beta * dist)
    np.fill_diagonal(exp_term, 0.0)
    
    b_weights = np.exp(-beta * boundary_dist)
    
    sum_pair_w = 0.5 * np.sum(exp_term)
    sum_b_w = np.sum(b_weights)
    sum_w = sum_pair_w + sum_b_w
    
    if sum_w < 1e-15:
        return np.zeros_like(points)
        
    unit_diff = diff / safe_dist[:, :, np.newaxis]
    grad_pair = np.sum(exp_term[:, :, np.newaxis] * unit_diff, axis=1)
    
    unit_center = points / safe_center_dist[:, np.newaxis]
    grad_b = b_weights[:, np.newaxis] * (-unit_center)
    
    grad_m = (grad_pair + grad_b) / sum_w
    
    return grad_m

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    points = _hex_lattice_init(N)
    
    def _score(pts):
        if len(pts) != N:
            return 0.0
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        min_pair = np.min(d[np.triu_indices_from(d, k=1)])
        min_bound = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_pair / 2.0, min_bound)
        
    best_points = points.copy()
    best_score = _score(best_points)
    
    beta = 20.0
    lr = 0.01
    t = 0
    
    while time.time() < deadline:
        beta = 20.0 + 0.05 * t
        if beta > 100.0:
            beta = 100.0
            
        lr = 0.01 / (1.0 + 0.005 * t)
        
        grads = _compute_gradients(points, beta)
        
        points = points + lr * grads
        
        radii = np.linalg.norm(points, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            points[mask] = points[mask] * (1.0 / radii[mask])[:, np.newaxis]
            
        current_score = _score(points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        t += 1
        
    return best_points

# EVOLVE-BLOCK-END
