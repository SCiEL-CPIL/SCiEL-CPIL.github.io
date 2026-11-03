import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # Target spacing for N=300 in unit circle.
    # N ~ 2*pi / (sqrt(3)*s^2) => s ~ sqrt(2*pi / (sqrt(3)*N))
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    
    # Generate a slightly denser lattice to have more points, then prune
    s_gen = s_init * 0.95
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2
    
    points = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 1.0:
                points = np.vstack([points, p])
        points = points[:N]

    points = points.copy()
    best_points = points.copy()
    best_score = 0.0

    # 2. Optimization: Soft-Min Gradient Ascent
    alpha = 0.1
    cooling = 0.998
    k_soft = 100.0  # Sharper soft-min
    
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    
    while time.time() - start_time < time_limit:
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Compute boundary distances
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        # All constraint values
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        current_min = np.min(all_vals)
        
        if current_min > best_score:
            best_score = current_min
            best_points = points.copy()
        
        # Compute Soft-Min Weights
        shifted_vals = all_vals - current_min
        weights = np.exp(-k_soft * shifted_vals)
        weights /= np.sum(weights)
        
        # Compute Gradients
        grads = np.zeros_like(points)
        
        # Pair contributions
        pair_weights = weights[:len(pair_vals)]
        vecs = (diff[i_idx, j_idx] / dists[i_idx, j_idx, np.newaxis]) * pair_weights[:, np.newaxis]
        
        np.add.at(grads, i_idx, vecs)
        np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions
        bound_weights = weights[len(pair_vals):]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        # Update points
        points += alpha * grads
        
        # Project back inside circle
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
            
        # Cool down
        alpha *= cooling
        
        # Random perturbation to escape local minima
        if np.random.rand() < 0.05:
            points += np.random.normal(0, alpha * 2.0, size=points.shape)
            rad_new = np.linalg.norm(points, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                points[mask] /= rad_new[mask, np.newaxis]

    return best_points
