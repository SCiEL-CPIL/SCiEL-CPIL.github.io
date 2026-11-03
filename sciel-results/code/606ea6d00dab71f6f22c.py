import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # 1. Initialization: Hexagonal Lattice
    # Target spacing for N=300 in unit circle.
    # Area of hexagon with side s is (3*sqrt(3)/2) * s^2.
    # Number of hexagons ~ Area / HexArea = pi / ((3*sqrt(3)/2) * s^2)
    # N ~ pi / ((3*sqrt(3)/2) * s^2) => s ~ sqrt(2*pi / (3*sqrt(3)*N))
    s_init = np.sqrt(2 * np.pi / (3 * np.sqrt(3) * N))
    
    # Generate a slightly denser lattice to have more points, then prune
    s_gen = s_init * 0.98
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 1.0:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        # Select N points closest to the center to maintain symmetry and density
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        # Fallback: add random points
        while len(points) < N:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 1.0:
                points = np.vstack([points, p])
        points = points[:N]

    points = points.copy()
    best_points = points.copy()
    best_score = 0.0

    # Precompute indices for pairwise distances
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    n_pairs = len(i_idx)

    # Optimization Parameters
    # Phase 1: Coarse search
    # Phase 2: Fine search
    # Phase 3: Polish
    
    # We will run a single loop that adapts its parameters over time.
    # Or we can split into two loops. Splitting is safer.
    
    # --- Phase 1: Coarse Optimization ---
    alpha = 0.05
    cooling = 0.999
    k_soft = 50.0
    
    while time.time() - start_time < time_limit * 0.5:
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
        # Use a stable log-sum-exp trick if possible, but simple exp is fine for moderate k
        shifted_vals = all_vals - current_min
        weights = np.exp(-k_soft * shifted_vals)
        sum_w = np.sum(weights)
        if sum_w == 0:
            break
        weights /= sum_w
        
        # Compute Gradients
        grads = np.zeros_like(points)
        
        # Pair contributions
        pair_weights = weights[:n_pairs]
        # Avoid division by zero
        safe_dists = np.where(dists[i_idx, j_idx] > 1e-9, dists[i_idx, j_idx], 1.0)
        vecs = (diff[i_idx, j_idx] / safe_dists[:, np.newaxis]) * pair_weights[:, np.newaxis]
        
        np.add.at(grads, i_idx, vecs)
        np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions
        bound_weights = weights[n_pairs:]
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
        if np.random.rand() < 0.02:
            points += np.random.normal(0, 0.01, size=points.shape)
            rad_new = np.linalg.norm(points, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                points[mask] /= rad_new[mask, np.newaxis]

    # --- Phase 2: Fine Optimization / Polish ---
    # Use a much sharper soft-min to approximate the exact min gradient
    alpha = 0.01
    cooling = 0.9995
    k_soft = 500.0
    
    # Reset best tracking in case Phase 1 didn't find the global best yet
    # (It should, but just in case)
    current_min = np.min(np.concatenate([
        np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)[i_idx, j_idx] / 2.0,
        1.0 - np.linalg.norm(points, axis=1)
    ]))
    if current_min > best_score:
        best_score = current_min
        best_points = points.copy()

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
        sum_w = np.sum(weights)
        if sum_w == 0:
            break
        weights /= sum_w
        
        # Compute Gradients
        grads = np.zeros_like(points)
        
        # Pair contributions
        pair_weights = weights[:n_pairs]
        safe_dists = np.where(dists[i_idx, j_idx] > 1e-9, dists[i_idx, j_idx], 1.0)
        vecs = (diff[i_idx, j_idx] / safe_dists[:, np.newaxis]) * pair_weights[:, np.newaxis]
        
        np.add.at(grads, i_idx, vecs)
        np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions
        bound_weights = weights[n_pairs:]
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

    return best_points
