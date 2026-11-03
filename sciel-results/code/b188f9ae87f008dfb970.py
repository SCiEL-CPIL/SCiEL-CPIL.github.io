import numpy as np
import time
import math
from scipy.optimize import least_squares

N = 300

def compute_score(points):
    """
    Compute the packing radius for a set of points.
    """
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    score = min(min_pair / 2.0, min_bound)
    return score

def soft_min_value(points, k=100.0):
    """
    Compute the smooth minimum of pairwise distances and boundary distances.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    rad = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - rad
    
    pair_dists = dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Soft min: -1/k * log(sum(exp(-k * d)))
    d_min = np.min(all_dists)
    exp_terms = np.exp(-k * (all_dists - d_min))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return d_min
    
    soft_min = d_min - (1.0 / k) * np.log(sum_exp)
    return soft_min

def fast_score(points):
    """
    Optimized score calculation for Simulated Annealing.
    Uses a local neighborhood check to avoid O(N^2) full matrix operations.
    """
    # Boundary check
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    # Pairwise distance check with early exit
    # For N=300, checking all pairs is ~45k ops. 
    # We can vectorize the check for a specific point or just do a fast full check.
    # Given N=300, a full check is actually quite fast in NumPy (< 1ms).
    # However, SA might run thousands of times. 
    # Let's stick to a slightly optimized full check or a local check.
    # For simplicity and robustness in SA, we'll use a local check for the moved points.
    
    # Actually, for N=300, even full check is fast. Let's use a hybrid:
    # Check boundary.
    # Check distances for moved points vs all others.
    # This requires knowing which points moved.
    
    # Since 'fast_score' is called on the whole array, we can't easily know what moved 
    # unless we pass indices. Let's modify the SA loop to use a specialized score function 
    # that takes points and indices of moved points.
    
    # For now, this function is a fallback or for initial eval.
    # The SA loop will use a specialized internal function.
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_pair / 2.0, min_bound)

def sa_score_local(points, moved_indices):
    """
    Calculate score assuming only points at moved_indices changed.
    We need to check:
    1. Boundary for moved points.
    2. Distances between moved points and all other points.
    3. Distances between moved points themselves.
    4. We assume the rest of the points are valid and their mutual distances were valid before.
       However, if we moved points, the global min might have been defined by non-moved points 
       previously, but now the bottleneck might be elsewhere? 
       No, if we only move a few points, the new global min is either:
       - The old global min (if it didn't involve moved points)
       - A new min involving moved points.
       
       To be safe and simple, and since N=300 is small, we will just do the full calculation 
       but optimized. Actually, 300x300 distance matrix is 90,000 elements. 
       Computing it takes ~0.1-0.5ms. 
       If SA runs 10,000 times, that's 1-5 seconds. This is acceptable.
       
       So, we don't need a complex local score. We can just use compute_score.
       The previous "fast" attempt was over-engineering.
    """
    return compute_score(points)

def simulated_annealing(points, time_budget, initial_temp=0.015, cooling_rate=0.999, step_size=0.01):
    """
    Simulated Annealing to escape local minima.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    T = initial_temp
    
    while time.time() - start_time < time_budget:
        # Generate perturbation
        # Moving a small subset allows finer adjustments
        num_moves = np.random.randint(1, 5)
        indices = np.random.choice(N, num_moves, replace=False)
        
        proposed_points = current_points.copy()
        noise = np.random.normal(0, step_size, size=(num_moves, 2))
        proposed_points[indices] += noise
        
        # Project to unit circle
        rad = np.linalg.norm(proposed_points, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            proposed_points[mask] /= rad[mask, np.newaxis]
            
        proposed_score = compute_score(proposed_points)
        
        # Acceptance criterion
        delta = proposed_score - current_score
        if delta > 0:
            current_points = proposed_points
            current_score = proposed_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Accept with probability exp(-delta / T)
            # delta is negative, so -delta is positive
            if T > 1e-8:
                prob = np.exp(-delta / T)
                if np.random.rand() < prob:
                    current_points = proposed_points
                    current_score = proposed_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points.copy()
        
        # Cool down
        T *= cooling_rate
        if T < 1e-6:
            T = 1e-6
            
    return best_points, best_score

def polish_least_squares(points, time_budget, k=100.0, R_target=0.055):
    """
    Polish using scipy.optimize.least_squares.
    """
    start_time = time.time()
    
    def residuals(x_flat):
        x = x_flat.reshape(N, 2)
        # Project inside container to ensure validity during optimization
        rad = np.linalg.norm(x, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            x = x.copy()
            x[mask] /= rad[mask, np.newaxis]
        
        s_val = soft_min_value(x, k)
        return np.array([R_target - s_val])

    x0 = points.flatten()
    
    try:
        # Use trf (trust-region reflective)
        result = least_squares(
            residuals, 
            x0, 
            method='trf', 
            max_nfev=5000,
            xtol=1e-12, 
            ftol=1e-12
        )
        
        x_opt = result.x.reshape(N, 2)
        score_opt = compute_score(x_opt)
        
        return x_opt, score_opt
        
    except Exception as e:
        return points, compute_score(points)

def _hex_init(n_points, seed=0):
    """
    Generate multiple hexagonal lattice candidates and return the best one.
    """
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    
    # Try a few different spacings/seeds
    for attempt in range(5):
        # Estimate spacing
        # Area per point ~ pi / n
        # Hex cell area ~ s^2 * sqrt(3)/2
        s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
        # Add some variation
        s *= (0.95 + 0.05 * attempt)
        
        points_list = []
        max_j = int(1.0 / (s * np.sqrt(3) / 2)) + 2
        max_i = int(1.0 / s) + 2
        
        for j in range(-max_j, max_j + 1):
            y = j * s * np.sqrt(3) / 2
            offset = 0 if j % 2 == 0 else s / 2
            for i in range(-max_i, max_i + 1):
                x = i * s + offset
                if np.hypot(x, y) < 1.0:
                    points_list.append([x, y])
                    
        points = np.array(points_list)
        
        if len(points) < n_points:
            continue
            
        # Select n_points closest to center
        dists_to_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_to_center)[:n_points]
        points = points[idx]
        
        # Add small noise to break symmetry for attempts > 0
        if attempt > 0:
            points = points + rng.normal(0, s * 0.01, size=(n_points, 2))
            # Project back
            rad = np.linalg.norm(points, axis=1)
            mask = rad > 1.0
            if np.any(mask):
                points[mask] /= rad[mask, np.newaxis]
        
        score = compute_score(points)
        if score > best_r:
            best_r = score
            best_pts = points.copy()
            
    if best_pts is None:
        # Fallback: random
        angles = np.random.uniform(0, 2 * np.pi, n_points)
        radii = np.sqrt(np.random.uniform(0, 1, n_points))
        best_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
    return best_pts

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Initialization
    best_points = _hex_init(N, seed=42)
    best_score = compute_score(best_points)
    
    # 2. Optimization Phase 1: Simulated Annealing
    # SA helps escape the local minima of the lattice structure.
    sa_budget = 40.0
    sa_elapsed = time.time() - start_time
    if time_limit - sa_elapsed > sa_budget:
        sa_points, sa_score = simulated_annealing(best_points, sa_budget)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points
    
    # 3. Optimization Phase 2: Gradient Polish (Least Squares)
    # LS refines the solution using gradients of the soft-min objective.
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    # Run one main polish
    if remaining > 20.0:
        polished_points, polished_score = polish_least_squares(best_points, 15.0)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points
            
    # 4. Optimization Phase 3: Perturbed Polish
    # Try a few more polishes from slightly perturbed states to ensure we are at a local optimum.
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    for i in range(3):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        noise_scale = 0.005 * (1 + i * 0.5)
        noise = np.random.normal(0, noise_scale, size=best_points.shape)
        current_points = best_points + noise
        rad = np.linalg.norm(current_points, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            current_points[mask] /= rad[mask, np.newaxis]
            
        run_time_budget = min(10.0, time_limit - (time.time() - start_time) - 2.0)
        
        if run_time_budget < 1.0:
            continue
        
        polished_points, polished_score = polish_least_squares(current_points, run_time_budget)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points
            
    # Final validation
    rad_final = np.linalg.norm(best_points, axis=1)
    if np.any(rad_final > 1.0) or not np.all(np.isfinite(best_points)):
        return np.zeros((N, 2))
        
    return best_points
