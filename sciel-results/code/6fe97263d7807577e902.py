import numpy as np
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600 - 10  # 590 seconds to be safe
    
    # Fast distance calculation using broadcasting
    # Returns array of shape (n*(n-1)/2,)
    def fast_dists(points):
        # points: (n, d)
        # diff: (n, n, d)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        # dist_sq: (n, n)
        dist_sq = np.sum(diff * diff, axis=-1)
        # Extract upper triangle (excluding diagonal)
        # indices for upper triangle
        i, j = np.triu_indices(n, k=1)
        return dist_sq[i, j]

    def score(points):
        dists = fast_dists(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    # Simulated Annealing
    def simulated_annealing(init_points, t_start=1.0, t_end=0.001, steps=50000, alpha=0.99995):
        current_points = init_points.copy()
        current_score = score(current_points)
        best_points = current_points.copy()
        best_score = current_score
        
        # Pre-calculate cooling factors
        # We want T to go from t_start to t_end over 'steps'
        # T_k = t_start * alpha^k
        # We can just use a loop with time checks
        
        # For efficiency, we'll do chunks of steps and check time
        chunk_size = 1000
        
        # Initial step size
        step_size = 0.1
        current_temp = t_start
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Update temperature
            # Using a linear or exponential decay. Exponential is standard.
            # But we have a hard time limit. Let's just decay each step.
            current_temp *= alpha
            
            # Generate candidate
            # Perturb one or two points randomly
            num_perturbed = np.random.choice([1, 2], p=[0.7, 0.3])
            indices = np.random.choice(n, num_perturbed, replace=False)
            
            candidate_points = current_points.copy()
            # Gaussian noise
            noise = np.random.normal(0, step_size, (num_perturbed, d))
            candidate_points[indices] += noise
            
            # Calculate new score
            new_score = score(candidate_points)
            
            # Acceptance criterion
            if new_score > current_score:
                current_points = candidate_points
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = candidate_points.copy()
            else:
                # Accept worse solution with probability exp((new - current) / T)
                # Since we maximize, delta = new - current < 0
                delta = new_score - current_score
                if current_temp > 1e-15:
                    prob = np.exp(delta / current_temp)
                    if np.random.random() < prob:
                        current_points = candidate_points
                        current_score = new_score
                # Else reject
                
            # Adaptive step size (optional, but helps)
            # If we are stuck, reduce step size. If we are moving a lot, maybe keep it.
            # Simple heuristic: reduce step size slowly
            step_size *= 0.99999

        return best_points, best_score

    # Multi-start
    best_points_overall = None
    best_score_overall = -np.inf

    # Define several diverse initial configurations
    inits = []
    
    # 1. Grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    inits.append(np.column_stack((grid_x.ravel(), grid_y.ravel())))
    
    # 2. Circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    inits.append(np.column_stack((np.cos(angles), np.sin(angles))))
    
    # 3. Random (seeded for reproducibility in structure, but SA will move it)
    np.random.seed(42)
    inits.append(np.random.randn(n, d))
    
    # 4. Two concentric circles (common structure for packing)
    # 8 on outer, 8 on inner? Or 1 center + 15?
    # Let's try 1 center + 15 on circle
    center = np.zeros((1, 2))
    angles_15 = np.linspace(0, 2*np.pi, 15, endpoint=False)
    circle_15 = np.column_stack((np.cos(angles_15), np.sin(angles_15)))
    inits.append(np.vstack([center, circle_15]))
    
    # 5. Random restarts from scratch
    for i in range(5):
        np.random.seed(100 + i)
        inits.append(np.random.randn(n, d) * 0.5)

    # Run SA for each initial config
    # We allocate time budget dynamically
    total_inits = len(inits)
    # We will run SA until the deadline is hit.
    # To be fair, we can just let the first few run longer if they are promising, 
    # or just run them sequentially until time is up.
    # Given the non-convexity, diversity is key.
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        # Give each run a reasonable number of steps, but bounded by time
        # 50k steps per run is fast with vectorized dists.
        # Let's estimate: 50k steps * ~10us/step = 0.5s. So we can do many runs.
        # Let's do 100k steps per run to be safe on quality, but check time.
        
        res_pts, res_score = simulated_annealing(init, steps=100000)
        
        if res_score > best_score_overall:
            best_score_overall = res_score
            best_points_overall = res_pts

    # Final local refinement using Nelder-Mead on the best point found by SA
    # This helps polish the result to machine precision
    if best_points_overall is not None and time.time() < deadline - 10:
        try:
            import scipy.optimize
            # Minimize negative score
            res_nm = scipy.optimize.minimize(
                lambda x: -score(x.reshape(n, d)), 
                best_points_overall.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            nm_score = -res_nm.fun
            if nm_score > best_score_overall:
                best_score_overall = nm_score
                best_points_overall = res_nm.x.reshape(n, d)
        except Exception:
            pass

    # Fallback
    if best_points_overall is None:
        best_points_overall = inits[0]

    return best_points_overall
