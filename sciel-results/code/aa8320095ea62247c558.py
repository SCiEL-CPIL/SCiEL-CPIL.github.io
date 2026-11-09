# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 600  # safety margin before 630s kill

    def get_score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def sa_optimize(init_pts, time_budget):
        """
        Run Simulated Annealing to optimize the points.
        """
        current_pts = init_pts.copy()
        current_score = get_score(current_pts)
        best_pts = current_pts.copy()
        best_score = current_score
        
        # Initial temperature
        # Score range is [0, 1]. A good initial T allows accepting scores 0.5 vs 1.0
        # Delta = 0.5. exp(-0.5/T) > 0.5 => T > 0.7.
        T = 0.5 
        T_min = 1e-6
        cooling_rate = 0.9995 # Slow cooling to explore more
        
        # Move magnitude
        # Start with relatively large moves
        move_sigma = 0.2
        
        steps = 0
        no_improve_steps = 0
        max_no_improve = 5000
        
        while time.time() - start_time < time_limit - 5:
            # Check if we should restart or adjust parameters
            if no_improve_steps > max_no_improve:
                # Restart from best point with higher T
                current_pts = best_pts.copy()
                current_score = best_score
                T = 0.5
                no_improve_steps = 0
                move_sigma = 0.2
                continue

            # Generate a move: perturb one random point
            idx = np.random.randint(n)
            move = np.random.randn(d) * move_sigma
            new_pts = current_pts.copy()
            new_pts[idx] += move
            
            new_score = get_score(new_pts)
            
            # Acceptance criterion
            if new_score >= current_score:
                current_pts = new_pts
                current_score = new_score
                no_improve_steps = 0
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            else:
                # Accept with probability exp((new - current) / T)
                delta = new_score - current_score
                # If delta is very negative, exp will be 0
                if delta > -10 * T: # Optimization: avoid underflow
                    prob = np.exp(delta / T)
                    if np.random.rand() < prob:
                        current_pts = new_pts
                        current_score = new_score
                no_improve_steps += 1
            
            # Cooling
            T *= cooling_rate
            if T < T_min:
                T = T_min
                # If T is at min, we are effectively doing local search (hill climbing)
                # Reduce move size to polish
                move_sigma *= 0.95
                if move_sigma < 1e-5:
                    # Try to restart with larger moves if stuck
                    if no_improve_steps > 1000:
                         T = 0.1
                         move_sigma = 0.1
                         no_improve_steps = 0

            steps += 1
            
            # Adaptive move size: if success rate is too low, increase sigma; if too high, decrease
            # Simple heuristic: every 1000 steps, check
            # For simplicity, we stick to the cooling schedule and restart logic.
            
        return best_pts, best_score

    # Initializations
    initializations = []
    
    # 1. Icosahedron + Poles (normalized)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms_ico = np.linalg.norm(ico_verts, axis=1)
    ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
    base_points = np.vstack([ico_verts_norm, np.array([[0, 0, 1.0], [0, 0, -1.0]])])
    initializations.append(base_points)
    
    # 2. Random points on sphere
    np.random.seed(42)
    for i in range(10):
        rand_pts = np.random.randn(n, d)
        norms = np.linalg.norm(rand_pts, axis=1)
        rand_pts = rand_pts / norms[:, np.newaxis]
        initializations.append(rand_pts)
        
        # 3. Random points in cube (sometimes interior points help if scale is not fixed, 
        #    but ratio is scale invariant, so sphere is usually a good start)
        if i % 2 == 0:
            rand_pts_cube = np.random.rand(n, d) * 2 - 1
            initializations.append(rand_pts_cube)

    best_points_overall = initializations[0].copy()
    best_score_overall = get_score(best_points_overall)
    
    # Allocate time to each initialization
    # We have ~600s. 13 initializations.
    # Let's give the first one (icosahedron) more time, and others less.
    # Or just run them sequentially until time is up.
    
    for init_idx, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:
            break
            
        # Run SA for this initialization
        # Give each init a fixed chunk of time, or just run until time is up?
        # Running until time is up for the first init might be better if it's good.
        # But we want to explore multiple basins.
        # Let's allocate 45s per init (13 * 45 = 585s).
        time_budget_for_init = 45.0
        target_end_time = time.time() + time_budget_for_init
        
        # Modify sa_optimize to respect a specific end time or budget
        # My sa_optimize uses global start_time and time_limit. 
        # I need to pass a specific deadline.
        
        # Let's refactor sa_optimize to take a deadline
        pass

    # Refactor: I will implement a local SA loop inside the main function to control time per init more easily.
    
    best_points = initializations[0].copy()
    best_score = get_score(best_points)
    
    for init_idx, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 20:
            break
            
        # Run SA for this init
        current_pts = init_pts.copy()
        current_score = get_score(current_pts)
        local_best_pts = current_pts.copy()
        local_best_score = current_score
        
        T = 0.5
        T_min = 1e-6
        cooling_rate = 0.9995
        move_sigma = 0.2
        
        no_improve_steps = 0
        max_no_improve = 3000 # Restart within this init's run
        
        # Time budget for this init: 40 seconds
        init_deadline = time.time() + 40.0
        
        while time.time() < init_deadline:
            if no_improve_steps > max_no_improve:
                # Restart from local best
                current_pts = local_best_pts.copy()
                current_score = local_best_score
                T = 0.5
                move_sigma = 0.2
                no_improve_steps = 0
                continue

            idx = np.random.randint(n)
            move = np.random.randn(d) * move_sigma
            new_pts = current_pts.copy()
            new_pts[idx] += move
            
            new_score = get_score(new_pts)
            
            if new_score >= current_score:
                current_pts = new_pts
                current_score = new_score
                no_improve_steps = 0
                if current_score > local_best_score:
                    local_best_score = current_score
                    local_best_pts = current_pts.copy()
            else:
                delta = new_score - current_score
                if delta > -10 * T:
                    prob = np.exp(delta / T)
                    if np.random.rand() < prob:
                        current_pts = new_pts
                        current_score = new_score
                no_improve_steps += 1
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
                move_sigma *= 0.95
                if move_sigma < 1e-5:
                    if no_improve_steps > 500:
                        T = 0.1
                        move_sigma = 0.1
                        no_improve_steps = 0

        if local_best_score > best_score:
            best_score = local_best_score
            best_points = local_best_pts.copy()
            
    return best_points

# EVOLVE-BLOCK-END
