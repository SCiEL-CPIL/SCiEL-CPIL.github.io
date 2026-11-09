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
        """Compute the actual metric (d_min / d_max)^2."""
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

    def get_smooth_score(pts, beta):
        """
        Compute a smooth surrogate: log(softmin) - log(softmax)
        softmin = -1/beta * log(sum(exp(-beta * d)))
        softmax = 1/beta * log(sum(exp(beta * d)))
        We maximize this, so return it (higher is better).
        """
        if not np.all(np.isfinite(pts)):
            return -np.inf
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return -np.inf
        
        # Softmin: -1/beta * log(sum(exp(-beta * d)))
        # Use log-sum-exp trick for stability
        neg_beta_d = -beta * dists
        max_neg = np.max(neg_beta_d)
        log_sum_exp_neg = max_neg + np.log(np.sum(np.exp(neg_beta_d - max_neg)))
        softmin = -log_sum_exp_neg / beta
        
        # Softmax: 1/beta * log(sum(exp(beta * d)))
        beta_d = beta * dists
        max_beta = np.max(beta_d)
        log_sum_exp_pos = max_beta + np.log(np.sum(np.exp(beta_d - max_beta)))
        softmax = log_sum_exp_pos / beta
        
        if softmin <= 0 or softmax <= 0:
            return -np.inf
        
        return np.log(softmin) - np.log(softmax)

    def sa_optimize(init_pts, time_budget, use_smooth=True):
        """
        Run Simulated Annealing to optimize the points.
        Uses smooth surrogate for acceptance decisions when use_smooth=True.
        """
        current_pts = init_pts.copy()
        current_score = get_score(current_pts)
        current_smooth = get_smooth_score(current_pts, 5.0) if use_smooth else current_score
        best_pts = current_pts.copy()
        best_score = current_score
        
        # Initial temperature for smooth score (range is roughly [-10, 0])
        T = 0.3 
        T_min = 1e-6
        cooling_rate = 0.9998  # Very slow cooling
        
        # Move magnitude
        move_sigma = 0.15
        
        steps = 0
        no_improve_steps = 0
        max_no_improve = 4000
        
        # Adaptive beta: start moderate, increase for sharper approximation
        beta = 5.0
        beta_increase_interval = 2000
        beta_max = 50.0
        
        while time.time() - start_time < time_limit - 10:
            if no_improve_steps > max_no_improve:
                # Restart from best point with higher T
                current_pts = best_pts.copy()
                current_score = get_score(current_pts)
                current_smooth = get_smooth_score(current_pts, beta) if use_smooth else current_score
                T = 0.3
                no_improve_steps = 0
                move_sigma = 0.15
                continue

            # Generate a move: perturb one random point
            idx = np.random.randint(n)
            move = np.random.randn(d) * move_sigma
            new_pts = current_pts.copy()
            new_pts[idx] += move
            
            new_score = get_score(new_pts)
            new_smooth = get_smooth_score(new_pts, beta) if use_smooth else new_score
            
            # Acceptance criterion based on smooth score
            if use_smooth:
                delta = new_smooth - current_smooth
                accept = (new_smooth >= current_smooth) or (
                    delta > -10 * T and np.random.rand() < np.exp(delta / T)
                )
            else:
                delta = new_score - current_score
                accept = (new_score >= current_score) or (
                    delta > -10 * T and np.random.rand() < np.exp(delta / T)
                )
            
            if accept:
                current_pts = new_pts
                current_score = new_score
                current_smooth = new_smooth
                no_improve_steps = 0
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            else:
                no_improve_steps += 1
            
            # Cooling
            T *= cooling_rate
            if T < T_min:
                T = T_min
                move_sigma *= 0.97
                if move_sigma < 1e-4:
                    if no_improve_steps > 1000:
                        T = 0.1
                        move_sigma = 0.08
                        no_improve_steps = 0

            # Periodically increase beta for sharper approximation
            if use_smooth and steps % beta_increase_interval == 0 and beta < beta_max:
                beta *= 1.5
                # Re-evaluate current smooth score with new beta
                current_smooth = get_smooth_score(current_pts, beta)

            steps += 1
            
        return best_pts, best_score

    def polish(pts, time_budget):
        """
        Final high-precision polish: very small moves, hill climbing only.
        """
        current_pts = pts.copy()
        current_score = get_score(current_pts)
        best_pts = current_pts.copy()
        best_score = current_score
        
        move_sigma = 0.01
        no_improve = 0
        max_no_improve = 500
        
        while time.time() - start_time < time_limit - 5:
            if no_improve > max_no_improve:
                # Try slightly larger moves
                move_sigma *= 1.5
                no_improve = 0
                if move_sigma > 0.1:
                    break
                continue
            
            idx = np.random.randint(n)
            move = np.random.randn(d) * move_sigma
            new_pts = current_pts.copy()
            new_pts[idx] += move
            
            new_score = get_score(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            else:
                no_improve += 1
            
            # Gradually decrease move size
            move_sigma *= 0.999
            
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
        
        # 3. Random points in cube
        if i % 2 == 0:
            rand_pts_cube = np.random.rand(n, d) * 2 - 1
            initializations.append(rand_pts_cube)

    best_points_overall = initializations[0].copy()
    best_score_overall = get_score(best_points_overall)
    
    # Run SA for each initialization with smooth surrogate
    for init_idx, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 30:
            break
            
        # Time budget for this init
        remaining = time_limit - 10 - elapsed
        time_per_init = min(40.0, remaining / (len(initializations) - init_idx))
        
        if time_per_init < 5:
            break
        
        local_best_pts, local_best_score = sa_optimize(init_pts, time_per_init, use_smooth=True)
        
        if local_best_score > best_score_overall:
            best_score_overall = local_best_score
            best_points_overall = local_best_pts.copy()
    
    # Final polish phase
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10:
        polish_time = min(60.0, time_limit - 10 - elapsed)
        if polish_time > 5:
            best_points_overall, best_score_overall = polish(best_points_overall, polish_time)
    
    return best_points_overall

# EVOLVE-BLOCK-END
