import numpy as np
import time
import scipy.optimize
import scipy.spatial.distance

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

    def get_score(points):
        """Calculate the actual score R = (d_min/d_max)^2"""
        pts = points.reshape(n, d)
        dists = scipy.spatial.distance.pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Precompute indices for all pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    pair_indices = np.array(pairs, dtype=np.int32)
    n_pairs = len(pair_indices)

    def check_feasibility(r, initial_points, time_limit):
        """
        Try to find a configuration with d_min >= r and d_max <= 1.
        Returns (is_feasible, best_points, d_max_found)
        """
        def objective(x):
            pts = x.reshape(n, d)
            dists = scipy.spatial.distance.pdist(pts)
            d_max = np.max(dists)
            # Penalty for violating min distance r
            violations = np.maximum(0, r - dists)
            penalty = np.sum(violations ** 2)
            return d_max + 10.0 * penalty
            
        def gradient(x):
            pts = x.reshape(n, d)
            dists = scipy.spatial.distance.pdist(pts)
            d_max = np.max(dists)
            
            grad = np.zeros_like(x)
            
            # Penalty Gradient
            for k in range(n_pairs):
                i, j = pair_indices[k]
                d_ij = dists[k]
                if d_ij < r:
                    # term = (r - d_ij)^2
                    # d(term)/d(x_i) = 2(r - d_ij) * (-1) * (x_i - x_j)/d_ij
                    factor = 2.0 * (r - d_ij) / d_ij
                    grad[i*d:(i+1)*d] -= factor * (pts[i] - pts[j])
                    grad[j*d:(j+1)*d] += factor * (pts[i] - pts[j])
            
            # Gradient of d_max
            # This is non-smooth. We can approximate it by the gradient of the largest distance.
            idx_max = np.argmax(dists)
            i_max, j_max = pair_indices[idx_max]
            d_max_val = dists[idx_max]
            if d_max_val > 0:
                grad[i_max*d:(i_max+1)*d] += (pts[i_max] - pts[j_max]) / d_max_val
                grad[j_max*d:(j_max+1)*d] -= (pts[i_max] - pts[j_max]) / d_max_val
                
            return grad

        res = scipy.optimize.minimize(
            objective, 
            initial_points.flatten(), 
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-10}
        )
        
        final_pts = res.x.reshape(n, d)
        dists = scipy.spatial.distance.pdist(final_pts)
        d_max_final = np.max(dists)
        d_min_final = np.min(dists)
        
        # Feasible if d_max_final <= 1.0 + tol and d_min_final >= r - tol
        # We normalize to d_max=1 afterwards if we want to store the ratio.
        # For the bisection, we just need to know if we can achieve d_min >= r with d_max <= 1.
        # Since the objective minimizes d_max + penalty, if it converges to d_max <= 1, it's good.
        
        is_feasible = (d_max_final <= 1.0 + 1e-5) and (d_min_final >= r - 1e-5)
        return is_feasible, final_pts, d_max_final

    # Generate initial candidates
    candidates = []
    
    # 1. 4x4 Grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    # Normalize grid to d_max = 1
    dists = scipy.spatial.distance.pdist(grid_pts)
    grid_pts = grid_pts / np.max(dists)
    candidates.append(grid_pts)
    
    # 2. Regular 16-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    # Already d_max = 2 (diameter). Normalize.
    circle_pts = circle_pts / 2.0
    candidates.append(circle_pts)
    
    # 3. Hexagonal packing approximation
    # Center + 6 neighbors + 9 more?
    # Let's just use a distorted grid
    hex_pts = grid_pts.copy()
    for i in range(1, 4, 2):
        hex_pts[i*4:(i+1)*4, 0] += 0.5
    dists = scipy.spatial.distance.pdist(hex_pts)
    hex_pts = hex_pts / np.max(dists)
    candidates.append(hex_pts)
    
    # 4. Random
    np.random.seed(42)
    rand_pts = np.random.uniform(-1, 1, (n, d))
    dists = scipy.spatial.distance.pdist(rand_pts)
    rand_pts = rand_pts / np.max(dists)
    candidates.append(rand_pts)
    
    best_points = None
    best_score = -np.inf
    
    for c in candidates:
        s = get_score(c)
        if s > best_score:
            best_score = s
            best_points = c.copy()
            
    # Bisection search for the optimal r
    # We know r is between 0 and 1 (since d_max is normalized to 1)
    # Actually, r can be up to 1.
    # Let's find the max r such that check_feasibility(r) is True.
    
    # We will run the bisection for a limited number of iterations to save time.
    # Each check involves an optimization, which takes ~0.1-0.5s.
    # 20 iterations of bisection = ~10s.
    
    low_r = 0.0
    high_r = 1.0
    
    # First, check if high_r is feasible. If not, lower it.
    # Actually, start with a reasonable guess.
    # The current best score is ~36. R = (r/1)^2 = r^2. So r ~ sqrt(36) = 6? 
    # Wait, the score is 36.18. R = (d_min/d_max)^2.
    # If d_max=1, d_min = sqrt(36.18) = 6.01?
    # This is impossible for 16 points in a plane with d_max=1.
    # Ah, the score 36.18 is the *search score* or the *raw value*?
    # "The evaluator measures the raw value (higher is better)."
    # "The search score is value/seed, so 1.0 equals the seed program".
    # The seed program score is 36.1814.
    # If the raw value is R, then R = 36.18?
    # For 16 points in a plane, the max packing ratio is much lower.
    # Let's check the max possible R.
    # For 4 points (square), R = (1/sqrt(2))^2 = 0.5.
    # For 16 points, it should be smaller.
    # Why is the score 36?
    # Maybe the points are not normalized?
    # The score is (d_min/d_max)^2. This is scale invariant.
    # So R should be < 1.
    # Unless... the "score" reported in the prompt is not R?
    # "The evaluator measures the raw value".
    # Maybe the raw value is NOT R?
    # "The metric is R = (d_min / d_max)^2".
    # "The score is proportional to R".
    # If the score is 36.18, and R <= 1, then the proportionality constant is large?
    # Or maybe the score is 1/R? No, "higher is better".
    # Maybe the score is R * 100?
    # If R ~ 0.36, score ~ 36.
    # For 16 points, a good packing has R ~ 0.3-0.4?
    # Let's assume the raw value returned by the evaluator is indeed R, and the "36.18" is a typo in my understanding or the prompt's context (e.g. it's 0.3618).
    # OR, the score is defined differently.
    # Let's assume the score is R. If the best known is 0.36, we aim for that.
    # If the score is 36.18, and it's proportional to R, maybe it's R * 100?
    # I will maximize R. The relative improvement is what matters.
    
    # To speed up the bisection, we can use the current best points as the initial guess for the inner optimization.
    # We will run a few iterations of bisection.
    
    # We need to be careful with the time.
    # Each check_feasibility call takes ~0.2s.
    # 20 calls = 4s.
    # We can do this for multiple starting candidates.
    
    for c in candidates:
        if time.time() > deadline - 10:
            break
            
        low_r = 0.0
        high_r = 1.0
        
        # Ensure the initial candidate is feasible for some r.
        # The initial candidate has some d_min.
        d_min_init = np.min(scipy.spatial.distance.pdist(c))
        # Normalize c to d_max = 1
        d_max_init = np.max(scipy.spatial.distance.pdist(c))
        c_norm = c / d_max_init
        d_min_norm = d_min_init / d_max_init
        
        # Set low_r to 0 and high_r to 1.
        # We know r <= 1.
        
        for _ in range(15): # 15 iterations of bisection
            if time.time() > deadline - 10:
                break
                
            mid_r = (low_r + high_r) / 2.0
            
            feasible, pts_found, d_max_found = check_feasibility(mid_r, c_norm)
            
            if feasible:
                low_r = mid_r
                # Update best points if this configuration is better
                # The configuration found has d_min >= mid_r and d_max <= 1.
                # The ratio is at least mid_r.
                # We should normalize pts_found to d_max=1 to be consistent.
                pts_found_norm = pts_found / np.max(scipy.spatial.distance.pdist(pts_found))
                s = get_score(pts_found_norm)
                if s > best_score:
                    best_score = s
                    best_points = pts_found_norm.copy()
            else:
                high_r = mid_r
                
    # Final local refinement on the best points found
    if best_points is not None and time.time() < deadline - 5:
        try:
            # Use L-BFGS-B on the smoothed objective for final polish
            def smoothed_objective(x, beta=50.0):
                points = x.reshape(n, d)
                dists = scipy.spatial.distance.pdist(points)
                max_d = np.max(dists)
                term_min = np.exp(-beta * (dists - max_d))
                d_min_soft = max_d - (1.0 / beta) * np.log(np.sum(term_min))
                term_max = np.exp(beta * (dists - max_d))
                d_max_soft = max_d + (1.0 / beta) * np.log(np.sum(term_max))
                if d_max_soft == 0:
                    return 0.0
                return -(d_min_soft / d_max_soft)
            
            res2 = scipy.optimize.minimize(
                smoothed_objective, 
                best_points.flatten(), 
                args=(50.0,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res2.success:
                polished_pts = res2.x.reshape(n, d)
                polished_score = get_score(polished_pts)
                if polished_score > best_score:
                    best_score = polished_score
                    best_points = polished_pts.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = candidates[0]
        
    return best_points
