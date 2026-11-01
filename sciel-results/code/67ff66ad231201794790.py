import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

def construct():
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer

    # Precompute indices for triangle area calculation
    # We need all pairs (i, j) with i < j to compute vectors, but for areas we need triples (i, j, k).
    # Vectorized approach:
    # Let P be (N, 2).
    # We want area of triangle (i, j, k) = 0.5 * |det(P[j]-P[i], P[k]-P[i])|.
    # This can be done by broadcasting.
    
    # Indices for all combinations of 3 from N
    # Number of combinations is C(14, 3) = 364. This is small enough to store.
    i_indices, j_indices, k_indices = np.indices((N, N, N)).reshape(3, -1).T
    mask = (i_indices < j_indices) & (j_indices < k_indices)
    i_idx = i_indices[mask]
    j_idx = j_indices[mask]
    k_idx = k_indices[mask]
    
    def compute_areas(pts):
        """
        Vectorized computation of all triangle areas.
        pts: (N, 2)
        Returns: array of shape (M,) where M is number of triples
        """
        p_i = pts[i_idx]
        p_j = pts[j_idx]
        p_k = pts[k_idx]
        
        v1 = p_j - p_i
        v2 = p_k - p_i
        
        # det = v1_x * v2_y - v1_y * v2_x
        dets = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(dets)
        return areas

    def compute_hull_area(pts):
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1e-10 # Return small area to penalize invalid configs

    def smoothed_objective(x, beta):
        """
        Minimize: (1/beta) * log(sum(exp(-beta * areas))) / hull_area
        This is an approximation of -min(areas)/hull_area.
        """
        pts = np.zeros((N, 2))
        # First two points fixed? 
        # The current program fixes first two points to [0,0] and [1,0].
        # This is a valid normalization (translation and scaling invariance).
        # We can keep this to reduce dimensionality.
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        
        areas = compute_areas(pts)
        
        # Log-sum-exp trick for stability
        # LSE(y) = log(sum(exp(y)))
        # We want log(sum(exp(-beta * areas)))
        # = -beta * max(areas) + log(sum(exp(-beta * (areas - max(areas)))))
        max_a = np.max(areas)
        if max_a < 1e-10:
            return 1e6 # Penalize zero area
        
        # Avoid overflow in exp
        shifted = -beta * (areas - max_a)
        # shifted values are <= 0, so exp is <= 1
        sum_exp = np.sum(np.exp(shifted))
        if sum_exp == 0:
            return 1e6
            
        lse_val = -beta * max_a + np.log(sum_exp)
        
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1e6
            
        # Objective to minimize
        return lse_val / hull_area

    def raw_score(pts):
        areas = compute_areas(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 0.0
        return min_area / hull_area

    # Optimization setup
    dim = 2 * (N - 2)
    bounds = [(-3.0, 3.0)] * dim
    
    best_score = 0.0
    best_x = None
    
    # Generate multiple diverse initial guesses
    def get_initial_guesses():
        guesses = []
        # 1. Circle
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        for i in range(N - 2):
            guesses.append([np.cos(angles[i+2]), np.sin(angles[i+2])])
        
        # 2. Random
        for _ in range(5):
            guesses.append(np.random.uniform(-1, 1, dim))
            
        # 3. Grid-like
        x_grid = np.linspace(-1, 1, N)
        y_grid = np.linspace(-1, 1, N)
        # Take first N-2 points from a diagonal or similar
        g = []
        for i in range(N-2):
            g.extend([x_grid[i+2], y_grid[i+2]])
        guesses.append(g)
        
        return [np.array(g) for g in guesses]

    initial_guesses = get_initial_guesses()
    
    # Annealing schedule for beta
    # Start low (smooth) to find global basin, end high (sharp) for precision
    betas = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    for guess in initial_guesses:
        if time.time() - start_time > time_limit:
            break
            
        current_x = guess.copy()
        
        for beta in betas:
            if time.time() - start_time > time_limit:
                break
                
            try:
                res = minimize(
                    smoothed_objective,
                    current_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                current_x = res.x
            except Exception:
                pass
        
        # Final evaluation with raw score
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = current_x[2 * i]
            pts[i + 2, 1] = current_x[2 * i + 1]
            
        score = raw_score(pts)
        if score > best_score:
            best_score = score
            best_x = current_x
            
        # Local refinement on raw objective using Nelder-Mead?
        # The raw objective is non-smooth. 
        # However, we can use the high-beta smoothed objective as a proxy for final polish.
        if time.time() - start_time > time_limit - 10:
            continue
            
        try:
            # One more step with very high beta
            res_final = minimize(
                smoothed_objective,
                current_x,
                args=(1000.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14}
            )
            current_x = res_final.x
            
            pts = np.zeros((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            for i in range(N - 2):
                pts[i + 2, 0] = current_x[2 * i]
                pts[i + 2, 1] = current_x[2 * i + 1]
                
            score_final = raw_score(pts)
            if score_final > best_score:
                best_score = score_final
                best_x = current_x
        except Exception:
            pass

    if best_x is None:
        # Fallback
        best_x = np.zeros(dim)
        
    # Construct final points
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    for i in range(N - 2):
        final_pts[i + 2, 0] = best_x[2 * i]
        final_pts[i + 2, 1] = best_x[2 * i + 1]
        
    return final_pts
