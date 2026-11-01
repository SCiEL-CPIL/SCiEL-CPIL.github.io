import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

def construct():
    def get_points(x):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def compute_areas(pts):
        """Vectorized computation of all triangle areas"""
        # N=14, C(14,3) = 364 triangles. Vectorization might be overkill but good for speed.
        # Actually, 364 is small. A loop is fine, but let's try to make it efficient.
        areas = np.zeros(364)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    v1 = pts[j] - pts[i]
                    v2 = pts[k] - pts[i]
                    det = v1[0] * v2[1] - v1[1] * v2[0]
                    areas[idx] = 0.5 * abs(det)
                    idx += 1
        return areas

    def compute_hull_area(pts):
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 0.0

    def raw_score(x):
        pts = get_points(x)
        areas = compute_areas(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return -np.inf
        return min_area / hull_area

    def smoothed_objective(x, beta):
        pts = get_points(x)
        areas = compute_areas(pts)
        
        # Penalty for small hull area to avoid division by zero and degenerate cases
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-6:
            return 1.0 + (1e-6 - hull_area)
            
        # Log-sum-exp approximation of min(areas)
        # min(a) approx -1/beta * log(sum(exp(-beta * a)))
        # To avoid overflow, shift by max
        max_area = np.max(areas)
        log_sum_exp = max_area + np.log(np.sum(np.exp(-beta * (areas - max_area))))
        smooth_min = -log_sum_exp / beta
        
        # We want to maximize smooth_min / hull_area
        # So we minimize -smooth_min / hull_area
        # Add a small penalty if points are too close to each other (optional, but helps stability)
        # Let's stick to the main term.
        return -(smooth_min / hull_area)

    # Time limit management
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    best_score = -np.inf
    best_x = None
    
    # Define bounds for the free points
    # The convex hull should be roughly bounded. 
    # Since we fix (0,0) and (1,0), the other points should be in a reasonable range.
    # A square of side 2 centered at (0.5, 0.5) is a good start.
    bounds = [(-1.0, 2.0)] * (2 * (N - 2))
    
    # Initializations
    # 1. Regular polygon on a circle
    # 2. Random points in a circle
    # 3. Points from a previous best solution (if we had one, but we start fresh)
    
    # Strategy: Multi-restart L-BFGS-B with annealing beta
    # Beta starts low (smooth) and increases (sharp)
    
    # Pre-allocate for speed in the loop? No, just call functions.
    
    num_restarts = 40
    betas = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    # We will iterate through restarts. For each restart, we will try a sequence of betas.
    # Or, we can do a global search with low beta, then refine with high beta.
    
    # Let's do a loop over restarts.
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Generate initial guess
        if r == 0:
            # Regular polygon
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            x0 = np.zeros(2 * (N - 2))
            for i in range(N - 2):
                idx = i + 2
                x0[2 * i] = 0.5 + 0.5 * np.cos(angles[idx])
                x0[2 * i + 1] = 0.5 * np.sin(angles[idx])
        else:
            # Random in a circle of radius 1.5 centered at (0.5, 0.5)
            x0 = np.random.uniform(-1.0, 2.0, size=2 * (N - 2))
            # Or random on a circle
            # angles = np.random.uniform(0, 2 * np.pi, N - 2)
            # radii = np.random.uniform(0.1, 1.5, N - 2)
            # x0[0::2] = 0.5 + radii * np.cos(angles)
            # x0[1::2] = 0.5 + radii * np.sin(angles)
            
        # Annealing loop for this restart
        current_x = x0
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
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                current_x = res.x
            except Exception:
                pass
        
        # Evaluate raw score for this candidate
        score = raw_score(current_x)
        if score > best_score:
            best_score = score
            best_x = current_x.copy()
            
        # Also try a final Nelder-Mead polish on the best raw objective?
        # NM is slow. Let's just rely on the high-beta L-BFGS-B result.
        # Or, do a quick NM on the raw objective for the top few candidates.
        
    # Final Polish: Use Nelder-Mead on the raw objective for the best candidate found
    if best_x is not None and time.time() - start_time < time_limit - 10:
        try:
            res_nm = minimize(
                lambda x: -raw_score(x),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            score_nm = -res_nm.fun
            if score_nm > best_score:
                best_score = score_nm
                best_x = res_nm.x
        except Exception:
            pass

    return get_points(best_x)
