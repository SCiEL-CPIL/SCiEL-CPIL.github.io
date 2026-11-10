import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import itertools
import time

N_POINTS = 13
TRIANGLE_IDX = np.array(list(itertools.combinations(range(N_POINTS), 3)))

def compute_areas(points):
    """
    Compute areas of all 286 triangles.
    Returns an array of shape (286,).
    """
    p1 = points[TRIANGLE_IDX[:, 0]]
    p2 = points[TRIANGLE_IDX[:, 1]]
    p3 = points[TRIANGLE_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    return areas

def compute_hull_area(points):
    """
    Compute the area of the convex hull.
    Returns 0.0 if hull calculation fails.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    """
    if len(points) < 3:
        return 0.0
    
    A_hull = compute_hull_area(points)
    if A_hull <= 1e-10:
        return 0.0
        
    areas = compute_areas(points)
    A_min = np.min(areas)
    
    return A_min / A_hull

def soft_min_objective(flat_points, T=0.1):
    """
    Objective function for optimization: negative soft-min of triangle areas,
    normalized by hull area.
    
    We want to MAXIMIZE A_min / A_hull.
    So we MINIMIZE - (A_min / A_hull).
    
    Using soft-min: S = -1/T * log(sum(exp(-A_i/T))) approximates -min(A_i).
    To handle the ratio, we compute the soft_min_area and divide by hull_area.
    If hull_area is 0, return a large penalty.
    """
    points = flat_points.reshape(N_POINTS, 2)
    
    # Basic validity check
    if not np.all(np.isfinite(points)):
        return 1e10
        
    A_hull = compute_hull_area(points)
    if A_hull < 1e-8:
        return 1e10

    areas = compute_areas(points)
    
    # Avoid log(0)
    areas = np.maximum(areas, 1e-10)
    
    # Soft min calculation
    # log(sum(exp(-A/T))) = -T * logsumexp(-A/T)
    # We want -soft_min = - ( -T * logsumexp(-A/T) ) ? No.
    # Soft min is defined as -T * log(sum(exp(-A_i/T))).
    # This is a lower bound on the min. As T->0, it approaches min(A_i).
    # We want to maximize the min area. So we minimize -SoftMinArea.
    
    # Numerical stability for logsumexp
    # max_val = np.max(areas) # not needed for -A/T if T is small and A positive
    # Let's use scipy.special.logsumexp if available, else manual
    # Manual:
    # log_sum_exp(x) = max(x) + log(sum(exp(x - max(x))))
    x = -areas / T
    max_x = np.max(x)
    log_sum_exp = max_x + np.log(np.sum(np.exp(x - max_x)))
    soft_min_area = -T * log_sum_exp
    
    score = soft_min_area / A_hull
    
    # We want to maximize score, so minimize -score
    return -score

def polish_local(points, time_limit=30.0):
    """
    Local optimization using Nelder-Mead on the hard-min objective.
    """
    start_time = time.time()
    
    # Initial score
    current_score = compute_score(points)
    best_score = current_score
    best_points = points.copy()
    
    # Objective to minimize: -score
    def obj(flat_pts):
        if time.time() > start_time + time_limit:
            return 1e10
        pts = flat_pts.reshape(N_POINTS, 2)
        s = compute_score(pts)
        if s < 1e-10:
            return 1e10
        return -s
    
    # Run Nelder-Mead
    # We might need to run multiple times with different initial perturbations if it gets stuck
    for i in range(3):
        if time.time() > start_time + time_limit:
            break
            
        # Slight perturbation for diversity in restarts
        if i > 0:
            rng = np.random.default_rng(i)
            perturbed = points + rng.normal(0, 0.01, size=points.shape)
        else:
            perturbed = points
            
        try:
            res = minimize(obj, perturbed.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            
            if res.success and res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape(N_POINTS, 2)
        except Exception:
            pass
            
    return best_points, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area ratio.
    Uses Differential Evolution with a soft-min objective, followed by local polish.
    """
    start_time = time.time()
    # Safety margin: return 10s before the 600s limit
    deadline = start_time + 590.0
    
    best_points_overall = None
    best_score_overall = -1.0
    
    # Strategy 1: Global Search with Differential Evolution
    # Boundaries: Points roughly in [-1, 1] to keep hull area manageable
    # We can center the points later
    bounds = [(-1.0, 1.0)] * (N_POINTS * 2)
    
    # DE parameters
    # popsize=10 means 10*26 = 260 evaluations per generation
    # maxiter=100 means 26000 evaluations.
    # Each evaluation is fast (~1ms), so 26s.
    # We have ~590s. We can afford much more.
    # Let's try popsize=20, maxiter=200. 20*26*200 = 104,000 evals. ~100s.
    # We can run multiple DE instances with different T (temperature) for soft-min.
    
    # We will run DE with a fixed T, then polish.
    # A good T is small enough to approximate min, but large enough for smoothness.
    # T=0.1 is a reasonable start.
    
    # To improve robustness, we can run DE for a few different T values or just one good one.
    # Let's stick to one strong DE run with adaptive T if possible, but DE doesn't support that directly.
    # We can run DE with T=0.05.
    
    T_de = 0.05
    print(f"Starting DE with T={T_de}")
    
    def de_objective(flat_pts):
        if time.time() > deadline - 60: # Stop DE if close to deadline
            return 1e10
        return soft_min_objective(flat_pts, T=T_de)
    
    try:
        # seed=42 for reproducibility
        res_de = differential_evolution(
            de_objective, 
            bounds, 
            seed=42, 
            maxiter=150, 
            popsize=15, 
            tol=1e-6, 
            mutation=(0.4, 1.0), 
            recombination=0.9, 
            polish=False, # We do our own polish
            init='sobol' # Better initial distribution
        )
        
        if res_de.success or res_de.fun < 1e9:
            de_points = res_de.x.reshape(N_POINTS, 2)
            de_score = compute_score(de_points)
            
            if de_score > best_score_overall:
                best_score_overall = de_score
                best_points_overall = de_points.copy()
                print(f"DE found score: {de_score}")
                
    except Exception as e:
        print(f"DE failed: {e}")
    
    # Strategy 2: Local Polish on the DE result
    if best_points_overall is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            print("Polishing DE result...")
            polished_pts, polished_score = polish_local(best_points_overall, time_limit=min(remaining_time - 5, 60))
            if polished_score > best_score_overall:
                best_score_overall = polished_score
                best_points_overall = polished_pts
                print(f"Polished score: {polished_score}")
    
    # Strategy 3: If DE failed or score is low, try the previous local search seeds as fallback
    if best_score_overall < 0.05: # Threshold for "bad" result
        print("Falling back to local search seeds...")
        
        # Quick local search from regular polygon
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        pts_reg = np.column_stack([np.cos(angles), np.sin(angles)])
        
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            polished_reg, score_reg = polish_local(pts_reg, time_limit=min(remaining_time - 5, 30))
            if score_reg > best_score_overall:
                best_score_overall = score_reg
                best_points_overall = polished_reg
                print(f"Regular polygon polished score: {score_reg}")
        
        # Quick local search from 12-gon + center
        angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
        pts12 = np.column_stack([np.cos(angles12), np.sin(angles12)])
        pts12c = np.vstack([np.array([0.0, 0.0]), pts12])
        
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            polished_12c, score_12c = polish_local(pts12c, time_limit=min(remaining_time - 5, 30))
            if score_12c > best_score_overall:
                best_score_overall = score_12c
                best_points_overall = polished_12c
                print(f"12-gon+center polished score: {score_12c}")

    # Final safety check
    if best_points_overall is None:
        # Return a known valid configuration
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_points_overall = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Ensure finite
    if not np.all(np.isfinite(best_points_overall)):
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_points_overall = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Ensure convex hull exists
    try:
        hull = ConvexHull(best_points_overall)
        if hull.volume <= 1e-10:
            angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
            best_points_overall = np.column_stack([np.cos(angles), np.sin(angles)])
    except:
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_points_overall = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points_overall

if __name__ == "__main__":
    # Quick test
    import time
    start = time.time()
    pts = heilbronn_convex13()
    score = compute_score(pts)
    print(f"Time taken: {time.time() - start:.2f}s")
    print(f"Final Score: {score}")
    print(f"Points:\n{pts}")
