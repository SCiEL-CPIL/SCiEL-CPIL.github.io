import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools
import time

N = 13
# Precompute all triangle indices
_TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))

def score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        if len(points) < 3:
            return 0.0
        
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
            
        idx = _TRI_IDX
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        return min_area / hull_area
    except Exception:
        return 0.0

def neg_score(points_flat):
    points = points_flat.reshape(N, 2)
    return -score(points)

def neg_score_softmin(points_flat, beta):
    """
    Smoothed objective using soft-min of triangle areas / hull area.
    Minimizes - (softmin(areas) / hull_area)
    """
    try:
        points = points_flat.reshape(N, 2)
        
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 1e6 # Large penalty for degenerate hull
        except Exception:
            return 1e6
            
        idx = _TRI_IDX
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        # Soft min calculation
        # soft_min = -1/beta * log(sum(exp(-beta * areas)))
        # To avoid overflow, shift by max area
        max_area = np.max(areas)
        shifted = areas - max_area
        log_sum = np.log(np.sum(np.exp(-beta * shifted)))
        soft_min_val = max_area - (1.0 / beta) * log_sum
        
        return -soft_min_val / hull_area
    except Exception:
        return 1e6

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_score = 0.0
    best_points = None
    
    def update_best(points):
        nonlocal best_score, best_points
        s = score(points)
        if s > best_score:
            best_score = s
            best_points = points

    # Initial points
    starting_points = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    starting_points.append(init_reg)
    
    # 2. Noisy Regular 13-gon
    rng = np.random.default_rng(42)
    for _ in range(5):
        noise = rng.normal(0, 0.1, (N, 2))
        starting_points.append(init_reg + noise)
        
    # 3. Two rings: 7 outer, 6 inner
    for r_in in [0.3, 0.5, 0.7]:
        angles_7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        angles_6 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        p7 = np.column_stack((np.cos(angles_7), np.sin(angles_7)))
        p6 = np.column_stack((r_in * np.cos(angles_6), r_in * np.sin(angles_6)))
        starting_points.append(np.vstack([p7, p6]))
        
    # 4. Three rings: 5 outer, 4 mid, 4 inner
    for r_mid, r_in in [(0.6, 0.3), (0.7, 0.5)]:
        angles_5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        angles_4 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        p5 = np.column_stack((np.cos(angles_5), np.sin(angles_5)))
        p4m = np.column_stack((r_mid * np.cos(angles_4), r_mid * np.sin(angles_4)))
        p4i = np.column_stack((r_in * np.cos(angles_4 + np.pi/4), r_in * np.sin(angles_4 + np.pi/4)))
        starting_points.append(np.vstack([p5, p4m, p4i]))

    bounds = [(-2.0, 2.0)] * (N * 2)

    # Strategy 1: Differential Evolution with soft-min
    # Use a moderate beta for DE to keep the landscape smooth
    beta_de = 20.0
    try:
        if time.time() < deadline - 60:
            de_result = differential_evolution(
                neg_score_softmin,
                bounds=bounds,
                args=(beta_de,),
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-5,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='immediate'
            )
            
            if de_result.fun < 1e5: # Check if valid
                de_points = de_result.x.reshape(N, 2)
                update_best(de_points)
    except Exception:
        pass
    
    # Strategy 2: Local Optimization (L-BFGS-B) with Annealing
    # We will polish the best points found so far and the initial points
    
    # Collect candidates for polishing
    candidates = list(starting_points)
    if best_points is not None:
        candidates.append(best_points)
        
    # Also add noisy versions of the best points found
    if best_points is not None:
        for _ in range(5):
            noise = rng.normal(0, 0.05, (N, 2))
            candidates.append(best_points + noise)

    # Annealing schedule for beta
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    for start in candidates:
        if time.time() > deadline - 10:
            break
            
        current_points = start.flatten()
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
                
            try:
                res = minimize(
                    neg_score_softmin,
                    current_points,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                if res.fun < 1e5:
                    candidate_points = res.x.reshape(N, 2)
                    update_best(candidate_points)
                    current_points = res.x
            except Exception:
                pass

    # Strategy 3: Final Polish with True Objective
    # The soft-min is an approximation. The true objective is sharper.
    # L-BFGS-B on the non-smooth function might work if we start close to the optimum.
    if best_points is not None and time.time() < deadline - 5:
        try:
            res_true = minimize(
                neg_score,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if res_true.fun < 0:
                update_best(res_true.x.reshape(N, 2))
        except Exception:
            pass

    # Strategy 4: If we have time, do another DE round with tighter bounds around best
    if best_points is not None and time.time() < deadline - 30:
        try:
            center = best_points.mean(axis=0)
            spread = np.abs(best_points - center).max()
            b = max(spread * 1.5, 0.5)
            bounds_tight = [(center[0]-b, center[0]+b), (center[1]-b, center[1]+b)] * N
            # Flatten bounds
            bounds_tight_flat = []
            for i in range(N):
                bounds_tight_flat.append((best_points[i,0]-b, best_points[i,0]+b))
                bounds_tight_flat.append((best_points[i,1]-b, best_points[i,1]+b))
            
            de2 = differential_evolution(
                neg_score_softmin,
                bounds=bounds_tight_flat,
                args=(100.0,),
                seed=123,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                mutation=(0.2, 0.8),
                recombination=0.7,
                polish=False,
                init='latinhypercube'
            )
            
            if de2.fun < 1e5:
                de2_points = de2.x.reshape(N, 2)
                # Polish
                res_p = minimize(
                    neg_score_softmin,
                    de2_points.flatten(),
                    args=(1000.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                if res_p.fun < 1e5:
                    update_best(res_p.x.reshape(N, 2))
        except Exception:
            pass

    if best_points is None:
        best_points = init_reg

    return best_points
