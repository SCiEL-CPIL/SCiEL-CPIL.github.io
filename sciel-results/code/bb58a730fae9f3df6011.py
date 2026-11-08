import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 13

def score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        # Check for degenerate points
        if len(points) < 3:
            return 0.0
        
        # Compute Convex Hull Area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
            
        # Compute Min Triangle Area
        idx = np.array(list(itertools.combinations(range(N), 3)))
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

def neg_score_smooth(points_flat, k=5):
    """
    Smoothed objective: use k-th smallest triangle area instead of minimum.
    """
    try:
        points = points_flat.reshape(N, 2)
        
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
            
        idx = np.array(list(itertools.combinations(range(N), 3)))
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        k = min(k, len(areas))
        kth_area = np.partition(areas, k-1)[k-1]
        
        return -kth_area / hull_area
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    best_score = 0.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    tri_idx = np.array(list(itertools.combinations(range(N), 3)))
    
    bounds = [(-2.0, 2.0)] * (N * 2)
    
    # Strategy 1: Differential Evolution with smoothed objective
    try:
        if time.time() < deadline - 60:
            de_result = differential_evolution(
                neg_score_smooth,
                bounds=bounds,
                args=(5,),
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='immediate'
            )
            
            if de_result.success and de_result.fun < 0:
                de_points = de_result.x.reshape(N, 2)
                de_score = score(de_points)
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points
    except Exception:
        pass
    
    # Strategy 2: Generate diverse starting points
    starting_points = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_13 = np.column_stack((np.cos(angles), np.sin(angles)))
    starting_points.append(init_13)
    
    # 2. Regular 12-gon + Center
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts_12 = np.column_stack((np.cos(angles_12), np.sin(angles_12)))
    init_12c = np.vstack([pts_12, np.array([[0.0, 0.0]])])
    starting_points.append(init_12c)
    
    # 3. Two rings: 7 outer, 6 inner
    angles_7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_6 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r_inner = 0.5
    pts_7 = np.column_stack((np.cos(angles_7), np.sin(angles_7)))
    pts_6 = np.column_stack((r_inner * np.cos(angles_6), r_inner * np.sin(angles_6)))
    init_2ring = np.vstack([pts_7, pts_6])
    starting_points.append(init_2ring)
    
    # 4. More two-ring configurations
    for r_in in [0.3, 0.4, 0.6, 0.7]:
        angles_8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        angles_5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        p8 = np.column_stack((np.cos(angles_8), np.sin(angles_8)))
        p5 = np.column_stack((r_in * np.cos(angles_5), r_in * np.sin(angles_5)))
        starting_points.append(np.vstack([p8, p5]))
    
    # 5. Noisy versions of regular 13-gon
    for _ in range(6):
        noise = rng.normal(0, 0.25, (N, 2))
        starting_points.append(init_13 + noise)
    
    # 6. If DE found a good point, add it and noisy versions
    if best_points is not None:
        starting_points.append(best_points)
        for _ in range(3):
            noise = rng.normal(0, 0.1, (N, 2))
            starting_points.append(best_points + noise)
    
    # L-BFGS-B optimization from each starting point
    for i, start in enumerate(starting_points):
        if time.time() > deadline - 8:
            break
            
        # Direct optimization with true objective
        try:
            res = minimize(
                neg_score, 
                start.flatten(), 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            cur_score = -res.fun
            if cur_score > best_score:
                best_score = cur_score
                best_points = res.x.reshape(N, 2)
                
        except Exception:
            pass
        
        # Smoothed then polished
        if time.time() > deadline - 15:
            break
        try:
            res_smooth = minimize(
                neg_score_smooth, 
                start.flatten(), 
                args=(3,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            res_final = minimize(
                neg_score, 
                res_smooth.x, 
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
            )
            
            cur_score = -res_final.fun
            if cur_score > best_score:
                best_score = cur_score
                best_points = res_final.x.reshape(N, 2)
                
        except Exception:
            pass

    # Strategy 3: If we have time, do a second round of DE with tighter bounds
    if best_points is not None and time.time() < deadline - 30:
        try:
            # Tighter bounds around best solution
            center = best_points.mean(axis=0)
            spread = np.abs(best_points - center).max()
            b = max(spread * 1.5, 0.5)
            bounds_tight = [(-b, b)] * (N * 2)
            
            de2 = differential_evolution(
                neg_score_smooth,
                bounds=bounds_tight,
                args=(3,),
                seed=123,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                mutation=(0.2, 0.8),
                recombination=0.7,
                polish=False,
                init='latinhypercube'
            )
            
            if de2.success and de2.fun < 0:
                de2_points = de2.x.reshape(N, 2)
                # Polish
                res_p = minimize(
                    neg_score,
                    de2_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40}
                )
                s = -res_p.fun
                if s > best_score:
                    best_score = s
                    best_points = res_p.x.reshape(N, 2)
        except Exception:
            pass
    
    # Strategy 4: Final high-precision polish of the best solution
    if best_points is not None and time.time() < deadline - 5:
        try:
            res_hip = minimize(
                neg_score,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            s = -res_hip.fun
            if s > best_score:
                best_score = s
                best_points = res_hip.x.reshape(N, 2)
        except Exception:
            pass

    if best_points is None or best_score <= 0:
        best_points = init_13
    else:
        final_score = score(best_points)
        if final_score <= 0:
            best_points = init_13

    return best_points
