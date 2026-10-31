import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 13

def _polygon_area(pts):
    """
    Computes the area of the polygon formed by the points in the given order.
    Assumes points are ordered (e.g., by angle).
    """
    x = pts[:, 0]
    y = pts[:, 1]
    # Shoelace formula
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _min_triangle_area_soft(pts, beta):
    """
    Computes the soft-min of triangle areas.
    """
    n = len(pts)
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                xi, yi = pts[i]
                xj, yj = pts[j]
                xk, yk = pts[k]
                area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
                areas.append(area)
    areas = np.array(areas)
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # For numerical stability, subtract max area before exp? 
    # min(A) = -1/beta * log(sum(exp(-beta*A)))
    # Let m = max(A). sum(exp(-beta*A)) = exp(-beta*m) * sum(exp(-beta*(A-m)))
    # log(sum(...)) = -beta*m + log(sum(exp(-beta*(A-m))))
    # So soft_min = m - 1/beta * log(sum(exp(-beta*(A-m))))
    m = np.max(areas)
    # If beta is large, exp(-beta*(A-m)) underflows for A >> m, which is fine.
    # For A close to m, it's ~1.
    # Actually, we want the MIN.
    # The formula -1/beta * log(sum(exp(-beta*A))) approximates min(A).
    # Let's check: if A1 << A2, exp(-beta*A1) >> exp(-beta*A2). Sum ~ exp(-beta*A1). Log ~ -beta*A1. Result ~ A1. Correct.
    
    # To avoid overflow/underflow, use the max shift trick on the negative exponent?
    # We are summing exp(-beta * A). The largest term corresponds to the smallest A.
    # Let A_min = min(areas).
    # sum(exp(-beta*A)) = exp(-beta*A_min) * sum(exp(-beta*(A - A_min)))
    # log(sum) = -beta*A_min + log(sum(exp(-beta*(A - A_min))))
    # soft_min = A_min - (1/beta) * log(sum(exp(-beta*(A - A_min))))
    
    a_min = np.min(areas)
    shifted = -beta * (areas - a_min)
    # shifted <= 0. max is 0.
    log_sum = np.log(np.sum(np.exp(shifted)))
    return a_min - log_sum / beta

def _score_soft(pts, beta):
    """
    Score using soft-min area and polygon area (assuming convex order).
    """
    # Sort points by angle around centroid to maintain consistent polygon order
    centroid = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)
    pts_sorted = pts[order]
    
    hull_area = _polygon_area(pts_sorted)
    if hull_area < 1e-12:
        return 0.0
    
    min_area = _min_triangle_area_soft(pts_sorted, beta)
    return min_area / hull_area

def _neg_score_soft(params, beta):
    pts = params.reshape(N, 2)
    return -_score_soft(pts, beta)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: regular 13-gon on unit circle
    angles = 2 * np.pi * np.arange(N) / N
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    best_points = x0.reshape(N, 2).copy()
    # Use a large beta for initial scoring to get a stable value
    best_score = _score_soft(best_points, beta=10.0)
    
    # 1. Global Search with Differential Evolution
    # DE uses the raw score (hard min) or soft min? 
    # DE is derivative-free. We can use the hard min for DE to be precise, 
    # but it's slow. Let's use a moderate beta for DE to be faster.
    beta_de = 10.0
    
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    try:
        result_de = differential_evolution(
            lambda x: _neg_score_soft(x, beta_de),
            bounds,
            seed=42,
            maxiter=30,
            tol=1e-4,
            polish=False,
            init='sobol',
            popsize=15
        )
        
        if result_de.fun < -best_score:
            candidate = result_de.x.reshape(N, 2)
            s = _score_soft(candidate, beta=10.0)
            if s > best_score:
                best_score = s
                best_points = candidate.copy()
                
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_points
    
    # 2. Local Refinement with L-BFGS-B and Annealing Beta
    # We will run L-BFGS-B with decreasing beta (increasing sharpness)
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    current_points = best_points.flatten()
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
            
        try:
            result_lm = minimize(
                lambda x: _neg_score_soft(x, beta),
                current_points,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result_lm.fun < -best_score:
                candidate = result_lm.x.reshape(N, 2)
                # Evaluate with a very high beta (approx hard min) to check true score
                s_true = _score_soft(candidate, beta=10000.0)
                if s_true > best_score:
                    best_score = s_true
                    best_points = candidate.copy()
                    current_points = candidate.flatten()
            else:
                # Even if fun didn't improve in soft sense, maybe the points are better for hard min?
                # Just update current points to the result to continue chain
                current_points = result_lm.x
                
        except Exception:
            pass
            
    # Final check with hard min
    final_score = _score_soft(best_points, beta=10000.0)
    if final_score > best_score:
        best_score = final_score
        
    return best_points
