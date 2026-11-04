import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

N = 13

# EVOLVE-BLOCK-START

def compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Calculate all triangle areas
    min_area = float('inf')
    
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                # Area is 0.5 * |cross product|
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0
    
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def decode_params(params):
    """
    Decode parameter vector into 13 (x, y) points.
    params[0:13] are radial offsets from a base circle.
    Angles are fixed at linspace(0, 2*pi, 13, endpoint=False).
    """
    radii = params[:N]
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    return np.column_stack((x, y))

def objective(params):
    """
    Objective function for minimization (negative score).
    """
    try:
        points = decode_params(params)
        score = compute_score(points)
        return -score
    except Exception:
        return 1e6

def local_search(points, max_iters=2000, time_limit=10.0, start_temp=0.05, cool_rate=0.98):
    """
    Simulated Annealing to maximize the score.
    """
    current_points = points.copy()
    current_score = compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    temp = start_temp
    start_time = time.time()
    no_improve_count = 0
    
    import random
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Perturbation: move 1 point
        idx = random.randint(0, N - 1)
        
        new_points = current_points.copy()
        # Gaussian noise
        noise = np.random.normal(0, 0.02, 2)
        new_points[idx] = current_points[idx] + noise
        
        new_score = compute_score(new_points)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / temp):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        # Cool down
        temp *= cool_rate
        
        # If stuck for a long time, kick
        if no_improve_count > 50:
             kick_idx = random.randint(0, N - 1)
             new_points = current_points.copy()
             new_points[kick_idx] = current_points[kick_idx] + np.random.normal(0, 0.1, 2)
             new_score = compute_score(new_points)
             if new_score > current_score:
                 current_points = new_points
                 current_score = new_score
                 if new_score > best_score:
                     best_score = new_score
                     best_points = current_points.copy()
                 no_improve_count = 0
                 temp = 0.05 # Reheat
             else:
                 temp *= 0.9

    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    best_global_points = None
    best_global_score = 0.0
    
    # Strategy 1: Differential Evolution on radial parameters
    # Initial guess: regular polygon (radius 1)
    # Bounds: radii between 0.5 and 1.5 to allow some flexibility but keep shape sane
    # Actually, let's allow wider range: 0.2 to 2.0
    bounds = [(0.2, 2.0) for _ in range(N)]
    
    try:
        # Use differential evolution
        # popsize=10, maxiter=100, tol=1e-6
        # We need to be careful with time. DE can be slow.
        # Let's limit it.
        de_start = time.time()
        de_time_limit = 60.0
        
        # Custom callback to check time? No, DE doesn't support easy time-based stop in all versions easily without wrapper.
        # We can just run it with limited iterations.
        # 13 variables. popsize 10 -> 130 evals per iter.
        # 50 iters -> 6500 evals.
        # Each eval is ~1ms? 6.5s. Safe.
        
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15, 
            tol=1e-6,
            mutation=(0.5, 1.5),
            recombination=0.9
        )
        
        if time.time() < deadline - 20.0:
            best_params = result.x
            best_points = decode_params(best_params)
            best_score = compute_score(best_points)
            
            if best_score > best_global_score:
                best_global_score = best_score
                best_global_points = best_points.copy()
                
    except Exception as e:
        pass

    # Strategy 2: If DE didn't run or failed, fallback to simple local search from regular polygon
    if best_global_points is None or best_global_score < 1.0:
        # Fallback init
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        radius = 1.0
        init_points = np.array([[radius * np.cos(t), radius * np.sin(t)] for t in theta])
        
        if time.time() < deadline - 10.0:
            pts, score = local_search(init_points, time_limit=min(20.0, deadline - time.time() - 5.0))
            if score > best_global_score:
                best_global_score = score
                best_global_points = pts

    # Strategy 3: Final polishing on the best point found so far
    if best_global_points is not None and time.time() < deadline - 5.0:
        remaining_time = deadline - time.time() - 2.0
        if remaining_time > 1.0:
            final_pts, final_score = local_search(best_global_points, time_limit=remaining_time, start_temp=0.01, cool_rate=0.99)
            if final_score > best_global_score:
                best_global_score = final_score
                best_global_points = final_pts

    if best_global_points is None:
        # Absolute fallback
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        radius = 1.0
        best_global_points = np.array([[radius * np.cos(t), radius * np.sin(t)] for t in theta])
        
    return best_global_points
# EVOLVE-BLOCK-END
