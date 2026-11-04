import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import random

N = 13
# Safety margin: evaluator kills at 150s. We target 135s to be safe.
DEADLINE = 135.0

def compute_score(points):
    """
    Compute the raw score: min triangle area / convex hull area.
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
    
    # Vectorized approach for speed
    # i, j, k where i < j < k
    # C(13, 3) = 286 triples.
    # We can compute all areas at once using broadcasting or loops.
    # 286 is small, so a simple loop is fast enough and robust.
    
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0
    
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def smooth_objective(params):
    """
    Objective function to minimize.
    We want to maximize min_area / hull_area.
    Equivalent to minimizing -log(min_area / hull_area) = log(hull_area) - log(min_area).
    
    To make it differentiable, we approximate log(min_area) using log-sum-exp of negative areas?
    Actually, min(A) = -logsumexp(-A / T) * T as T->0.
    So log(min(A)) ≈ log(-logsumexp(-A/T) * T) = -log(logsumexp(-A/T)) + log(T).
    
    Let's define the surrogate:
    Score = min_area / hull_area
    We want to maximize Score.
    Minimize -log(Score) = -log(min_area) + log(hull_area).
    
    Using soft-min for min_area:
    S_min = -T * log( sum( exp(-A_i / T) ) )
    log(S_min) = -log(T) - log( sum( exp(-A_i / T) ) )
    
    So -log(S_min) = log(T) + log( sum( exp(-A_i / T) ) )
    
    Total Objective to minimize:
    J = log(T) + log( sum( exp(-A_i / T) ) ) + log(hull_area)
    
    Wait, hull_area is also variable.
    J(params) = log(T) + log( sum_{i<j<k} exp(-Area(i,j,k) / T) ) + log(HullArea(params))
    
    As T -> 0, sum(exp(-A_i/T)) -> exp(-min(A_i)/T).
    log(sum(...)) -> -min(A_i)/T.
    J -> log(T) - min(A_i)/T + log(HullArea).
    This goes to -infinity if min(A_i) > 0.
    Minimizing J maximizes min(A_i)/T roughly?
    
    Let's stick to a simpler, robust surrogate that is smooth:
    Minimize: 1/min_area - 1/hull_area? No.
    
    Let's use: Minimize (sum(1/Area_i^p))^(1/p) / HullArea?
    Or simpler: Minimize -log(soft_min_area) + log(hull_area).
    
    Let's implement:
    soft_min_area = -T * log( sum( exp(-Area / T) ) )
    obj = -log(soft_min_area) + log(hull_area)
    
    Gradient will be computed by finite differences or analytically?
    L-BFGS-B can use finite differences, but it's slow.
    Let's try to compute gradients analytically if possible, or rely on scipy's numerical grad with a good step size.
    Given 26 parameters, numerical grad is 27 evaluations per step.
    If we do 1000 steps, that's 27000 evaluations.
    Each eval is ~286 area calcs + 1 hull calc.
    Hull calc is O(N log N) ~ 13*3. Fast.
    Area calc is 286 ops. Fast.
    27000 * (small overhead) should be well within 135s.
    """
    T = 0.01 # Temperature for soft-min
    
    points = params.reshape(N, 2)
    
    # Calculate Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 1e10 # Large penalty
    except Exception:
        return 1e10
        
    # Calculate all triangle areas
    areas = []
    n = len(points)
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < 1e-10:
                    return 1e10 # Penalty for degenerate triangles
                areas.append(area)
                
    areas = np.array(areas)
    
    # Soft min
    # To avoid overflow in exp, shift by min area
    min_a = np.min(areas)
    exp_sum = np.sum(np.exp(-(areas - min_a) / T))
    soft_min_area = min_a - T * np.log(exp_sum)
    
    if soft_min_area < 1e-10:
        return 1e10
        
    # Objective: minimize -log(soft_min_area) + log(hull_area)
    # This is equivalent to maximizing soft_min_area / hull_area
    obj = -np.log(soft_min_area) + np.log(hull_area)
    
    return obj

def construct():
    start_time = time.time()
    
    # Initial guess: Regular polygon
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Add a small jitter to break symmetry and help optimization
    # But regular polygon is a good local minimum.
    # Maybe use a slightly distorted polygon.
    
    best_score = 0.0
    best_points = None
    
    # Try a few different initializations
    # 1. Regular polygon
    # 2. Halton sequence (normalized to fit in unit area roughly)
    # 3. Random points
    
    inits = []
    
    # Regular polygon
    pts_reg = np.column_stack([np.cos(theta), np.sin(theta)])
    # Normalize to unit area?
    # Area of regular N-gon with radius R is 0.5 * N * R^2 * sin(2pi/N)
    # We want area = 1.
    # R^2 = 2 / (N * sin(2pi/N))
    R = np.sqrt(2.0 / (N * np.sin(2.0 * np.pi / N)))
    pts_reg *= R
    inits.append(pts_reg)
    
    # Halton
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
        
    pts_hal = np.array([[halton(i, 2), halton(i, 3)] for i in range(1, N+1)])
    # Center and scale to unit area
    pts_hal -= np.mean(pts_hal, axis=0)
    try:
        hull = ConvexHull(pts_hal)
        area = hull.volume
        scale = np.sqrt(1.0 / area)
        pts_hal *= scale
    except:
        pts_hal = pts_reg # Fallback
    inits.append(pts_hal)
    
    # Random
    np.random.seed(42)
    pts_rand = np.random.rand(N, 2)
    pts_rand -= np.mean(pts_rand, axis=0)
    try:
        hull = ConvexHull(pts_rand)
        area = hull.volume
        scale = np.sqrt(1.0 / area)
        pts_rand *= scale
    except:
        pts_rand = pts_reg
    inits.append(pts_rand)
    
    # Optimization Loop
    # We have ~135s.
    # Let's allocate time per restart.
    # 3 restarts * 40s each?
    
    for i, init_pts in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        x0 = init_pts.flatten()
        
        # L-BFGS-B
        # We don't have bounds strictly, but we can set loose bounds to prevent explosion
        # Or no bounds.
        
        try:
            # Use finite difference gradients
            # maxiter: limit iterations to ensure time bound
            # maxfun: limit function evaluations
            # Since each eval is fast, we can allow many.
            # But we need to be careful with time.
            
            # Let's use a callback to check time
            time_check = {'stop': False}
            
            def callback(xk):
                if time.time() - start_time > DEADLINE - 5:
                    time_check['stop'] = True
                    return True
                return False
            
            res = minimize(
                smooth_objective, 
                x0, 
                method='L-BFGS-B', 
                options={
                    'maxiter': 1000,
                    'maxfun': 5000,
                    'ftol': 1e-10,
                    'gtol': 1e-6
                },
                callback=callback
            )
            
            if res.success or time_check['stop']:
                opt_pts = res.x.reshape(N, 2)
                # Verify score with exact metric
                score = compute_score(opt_pts)
                if score > best_score:
                    best_score = score
                    best_points = opt_pts.copy()
                    
        except Exception as e:
            pass # Ignore optimization errors for this restart

    # If no points found (should not happen), return regular polygon
    if best_points is None:
        best_points = inits[0]
        
    return best_points

# EVOLVE-BLOCK-END
