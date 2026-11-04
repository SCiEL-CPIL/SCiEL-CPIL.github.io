import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def _compute_score(points):
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
    # We can vectorize this for speed
    # Indices for all triples
    min_area = float('inf')
    
    # Vectorized approach
    # i, j, k where i < j < k
    # Number of triples is C(13, 3) = 286.
    # Vectorizing might be overkill and memory heavy if not careful, but 286 is small.
    # Let's stick to a loop but optimized with numpy ops if possible, or just pure python if fast enough.
    # 286 iterations is very fast.
    
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

def _decode_params(params):
    """
    Decode 26 parameters (13 radii, 13 angles) to 13 points.
    Angles are sorted to ensure convex order.
    Radii are constrained to be positive.
    """
    radii = params[:N]
    angles = params[N:]
    
    # Ensure radii are positive (softplus or just exp if we want strict positivity, 
    # but simple clipping or abs might be better for L-BFGS-B bounds)
    # Let's use exp for radii to ensure they are > 0 and smooth.
    # However, exp can blow up. Let's use a scaled version.
    # Or simply use bounds in L-BFGS-B and pass raw values.
    # The prompt says "small, independent radial perturbation".
    # Let's assume params are raw radii and raw angles.
    # We will sort angles.
    
    # Sort angles to maintain convex order
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]
    sorted_radii = radii[sorted_indices]
    
    # Convert to Cartesian
    x = sorted_radii * np.cos(sorted_angles)
    y = sorted_radii * np.sin(sorted_angles)
    
    points = np.column_stack((x, y))
    return points

def _soft_min(values, beta):
    """
    Smooth approximation of the minimum using log-sum-exp.
    min(v) ≈ -1/beta * log(sum(exp(-beta * v)))
    """
    # For numerical stability, shift by max
    # Since we are minimizing -log-sum-exp of areas, we want to maximize the min area.
    # Let's define the objective as minimizing -soft_min(areas).
    # soft_min(areas) = -1/beta * log(sum(exp(-beta * areas)))
    # This is a lower bound on the true min, and converges to it as beta -> inf.
    # To maximize soft_min, we minimize -soft_min.
    
    # Use log-sum-exp trick for stability
    max_val = np.max(values)
    # If beta is large, exp(-beta * values) can underflow/overflow.
    # log(sum(exp(-beta * v))) = -beta * max_val + log(sum(exp(-beta * (v - max_val))))
    # Let u = -beta * (v - max_val). Since v <= max_val, u >= 0.
    # Actually, standard logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    # Here x = -beta * v.
    # max(x) = -beta * min(v).
    # logsumexp(-beta * v) = -beta * min(v) + log(sum(exp(-beta * v + beta * min(v))))
    # Let m = min(v).
    # sum(exp(-beta * (v - m)))
    
    m = np.min(values)
    exp_terms = np.exp(-beta * (values - m))
    log_sum = np.log(np.sum(exp_terms))
    logsumexp_val = -beta * m + log_sum
    
    # soft_min = -1/beta * logsumexp_val
    soft_min_val = -1.0 / beta * logsumexp_val
    
    return soft_min_val

def _objective(params, beta):
    """
    Objective function for L-BFGS-B.
    Minimizes - (soft_min(areas) / hull_area)
    """
    points = _decode_params(params)
    
    # Check for degeneracy
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-6:
            return 1e6 # Penalty for non-convex or degenerate
    except Exception:
        return 1e6
        
    # Calculate areas
    n = len(points)
    areas = []
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
                
    areas = np.array(areas)
    
    # If any area is 0, return penalty
    if np.min(areas) < 1e-6:
        return 1e6
        
    soft_min_area = _soft_min(areas, beta)
    score = soft_min_area / hull_area
    
    return -score

def _construct_initial_params(method='regular'):
    """
    Generate initial parameters (radii, angles).
    """
    if method == 'regular':
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.ones(N) * 0.5
        # Add small noise
        angles += np.random.normal(0, 0.01, N)
        radii += np.random.normal(0, 0.01, N)
    elif method == 'halton':
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        radii = []
        angles = []
        for i in range(1, N + 1):
            r = halton(i, 2) * 0.5 + 0.25 # Map to [0.25, 0.75]
            a = halton(i, 3) * 2 * np.pi
            radii.append(r)
            angles.append(a)
        radii = np.array(radii)
        angles = np.array(angles)
    else:
        radii = np.random.uniform(0.2, 0.8, N)
        angles = np.random.uniform(0, 2*np.pi, N)
        
    return np.concatenate([radii, angles])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_score = 0.0
    best_points = None
    
    strategies = ['regular', 'halton', 'regular', 'halton', 'random']
    np.random.shuffle(strategies)
    
    for strat in strategies:
        if time.time() > deadline - 10.0:
            break
            
        init_params = _construct_initial_params(strat)
        
        # Bounds for L-BFGS-B
        # Radii: [0.01, 2.0]
        # Angles: [-10, 10] (wide range to allow wrapping, though sorting handles order)
        # Actually, angles should be in a range that covers the circle.
        # Since we sort them, the absolute value doesn't matter as much, 
        # but we need them to be distinct enough.
        # Let's use bounds that encourage spread.
        bounds = []
        for i in range(N):
            bounds.append((0.01, 2.0)) # Radii
        for i in range(N):
            bounds.append((-5.0, 5.0)) # Angles
            
        # Multi-phase beta annealing
        betas = [10.0, 50.0, 200.0, 1000.0]
        
        current_params = init_params.copy()
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            # Optimize with current beta
            res = minimize(
                _objective, 
                current_params, 
                args=(beta,), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res.success or res.fun < _objective(current_params, beta):
                current_params = res.x
                
        # Evaluate final score with hard min
        final_points = _decode_params(current_params)
        final_score = _compute_score(final_points)
        
        if final_score > best_score:
            best_score = final_score
            best_points = final_points.copy()
            
    if best_points is None:
        # Fallback
        best_points = _decode_params(_construct_initial_params('regular'))
        
    return best_points
