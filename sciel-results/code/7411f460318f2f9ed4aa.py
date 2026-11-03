import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START

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
    min_area = float('inf')
    
    # Vectorized approach for speed
    # We compute areas for all triples (i, j, k) with i < j < k
    # Use broadcasting to compute cross products for all pairs at once? 
    # Actually, 286 triples is small, but let's do it efficiently.
    
    # Pre-compute all pairwise cross products? No, let's just loop but it's very fast.
    # To be safe and fast, let's use a numpy-based approach for the inner loop.
    
    areas = np.zeros((286,), dtype=np.float64)
    idx = 0
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas[idx] = area
                idx += 1
    
    min_area = np.min(areas)
    
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def _objective_soft(points, beta):
    """
    Soft minimum of triangle areas divided by hull area.
    We want to maximize the min area, so we minimize the negative.
    f(x) = - (softmin(areas) / hull_area)
    softmin(a) = (1/beta) * log(sum(exp(-beta * a_i)))
    """
    n = len(points)
    
    # Hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 1e6
    except Exception:
        return 1e6
    
    # Triangle areas
    areas = np.zeros((286,), dtype=np.float64)
    idx = 0
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas[idx] = area
                idx += 1
    
    # Soft min
    # To avoid overflow, subtract max (which is min in -beta*a)
    # We want softmin(areas).
    # logsumexp(-beta * areas) = log(sum(exp(-beta * areas)))
    # Let M = min(areas). Then -beta * M is the max of -beta * areas.
    # softmin = (1/beta) * log(sum(exp(-beta * (areas - M)))) - M
    M = np.min(areas)
    if M < 1e-10:
        return 1e6
        
    exp_args = -beta * (areas - M)
    # Clip for stability
    exp_args = np.clip(exp_args, -500, 0)
    
    log_sum = np.log(np.sum(np.exp(exp_args)))
    soft_min_val = M + (1.0 / beta) * log_sum
    
    return -(soft_min_val / hull_area)

def _objective_hard(points):
    """
    Hard minimum of triangle areas divided by hull area.
    Used for final evaluation or when beta is effectively infinite.
    """
    return -_compute_score(points)

def _generate_halton(n):
    """
    Generate n points using Halton sequences for low-discrepancy initialization.
    """
    def halton(i, base):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        points.append([x, y])
    return np.array(points)

def _generate_polygon(n):
    """
    Generate n points on a regular polygon.
    """
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 0.5
    center = [0.5, 0.5]
    points = np.array([[center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)] for t in theta])
    return points

def _optimize_with_annealing(points, time_limit, start_temp=0.05, cool_rate=0.95, beta_start=10.0, beta_end=500.0):
    """
    Simulated Annealing with increasing beta (soft-min sharpness).
    """
    current = points.copy().flatten()
    best = current.copy()
    
    start_time = time.time()
    n = N
    dim = n * 2
    
    # Pre-compute indices for faster area calculation in the loop if needed?
    # The objective function is the bottleneck.
    # Let's inline the objective calculation for speed in the inner loop.
    
    def _fast_objective(x, beta):
        pts = x.reshape(-1, 2)
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 1e6
        except Exception:
            return 1e6
            
        areas = np.zeros((286,), dtype=np.float64)
        idx = 0
        for i in range(n - 2):
            p1 = pts[i]
            for j in range(i + 1, n - 1):
                p2 = pts[j]
                v1 = p2 - p1
                for k in range(j + 1, n):
                    p3 = pts[k]
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    areas[idx] = area
                    idx += 1
        
        M = np.min(areas)
        if M < 1e-10:
            return 1e6
            
        exp_args = -beta * (areas - M)
        exp_args = np.clip(exp_args, -500, 0)
        log_sum = np.log(np.sum(np.exp(exp_args)))
        soft_min_val = M + (1.0 / beta) * log_sum
        
        return -(soft_min_val / hull_area)

    current_obj = _fast_objective(current, beta_start)
    best_obj = current_obj
    best = current.copy()
    
    temp = start_temp
    iters = 0
    total_iters_target = 10000 # Just a counter for beta scheduling if time is not the only factor
    
    while time.time() - start_time < time_limit:
        iters += 1
        
        # Update beta
        progress = min(1.0, iters / total_iters_target)
        beta = beta_start * (beta_end / beta_start) ** progress
        
        # Perturbation
        # Move 1 or 2 points
        n_moves = 1 if np.random.random() < 0.7 else 2
        indices = np.random.choice(dim, size=n_moves, replace=False)
        
        new_x = current.copy()
        step_size = 0.05 * (0.5 + 0.5 * (1 - progress)) # Shrink step size
        for idx in indices:
            new_x[idx] += np.random.normal(0, step_size)
            
        # Keep points within reasonable bounds? 
        # The problem is scale invariant, so we don't strictly need bounds,
        # but keeping them in [0,1] helps numerical stability of Hull.
        # Actually, ConvexHull doesn't care about scale, just geometry.
        # Let's allow free movement but maybe penalize if they get too far?
        # No, let's just let them move. The objective is scale invariant.
        
        new_obj = _fast_objective(new_x, beta)
        
        delta = new_obj - current_obj
        if delta < 0 or np.random.random() < np.exp(-delta / temp):
            current = new_x
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_obj = current_obj
                best = current.copy()
        
        temp *= cool_rate
        if temp < 1e-4:
            temp = 1e-4
            
    return best.reshape(-1, 2), _compute_score(best.reshape(-1, 2))

def _optimize_with_lbfgs(points, time_limit, beta=500.0):
    """
    L-BFGS-B optimization on the soft-min objective.
    """
    x0 = points.flatten()
    
    def obj(x):
        pts = x.reshape(-1, 2)
        return _objective_soft(pts, beta)
        
    def jac(x):
        # Numerical gradient
        h = 1e-5
        grad = np.zeros_like(x)
        f0 = obj(x)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += h
            fp = obj(xp)
            grad[i] = (fp - f0) / h
        return grad
        
    start_time = time.time()
    
    # We can't easily pass a time limit to L-BFGS-B directly.
    # We'll just run it and hope it finishes, or check time in callback.
    # scipy minimize doesn't have a great time callback that stops cleanly without error.
    # Let's just run it for a limited number of iterations.
    
    try:
        res = minimize(
            obj, 
            x0, 
            jac=jac,
            method='L-BFGS-B', 
            options={'maxiter': 50, 'gtol': 1e-4}
        )
        return res.x.reshape(-1, 2), _compute_score(res.x.reshape(-1, 2))
    except Exception:
        return points, _compute_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_global_points = None
    best_global_score = 0.0
    
    # Strategy 1: Halton + Annealing
    if time.time() < deadline - 20.0:
        init_pts = _generate_halton(N)
        pts, score = _optimize_with_annealing(init_pts, time_limit=min(20.0, deadline - time.time() - 10.0), beta_start=10.0, beta_end=500.0)
        if score > best_global_score:
            best_global_score = score
            best_global_points = pts.copy()
        
        # Refine with L-BFGS
        if time.time() < deadline - 10.0:
            pts, score = _optimize_with_lbfgs(pts, time_limit=min(10.0, deadline - time.time() - 5.0), beta=500.0)
            if score > best_global_score:
                best_global_score = score
                best_global_points = pts.copy()
                
    # Strategy 2: Polygon + Annealing (different start)
    if time.time() < deadline - 20.0:
        init_pts = _generate_polygon(N)
        # Perturb slightly to break symmetry
        init_pts += np.random.normal(0, 0.01, init_pts.shape)
        pts, score = _optimize_with_annealing(init_pts, time_limit=min(20.0, deadline - time.time() - 10.0), beta_start=10.0, beta_end=500.0)
        if score > best_global_score:
            best_global_score = score
            best_global_points = pts.copy()
            
        # Refine with L-BFGS
        if time.time() < deadline - 10.0:
            pts, score = _optimize_with_lbfgs(pts, time_limit=min(10.0, deadline - time.time() - 5.0), beta=500.0)
            if score > best_global_score:
                best_global_score = score
                best_global_points = pts.copy()

    # Strategy 3: Random + Annealing
    if time.time() < deadline - 20.0:
        init_pts = np.random.rand(N, 2)
        pts, score = _optimize_with_annealing(init_pts, time_limit=min(20.0, deadline - time.time() - 10.0), beta_start=10.0, beta_end=500.0)
        if score > best_global_score:
            best_global_score = score
            best_global_points = pts.copy()
            
        # Refine with L-BFGS
        if time.time() < deadline - 10.0:
            pts, score = _optimize_with_lbfgs(pts, time_limit=min(10.0, deadline - time.time() - 5.0), beta=500.0)
            if score > best_global_score:
                best_global_score = score
                best_global_points = pts.copy()

    # Final Polish: Use L-BFGS with higher beta on the best result
    if best_global_points is not None and time.time() < deadline - 5.0:
        pts, score = _optimize_with_lbfgs(best_global_points, time_limit=min(5.0, deadline - time.time() - 2.0), beta=1000.0)
        if score > best_global_score:
            best_global_score = score
            best_global_points = pts.copy()

    if best_global_points is None:
        # Fallback
        best_global_points = _generate_halton(N)
        
    return best_global_points

# EVOLVE-BLOCK-END
