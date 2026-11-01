import numpy as np
import scipy.spatial as spat
import scipy.optimize as opt
import time
import itertools

N = 14
DEADLINE = 145  # safety margin before 150s kill

def _hull_area(points):
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _calc_min_tri_area(points):
    """
    Compute minimum triangle area.
    """
    n = len(points)
    min_area = np.inf
    # Vectorized calculation for speed
    # n=14, so 14 choose 3 = 364 triangles.
    # We can just loop or vectorize. Vectorizing is safer for consistency.
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _score(points):
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return 0.0
    min_tri = _calc_min_tri_area(points)
    return min_tri / hull_a

def _generate_ring_config(params):
    """
    params: array of shape (2 * n_rings,)
    First half: radii
    Second half: angular offsets
    We assume a fixed structure: 7 points in inner ring, 7 in outer? 
    Or 6/8? 
    The regular polygon is 14 points.
    A common good configuration for TSP/Packing is concentric rings.
    Let's try 2 rings: 6 and 8? Or 7 and 7?
    Let's try 2 rings with variable counts? No, fixed counts for optimization simplicity.
    Let's try 2 rings: 7 and 7.
    Or maybe 3 rings: 4, 6, 4?
    Let's stick to 2 rings for simplicity first, but allow rotation.
    Actually, let's allow the number of points per ring to be fixed.
    Config: Ring 0 has k0 points, Ring 1 has k1 points, etc.
    Let's try 2 rings: 7 and 7.
    """
    n_rings = 2
    counts = [7, 7]
    radii = params[:n_rings]
    offsets = params[n_rings:]
    
    points = []
    for i, count in enumerate(counts):
        r = radii[i]
        theta0 = offsets[i]
        angles = theta0 + np.linspace(0, 2 * np.pi, count, endpoint=False)
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        points.append(np.column_stack([xs, ys]))
        
    return np.vstack(points)

def _objective_ring(params):
    """
    Objective for L-BFGS-B. We want to maximize score, so minimize -score.
    To make it smooth, we use a soft-min approximation for the triangle areas
    and a smooth approximation for the hull area?
    Hull area is piecewise linear (convex function), so it's non-smooth at boundaries.
    However, for a fixed combinatorial structure of the hull, it is smooth.
    Assuming the hull structure doesn't change during optimization (which is likely for small perturbations),
    we can just use the standard hull area.
    
    The min triangle area is the bottleneck.
    Let's use a soft-min on the triangle areas.
    f(x) = - (soft_min(areas) / hull_area)
    
    soft_min(areas) = -1/beta * log(sum(exp(-b_i * areas_i)))
    where b_i = 1.
    """
    points = _generate_ring_config(params)
    
    # Check for degeneracy
    hull_a = _hull_area(points)
    if hull_a < 1e-6:
        return 1e6
    
    min_tri = _calc_min_tri_area(points)
    
    # For the gradient-based optimizer, we need a differentiable objective.
    # _calc_min_tri_area is non-smooth.
    # Let's implement a soft-min version inside the objective for the optimizer.
    # But _generate_ring_config and _hull_area are called every time.
    
    # To get gradients, we would need to implement the soft-min and its gradient.
    # Since we are using L-BFGS-B, we can provide the gradient or let it approximate.
    # Approximating gradient on a non-smooth function is risky.
    # Let's implement a differentiable soft-min score.
    
    n = len(points)
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min with beta
    beta = 50.0
    # To avoid overflow, shift by min
    min_a = np.min(areas)
    exps = np.exp(-beta * (areas - min_a))
    log_sum = np.log(np.sum(exps))
    soft_min_area = min_a - (1.0 / beta) * log_sum
    
    score = soft_min_area / hull_a
    return -score

def _objective_ring_grad(params):
    """
    Gradient of _objective_ring.
    This is complex to derive analytically due to the abs() and hull area.
    Let's rely on numerical gradient or use a simpler approach.
    Actually, L-BFGS-B with numerical gradient is often sufficient if the function is smooth enough.
    The soft-min makes the min area smooth.
    The hull area is piecewise linear. If we assume the hull vertices don't change, it's smooth.
    """
    # We will let scipy compute the gradient numerically.
    return None

def _optimize_ring(params_init, time_limit=10):
    """
    Optimize ring parameters using L-BFGS-B.
    """
    start_time = time.time()
    
    # Bounds for radii (positive) and offsets (0 to 2pi)
    n_rings = 2
    bounds = [(1e-6, 2.0)] * n_rings + [(0, 2 * np.pi)] * n_rings
    
    # L-BFGS-B
    res = opt.minimize(
        _objective_ring, 
        params_init, 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': 500, 'maxfun': 5000}
    )
    
    if time.time() - start_time > time_limit:
        # This is tricky, minimize doesn't support time limits easily.
        # We'll just check after.
        pass
        
    return res.x

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    best_score = -np.inf
    best_points = None
    
    # Strategy:
    # 1. Try different ring configurations (2 rings, 3 rings)
    # 2. For each, do a coarse grid search or random restarts
    # 3. Polish with L-BFGS-B
    
    # Configuration 1: 2 rings (7, 7)
    # Initial guess: Regular polygon is one ring.
    # Let's try r1=0.5, r2=1.0
    configs = []
    
    # 2 rings: 7, 7
    # params: [r1, r2, theta1, theta2]
    for r1 in [0.3, 0.5, 0.7]:
        for r2 in [0.9, 1.0, 1.2]:
            for t1 in [0, np.pi/7]:
                for t2 in [0, np.pi/7]:
                    params = np.array([r1, r2, t1, t2])
                    configs.append(params)
                    
    # 3 rings: 4, 6, 4
    # params: [r1, r2, r3, t1, t2, t3]
    def _generate_ring_config_3(params):
        n_rings = 3
        counts = [4, 6, 4]
        radii = params[:n_rings]
        offsets = params[n_rings:]
        
        points = []
        for i, count in enumerate(counts):
            r = radii[i]
            theta0 = offsets[i]
            angles = theta0 + np.linspace(0, 2 * np.pi, count, endpoint=False)
            xs = r * np.cos(angles)
            ys = r * np.sin(angles)
            points.append(np.column_stack([xs, ys]))
            
        return np.vstack(points)
        
    def _objective_ring_3(params):
        points = _generate_ring_config_3(params)
        hull_a = _hull_area(points)
        if hull_a < 1e-6:
            return 1e6
        
        n = len(points)
        i_idx, j_idx, k_idx = np.indices((n, n, n))
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        k_idx = k_idx[mask]
        
        p_i = points[i_idx]
        p_j = points[j_idx]
        p_k = points[k_idx]
        
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        beta = 50.0
        min_a = np.min(areas)
        exps = np.exp(-beta * (areas - min_a))
        log_sum = np.log(np.sum(exps))
        soft_min_area = min_a - (1.0 / beta) * log_sum
        
        score = soft_min_area / hull_a
        return -score

    for r1 in [0.3, 0.5]:
        for r2 in [0.7, 0.9]:
            for r3 in [1.0, 1.2]:
                params = np.array([r1, r2, r3, 0, 0, 0])
                # We need a separate optimizer for 3 rings
                # Let's just add to a generic list of (params, generator, objective)
                pass
                
    # Let's stick to 2 rings for the main optimization loop to keep it simple and fast,
    # and add a few random 28-dim searches if time permits.
    
    # Optimize 2-ring configs
    for params_init in configs:
        if time.time() > DEADLINE - 20:
            break
            
        try:
            res = opt.minimize(
                _objective_ring, 
                params_init, 
                method='L-BFGS-B', 
                bounds=[(1e-6, 2.0)] * 2 + [(0, 2 * np.pi)] * 2,
                options={'maxiter': 200, 'maxfun': 2000}
            )
            
            opt_params = res.x
            points = _generate_ring_config(opt_params)
            
            # Normalize to unit hull area for final score?
            # The score function handles normalization implicitly via ratio.
            # But the problem says "normalized to unit area".
            # The score is min_tri / hull_area.
            # So we just compute the score.
            
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
                
        except Exception as e:
            continue
            
    # If we have time, try a few random 28-dim starts with a quick polish
    if time.time() < DEADLINE - 30:
        for _ in range(5):
            if time.time() > DEADLINE - 10:
                break
            # Random points in unit circle
            r = np.sqrt(rng.random(N))
            theta = rng.uniform(0, 2 * np.pi, N)
            pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            
            # Quick coordinate descent or Nelder-Mead?
            # Nelder-Mead is slow in 28D.
            # Let's just evaluate.
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()

    if best_points is None:
        # Fallback to regular polygon
        best_points = _regular_polygon(N)
        best_score = _score(best_points)
        
    return np.array(best_points)
