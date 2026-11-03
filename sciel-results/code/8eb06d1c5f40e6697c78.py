import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 50
START_TIME = time.time()
# Leave a safety margin for the evaluator's kill switch
DEADLINE = 140.0 

def to_sphere(params):
    """Convert raw params to unit vectors."""
    v = params.reshape(-1, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return v / norms

def compute_covering_radius_exact(points):
    """Compute the covering radius (in radians) using ConvexHull.
    This is the ground truth for the score.
    """
    points = np.asarray(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # Worst case: all points in a hemisphere or degenerate
    
    max_circum = 0.0
    for simplex in hull.simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        # Compute circumcenter of triangle in 3D
        ab = b - a
        ac = c - a
        abab = np.dot(ab, ab)
        acac = np.dot(ac, ac)
        abac = np.dot(ab, ac)
        
        denom = abab * acac - abac * abac
        if abs(denom) < 1e-12:
            continue
            
        u1 = (acac * abac - abab * abac) / denom
        u2 = (abab * abac - acac * abac) / denom
        circumcenter = a + u1 * ab + u2 * ac
        norm_cc = np.linalg.norm(circumcenter)
        
        if norm_cc < 1e-10:
            continue
            
        pole = circumcenter / norm_cc
        theta_a = np.arccos(np.clip(np.dot(pole, a), -1, 1))
        theta_b = np.arccos(np.clip(np.dot(pole, b), -1, 1))
        theta_c = np.arccos(np.clip(np.dot(pole, c), -1, 1))
        
        circum_angular = max(theta_a, theta_b, theta_c)
        max_circum = max(max_circum, circum_angular)
        
    return max_circum

def smooth_covering_radius(points, beta=100.0):
    """
    Differentiable approximation of the covering radius.
    Uses log-sum-exp to approximate max distance.
    We sample a set of points on the sphere and compute the max min-distance.
    To make it differentiable, we approximate max with log-sum-exp.
    """
    # Generate a grid of test points on the sphere
    # Using a Fibonacci spiral for good coverage
    def fib_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    # Use a moderate number of samples for speed in optimization loops
    # This number might need tuning. 1000 is a good balance.
    if not hasattr(smooth_covering_radius, "test_points"):
        smooth_covering_radius.test_points = fib_sphere(2000)
        
    test_pts = smooth_covering_radius.test_points
    # Compute distances from each test point to all configuration points
    # dist[i, j] = angle between test_pts[i] and points[j]
    dots = test_pts @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots) # Shape: (n_test, N)
    
    # min_dist[i] = min_j dists[i, j]
    min_dists = np.min(dists, axis=1)
    
    # Approximate max(min_dists) using log-sum-exp
    # max(x) ≈ (1/beta) * log(sum(exp(beta*x)))
    # To avoid overflow, subtract max
    max_val = np.max(min_dists)
    if max_val == 0:
        return 0.0
    exps = np.exp(beta * (min_dists - max_val))
    soft_max = max_val + np.log(np.sum(exps)) / beta
    return soft_max

def objective_and_grad(params):
    """Objective function and gradient for L-BFGS-B."""
    points = to_sphere(params)
    val = smooth_covering_radius(points, beta=100.0)
    # Note: L-BFGS-B does not require gradient for 'Powell', but we can use
    # 'L-BFGS-B' if we provide a gradient. However, numerical gradient of
    # the smooth objective is expensive. Let's stick to Powell or BFGS 
    # on the smooth objective first, or use the smooth objective with 
    # a gradient-free method that is fast.
    # Actually, let's implement a simple gradient-free optimization on the 
    # smooth objective, as it is much faster than the exact one.
    return val

def local_optimize_smooth(init_points, max_iter=200):
    """Optimize using the smooth objective. This is fast."""
    params = init_points.flatten()
    
    def obj(p):
        return smooth_covering_radius(to_sphere(p), beta=100.0)
    
    try:
        # Powell is robust for unconstrained optimization
        result = minimize(obj, params, method='Powell',
                          options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
        if result.success or result.fun < obj(params):
            return to_sphere(result.x), result.fun
    except Exception:
        pass
    return init_points, obj(params)

def high_precision_polish(points, max_iter=50):
    """Final polish using the exact convex hull objective.
    This is slower but ensures the score is accurate.
    """
    params = points.flatten()
    
    def obj_exact(p):
        return compute_covering_radius_exact(to_sphere(p))
    
    try:
        # Use a very tight tolerance
        result = minimize(obj_exact, params, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        if result.fun < obj_exact(params):
            return to_sphere(result.x), result.fun
    except Exception:
        pass
    return points, obj_exact(params)

def simulated_annealing(init_points, steps=500, T_start=0.1, T_end=0.001):
    """Simulated annealing on the smooth objective."""
    current_points = init_points.copy()
    current_val = smooth_covering_radius(current_points, beta=50.0)
    best_points = current_points.copy()
    best_val = current_val
    
    T = T_start
    for i in range(steps):
        if time.time() > DEADLINE - 10:
            break
            
        T = T_start * (T_end / T_start) ** (i / steps)
        
        # Perturb
        noise = np.random.randn(N, 3) * 0.05
        new_points = current_points + noise
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_val = smooth_covering_radius(new_points, beta=50.0)
        
        # Accept if better, or with probability exp(-delta/T)
        if new_val < current_val or np.random.rand() < np.exp(-(new_val - current_val) / max(T, 1e-9)):
            current_points = new_points
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_points = current_points.copy()
                
    return best_points, best_val

def random_init(n_points, rng):
    v = rng.randn(n_points, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def fibonacci_spiral(n_points):
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def icosahedron_based_init(n_points, seed=42):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    if n_points <= 12:
        return verts[:n_points]
        
    points = verts.copy()
    rng = np.random.RandomState(seed)
    while len(points) < n_points:
        best_dist = 0
        best_point = None
        for _ in range(100):
            candidate = rng.randn(3)
            candidate = candidate / np.linalg.norm(candidate)
            dists = np.arccos(np.clip(points @ candidate, -1, 1))
            min_dist = np.min(dists)
            if min_dist > best_dist:
                best_dist = min_dist
                best_point = candidate
        if best_point is not None:
            points = np.vstack([points, best_point])
    return points[:n_points]

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    np.random.seed(42)
    rng = np.random.RandomState(42)
    
    inits = [
        fibonacci_spiral(N),
        icosahedron_based_init(N),
        random_init(N, rng),
        fibonacci_spiral(N) + 0.1 * rng.randn(N, 3)
    ]
    inits = [p / np.linalg.norm(p, axis=1, keepdims=True) for p in inits]
    
    best_points = inits[0]
    best_exact_cr = compute_covering_radius_exact(best_points)
    best_smooth_cr = smooth_covering_radius(best_points, beta=100.0)
    
    for init in inits:
        if time.time() > DEADLINE - 30:
            break
            
        # 1. Fast local optimization on smooth objective
        smooth_points, smooth_val = local_optimize_smooth(init, max_iter=100)
        
        # 2. Simulated Annealing on smooth objective
        sa_points, sa_val = simulated_annealing(smooth_points, steps=200)
        
        # Choose the better of the two smooth results
        candidate_smooth = sa_points if sa_val < smooth_val else smooth_points
        candidate_val = min(sa_val, smooth_val)
        
        # 3. High-precision polish using exact objective
        # Only do this if the smooth value is promising
        exact_points, exact_val = high_precision_polish(candidate_smooth, max_iter=20)
        
        # Update global best based on exact evaluation
        if exact_val < best_exact_cr:
            best_exact_cr = exact_val
            best_points = exact_points
            best_smooth_cr = candidate_val
            
    # Final check: if time permits, do one more polish on the best points
    if time.time() < DEADLINE - 5:
        final_points, final_val = high_precision_polish(best_points, max_iter=50)
        if final_val < best_exact_cr:
            best_points = final_points
            best_exact_cr = final_val
            
    # Ensure unit sphere and no coincident points
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-8:
                best_points[j] = best_points[j] + 1e-8 * rng.randn(3)
                best_points[j] /= np.linalg.norm(best_points[j])
                
    return best_points
