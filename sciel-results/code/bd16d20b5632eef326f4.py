# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time

def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _neg_smooth_objective(x, k=10):
    """
    Objective function to minimize (negative of score proxy).
    We want to maximize the minimum distance relative to the maximum.
    
    Strategy:
    1. Normalize the points so that the maximum distance is 1. 
       This makes the problem scale-invariant and focuses optimization on shape.
    2. Maximize the sum of the smallest k pairwise distances.
       Maximizing the sum of the smallest distances ensures that no pair is 
       arbitrarily close, effectively pushing up the d_min.
       Since the scale is fixed (d_max=1), maximizing d_min is equivalent to 
       maximizing the ratio d_min/d_max.
       
    We minimize the negative of this sum.
    """
    n = len(x)
    points = x.reshape((n, 2))
    
    # Compute pairwise distances
    d = pdist(points)
    
    # Handle degenerate cases
    if np.allclose(d, 0):
        return 1e10
    
    d_max = np.max(d)
    if d_max < 1e-8:
        return 1e10
        
    # Normalize distances so d_max = 1
    d_norm = d / d_max
    
    # We want to maximize the sum of the smallest k distances
    d_sorted = np.sort(d_norm)
    
    # Sum of the smallest k
    obj_val = -np.sum(d_sorted[:k])
    
    return obj_val

def _local_refine(points, max_iter=200):
    """
    Local refinement using L-BFGS-B starting from given points.
    """
    n = len(points)
    x0 = points.flatten()
    
    try:
        res = minimize(
            _neg_smooth_objective, 
            x0, 
            args=(10,),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        opt_points = res.x.reshape((n, 2))
        score = _compute_score(opt_points)
        return opt_points, score
    except Exception:
        return points, _compute_score(points)

def _generate_initial_population(n_points=16, pop_size=50, rng=None):
    """
    Generate diverse initial configurations for DE population.
    """
    if rng is None:
        rng = np.random.default_rng(42)
        
    pop = []
    
    # 1. Regular Polygon
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    pop.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. 4x4 Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    pop.append(np.column_stack([xx.ravel(), yy.ravel()]))
    
    # 3. Hexagonal Lattice
    pts_hex = []
    for row in range(4):
        for col in range(4):
            x = col * 1.0
            y = row * np.sqrt(3)/2
            if col % 2 == 1:
                y += np.sqrt(3)/4
            pts_hex.append([x, y])
    pts_hex = np.array(pts_hex)
    pts_hex -= pts_hex.mean(axis=0)
    pop.append(pts_hex)
    
    # 4. Two Concentric Circles
    r1, r2 = 0.5, 1.5
    a1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    a2 = np.linspace(np.pi/8, 2*np.pi + np.pi/8, 8, endpoint=False)
    p1 = r1 * np.column_stack([np.cos(a1), np.sin(a1)])
    p2 = r2 * np.column_stack([np.cos(a2), np.sin(a2)])
    pop.append(np.vstack([p1, p2]))
    
    # 5. Three Concentric Rings (4+6+6)
    r1, r2, r3 = 0.3, 0.7, 1.2
    a1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    a2 = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 6, endpoint=False)
    a3 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    p1 = r1 * np.column_stack([np.cos(a1), np.sin(a1)])
    p2 = r2 * np.column_stack([np.cos(a2), np.sin(a2)])
    p3 = r3 * np.column_stack([np.cos(a3), np.sin(a3)])
    pop.append(np.vstack([p1, p2, p3]))
    
    # 6. Random Seeds
    for _ in range(pop_size - len(pop)):
        pts = rng.standard_normal((n_points, 2))
        d = pdist(pts)
        if np.max(d) > 0:
            pts = pts / np.max(d)
        pop.append(pts)
        
    return pop

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 590  # 40s safety margin
    
    n_points = 16
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = 0.0
    
    # Phase 1: Differential Evolution for global search
    # DE explores the entire search space without getting stuck in local minima
    bounds = [(-2, 2)] * (n_points * 2)
    
    # Use a callback to track best solution and check time
    de_best = [0.0]
    
    def de_callback(xk, convergence):
        """Callback for DE to track best and check time."""
        if time.time() > deadline - 30:
            raise TimeoutError("DE time limit reached")
        pts = xk.reshape((n_points, 2))
        score = _compute_score(pts)
        if score > de_best[0]:
            de_best[0] = score
    
    try:
        # Run DE with moderate settings
        de_result = differential_evolution(
            _neg_smooth_objective,
            bounds=bounds,
            args=(10,),
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-8,
            polish=True,
            init='latinhypercube',
            updating='immediate',
            callback=de_callback,
            constraints=()
        )
        
        if de_result.x is not None:
            de_points = de_result.x.reshape((n_points, 2))
            de_score = _compute_score(de_points)
            if de_score > best_score:
                best_score = de_score
                best_points = de_points.copy()
                
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # Phase 2: Local refinement from multiple diverse starting points
    # Generate diverse seeds and refine each locally
    seeds = _generate_initial_population(n_points=n_points, pop_size=30, rng=rng)
    
    # Sort seeds by score to prioritize better starts
    seed_scores = [_compute_score(s) for s in seeds]
    seed_order = np.argsort(seed_scores)[::-1]
    
    for idx in seed_order:
        if time.time() > deadline - 5:
            break
        
        seed = seeds[idx]
        try:
            opt_pts, score = _local_refine(seed, max_iter=300)
            if score > best_score:
                best_score = score
                best_points = opt_pts.copy()
        except Exception:
            continue
    
    # Phase 3: If time remains, do additional random restarts with local refinement
    while time.time() < deadline - 10:
        seed = rng.standard_normal((n_points, 2))
        d = pdist(seed)
        if np.max(d) > 0:
            seed = seed / np.max(d)
            
        try:
            opt_pts, score = _local_refine(seed, max_iter=200)
            if score > best_score:
                best_score = score
                best_points = opt_pts.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
