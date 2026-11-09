from scipy.optimize import minimize
import numpy as np
from scipy.spatial import ConvexHull
import itertools
import time

# Precompute triangle indices for 13 points
N_POINTS = 13
TRIANGLE_IDX = np.array(list(itertools.combinations(range(N_POINTS), 3)))
N_TRIS = len(TRIANGLE_IDX)

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0.0 if the points are collinear or invalid.
    """
    if len(points) < 3:
        return 0.0
    
    # Check for valid hull
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 1e-10:
        return 0.0
        
    # Calculate all triangle areas efficiently using precomputed indices
    p1 = points[TRIANGLE_IDX[:, 0]]
    p2 = points[TRIANGLE_IDX[:, 1]]
    p3 = points[TRIANGLE_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def generate_seeds():
    """
    Generate a variety of high-quality initial configurations.
    """
    seeds = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    seeds.append(pts)
    
    # 2. Regular 12-gon + Center
    angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts12 = np.column_stack([np.cos(angles12), np.sin(angles12)])
    pts12_center = np.vstack([np.array([0.0, 0.0]), pts12])
    seeds.append(pts12_center)
    
    # 3. Two Concentric Rings (7 outer, 6 inner)
    ang7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    ang6 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    r_out = 1.0
    r_in = 0.6
    p_out = np.column_stack([r_out * np.cos(ang7), r_out * np.sin(ang7)])
    p_in = np.column_stack([r_in * np.cos(ang6), r_in * np.sin(ang6)])
    seeds.append(np.vstack([p_out, p_in]))
    
    # 4. Two Concentric Rings (6 outer, 7 inner)
    ang6b = np.linspace(0, 2*np.pi, 6, endpoint=False)
    ang7b = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    p_out_b = np.column_stack([1.0 * np.cos(ang6b), 1.0 * np.sin(ang6b)])
    p_in_b = np.column_stack([0.6 * np.cos(ang7b), 0.6 * np.sin(ang7b)])
    seeds.append(np.vstack([p_out_b, p_in_b]))
    
    # 5. Distorted Grid (4x3 + 1)
    xs = np.array([0, 1, 2, 3])
    ys = np.array([0, 1, 2])
    grid_pts = []
    for x in xs:
        for y in ys:
            grid_pts.append([x, y])
    grid_pts.append([1.5, 1.0])
    grid_pts = np.array(grid_pts)
    grid_pts = (grid_pts - grid_pts.mean(axis=0))
    rng = np.random.default_rng(42)
    grid_pts += rng.normal(0, 0.2, size=grid_pts.shape)
    seeds.append(grid_pts)
    
    # 6. Fibonacci Sphere projection (2D variant)
    n = 13
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    angles = np.arange(n) * golden_angle
    radii = np.sqrt(np.arange(n) / n)
    pts_fib = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    seeds.append(pts_fib)

    # 7. Alternating Radii (Star-like)
    angles_alt = np.linspace(0, 2*np.pi, 13, endpoint=False)
    radii_alt = np.ones(13)
    radii_alt[::2] = 1.0
    radii_alt[1::2] = 0.5
    pts_alt = np.column_stack([radii_alt * np.cos(angles_alt), radii_alt * np.sin(angles_alt)])
    seeds.append(pts_alt)
    
    # 8. Triangular Lattice (subset)
    pts_lattice = []
    for i in range(4):
        for j in range(4 - i):
            x = i + 0.5 * j
            y = (np.sqrt(3) / 2) * j
            pts_lattice.append([x, y])
    pts_lattice = np.array(pts_lattice)
    if len(pts_lattice) >= 13:
        pts_lattice = pts_lattice[:13]
    else:
        # Pad if needed
        while len(pts_lattice) < 13:
            pts_lattice.append([0, 0])
    pts_lattice = np.array(pts_lattice)
    pts_lattice = (pts_lattice - pts_lattice.mean(axis=0))
    max_norm = np.max(np.linalg.norm(pts_lattice, axis=1))
    if max_norm > 1e-10:
        pts_lattice /= max_norm
    seeds.append(pts_lattice)
    
    return seeds

def nelder_mead_optimize(points, maxiter=200):
    """
    Optimize points using Nelder-Mead method.
    """
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(N_POINTS, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        score = compute_score(pts)
        if score == 0.0:
            return 1e10
        return -score
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-6})
        
        best_x = result.x
        current_score = compute_score(best_x.reshape(N_POINTS, 2))
        
        if current_score <= compute_score(points):
            return points, compute_score(points)
            
        return best_x.reshape(N_POINTS, 2), current_score
        
    except Exception:
        return points, compute_score(points)

def soft_min_optimize(points, beta_start=10.0, beta_end=100.0, maxiter=500):
    """
    Optimize using a soft-min surrogate to smooth the objective landscape.
    """
    x0 = points.flatten()
    
    def objective(x, beta):
        pts = x.reshape(N_POINTS, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        try:
            hull = ConvexHull(pts)
            A_hull = hull.volume
        except Exception:
            return 1e10
        if A_hull <= 1e-10:
            return 1e10
        
        p1 = pts[TRIANGLE_IDX[:, 0]]
        p2 = pts[TRIANGLE_IDX[:, 1]]
        p3 = pts[TRIANGLE_IDX[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = np.abs(cross) * 0.5
        
        m = np.min(areas)
        # Soft min
        exp_terms = np.exp(-beta * (areas - m))
        log_sum = np.log(np.sum(exp_terms))
        softmin_val = m - (1.0/beta) * log_sum
        
        score = softmin_val / A_hull
        if not np.isfinite(score):
            return 1e10
        return -score
    
    # Anneal beta
    n_steps = 10
    betas = np.linspace(beta_start, beta_end, n_steps)
    current_x = x0
    
    for beta in betas:
        try:
            res = minimize(objective, current_x, args=(beta,), method='Nelder-Mead',
                           options={'maxiter': maxiter // n_steps, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.success or res.fun < objective(current_x, beta):
                current_x = res.x
        except Exception:
            pass
    
    final_pts = current_x.reshape(N_POINTS, 2)
    final_score = compute_score(final_pts)
    
    if final_score > compute_score(points):
        return final_pts, final_score
    return points, compute_score(points)

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area ratio.
    Uses Multi-Start Nelder-Mead optimization combined with Soft-Min Surrogate annealing.
    """
    start_time = time.time()
    deadline = start_time + 590.0

    seeds = generate_seeds()
    
    best_points_overall = None
    best_score_overall = -1.0
    
    # Add random seeds to increase diversity
    rng_master = np.random.default_rng(12345)
    for _ in range(10):
        random_seed = rng_master.uniform(-1, 1, size=(13, 2))
        if compute_score(random_seed) > 0.01:
             seeds.append(random_seed)

    n_seeds = len(seeds)
    
    # Phase 1: Quick optimization on all seeds using standard Nelder-Mead
    maxiter_phase1 = 100
    
    for i, seed in enumerate(seeds):
        if time.time() >= deadline - 120.0:
            break
            
        optimized_pts, score = nelder_mead_optimize(seed, maxiter=maxiter_phase1)
        
        if score > best_score_overall:
            best_score_overall = score
            best_points_overall = optimized_pts

    # Phase 2: Deep optimization using Soft-Min Surrogate on the best found
    if best_points_overall is not None:
        # Run soft-min optimization
        # This approach smooths the objective, allowing the optimizer to navigate
        # past local minima that trap standard hard-min optimization.
        
        # Do a few restarts with perturbations
        for k in range(5):
            if time.time() >= deadline - 30.0:
                break
            
            # Perturb best point slightly
            rng = np.random.default_rng(k)
            perturbed = best_points_overall + rng.normal(0, 0.05, size=(N_POINTS, 2))
            
            optimized_pts, score = soft_min_optimize(perturbed, beta_start=10.0, beta_end=100.0, maxiter=500)
            
            if score > best_score_overall:
                best_score_overall = score
                best_points_overall = optimized_pts
                
            # Also try a quick Nelder-Mead polish on the soft-min result
            if time.time() < deadline - 10.0:
                polished_pts, polished_score = nelder_mead_optimize(optimized_pts, maxiter=200)
                if polished_score > best_score_overall:
                    best_score_overall = polished_score
                    best_points_overall = polished_pts

    # Fallback
    if best_points_overall is None:
        best_points_overall = seeds[0]
        
    # Ensure validity
    if not np.all(np.isfinite(best_points_overall)):
        best_points_overall = seeds[0]
        
    # One last check to ensure ConvexHull doesn't fail on the returned result
    try:
        hull = ConvexHull(best_points_overall)
        if hull.volume <= 1e-10:
            return seeds[0]
    except:
        return seeds[0]

    return best_points_overall
