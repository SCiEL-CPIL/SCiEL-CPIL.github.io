# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import time
import itertools

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull (vectorized)
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for valid hull
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
        
    # Calculate all triangle areas efficiently using combinations
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def soft_min_score(points, temperature=0.05):
    """
    Smooth surrogate: -log(sum(exp(-area/temperature))) / A_hull
    This is differentiable and helps with gradient-based optimization.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    idx = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    # Soft minimum via log-sum-exp
    scaled = areas / temperature
    max_scaled = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - max_scaled))) + max_scaled
    soft_min = temperature * log_sum
    
    return -soft_min / A_hull

def nelder_mead_optimize(points, objective, max_iter=200, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0, tol=1e-10):
    """
    Nelder-Mead simplex optimization.
    """
    n = len(points)
    x0 = points.flatten()
    dim = len(x0)
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(dim):
        new_x = x0.copy()
        if x0[i] != 0:
            new_x[i] *= 1.05
        else:
            new_x[i] = 0.05
        simplex.append(new_x)
    
    # Evaluate initial simplex
    fvals = [objective(np.array(x).reshape(n, 2)) for x in simplex]
    
    # Sort by function value (minimizing -score = maximizing score)
    order = np.argsort(fvals)
    simplex = [simplex[i] for i in order]
    fvals = [fvals[i] for i in order]
    
    best_f = fvals[0]
    
    for iteration in range(max_iter):
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = objective(np.array(x_r).reshape(n, 2))
        
        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = objective(np.array(x_e).reshape(n, 2))
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = objective(np.array(x_c).reshape(n, 2))
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = objective(np.array(simplex[i]).reshape(n, 2))
        
        # Sort
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if fvals[0] < best_f:
            best_f = fvals[0]
    
    return np.array(simplex[0]).reshape(n, 2), -best_f

def delaunay_repair(points, time_limit):
    """
    Iteratively repair points by moving vertices incident to minimum-area
    Delaunay triangles toward the centroid of adjacent triangles.
    """
    points = points.copy()
    n = len(points)
    
    current_score = compute_score(points)
    best_score = current_score
    best_points = points.copy()
    
    step_size = 0.1
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < time_limit and no_improve < max_no_improve:
        try:
            # Get Delaunay triangulation
            tri = Delaunay(points)
            triangles = tri.simplices
        except Exception:
            break
        
        if len(triangles) == 0:
            break
        
        # Compute areas of Delaunay triangles
        areas = []
        for t in triangles:
            p1, p2, p3 = points[t[0]], points[t[1]], points[t[2]]
            v1 = p2 - p1
            v2 = p3 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            areas.append(abs(cross) * 0.5)
        
        areas = np.array(areas)
        min_area_idx = np.argmin(areas)
        min_triangle = triangles[min_area_idx]
        
        # Find centroid of this triangle
        tri_centroid = np.mean(points[min_triangle], axis=0)
        
        # Move each vertex of the minimum triangle toward the centroid
        improved = False
        for vertex_idx in min_triangle:
            # Try moving toward centroid
            for frac in [0.3, 0.5, 0.7, 1.0]:
                new_points = points.copy()
                move = (tri_centroid - points[vertex_idx]) * frac
                new_points[vertex_idx] += move
                
                new_score = compute_score(new_points)
                
                if new_score > current_score:
                    points = new_points
                    current_score = new_score
                    improved = True
                    no_improve = 0
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_points = points.copy()
                    break
        
        if not improved:
            no_improve += 1
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    return best_points, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area ratio.
    
    Uses Delaunay-based repair and Nelder-Mead optimization.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    best_points = None
    best_score = -1.0
    
    # Generate initial candidates
    candidates = []
    
    # Candidate 1: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    pts1 = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(pts1)
    
    # Candidate 2: Regular 12-gon + center
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts2 = np.column_stack([np.cos(angles), np.sin(angles)])
    pts2 = np.vstack([np.array([0.0, 0.0]), pts2])
    candidates.append(pts2)
    
    # Candidate 3: Two concentric hexagons + center
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    pts_h1 = np.column_stack([0.5 * np.cos(angles1), 0.5 * np.sin(angles1)])
    angles2 = angles1 + np.pi/6
    pts_h2 = np.column_stack([1.0 * np.cos(angles2), 1.0 * np.sin(angles2)])
    pts3 = np.vstack([np.array([0.0, 0.0]), pts_h1, pts_h2])
    candidates.append(pts3)
    
    # Candidate 4: Two concentric polygons (7 and 6)
    def make_concentric_polygons(n1=7, n2=6, r1=1.0, r2=0.5):
        angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
        p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([p1, p2])
    
    candidates.append(make_concentric_polygons(7, 6, 1.0, 0.5))
    candidates.append(make_concentric_polygons(6, 7, 1.0, 0.6))
    
    # Candidate 5: Fibonacci-like distribution
    def make_fibonacci_2d():
        n = 13
        golden_ratio = (1 + np.sqrt(5)) / 2
        angles = 2 * np.pi * np.arange(n) / golden_ratio
        radii = np.sqrt(np.arange(n) / n)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return points
    
    candidates.append(make_fibonacci_2d())
    
    # Add some random starts
    rng = np.random.default_rng(12345)
    for _ in range(3):
        pts_rand = rng.random((13, 2)) * 2 - 1
        candidates.append(pts_rand)
    
    # For each candidate, try Delaunay repair then Nelder-Mead
    n_candidates = len(candidates)
    
    for i, init_pts in enumerate(candidates):
        if time.time() >= deadline:
            break
        
        # Allocate time: 40% for Delaunay, 60% for Nelder-Mead
        remaining_time = deadline - time.time()
        remaining_candidates = n_candidates - i
        candidate_time = remaining_time / remaining_candidates
        
        # Phase 1: Delaunay repair
        del_deadline = min(deadline, time.time() + 0.4 * candidate_time)
        if time.time() < del_deadline:
            pts_del, score_del = delaunay_repair(init_pts, del_deadline)
            if score_del > best_score:
                best_score = score_del
                best_points = pts_del
            current_pts = pts_del
        else:
            current_pts = init_pts
            score_del = compute_score(current_pts)
        
        # Phase 2: Nelder-Mead on the best from Delaunay
        nm_deadline = min(deadline, time.time() + 0.6 * candidate_time)
        if time.time() < nm_deadline and current_pts is not None:
            # Use soft min as objective for smoother optimization
            def nm_objective(pts):
                return -soft_min_score(pts, temperature=0.05)
            
            # Run Nelder-Mead with time check
            nm_start = time.time()
            nm_max_iter = 100
            nm_result_pts, nm_score = nelder_mead_optimize(
                current_pts, nm_objective, max_iter=nm_max_iter
            )
            
            # Verify with hard score
            hard_score = compute_score(nm_result_pts)
            if hard_score > best_score:
                best_score = hard_score
                best_points = nm_result_pts
    
    # Final polish with Nelder-Mead on the best found
    if best_points is not None and time.time() < deadline - 15.0:
        def nm_objective_final(pts):
            return -soft_min_score(pts, temperature=0.02)
        
        final_deadline = deadline - 5.0
        # Run a final Nelder-Mead polish
        nm_result_pts, nm_score = nelder_mead_optimize(
            best_points, nm_objective_final, max_iter=150
        )
        hard_score = compute_score(nm_result_pts)
        if hard_score > best_score:
            best_score = hard_score
            best_points = nm_result_pts
    
    # Final fallback
    if best_points is None:
        best_points = candidates[0]
    
    # Ensure all points are finite
    if not np.all(np.isfinite(best_points)):
        best_points = candidates[0]
    
    return best_points

# EVOLVE-BLOCK-END
