import numpy as np
import scipy.spatial as spat
import time
import itertools
from scipy.optimize import differential_evolution, minimize

N = 14
DEADLINE = 145  # Safety margin before 150s kill

def _calc_min_tri_area(points):
    """
    Compute the minimum triangle area over all triples.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Pre-compute indices for unique triples i < j < k
    # This is done once in the main flow if possible, but here we assume points change.
    # For n=14, 364 triples is small.
    idxs = np.array(list(itertools.combinations(range(n), 3)))
    
    p_i = points[idxs[:, 0]]
    p_j = points[idxs[:, 1]]
    p_k = points[idxs[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min triangle area / hull area. Higher is better."""
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return 0.0
    min_tri = _calc_min_tri_area(points)
    return min_tri / hull_a

def _smooth_score(points, t=1.0):
    """
    Smoothed objective for optimization.
    Uses a soft-min to make the function differentiable and navigate basins.
    score = exp(sum(log(area_i) / t)) / hull_area
    As t -> 0, this approaches min(area_i) / hull_area.
    """
    n = len(points)
    idxs = np.array(list(itertools.combinations(range(n), 3)))
    
    p_i = points[idxs[:, 0]]
    p_j = points[idxs[:, 1]]
    p_k = points[idxs[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Avoid log(0)
    areas = np.maximum(areas, 1e-10)
    
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return -1e10
    
    # Soft min: (prod(areas^(1/t)))^(1/n) ? No, sum of logs.
    # log(soft_min) = (1/t) * sum(log(areas)) / n_triples ? 
    # Actually, softmin(x) = -t * log(sum(exp(-x/t))).
    # Here we want to maximize the minimum.
    # Let's use: exp( (1/n) * sum( log(area_i) ) ) is geometric mean.
    # To bias towards min, we can use: exp( (1/n) * sum( log(area_i) ) ) is not min.
    # A common trick: maximize sum(log(area_i)) is equivalent to maximizing product, which favors balanced areas.
    # This is often better for "max-min" problems because it prevents any single triangle from being too small.
    
    log_sum = np.sum(np.log(areas))
    geo_mean = np.exp(log_sum / len(areas))
    
    return geo_mean / hull_a

def _neg_smooth_score(params, t=1.0):
    """Objective for minimizers (negative smooth score)."""
    points = params.reshape(N, 2)
    # Normalize to unit area to keep parameters in a reasonable range
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return 1e10
    points = points / np.sqrt(hull_a)
    
    s = _smooth_score(points, t)
    return -s

def _generate_initial_population():
    """
    Generate diverse initial configurations.
    1. Regular 14-gon
    2. Two nested heptagons (various rho, phi)
    3. Random points in unit disk
    4. 4 rings of 3-4 points
    """
    pop = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    pop.append(pts.flatten())
    
    # 2. Two nested heptagons
    for rho in [0.5, 0.7]:
        for phi in [np.pi/7, np.pi/14, 0.3]:
            angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            angles_inner = angles_outer + phi
            
            x_outer = np.cos(angles_outer)
            y_outer = np.sin(angles_outer)
            
            x_inner = rho * np.cos(angles_inner)
            y_inner = rho * np.sin(angles_inner)
            
            xs = np.concatenate([x_outer, x_inner])
            ys = np.concatenate([y_outer, y_inner])
            pts = np.column_stack([xs, ys])
            pop.append(pts.flatten())
            
    # 3. Random points
    rng = np.random.RandomState(42)
    for _ in range(5):
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = rng.uniform(0.5, 1.0, N)
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        pts = np.column_stack([xs, ys])
        pop.append(pts.flatten())
        
    # 4. 4 rings
    # 3-4-4-3 or 4-3-3-4 distribution
    # Ring 1: 4 pts, Ring 2: 4 pts, Ring 3: 3 pts, Ring 4: 3 pts
    for r1, r2, r3, r4 in [(0.2, 0.4, 0.6, 0.8), (0.3, 0.5, 0.7, 0.9)]:
        pts_list = []
        # Ring 1
        a1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
        pts_list.append(np.column_stack([r1*np.cos(a1), r1*np.sin(a1)]))
        # Ring 2
        a2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/8
        pts_list.append(np.column_stack([r2*np.cos(a2), r2*np.sin(a2)]))
        # Ring 3
        a3 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/9
        pts_list.append(np.column_stack([r3*np.cos(a3), r3*np.sin(a3)]))
        # Ring 4
        a4 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/12
        pts_list.append(np.column_stack([r4*np.cos(a4), r4*np.sin(a4)]))
        
        pts = np.vstack(pts_list)
        pop.append(pts.flatten())
        
    return pop

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = -np.inf
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            
    # Pre-compute triple indices for speed in local refinement
    idxs = np.array(list(itertools.combinations(range(N), 3)))
    
    # Phase 1: Global Search with Differential Evolution
    # Bounds: [-2, 2] for x and y (we normalize later)
    bounds = [(-2, 2)] * (2 * N)
    
    # Use initial population from diverse heuristics
    init_pop = _generate_initial_population()
    
    # DE might be slow if maxiter is high. 
    # We have ~145s. 
    # Let's run DE for a limited time or iterations.
    # DE with 14*2=28 dims.
    
    def de_objective(params):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30: # Leave 30s for local refinement
            return 1e10
        return _neg_smooth_score(params, t=5.0) # t=5 is quite smooth, helps global search
    
    try:
        # maxiter=100 might be enough if popsize is small
        # popsize=10 means 280 evaluations per generation.
        # 100 * 280 = 28000 evals.
        # Each eval: hull area (fast) + smooth score (fast).
        # Let's try maxiter=50 to be safe.
        res = differential_evolution(
            de_objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10,
            tol=1e-6,
            polish=False # We will do our own polish
        )
        
        if res.fun < 1e9:
            best_pts_de = res.x.reshape(N, 2)
            hull_a = _hull_area(best_pts_de)
            if hull_a > 1e-10:
                best_pts_de = best_pts_de / np.sqrt(hull_a)
                s_de = _score(best_pts_de)
                update_best(best_pts_de, s_de)
                
    except Exception:
        pass
    
    # If DE didn't find anything good, fall back to the best from initial pop
    if best_points is None or best_score < 40.0:
        for params in init_pop:
            pts = params.reshape(N, 2)
            hull_a = _hull_area(pts)
            if hull_a < 1e-10:
                continue
            pts = pts / np.sqrt(hull_a)
            s = _score(pts)
            update_best(pts, s)
            
        if best_points is None:
            # Fallback to regular polygon
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            best_points = np.column_stack([np.cos(angles), np.sin(angles)])
            best_score = _score(best_points)

    # Phase 2: Local Refinement with Annealing
    # Start from best point
    current_pts = best_points.copy()
    current_score = best_score
    
    # Annealing schedule for t in smooth score
    # Start with t=10 (smooth), end with t=0.1 (sharp)
    t_values = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
    
    # Use L-BFGS-B or Nelder-Mead on the smoothed objective
    # L-BFGS-B is better for high dims if gradient is available, but we don't have analytic gradient.
    # Nelder-Mead is robust but can be slow.
    # Let's use Nelder-Mead with the smoothed objective.
    
    for t in t_values:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
            
        def local_obj(params):
            return _neg_smooth_score(params, t)
        
        try:
            res = minimize(
                local_obj, 
                current_pts.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res.fun < 1e9:
                new_pts = res.x.reshape(N, 2)
                hull_a = _hull_area(new_pts)
                if hull_a > 1e-10:
                    new_pts = new_pts / np.sqrt(hull_a)
                    new_score = _score(new_pts)
                    if new_score > current_score:
                        current_pts = new_pts
                        current_score = new_score
                        update_best(current_pts, current_score)
                        
        except Exception:
            pass
            
    # Phase 3: Final Exact Polish
    # Use the exact score (min area) with a very small step Nelder-Mead or coordinate descent
    # Since Nelder-Mead on non-differentiable min can get stuck, let's try a few random perturbations
    # and refine the best one.
    
    rng = np.random.RandomState(123)
    
    for _ in range(5): # 5 restarts from current best with small noise
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 2:
            break
            
        noisy_pts = current_pts + rng.normal(0, 0.01, (N, 2))
        hull_a = _hull_area(noisy_pts)
        if hull_a < 1e-10:
            continue
        noisy_pts = noisy_pts / np.sqrt(hull_a)
        
        # Quick Nelder-Mead on exact score
        def exact_obj(params):
            pts = params.reshape(N, 2)
            h = _hull_area(pts)
            if h < 1e-10:
                return 1e10
            pts = pts / np.sqrt(h)
            return -_score(pts)
            
        try:
            res = minimize(
                exact_obj, 
                noisy_pts.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            
            if res.fun < 1e9:
                final_pts = res.x.reshape(N, 2)
                hull_a = _hull_area(final_pts)
                if hull_a > 1e-10:
                    final_pts = final_pts / np.sqrt(hull_a)
                    final_score = _score(final_pts)
                    if final_score > current_score:
                        current_pts = final_pts
                        current_score = final_score
                        update_best(current_pts, current_score)
                        
        except Exception:
            pass
            
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return np.array(best_points)

if __name__ == "__main__":
    pts = construct()
    print(_score(pts))
