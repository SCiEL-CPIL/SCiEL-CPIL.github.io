import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def compute_min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = len(points)
    if n < 3:
        return 0.0
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def get_score(points):
    """Compute score: min triangle area / convex hull area."""
    N = len(points)
    if N < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except:
        return 0.0
    min_area = compute_min_triangle_area(points)
    if min_area <= 0:
        return 0.0
    return min_area / hull_area

def soft_min(areas, tau):
    """Soft minimum with log-sum-exp stability."""
    min_val = np.min(areas)
    shifted = areas - min_val
    # Clip to avoid overflow in exp
    shifted = np.maximum(shifted, -50.0 / tau)
    log_sum = np.log(np.sum(np.exp(-tau * shifted)))
    return min_val - (1.0 / tau) * log_sum

def compute_all_areas(points):
    """Compute all triangle areas for n points (vectorized)."""
    n = len(points)
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    N = 13
    best_score = 0.0
    best_points = None

    def time_left():
        return deadline - time.time()

    def make_regular_ngon(n, center=None, radius=1.0, phase=0.0):
        angles = np.linspace(phase, phase + 2 * np.pi, n, endpoint=False)
        if center is None:
            center = np.array([0.0, 0.0])
        return np.column_stack((center[0] + radius * np.cos(angles),
                                center[1] + radius * np.sin(angles)))

    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Regular 13-gon
    initial_configs.append(make_regular_ngon(13, radius=1.0))
    
    # 2. Regular 12-gon + center
    ring12 = make_regular_ngon(12, radius=1.0)
    initial_configs.append(np.vstack([ring12, np.array([[0.0, 0.0]])]))
    
    # 3. Regular 11-gon + 2 interior (symmetric pair)
    ring11 = make_regular_ngon(11, radius=1.0)
    interior2 = np.array([[0.35, 0.0], [-0.35, 0.0]])
    initial_configs.append(np.vstack([ring11, interior2]))
    
    # 4. 7 outer + 6 inner concentric
    outer7 = make_regular_ngon(7, radius=1.0)
    inner6 = make_regular_ngon(6, radius=0.55, phase=np.pi/6)
    initial_configs.append(np.vstack([outer7, inner6]))
    
    # 5. 9 outer + 4 inner (square)
    outer9 = make_regular_ngon(9, radius=1.0)
    inner4 = np.array([[0.45, 0.45], [-0.45, 0.45], [-0.45, -0.45], [0.45, -0.45]])
    initial_configs.append(np.vstack([outer9, inner4]))
    
    # 6. 10 outer + 3 inner (triangle)
    outer10 = make_regular_ngon(10, radius=1.0)
    inner3 = make_regular_ngon(3, radius=0.45)
    initial_configs.append(np.vstack([outer10, inner3]))
    
    # 7. 8 outer + 5 inner
    outer8 = make_regular_ngon(8, radius=1.0)
    inner5 = make_regular_ngon(5, radius=0.5, phase=np.pi/5)
    initial_configs.append(np.vstack([outer8, inner5]))
    
    # 8. 12 outer + 1 center with slight offset
    ring12b = make_regular_ngon(12, radius=1.0, phase=np.pi/12)
    initial_configs.append(np.vstack([ring12b, np.array([[0.0, 0.0]])]))
    
    # 9. 11 outer + 2 inner (radial pair at different angles)
    ring11b = make_regular_ngon(11, radius=1.0, phase=np.pi/11)
    interior2b = np.array([[0.4, 0.0], [0.2, 0.2]])
    initial_configs.append(np.vstack([ring11b, interior2b]))
    
    # 10. Grid-like structure (4x3 + 1)
    grid = np.array([
        [-1, -0.8], [-0.3, -0.8], [0.4, -0.8], [1.1, -0.8],
        [-1, 0], [-0.3, 0], [0.4, 0], [1.1, 0],
        [-1, 0.8], [-0.3, 0.8], [0.4, 0.8], [1.1, 0.8],
        [0.05, 0.0]
    ])
    # Scale to roughly unit area
    grid = grid / np.linalg.norm(grid, axis=1).max()
    initial_configs.append(grid)
    
    # 11. Two concentric rings with different rotations
    outer6 = make_regular_ngon(6, radius=1.0)
    inner7 = make_regular_ngon(7, radius=0.6, phase=np.pi/7)
    initial_configs.append(np.vstack([outer6, inner7]))
    
    # 12. Random perturbations of regular 13-gon
    rng = np.random.default_rng(42)
    base = make_regular_ngon(13, radius=1.0)
    for _ in range(3):
        perturbed = base + rng.normal(0, 0.05, (13, 2))
        initial_configs.append(perturbed)

    # Evaluate all initial configs
    for cfg in initial_configs:
        s = get_score(cfg)
        if s > best_score:
            best_score = s
            best_points = cfg.copy()

    # Optimization loop
    # We use a smooth surrogate objective with soft-min
    def smooth_objective(pts_flat, tau):
        pts = pts_flat.reshape(N, 2)
        try:
            areas = compute_all_areas(pts)
            sm = soft_min(areas, tau)
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 1e6
            score = sm / hull_area
            return -score
        except:
            return 1e6

    def exact_objective(pts_flat):
        pts = pts_flat.reshape(N, 2)
        try:
            min_area = compute_min_triangle_area(pts)
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 1e6
            score = min_area / hull_area
            return -score
        except:
            return 1e6

    # Progressive annealing with multiple methods
    taus = [50, 100, 200, 400]
    
    for tau in taus:
        if time_left() < 10.0:
            break
            
        # Start from current best
        x0 = best_points.flatten()
        
        # Try L-BFGS-B first (fast, gradient-based)
        try:
            res = minimize(smooth_objective, x0, args=(tau,), method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < 1e5:
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = res.x.reshape(N, 2)
        except:
            pass
        
        # Try Powell (derivative-free, robust)
        if time_left() > 15.0:
            try:
                x_start = best_points.flatten()
                res2 = minimize(smooth_objective, x_start, args=(tau,), method='Powell',
                                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
                if res2.fun < 1e5:
                    current_score = -res2.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = res2.x.reshape(N, 2)
            except:
                pass

    # Perturb-and-reoptimize loop to escape local minima
    if time_left() > 30.0:
        rng = np.random.default_rng(123)
        for i in range(20): # 20 iterations
            if time_left() < 15.0:
                break
            
            # Perturb current best
            scale = 0.05 / (i + 1) # Decreasing perturbation
            x_perturbed = best_points.flatten() + rng.normal(0, scale, 2 * N)
            
            # Quick local optimization with high tau
            try:
                res = minimize(smooth_objective, x_perturbed, args=(400,), method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                if res.fun < 1e5:
                    current_score = -res.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = res.x.reshape(N, 2)
            except:
                pass

    # Final exact Nelder-Mead polish
    if time_left() > 10.0 and best_points is not None:
        x0_nm = best_points.flatten()
        try:
            res_final = minimize(exact_objective, x0_nm, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res_final.fun < 1e5:
                current_score = -res_final.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = res_final.x.reshape(N, 2)
        except:
            pass

    # Final exact evaluation
    if best_points is not None:
        final_score = get_score(best_points)
        if final_score > best_score:
            best_score = final_score

    return best_points
