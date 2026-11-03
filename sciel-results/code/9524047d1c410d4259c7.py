import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()
N_TRI = len(IDX_I)

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = np.min(_all_areas(points))
    return min_area / hull_area

def _symmetric_points(half_pts):
    return np.vstack([half_pts, -half_pts])

def _objective_soft_sym(flat, beta=50.0):
    """
    Objective for centrally symmetric parameterization.
    Minimizes hull_area / soft_min_area.
    This is equivalent to maximizing soft_min_area / hull_area.
    """
    pts_half = flat.reshape(N_HALF, 2)
    pts = _symmetric_points(pts_half)
    
    if np.any(~np.isfinite(pts)):
        return 1e10
        
    areas = _all_areas(pts)
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 1e10
        
    # Soft min of areas
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    
    if soft_min < 1e-12:
        return 1e10
        
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
        
    return hull_area / soft_min

def _objective_soft_full(flat, beta=50.0):
    """
    Objective for full 28D parameterization.
    Minimizes hull_area / soft_min_area.
    """
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
        
    areas = _all_areas(pts)
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 1e10
        
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
        
    if soft_min < 1e-12:
        return 1e10
        
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
        
    return hull_area / soft_min

def _objective_hard_sym(flat):
    pts_half = flat.reshape(N_HALF, 2)
    pts = _symmetric_points(pts_half)
    return -_score(pts)

def _objective_hard_full(flat):
    pts = flat.reshape(N, 2)
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def _elapsed():
        return time.time() - start_time

    # Helper to run L-BFGS-B on soft objective
    def _refine_soft(flat_init, is_symmetric=True, beta=50.0, maxiter=1000):
        if is_symmetric:
            obj_func = lambda x: _objective_soft_sym(x, beta=beta)
            def get_pts(x):
                pts_half = x.reshape(N_HALF, 2)
                return _symmetric_points(pts_half)
        else:
            obj_func = lambda x: _objective_soft_full(x, beta=beta)
            def get_pts(x):
                return x.reshape(N, 2)
        
        try:
            res = minimize(
                obj_func,
                flat_init,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            pts = get_pts(res.x)
            score = _score(pts)
            return pts, score
        except Exception:
            return None, -np.inf

    # Helper to run Nelder-Mead on hard objective
    def _refine_hard(flat_init, is_symmetric=True, maxiter=1000):
        if is_symmetric:
            obj_func = _objective_hard_sym
            def get_pts(x):
                pts_half = x.reshape(N_HALF, 2)
                return _symmetric_points(pts_half)
        else:
            obj_func = _objective_hard_full
            def get_pts(x):
                return x.reshape(N, 2)
        
        try:
            res = minimize(
                obj_func,
                flat_init,
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            pts = get_pts(res.x)
            score = _score(pts)
            return pts, score
        except Exception:
            return None, -np.inf

    # --- Phase 1: Generate diverse starting points in Symmetric Space ---
    # 1. Regular polygon (symmetrized)
    angles = np.linspace(0, np.pi, N_HALF, endpoint=False)
    radii = np.ones(N_HALF)
    start_1 = np.column_stack((radii * np.cos(angles), radii * np.sin(angles))).flatten()
    
    # 2. Random symmetric points
    np.random.seed(42)
    starts_sym = [start_1]
    for _ in range(15):
        pts = np.random.uniform(-1, 1, (N_HALF, 2))
        # Ensure no points are too close to origin to avoid degeneracy in symmetric pairs
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms < 0.1):
            pts[norms < 0.1] *= 10
        starts_sym.append(pts.flatten())
        
    # 3. Structured starts: Hexagonal-like
    # Try to mimic a hexagonal lattice
    for r_base in [0.8, 1.0, 1.2]:
        for angle_offset in [0.0, np.pi/6, np.pi/3]:
            pts = []
            # 7 points in upper half + symmetric
            # We need 7 points in the half-space.
            # Let's just use random with bias towards circle
            for _ in range(5):
                 r = r_base * np.random.uniform(0.8, 1.2)
                 th = np.random.uniform(0, np.pi)
                 pts.append([r * np.cos(th), r * np.sin(th)])
            # Add 2 more points closer to center?
            r_c = r_base * 0.5
            th_c1 = np.random.uniform(0, np.pi)
            th_c2 = np.random.uniform(0, np.pi)
            pts.append([r_c * np.cos(th_c1), r_c * np.sin(th_c1)])
            pts.append([r_c * np.cos(th_c2), r_c * np.sin(th_c2)])
            starts_sym.append(np.array(pts).flatten())

    # --- Phase 2: Local Optimization from Starts ---
    # We will alternate between soft (for landscape) and hard (for precision)
    # Start with soft to get into a good basin.
    
    for i, start in enumerate(starts_sym):
        if _elapsed() > deadline - 30.0:
            break
            
        # Step 1: Soft refinement with low beta (smooth landscape)
        pts1, score1 = _refine_soft(start, is_symmetric=True, beta=20.0, maxiter=500)
        if pts1 is not None:
            update_best(pts1, score1)
        
        # Step 2: Soft refinement with higher beta (sharper min)
        if pts1 is not None and _elapsed() < deadline - 25.0:
            flat1 = pts1[:N_HALF].flatten() # Extract symmetric half
            pts2, score2 = _refine_soft(flat1, is_symmetric=True, beta=100.0, maxiter=1000)
            if pts2 is not None:
                update_best(pts2, score2)
            
            # Step 3: Hard refinement (Nelder-Mead) to polish
            if pts2 is not None and _elapsed() < deadline - 20.0:
                flat2 = pts2[:N_HALF].flatten()
                pts3, score3 = _refine_hard(flat2, is_symmetric=True, maxiter=2000)
                if pts3 is not None:
                    update_best(pts3, score3)

    # --- Phase 3: Global Search with Differential Evolution (Symmetric) ---
    if _elapsed() < deadline - 40.0:
        def de_obj(flat):
            return _objective_soft_sym(flat, beta=30.0)
            
        try:
            bounds = [(-2.0, 2.0)] * (N_HALF * 2)
            # DE is slow. Use a small population and few iterations, relying on polish later.
            res_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=20,
                popsize=15,
                tol=1e-4,
                polish=False, # We will polish manually
                init='latinhypercube'
            )
            
            if res_de.fun < 1e9: # Valid
                pts_de = _symmetric_points(res_de.x.reshape(N_HALF, 2))
                score_de = _score(pts_de)
                update_best(pts_de, score_de)
                
                # Polish DE result
                if _elapsed() < deadline - 20.0:
                    flat_de = res_de.x
                    # Soft polish
                    pts_p1, score_p1 = _refine_soft(flat_de, is_symmetric=True, beta=100.0, maxiter=1000)
                    if pts_p1 is not None:
                        update_best(pts_p1, score_p1)
                        # Hard polish
                        if _elapsed() < deadline - 10.0:
                            flat_p1 = pts_p1[:N_HALF].flatten()
                            pts_p2, score_p2 = _refine_hard(flat_p1, is_symmetric=True, maxiter=3000)
                            if pts_p2 is not None:
                                update_best(pts_p2, score_p2)
                                
        except Exception:
            pass

    # --- Phase 4: Break Symmetry? ---
    # If we have a good symmetric solution, try to break symmetry in full 28D space.
    # This might find a slightly better non-symmetric configuration.
    if best_points is not None and _elapsed() < deadline - 15.0:
        flat_full = best_points.flatten()
        
        # Soft refine in full space
        pts_f1, score_f1 = _refine_soft(flat_full, is_symmetric=False, beta=50.0, maxiter=500)
        if pts_f1 is not None:
            update_best(pts_f1, score_f1)
            
            # Hard refine in full space
            if _elapsed() < deadline - 5.0:
                pts_f2, score_f2 = _refine_hard(flat_full, is_symmetric=False, maxiter=1000)
                if pts_f2 is not None:
                    update_best(pts_f2, score_f2)

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
