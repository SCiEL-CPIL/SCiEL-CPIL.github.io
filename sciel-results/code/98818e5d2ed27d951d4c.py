"""Heilbronn n=14: SLSQP max-min with active-set tracking and Fekete initialization."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# EVOLVE-BLOCK-START

def _all_triangle_areas(points):
    """Compute all C(N,3) triangle areas vectorized."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas.extend(0.5 * np.abs(cross).tolist())
    return np.array(areas)

def _min_triangle_area(points):
    """Compute minimum triangle area."""
    return np.min(_all_triangle_areas(points))

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute min triangle area / hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _generate_fekete_initial(n=14):
    """
    Generate a Fekete-like initial configuration.
    Fekete points maximize the product of pairwise distances.
    We approximate by starting with a regular polygon and running a few
    iterations of a simple repulsion-based optimization.
    """
    # Start with regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Run a few iterations of Fekete optimization (maximize log product of distances)
    def feketee_obj(x):
        p = x.reshape(n, 2)
        log_prod = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(p[i] - p[j])
                if d < 1e-12:
                    return -1e10
                log_prod += np.log(d)
        return -log_prod
    
    start = time.time()
    best_x = pts.flatten()
    best_val = feketee_obj(best_x)
    
    # Simple coordinate descent for Fekete
    for iteration in range(100):
        if time.time() - start > 5:
            break
        improved = False
        for i in range(n):
            for dim in range(2):
                # Try small perturbations
                for step in [0.01, -0.01, 0.005, -0.005]:
                    trial = best_x.copy()
                    trial[i*2+dim] += step
                    val = feketee_obj(trial)
                    if val < best_val:
                        best_val = val
                        best_x = trial
                        improved = True
        if not improved:
            break
    
    return best_x.reshape(n, 2)

def _generate_initial_points():
    """Generate multiple initial configurations for diversity."""
    inits = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Fekete-like
    inits.append(_generate_fekete_initial(N))
    
    # 3. Regular polygon with slight perturbation
    rng = np.random.RandomState(42)
    angles_pert = angles + rng.uniform(-0.05, 0.05, N)
    inits.append(np.column_stack([np.cos(angles_pert), np.sin(angles_pert)]))
    
    # 4. Two concentric rings (7+7)
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = angles_outer + np.pi/7  # offset
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner = np.column_stack([0.6*np.cos(angles_inner), 0.6*np.sin(angles_inner)])
    inits.append(np.vstack([outer, inner]))
    
    # 5. Three rings (5+5+4)
    angles1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    angles3 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    r1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    r2 = np.column_stack([0.7*np.cos(angles2), 0.7*np.sin(angles2)])
    r3 = np.column_stack([0.4*np.cos(angles3), 0.4*np.sin(angles3)])
    inits.append(np.vstack([r1, r2, r3]))
    
    return inits

def _objective(x, t_deadline):
    """
    Soft objective: minimize -(min_area/hull_area) with penalty for degenerate configs.
    """
    p = x.reshape(N, 2)
    
    # Check for degeneracy
    hull_area = _hull_area(p)
    if hull_area < 1e-8:
        return 1e10
    
    min_area = _min_triangle_area(p)
    ratio = min_area / hull_area
    
    if ratio < 1e-12:
        return 1e10
    
    return -ratio

def _objective_hard(x, t_deadline):
    """
    Hard objective with active-set tracking for max-min problem.
    We track which triangles are "active" (near the minimum) and focus optimization there.
    """
    p = x.reshape(N, 2)
    hull_area = _hull_area(p)
    if hull_area < 1e-8:
        return 1e10, None
    
    areas = _all_triangle_areas(p)
    min_area = np.min(areas)
    ratio = min_area / hull_area
    
    if ratio < 1e-12:
        return 1e10, None
    
    # Identify active constraints (triangles with area within 20% of min)
    threshold = min_area * 1.2
    active_mask = areas < threshold
    
    return -ratio, active_mask

def _refine_slsqp(points, t_deadline, max_iter=200):
    """
    Refine using SLSQP with time tracking.
    """
    start = time.time()
    
    def obj(x):
        if time.time() > t_deadline:
            return 1e10
        val, _ = _objective_hard(x, t_deadline)
        return val
    
    try:
        res = minimize(
            obj,
            points.flatten(),
            method='SLSQP',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return points

def _refine_nelder_mead(points, t_deadline, max_iter=500):
    """
    Refine using Nelder-Mead as a robust fallback.
    """
    start = time.time()
    
    def obj(x):
        if time.time() > t_deadline:
            return 1e10
        val, _ = _objective_hard(x, t_deadline)
        return val
    
    try:
        res = minimize(
            obj,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return points

def _refine_powell(points, t_deadline, max_iter=300):
    """
    Refine using Powell's method.
    """
    start = time.time()
    
    def obj(x):
        if time.time() > t_deadline:
            return 1e10
        val, _ = _objective_hard(x, t_deadline)
        return val
    
    try:
        res = minimize(
            obj,
            points.flatten(),
            method='Powell',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return points

def construct():
    """
    Generate 14 points to maximize min triangle area / convex hull area.
    
    Strategy:
    1. Generate diverse initial configurations (regular, Fekete, multi-ring).
    2. For each, perform multi-stage local refinement:
       a. SLSQP for gradient-based refinement
       b. Nelder-Mead for robustness
       c. Powell for final polish
    3. Track best result throughout.
    4. Normalize to unit area.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Generate initial configurations
    inits = _generate_initial_points()
    
    best_score = 0.0
    best_pts = None
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        current_pts = init_pts.copy()
        current_score = _score(current_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
        
        # Stage 1: SLSQP refinement
        time_left = deadline - time.time() - 5
        if time_left > 10:
            refined = _refine_slsqp(current_pts, time.time() + time_left * 0.4, max_iter=150)
            s = _score(refined)
            if s > current_score:
                current_pts = refined
                current_score = s
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
        
        # Stage 2: Nelder-Mead refinement
        time_left = deadline - time.time() - 5
        if time_left > 10:
            refined = _refine_nelder_mead(current_pts, time.time() + time_left * 0.3, max_iter=300)
            s = _score(refined)
            if s > current_score:
                current_pts = refined
                current_score = s
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
        
        # Stage 3: Powell refinement
        time_left = deadline - time.time() - 5
        if time_left > 10:
            refined = _refine_powell(current_pts, time.time() + time_left * 0.3, max_iter=200)
            s = _score(refined)
            if s > current_score:
                current_pts = refined
                current_score = s
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
        
        # If time remains, do another round of SLSQP from the best so far
        time_left = deadline - time.time() - 5
        if time_left > 20 and best_pts is not None:
            refined = _refine_slsqp(best_pts, time.time() + time_left * 0.5, max_iter=200)
            s = _score(refined)
            if s > best_score:
                best_score = s
                best_pts = refined
    
    # If we didn't find anything good, fall back to regular polygon
    if best_pts is None or best_score < 1e-6:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    hull_area = _hull_area(best_pts)
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = best_pts * scale
    else:
        # Fallback: regular polygon normalized
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        hull_area = _hull_area(pts)
        scale = np.sqrt(1.0 / hull_area)
        final_points = pts * scale
    
    return final_points

# EVOLVE-BLOCK-END
