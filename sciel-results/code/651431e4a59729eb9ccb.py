"""Combined program: SA Global Search + Barycentric SLSQP Refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute triangle indices for vectorized area computation
# There are C(11, 3) = 165 triangles
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = _TRI_IDX.shape[0]

def _all_areas(pts):
    """Fully vectorized computation of all triangle areas."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return areas

def _min_area(pts):
    """Compute the minimum area of any triangle formed by three of the points."""
    return float(np.min(_all_areas(pts)))

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle using barycentric coordinates."""
    x, y = float(p[0]), float(p[1])
    # c = 2y/sqrt(3)
    c = 2.0 * y / np.sqrt(3)
    # b = x - 0.5c
    b = x - 0.5 * c
    # a = 1 - b - c
    a = 1.0 - b - c
    
    eps = 1e-10
    a = max(eps, min(1.0 - 2.0*eps, a))
    b = max(eps, min(1.0 - 2.0*eps, b))
    c = max(eps, min(1.0 - 2.0*eps, c))
    
    s = a + b + c
    a /= s
    b /= s
    c /= s
    
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _generate_initial_config(seed):
    """Generate a random initial configuration inside the triangle."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        p = r1 * V1 + r2 * V2
        pts.append(p)
    return np.array(pts)

def _generate_symmetric_config():
    """
    Generate a symmetric initial configuration based on known good patterns.
    This provides a high-quality starting point for optimization.
    """
    pts = []
    eps = 0.01
    
    # 1. Three vertices (slightly inside)
    pts.append([eps, eps])
    pts.append([1 - eps, eps])
    pts.append([0.5, H - eps * 2])
    
    # 2. Three midpoints of edges (slightly inside)
    # Midpoint of V0-V1: (0.5, 0) -> inside: (0.5, eps)
    pts.append([0.5, eps])
    # Midpoint of V1-V2: (0.75, H/2) -> inside: scale towards centroid
    # Centroid is (0.5, H/3). Vector from centroid to mid is (0.25, H/6).
    # Move inwards by factor (1-eps)
    cx, cy = 0.5, H/3
    mx, my = 0.75, H/2
    pts.append([cx + (1-eps)*(mx-cx), cy + (1-eps)*(my-cy)])
    # Midpoint of V0-V2: (0.25, H/2)
    mx, my = 0.25, H/2
    pts.append([cx + (1-eps)*(mx-cx), cy + (1-eps)*(my-cy)])
    
    # 3. Three interior points forming a smaller triangle around centroid
    # Centroid is (0.5, H/3).
    # Place them on a circle of radius r around centroid.
    # To maximize min area, they should be somewhat spread out.
    r = 0.15 
    for k in range(3):
        angle = 2 * np.pi * k / 3 + np.pi/6 # Rotate to align with symmetry
        pts.append([cx + r * np.cos(angle), cy + r * np.sin(angle)])
    
    return _clamp_all(np.array(pts))

def _simulated_annealing(start_points, time_limit, seed):
    """
    Perform Simulated Annealing to maximize the minimum triangle area.
    """
    rng = np.random.default_rng(seed)
    
    points = np.array(start_points, dtype=float)
    points = _clamp_all(points)
    current_score = _min_area(points)
    
    best_points = points.copy()
    best_score = current_score
    
    # SA Parameters
    T = 0.005
    T_min = 1e-10
    alpha = 0.9997
    
    step_size = 0.03
    min_step = 0.002
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            T = T_min * 5
            step_size = max(min_step, step_size * 0.7)
        
        # Pick a random point to perturb
        idx = rng.integers(N)
        
        # Perturb with Gaussian noise
        noise = rng.normal(0, step_size, size=2)
        new_point = points[idx] + noise
        new_point = _clamp_to_triangle(new_point)
        
        old_point = points[idx].copy()
        points[idx] = new_point
        
        new_score = _min_area(points)
        
        delta = new_score - current_score
        if delta > 0 or (T > 0 and rng.random() < np.exp(delta / (T + 1e-15))):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            points[idx] = old_point
            
        T *= alpha
        step_size *= 0.99995
        if step_size < min_step:
            step_size = min_step

    return best_points, best_score

def _barycentric_refine(pts, time_budget=10.0):
    """
    Local refinement using SLSQP with barycentric parameterization.
    This handles boundary constraints naturally.
    """
    # Convert to barycentric coordinates (a, b) for each point, c = 1-a-b
    bary = []
    for p in pts:
        x, y = float(p[0]), float(p[1])
        c = 2.0 * y / np.sqrt(3)
        b = x - 0.5 * c
        a = 1.0 - b - c
        
        # Clamp to valid range [0, 1]
        a = max(0.0, min(1.0, a))
        b = max(0.0, min(1.0, b))
        c = max(0.0, min(1.0, c))
        
        s = a + b + c
        if s < 1e-15:
            a, b, c = 1.0/3, 1.0/3, 1.0/3
        else:
            a /= s; b /= s; c /= s
        bary.append([a, b])
    bary = np.array(bary)
    
    def _bary_to_cartesian(flat_bary):
        """Convert flat barycentric array to Cartesian points."""
        pts = []
        for i in range(N):
            a = flat_bary[2*i]
            b = flat_bary[2*i+1]
            c = 1.0 - a - b
            
            # Clamp to ensure validity
            a = max(1e-12, a)
            b = max(1e-12, b)
            c = max(1e-12, c)
            s = a + b + c
            a /= s; b /= s; c /= s
            pts.append(a * V0 + b * V1 + c * V2)
        return np.array(pts)
    
    def _objective(flat_bary):
        pts = _bary_to_cartesian(flat_bary)
        return -_min_area(pts)
    
    # Constraints for SLSQP: a >= 0, b >= 0, a+b <= 1 (which implies c >= 0)
    constraints = []
    for i in range(N):
        # a_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
        # b_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        # a_i + b_i <= 1  => 1 - a_i - b_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
    
    try:
        res = minimize(
            _objective,
            bary.flatten(),
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
        )
        if res.success or res.fun < _objective(bary.flatten()):
            res_pts = _bary_to_cartesian(res.x)
            res_pts = _clamp_all(res_pts)
            return res_pts, _min_area(res_pts)
    except Exception:
        pass
    
    # Fallback: Nelder-Mead on Cartesian if SLSQP fails
    try:
        res_nm = minimize(
            lambda flat: -_min_area(_clamp_all(flat.reshape(-1, 2))),
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        nm_pts = _clamp_all(res_nm.x.reshape(-1, 2))
        nm_score = _min_area(nm_pts)
        if nm_score > _min_area(pts):
            return nm_pts, nm_score
    except Exception:
        pass
    
    return pts, _min_area(pts)

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. We use 140s as a hard deadline to be safe.
    deadline = start_time + 140.0
    
    def time_left():
        return deadline - time.time()
        
    best_pts = None
    best_score = -1.0
    
    # Phase 1: Global Search using Simulated Annealing
    # We run SA from a few diverse random starts and the symmetric start.
    
    # 1. Symmetric start (high quality)
    # 2. Random starts (diversity)
    starts = [
        _generate_symmetric_config(),
        _generate_initial_config(42),
        _generate_initial_config(123),
        _generate_initial_config(789),
        _generate_initial_config(101)
    ]
    
    for i, start_pts in enumerate(starts):
        if time_left() < 15.0: # Leave 15s for refinement
            break
            
        # Allocate time to this SA run. 
        # First run (symmetric) gets more time as it's likely better.
        time_budget = 20.0
        if i == 0:
            time_budget = 30.0
            
        try:
            sa_pts, sa_score = _simulated_annealing(start_pts, time_budget, seed=seed+i+1000)
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts
        except Exception as e:
            pass

    # If no points found (should not happen), use a symmetric fallback
    if best_pts is None:
        best_pts = _generate_symmetric_config()
        best_score = _min_area(best_pts)

    # Phase 2: Local Refinement using Barycentric SLSQP
    # This is the key improvement from the inspiration.
    
    if best_pts is not None and time_left() > 5.0:
        try:
            refined_pts, refined_score = _barycentric_refine(best_pts, time_budget=min(15.0, time_left() - 5.0))
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts
        except Exception:
            pass

    # Phase 3: Final Polish / Verification
    # A very short NM run in Cartesian to catch any last numerical issues.
    if best_pts is not None and time_left() > 2.0:
        try:
            res_final = minimize(
                lambda flat: -_min_area(_clamp_all(flat.reshape(-1, 2))),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 500, 
                    'xatol': 1e-12, 
                    'fatol': 1e-14, 
                    'adaptive': True
                }
            )
            final_pts = _clamp_all(res_final.x.reshape(-1, 2))
            final_score = _min_area(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts
        except Exception:
            pass

    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END
