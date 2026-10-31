"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute triangle indices for vectorized area computation
# There are C(11, 3) = 165 triangles
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _all_areas(pts):
    """Fully vectorized computation of all triangle areas using broadcasting."""
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

def _soft_min_area(pts, beta=50.0):
    """
    Smooth approximation of min area using log-sum-exp trick.
    This is differentiable and suitable for gradient-based optimizers like L-BFGS-B.
    """
    areas = _all_areas(pts)
    m = np.min(areas)
    if m <= 1e-15:
        return m
    # shifted = -beta * (areas - m)
    # To avoid overflow in exp, we subtract the max of the shifted terms (which is 0)
    # sum(exp(-beta*(a_i - m))) = exp(-beta*(a_max - m)) * sum(exp(-beta*(a_i - a_max)))
    # Since a_max >= a_i, a_max - m >= 0.
    # Actually, simpler:
    # soft_min = -1/beta * log(sum(exp(-beta * a_i)))
    # This is numerically unstable if a_i are small.
    # Let's use the shift by min:
    # log(sum(exp(-beta * (a_i - m)))) = log(exp(0) + sum(exp(-beta*(a_i-m))))
    # = log(1 + sum(exp(-beta*(a_i-m))))
    # soft_min = m - 1/beta * log(1 + sum(exp(-beta*(a_i-m))))
    
    shifted = -beta * (areas - m)
    # shifted values are <= 0. exp(shifted) is <= 1.
    # Sum is between 1 and 165.
    log_sum = np.log(1.0 + np.sum(np.exp(shifted)))
    soft_min_val = m - log_sum / beta
    return soft_min_val

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle using barycentric coordinates."""
    x, y = p
    # Convert to barycentric coordinates (a, b, c)
    # p = a*V0 + b*V1 + c*V2
    # x = b + 0.5c
    # y = c * H  => c = y / H = 2y/sqrt(3)
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    
    eps = 1e-9
    # Clamp barycentric coordinates to [eps, 1-2*eps] to ensure strict interior
    a = max(eps, min(1 - 2*eps, a))
    b = max(eps, min(1 - 2*eps, b))
    c = max(eps, min(1 - 2*eps, c))
    
    # Renormalize
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
    Generate a highly symmetric initial configuration.
    1 center, 3 near vertices, 3 mid-edge, 3 mid-mid.
    This helps SA start in a promising basin.
    """
    pts = []
    # Center
    pts.append(np.array([1/3, H/3]))
    
    # 3 points near vertices (pushed in slightly)
    # Vertex 0: (0,0) -> (0.1, 0.05)
    # Vertex 1: (1,0) -> (0.9, 0.05)
    # Vertex 2: (0.5, H) -> (0.5, H-0.05)
    pts.append(np.array([0.1, 0.05]))
    pts.append(np.array([0.9, 0.05]))
    pts.append(np.array([0.5, H - 0.05]))
    
    # 3 points on edges (midpoints pushed in)
    # Edge 0-1 mid: (0.5, 0) -> (0.5, 0.05)
    # Edge 1-2 mid: (0.75, H/2) -> (0.72, H/2 - 0.02)
    # Edge 2-0 mid: (0.25, H/2) -> (0.28, H/2 - 0.02)
    pts.append(np.array([0.5, 0.05]))
    pts.append(np.array([0.72, H/2 - 0.02]))
    pts.append(np.array([0.28, H/2 - 0.02]))
    
    # 3 points in the "inner triangle" formed by the midpoints
    # This is roughly the medial triangle.
    # We can place them symmetrically around the center, slightly offset.
    # Or just random jitter around center? Let's use a small triangle around center.
    # Center is (1/3, H/3).
    # Let's place 3 points in a small triangle around the center.
    r = 0.05
    for i in range(3):
        angle = i * 2 * np.pi / 3
        px = 1/3 + r * np.cos(angle)
        py = H/3 + r * np.sin(angle)
        pts.append(np.array([px, py]))
        
    return np.array(pts)

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
    T = 0.01 
    T_min = 1e-8
    alpha = 0.9995  # Cooling rate
    
    # Initial step size
    step_size = 0.05 
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            T = T_min * 10 
            step_size = max(0.01, step_size * 0.5)
            
        # Pick a random point to perturb
        idx = rng.integers(N)
        
        # Perturb with Gaussian noise
        noise = rng.normal(0, step_size, size=2)
        new_point = points[idx] + noise
        
        # Clamp the new point
        new_point = _clamp_to_triangle(new_point)
        
        # Update points
        old_point = points[idx].copy()
        points[idx] = new_point
        
        # Evaluate
        new_score = _min_area(points)
        
        # Acceptance criterion: Metropolis
        delta = new_score - current_score
        if delta > 0 or (T > 0 and rng.random() < np.exp(delta / T)):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            # Revert
            points[idx] = old_point
            
        # Cool down
        T *= alpha
        # Slightly reduce step size to allow fine tuning later
        step_size *= 0.9999
        if step_size < 0.005:
            step_size = 0.005

    return best_points, best_score

def _objective_hard(flat_pts):
    """Objective to minimize: negative min area."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_min_area(pts)

def _objective_soft(flat_pts, beta=50.0):
    """Objective to minimize: negative soft min area."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_soft_min_area(pts, beta=beta)

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. We use 140s as a hard deadline to be safe.
    deadline = start_time + 140.0
    
    def time_left():
        return deadline - time.time()
        
    best_pts = None
    best_score = -1.0
    
    # Phase 1: Global Search using Simulated Annealing
    # We run SA from a few diverse random starts and one symmetric start.
    
    seeds = [42, 123, 456, 789, 101]
    
    for i, seed in enumerate(seeds):
        if time_left() < 20.0: # Leave 20s for refinement
            break
            
        # Allocate time to this SA run. 
        time_budget = 20.0
        if i == 0:
            time_budget = 25.0 # First run gets a bit more
            
        # Use symmetric start for the first run, random for others
        if i == 0:
            init_pts = _generate_symmetric_config()
        else:
            init_pts = _generate_initial_config(seed)
            
        try:
            sa_pts, sa_score = _simulated_annealing(init_pts, time_budget, seed=seed + 1000)
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts
        except Exception:
            pass

    # If no points found, use symmetric fallback
    if best_pts is None:
        best_pts = _generate_symmetric_config()
        best_score = _min_area(best_pts)

    # Phase 2: Local Refinement
    # The inspiration suggests that a combination of L-BFGS-B (smooth) and Powell/NM (hard) is best.
    
    if best_pts is not None and time_left() > 10.0:
        # Stage 1: L-BFGS-B on soft objective
        # This uses gradients to quickly converge to the local optimum of the smooth approximation.
        try:
            res_lbfgs = minimize(
                lambda x: _objective_soft(x, beta=50.0),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={
                    'maxiter': 200, 
                    'ftol': 1e-12, 
                    'gtol': 1e-10
                }
            )
            lbfgs_pts = _clamp_all(res_lbfgs.x.reshape(-1, 2))
            lbfgs_score = _min_area(lbfgs_pts)
            if lbfgs_score > best_score:
                best_score = lbfgs_score
                best_pts = lbfgs_pts
        except Exception:
            pass

    if best_pts is not None and time_left() > 5.0:
        # Stage 2: Powell on hard objective
        # Powell is derivative-free and often handles the non-smoothness of min() better than NM.
        try:
            res_powell = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='Powell',
                options={
                    'maxiter': 500, 
                    'ftol': 1e-12, 
                    'xtol': 1e-10
                }
            )
            powell_pts = _clamp_all(res_powell.x.reshape(-1, 2))
            powell_score = _min_area(powell_pts)
            if powell_score > best_score:
                best_score = powell_score
                best_pts = powell_pts
        except Exception:
            pass

    if best_pts is not None and time_left() > 5.0:
        # Stage 3: Nelder-Mead on hard objective
        # Final polish with NM.
        try:
            res_nm = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000, 
                    'xatol': 1e-10, 
                    'fatol': 1e-12, 
                    'adaptive': True
                }
            )
            nm_pts = _clamp_all(res_nm.x.reshape(-1, 2))
            nm_score = _min_area(nm_pts)
            if nm_score > best_score:
                best_score = nm_score
                best_pts = nm_pts
        except Exception:
            pass

    # Phase 3: Final Polish / Verification
    # Sometimes a second round of soft/hard refinement helps if the first one got stuck.
    if best_pts is not None and time_left() > 5.0:
        try:
            # Another L-BFGS-B with higher beta (closer to hard min)
            res_lbfgs2 = minimize(
                lambda x: _objective_soft(x, beta=100.0),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={
                    'maxiter': 100, 
                    'ftol': 1e-14, 
                    'gtol': 1e-12
                }
            )
            lbfgs2_pts = _clamp_all(res_lbfgs2.x.reshape(-1, 2))
            lbfgs2_score = _min_area(lbfgs2_pts)
            if lbfgs2_score > best_score:
                best_score = lbfgs2_score
                best_pts = lbfgs2_pts
        except Exception:
            pass

    if best_pts is not None and time_left() > 2.0:
        try:
            # Final NM
            res_nm2 = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 500, 
                    'xatol': 1e-12, 
                    'fatol': 1e-14, 
                    'adaptive': True
                }
            )
            final_pts = _clamp_all(res_nm2.x.reshape(-1, 2))
            final_score = _min_area(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts
        except Exception:
            pass

    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END
