import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0

# Precompute triangle indices for vectorized area calculation
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])

def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles."""
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def _boundary_penalty(points):
    """
    Penalty for points outside the equilateral triangle.
    Vertices: (0,0), (1,0), (0.5, H)
    Constraints:
    y >= 0
    y <= H
    x >= 0
    x <= 1
    y <= sqrt(3) * x  => sqrt(3)*x - y >= 0
    y <= sqrt(3) * (1-x) => sqrt(3)*(1-x) - y >= 0
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Slanted side 1: y <= sqrt(3) * x
    # Violation: y - sqrt(3)*x > 0
    viol1 = np.maximum(0, y - np.sqrt(3) * x)
    
    # Slanted side 2: y <= sqrt(3) * (1-x)
    # Violation: y - sqrt(3)*(1-x) > 0
    viol2 = np.maximum(0, y - np.sqrt(3) * (1 - x))
    
    # Bottom side: y >= 0 (handled by bounds if using L-BFGS-B, but good for safety)
    viol3 = np.maximum(0, -y)
    
    # Top side: y <= H (handled by bounds)
    # Left/Right x bounds handled by bounds.
    
    # Sum of squared violations
    return np.sum(viol1**2) + np.sum(viol2**2) + np.sum(viol3**2)

def _objective_soft_min(x, beta=20.0):
    """
    Objective for L-BFGS-B: Minimize -soft_min(areas) + penalty.
    soft_min(A) = -1/beta * ln(sum(exp(-beta * A)))
    Minimizing -soft_min is equivalent to maximizing soft_min.
    """
    points = x.reshape(N, 2)
    
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    # LogSumExp trick for stability
    max_area = np.max(areas)
    # soft_min = -1/beta * ln(sum(exp(-beta * (A - max_A)))) - max_A
    # Wait, min(A) approx -1/beta * ln(sum(exp(-beta A)))
    # To be stable:
    # sum(exp(-beta A)) = exp(-beta max_A) * sum(exp(-beta (A - max_A)))
    # ln(sum) = -beta max_A + ln(sum(exp(-beta (A - max_A))))
    # soft_min = -1/beta * (-beta max_A + ln(...)) = max_A - 1/beta * ln(sum(exp(-beta (A - max_A))))
    # We want to MAXIMIZE soft_min, so we MINIMIZE -soft_min.
    
    # Actually, we want to maximize the MINIMUM area.
    # Let's just minimize the negative of the approximation of the min.
    # Approx min = -1/beta * logsumexp(-beta * areas)
    
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    approx_min = -log_sum_exp / beta
    
    penalty = _boundary_penalty(points)
    
    # Return negative approx min (to maximize it) plus penalty
    return -approx_min + 10.0 * penalty

def _objective_true(x):
    """
    Objective for Powell: Minimize -min_area + penalty.
    """
    points = x.reshape(N, 2)
    min_area = _min_triangle_area(points)
    penalty = _boundary_penalty(points)
    return -min_area + 10.0 * penalty

def _is_inside_triangle(p):
    """Check if point p (x,y) is inside the equilateral triangle."""
    x, y = p
    if y < -1e-9 or y > H + 1e-9:
        return False
    if x < -1e-9 or x > 1 + 1e-9:
        return False
    if x < y / np.sqrt(3) - 1e-9:
        return False
    if x > 1 - y / np.sqrt(3) + 1e-9:
        return False
    return True

def _project_to_triangle(p):
    """Project point p to the closest point inside the triangle."""
    x, y = p
    if _is_inside_triangle(p):
        return p
    
    A = np.array([0, 0])
    B = np.array([1, 0])
    C = np.array([0.5, H])
    
    def closest_point_on_segment(p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        return a + t * ab

    p_ab = closest_point_on_segment(p, A, B)
    p_bc = closest_point_on_segment(p, B, C)
    p_ca = closest_point_on_segment(p, C, A)
    
    d_ab = np.linalg.norm(p - p_ab)
    d_bc = np.linalg.norm(p - p_bc)
    d_ca = np.linalg.norm(p - p_ca)
    
    if d_ab == min(d_ab, d_bc, d_ca):
        return p_ab
    elif d_bc == min(d_ab, d_bc, d_ca):
        return p_bc
    else:
        return p_ca

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s buffer
    
    best_score = -np.inf
    best_points = None
    
    bounds = [(0, 1), (0, H)] * N
    
    # Initial configurations
    
    # 1. Symmetric Hexagonal-like
    def init_symmetric():
        pts = []
        # Center
        pts.append([0.5, H/3])
        # Inner ring
        r1 = 0.15
        for k in range(6):
            angle = np.pi/3 * k
            x = 0.5 + r1 * np.cos(angle)
            y = H/3 + r1 * np.sin(angle)
            pts.append([x, y])
        # Outer ring
        r2 = 0.3
        for k in range(3):
            angle = np.pi/3 * k + np.pi/6
            x = 0.5 + r2 * np.cos(angle)
            y = H/3 + r2 * np.sin(angle)
            pts.append([x, y])
        # Add 2 more? We have 1+6+3 = 10. Need 11.
        pts.append([0.5, H/3 + 0.05]) # Perturb center slightly? Or add a corner?
        # Let's just add a point near a corner
        pts.append([0.1, 0.1])
        
        # Project all to ensure validity
        pts = np.array([_project_to_triangle(p) for p in pts])
        return pts.flatten()

    # 2. Grid
    def init_grid():
        pts = []
        # 3x3 grid in the bounding box, projected
        for i in range(3):
            for j in range(3):
                x = 0.2 + i * 0.3
                y = 0.1 + j * 0.3
                pts.append([x, y])
        # Add 2 more
        pts.append([0.5, 0.5])
        pts.append([0.5, 0.2])
        
        pts = np.array([_project_to_triangle(p) for p in pts])
        return pts.flatten()

    # 3. Random
    def init_random(rng):
        def random_point():
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1.0:
                    x = u + 0.5 * v
                    y = H * v
                    return np.array([x, y])
        pts = np.array([random_point() for _ in range(N)])
        return pts.flatten()

    inits = [
        init_symmetric(),
        init_grid(),
        init_random(np.random.default_rng(42)),
        init_random(np.random.default_rng(123))
    ]
    
    for x0 in inits:
        if time.time() > deadline:
            break
            
        # Stage 1: L-BFGS-B on soft min
        try:
            res1 = minimize(_objective_soft_min, x0, args=(20.0,), method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10})
            x_current = res1.x
        except Exception:
            x_current = x0
            
        # Stage 2: Powell on true min
        try:
            res2 = minimize(_objective_true, x_current, method='Powell', 
                            bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
            x_final = res2.x
        except Exception:
            x_final = x_current
            
        # Verify and score
        final_points = x_final.reshape(N, 2)
        valid = True
        for p in final_points:
            if not _is_inside_triangle(p):
                valid = False
                break
        
        if valid:
            score = _min_triangle_area(final_points)
            if score > best_score:
                best_score = score
                best_points = final_points
        else:
            # If invalid, project and re-score
            proj_points = np.array([_project_to_triangle(p) for p in final_points])
            if np.all([_is_inside_triangle(p) for p in proj_points]):
                 score = _min_triangle_area(proj_points)
                 if score > best_score:
                     best_score = score
                     best_points = proj_points

    if best_points is None:
        best_points = np.array([_project_to_triangle(init_random(np.random.default_rng(0))[i*2:i*2+2]) for i in range(N)])
        
    return best_points
