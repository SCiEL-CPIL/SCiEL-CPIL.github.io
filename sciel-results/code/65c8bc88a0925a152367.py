"""Heilbronn triangles: place 11 points to maximize min triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0
_TOTAL_TIME_LIMIT = 140.0  # safety margin before 150s kill

# Precompute all C(11,3) = 165 triangle index triples
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points.
    pts has shape (N, 2)."""
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return np.min(np.abs(cross)) / 2.0


def _project_to_triangle(pts):
    """Project points to be inside the equilateral triangle (0,0),(1,0),(0.5, sqrt(3)/2).
    The triangle constraints are:
      x >= 0, y >= 0, y <= sqrt(3)*(1-x)  i.e., y <= sqrt(3) - sqrt(3)*x
    Or equivalently: x >= 0, y >= 0, x + y/sqrt(3) <= 1
    """
    pts = pts.copy()
    # Constraint 1: x >= 0
    pts[:, 0] = np.maximum(pts[:, 0], 0.0)
    # Constraint 2: y >= 0
    pts[:, 1] = np.maximum(pts[:, 1], 0.0)
    # Constraint 3: y <= sqrt(3) * (1 - x), i.e., y/sqrt(3) <= 1 - x, i.e., x + y/sqrt(3) <= 1
    # If violated, project onto the line y = sqrt(3)*(1-x)
    # The line is: sqrt(3)*x + y = sqrt(3)
    # Normal direction: (sqrt(3), 1), normalized: (sqrt(3)/2, 1/2)
    # For point (x,y), distance to line: (sqrt(3)*x + y - sqrt(3)) / 2
    # If positive, project back
    val = np.sqrt(3) * pts[:, 0] + pts[:, 1] - np.sqrt(3)
    mask = val > 0
    if np.any(mask):
        # Project: subtract val/2 * normal_direction
        # normal = (sqrt(3), 1) / 2
        pts[mask, 0] -= val[mask] * np.sqrt(3) / 2
        pts[mask, 1] -= val[mask] / 2
    return pts


def _soft_min_area(pts, beta):
    """Soft-min surrogate: -log-sum-exp(-beta * areas) / beta approximates min area.
    Returns a smooth objective to maximize."""
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = np.abs(cross) / 2.0
    # Use log-sum-exp for numerical stability
    max_area = np.max(areas)
    # soft_min = max_area - (1/beta) * log(sum(exp(-beta*(areas - max_area))))
    # We want to maximize soft_min, so minimize -soft_min
    exp_term = np.exp(-beta * (areas - max_area))
    soft_min = max_area - (1.0 / beta) * np.log(np.sum(exp_term))
    return -soft_min  # negative for minimization


def _objective(x, beta):
    """Objective for L-BFGS-B: flatten x to points, project, compute soft-min."""
    pts = x.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return _soft_min_area(pts, beta)


def _objective_grad(x, beta):
    """Gradient of the soft-min objective (numerical or analytic)."""
    pts = x.reshape(N, 2)
    pts = _project_to_triangle(pts)
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = np.abs(cross) / 2.0
    
    max_area = np.max(areas)
    exp_term = np.exp(-beta * (areas - max_area))
    sum_exp = np.sum(exp_term)
    
    # d(soft_min)/d(areas_i) = -(1/beta) * (-beta * exp_term_i / sum_exp) = exp_term_i / sum_exp
    # Wait: soft_min = max_area - (1/beta)*log(sum_exp)
    # d(soft_min)/d(areas_i) = -delta(max_area, areas_i) + (1/beta)*(1/sum_exp)*(-beta)*exp_term_i
    # = -delta + (-exp_term_i / sum_exp)
    # For maximizing soft_min, we minimize -soft_min
    # d(-soft_min)/d(areas_i) = delta - exp_term_i/sum_exp
    
    # But this is getting complex with the abs and max. Let's use numerical gradient.
    eps = 1e-7
    grad = np.zeros_like(x)
    for k in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[k] += eps
        xm[k] -= eps
        fp = _objective(xp, beta)
        fm = _objective(xm, beta)
        grad[k] = (fp - fm) / (2 * eps)
    return grad


def _symmetric_initial_config():
    """Generate a symmetric initial configuration for 11 points in the triangle."""
    pts = np.zeros((N, 2))
    # Place points in a symmetric pattern
    # Center point
    pts[0] = [0.5, _H / 3]
    
    # 6 points in a hexagonal-like arrangement around center
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r1 = 0.2
    for i in range(6):
        pts[i + 1] = [0.5 + r1 * np.cos(angles[i]), _H / 3 + r1 * np.sin(angles[i]) * 0.8]
    
    # 4 more points near the edges/corners
    pts[7] = [0.15, 0.1]
    pts[8] = [0.85, 0.1]
    pts[9] = [0.5, 0.55]
    pts[10] = [0.3, 0.35]
    
    pts = _project_to_triangle(pts)
    return pts.flatten()


def construct():
    start_time = time.time()
    best_pts = None
    best_val = 0.0
    
    # Try multiple initial configurations
    rng = np.random.RandomState(42)
    
    # Start with symmetric config
    inits = [_symmetric_initial_config()]
    
    # Add some random inits
    for _ in range(3):
        x = rng.rand(N * 2)
        # Map to triangle
        pts = np.zeros((N, 2))
        for i in range(N):
            u, v = x[2*i], x[2*i+1]
            if u + v < 1:
                pts[i] = [u + 0.5*v, _H*v]
            else:
                # Project
                pts[i] = [0.5, _H/3]
        inits.append(pts.flatten())
    
    # Also try a grid-like pattern
    grid_pts = np.zeros((N, 2))
    # 3x3 grid + 2 extra
    positions = [(0.2, 0.1), (0.5, 0.1), (0.8, 0.1),
                 (0.3, 0.3), (0.5, 0.3), (0.7, 0.3),
                 (0.4, 0.5), (0.6, 0.5), (0.5, 0.15),
                 (0.35, 0.45), (0.65, 0.45)]
    for i, (px, py) in enumerate(positions):
        grid_pts[i] = [px, py]
    grid_pts = _project_to_triangle(grid_pts)
    inits.append(grid_pts.flatten())
    
    for init in inits:
        if time.time() - start_time > _TOTAL_TIME_LIMIT - 20:
            break
        
        # Phase 1: L-BFGS-B with soft-min surrogate (increasing beta)
        x = init.copy()
        for beta in [10, 50, 100, 200]:
            if time.time() - start_time > _TOTAL_TIME_LIMIT - 15:
                break
            
            res = minimize(_objective, x, args=(beta,), method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            x = res.x
        
        # Phase 2: Nelder-Mead polish on true objective
        if time.time() - start_time > _TOTAL_TIME_LIMIT - 10:
            break
        
        def true_obj(x):
            pts = _project_to_triangle(x.reshape(N, 2))
            return -_min_triangle_area(pts)
        
        res2 = minimize(true_obj, x, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        x = res2.x
        
        pts = _project_to_triangle(x.reshape(N, 2))
        val = _min_triangle_area(pts)
        
        if val > best_val:
            best_val = val
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: return the symmetric config
        best_pts = _project_to_triangle(_symmetric_initial_config().reshape(N, 2))
    
    return best_pts
