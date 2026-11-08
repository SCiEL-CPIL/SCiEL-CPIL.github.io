"""Heilbronn triangles: place 11 points to maximize min triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0

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
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _project_to_triangle(pts):
    """Project points onto the triangle (0,0), (1,0), (0.5, sqrt(3)/2).
    Uses barycentric coordinate clipping."""
    h = _H
    # Barycentric coordinates relative to vertices A=(0,0), B=(1,0), C=(0.5,h)
    # For point (x,y):
    # b = y / h  (weight of C)
    # a = (x - 0.5*b) / 1  ... actually let's do it properly
    # Using formula: for triangle with vertices v0, v1, v2
    # barycentric coords can be computed as:
    # Let A=(0,0), B=(1,0), C=(0.5, h)
    # For point P=(x,y):
    #   w_C = y / h
    #   w_B = (x - 0.5 * w_C) / 1  ... no
    # Proper approach:
    #   w_C = y / h
    #   w_B = (x - 0.5 * w_C) / (1 - 0.5)  ... no
    # Let me use the standard formula.
    # Area of ABC = h/2
    # w_A = area(PBC)/area(ABC), w_B = area(APC)/area(ABC), w_C = area(ABP)/area(ABC)
    # w_C = y / h  (since ABP has base 1 and height y, area = y/2, ABC area = h/2)
    # w_B = (x - 0.5*w_C) ... let's compute directly
    # w_A = 1 - w_B - w_C
    # w_B: area of APC / area ABC
    # APC: A=(0,0), P=(x,y), C=(0.5,h)
    # area = 0.5*|x*h - y*0.5| = 0.5*|x*h - 0.5*y|
    # Since P is inside, x*h - 0.5*y >= 0 typically
    # w_B = (x*h - 0.5*y) / h = x - 0.5*y/h = x - 0.5*w_C
    # w_A = 1 - w_B - w_C = 1 - x + 0.5*w_C - w_C = 1 - x - 0.5*w_C
    
    w_C = pts[:, 1] / h
    w_B = pts[:, 0] - 0.5 * w_C
    w_A = 1.0 - w_B - w_C
    
    # Clip to [0, 1] and renormalize
    w = np.stack([w_A, w_B, w_C], axis=1)
    w = np.clip(w, 0.0, None)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum > 0, w_sum, 1.0)
    w = w / w_sum
    
    # Convert back to Cartesian
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, h])
    pts_proj = w[:, 0:1] * A + w[:, 1:2] * B + w[:, 2:3] * C
    return pts_proj


def _soft_min_objective(flat_pts, beta=20.0):
    """Soft-min surrogate: -logsumexp(beta * areas) / beta approximates -min(areas).
    We want to maximize min area, so minimize -min area.
    Using soft min: softmin(areas) = -1/beta * log(sum(exp(-beta*areas)))
    We minimize negative softmin = 1/beta * log(sum(exp(-beta*areas)))
    But that's for minimizing. We want to maximize min, so we minimize -softmax... 
    Actually, let's just use: objective = -soft_min(areas, beta)
    soft_min(x) = -1/beta * log(sum(exp(-beta*x_i)))
    So -soft_min = 1/beta * log(sum(exp(-beta*x_i)))
    Minimizing this is like minimizing a smooth approximation of -min(x)."""
    pts = flat_pts.reshape(N, 2)
    pts = _project_to_triangle(pts)
    
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Use log-sum-exp trick for numerical stability
    max_area = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_area)))) + (-beta * max_area)
    soft_min_val = -log_sum / beta
    return -soft_min_val


def _true_objective(flat_pts):
    """True objective: negative of min triangle area (to minimize)."""
    pts = flat_pts.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return -_min_triangle_area(pts)


def _random_initial_points(rng, n=N):
    """Generate random points inside the triangle."""
    pts = []
    for _ in range(n):
        u = rng.random()
        v = rng.random()
        while u + v >= 1.0:
            u = rng.random()
            v = rng.random()
        x = u + 0.5 * v
        y = _H * v
        pts.append([x, y])
    return np.array(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    best_pts = None
    best_val = -np.inf
    
    # Symmetric initial configuration: use a structured grid
    # Place points at vertices, edge midpoints, and centroid-like positions
    # For 11 points in an equilateral triangle, a good starting layout:
    # Row 0 (bottom): 4 points
    # Row 1: 3 points  
    # Row 2: 2 points
    # Row 3 (top): 1 point
    # Plus 1 at center
    
    def _structured_initial():
        pts = []
        # Bottom edge: 4 points at x = 0, 1/3, 2/3, 1
        for i in range(4):
            pts.append([i / 3.0, 0.0])
        # Second row: 3 points
        y1 = _H * 0.25
        for i in range(3):
            x_left = 0.25 + i / 3.0 * 0.5
            pts.append([x_left, y1])
        # Third row: 2 points
        y2 = _H * 0.5
        for i in range(2):
            x_left = 0.5 + i / 2.0 * 0.0  # centered
            pts.append([0.5 - 0.15 + i * 0.3, y2])
        # Top: 1 point
        pts.append([0.5, _H * 0.75])
        # Center
        pts.append([0.5, _H * 0.333])
        return np.array(pts[:N])
    
    init_pts = _structured_initial()
    init_pts = _project_to_triangle(init_pts)
    best_pts = init_pts.copy()
    best_val = _min_triangle_area(best_pts)
    
    # Try multiple random initializations and L-BFGS-B optimization
    rng = np.random.RandomState(42)
    
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            x0 = init_pts.flatten()
        else:
            # Random initialization with some structure
            x0 = _random_initial_points(rng).flatten()
        
        # First pass: L-BFGS-B with soft-min surrogate
        try:
            res1 = minimize(
                _soft_min_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res1.success or res1.fun < best_val:
                pts1 = _project_to_triangle(res1.x.reshape(N, 2))
                val1 = _min_triangle_area(pts1)
                if val1 > best_val:
                    best_val = val1
                    best_pts = pts1.copy()
        except Exception:
            pass
        
        if time.time() > deadline:
            break
        
        # Second pass: Nelder-Mead polish on true objective
        try:
            x1 = res1.x if 'res1' in dir() else x0
            res2 = minimize(
                _true_objective,
                x1,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            pts2 = _project_to_triangle(res2.x.reshape(N, 2))
            val2 = _min_triangle_area(pts2)
            if val2 > best_val:
                best_val = val2
                best_pts = pts2.copy()
        except Exception:
            pass
    
    # Final check
    if best_pts is None:
        best_pts = init_pts
    
    return best_pts
