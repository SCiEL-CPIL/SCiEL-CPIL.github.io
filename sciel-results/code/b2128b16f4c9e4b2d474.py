"""Heilbronn triangles: place 11 points to maximize min triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0
TRI_AREA = _H / 2.0  # area of the equilateral triangle with side 1

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
    return np.abs(cross) * 0.5


def _project_to_triangle(x):
    """Project a point to be inside the triangle (0,0), (1,0), (0.5, sqrt(3)/2).
    Returns (x_proj, was_projected)."""
    x, y = x
    # Check if inside triangle using barycentric-like checks
    # Edge 1: y >= 0
    # Edge 2: y <= sqrt(3) * x  (left edge from (0,0) to (0.5, sqrt(3)/2))
    # Edge 3: y <= sqrt(3) * (1 - x)  (right edge from (1,0) to (0.5, sqrt(3)/2))
    
    proj = np.array([x, y], dtype=float)
    was_proj = False
    
    if y < 0:
        proj[1] = 0
        was_proj = True
    if y > _H:
        proj[1] = _H
        was_proj = True
    if y > np.sqrt(3) * x:
        # Project onto left edge
        # Left edge: from (0,0) to (0.5, sqrt(3)/2), direction (0.5, sqrt(3)/2)
        # Parametric: t*(0.5, sqrt(3)/2), t in [0,1]
        # Closest point on line: project (x,y) onto direction
        d = np.array([0.5, _H])
        t = np.dot(proj, d) / np.dot(d, d)
        t = max(0, min(1, t))
        proj = t * d
        was_proj = True
    if y > np.sqrt(3) * (1 - x):
        # Project onto right edge
        # Right edge: from (1,0) to (0.5, sqrt(3)/2), direction (-0.5, sqrt(3)/2)
        # Point on edge: (1,0) + s*(-0.5, sqrt(3)/2), s in [0,1]
        d = np.array([-0.5, _H])
        v = proj - np.array([1.0, 0.0])
        s = np.dot(v, d) / np.dot(d, d)
        s = max(0, min(1, s))
        proj = np.array([1.0, 0.0]) + s * d
        was_proj = True
    
    # Also clamp x to [0, 1]
    if proj[0] < 0:
        proj[0] = 0
        was_proj = True
    if proj[0] > 1:
        proj[0] = 1
        was_proj = True
    
    return proj, was_proj


def _min_triangle_area_projected(x_flat):
    """Compute min triangle area after projecting points to triangle.
    x_flat has shape (2*N,)."""
    pts = x_flat.reshape(N, 2)
    projected = []
    for i in range(N):
        proj, _ = _project_to_triangle(pts[i])
        projected.append(proj)
    projected = np.array(projected)
    return _min_triangle_area(projected)


def _softmin(x, beta):
    """Soft minimum using log-sum-exp trick."""
    x = np.asarray(x)
    return -np.log(np.sum(np.exp(-beta * x))) / beta


def _objective_softmin(x_flat, beta):
    """Soft-min surrogate objective: negative of soft-min of triangle areas.
    We minimize this, so we want to maximize the soft-min."""
    pts = x_flat.reshape(N, 2)
    projected = np.array([_project_to_triangle(pts[i])[0] for i in range(N)])
    pi = projected[_idx_i]
    pj = projected[_idx_j]
    pk = projected[_idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = np.abs(cross) * 0.5
    return -_softmin(areas, beta)


def _objective_true(x_flat):
    """True objective: negative of min triangle area."""
    pts = x_flat.reshape(N, 2)
    projected = np.array([_project_to_triangle(pts[i])[0] for i in range(N)])
    return -_min_triangle_area(projected)


def _symmetric_initial_config():
    """Generate a symmetric initial configuration with 11 points."""
    pts = []
    # Center
    pts.append([0.5, _H / 3.0])
    
    # 6 points in a hexagonal pattern around center
    r = 0.25
    for k in range(6):
        angle = np.pi / 3 * k
        x = 0.5 + r * np.cos(angle)
        y = _H / 3.0 + r * np.sin(angle)
        # Project to triangle
        proj, _ = _project_to_triangle([x, y])
        pts.append(proj)
    
    # 4 more points
    r2 = 0.4
    for k in range(4):
        angle = np.pi / 4 * k + np.pi / 8
        x = 0.5 + r2 * np.cos(angle)
        y = _H / 3.0 + r2 * np.sin(angle)
        proj, _ = _project_to_triangle([x, y])
        pts.append(proj)
    
    return np.array(pts)


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin
    
    # Start with symmetric config
    best_pts = _symmetric_initial_config()
    best_val = _min_triangle_area(best_pts)
    
    # Multi-start with L-BFGS-B using soft-min surrogate
    rng = np.random.RandomState(42)
    
    # Beta annealing stages
    betas = [5, 20, 50, 100, 200, 500, 1000]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts for this beta
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize: either from best or random
            if restart == 0:
                x0 = best_pts.flatten()
            else:
                # Random points in triangle
                pts = []
                for _ in range(N):
                    u, v = rng.random(), rng.random()
                    if u + v > 1:
                        u, v = 1 - u, 1 - v
                    x = u + 0.5 * v
                    y = _H * v
                    pts.append([x, y])
                x0 = np.array(pts).flatten()
            
            # Optimize using L-BFGS-B
            try:
                res = minimize(
                    _objective_softmin,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 5000}
                )
            except Exception:
                continue
            
            x_opt = res.x
            pts_opt = x_opt.reshape(N, 2)
            projected = np.array([_project_to_triangle(pts_opt[i])[0] for i in range(N)])
            val = _min_triangle_area(projected)
            
            if val > best_val:
                best_val = val
                best_pts = projected.copy()
        
        # Polish with SLSQP on true objective
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(
                _objective_true,
                best_pts.flatten(),
                method='SLSQP',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            x_polished = res.x
            pts_polished = x_polished.reshape(N, 2)
            projected = np.array([_project_to_triangle(pts_polished[i])[0] for i in range(N)])
            val = _min_triangle_area(projected)
            if val > best_val:
                best_val = val
                best_pts = projected.copy()
        except Exception:
            pass
    
    # Final verification and return
    final_pts = np.array([_project_to_triangle(best_pts[i])[0] for i in range(N)])
    return final_pts
