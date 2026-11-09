import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 11
h = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, h])

# Precompute all triplets for vectorized area computation
_TRIPLETS = np.array(list(itertools.combinations(range(N), 3)), dtype=np.int32)  # shape (165, 3)

def project_to_triangle(p):
    """
    p: array of shape (N, 2)
    Returns projected points inside the triangle (0,0), (1,0), (0.5, h)
    """
    p = p.copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, h)
    
    # Edge AB: y >= 0
    p[:, 1] = np.maximum(p[:, 1], 0.0)
    
    # Edge AC: y <= sqrt(3)*x
    sqrt3 = np.sqrt(3)
    mask_ac = p[:, 1] > sqrt3 * p[:, 0]
    if np.any(mask_ac):
        px = p[mask_ac, 0]
        py = p[mask_ac, 1]
        dot = px + py * sqrt3
        px_proj = dot / 4.0
        py_proj = dot * sqrt3 / 4.0
        px_proj = np.clip(px_proj, 0.0, 0.5)
        py_proj = px_proj * sqrt3
        p[mask_ac, 0] = px_proj
        p[mask_ac, 1] = py_proj
    
    # Edge BC: y <= sqrt(3)*(1-x)
    mask_bc = p[:, 1] > sqrt3 * (1.0 - p[:, 0])
    if np.any(mask_bc):
        px = p[mask_bc, 0]
        py = p[mask_bc, 1]
        d = (sqrt3 * (px - 1.0) + py) / 4.0
        px_proj = px - d * sqrt3
        py_proj = py - d
        px_proj = np.clip(px_proj, 0.5, 1.0)
        py_proj = sqrt3 * (1.0 - px_proj)
        p[mask_bc, 0] = px_proj
        p[mask_bc, 1] = py_proj
        
    return p

def min_triangle_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    Vectorized using precomputed triplets.
    """
    # _TRIPLETS has shape (165, 3)
    p_i = points[_TRIPLETS[:, 0]]  # (165, 2)
    p_j = points[_TRIPLETS[:, 1]]  # (165, 2)
    p_k = points[_TRIPLETS[:, 2]]  # (165, 2)
    
    v12 = p_j - p_i  # (165, 2)
    v13 = p_k - p_i  # (165, 2)
    
    cross = v12[:, 0] * v13[:, 1] - v12[:, 1] * v13[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))

def objective_flat(x):
    """
    x: flattened array of shape (2*N,)
    Returns negative min area (to be minimized).
    """
    points = x.reshape(N, 2)
    points = project_to_triangle(points)
    return -min_triangle_area(points)

def objective_penalized(x):
    """
    x: flattened array of shape (2*N,)
    Returns a penalized objective for L-BFGS-B (no hard projection, smooth penalty).
    """
    points = x.reshape(N, 2)
    # Soft penalty for being outside triangle
    sqrt3 = np.sqrt(3)
    # Constraints: y >= 0, y <= sqrt3*x, y <= sqrt3*(1-x)
    viol1 = np.maximum(0, -points[:, 1])  # y < 0
    viol2 = np.maximum(0, points[:, 1] - sqrt3 * points[:, 0])  # y > sqrt3*x
    viol3 = np.maximum(0, points[:, 1] - sqrt3 * (1.0 - points[:, 0]))  # y > sqrt3*(1-x)
    penalty = 10.0 * np.sum(viol1**2 + viol2**2 + viol3**2)
    min_a = min_triangle_area(points)
    return penalty - min_a

def _local_refine(x0, deadline):
    """Run L-BFGS-B on a flattened starting point, return best area and points."""
    try:
        res = minimize(objective_penalized, x0, method='L-BFGS-B',
                       options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12})
        pts = project_to_triangle(res.x.reshape(N, 2))
        area = min_triangle_area(pts)
        return area, pts
    except Exception:
        return -1, None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # --- Phase 1: Differential Evolution (global search) ---
    # Bounds: x in [0,1], y in [0, h]
    bounds = [(0.0, 1.0)] * N + [(0.0, h)] * N
    
    # Run DE with moderate population
    try:
        de_res = differential_evolution(
            objective_flat,
            bounds=bounds,
            seed=42,
            maxiter=30,
            popsize=15,
            tol=1e-6,
            mutation=(0.3, 0.9),
            recombination=0.7,
            init='sobol',
            polish=True,
            polish_options={'maxiter': 100, 'method': 'L-BFGS-B'}
        )
        if de_res.success or de_res.fun < 0:
            pts = project_to_triangle(de_res.x.reshape(N, 2))
            area = min_triangle_area(pts)
            if area > best_score:
                best_score = area
                best_points = pts
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_points is not None:
            return best_points
        # Fallback
        pts = np.random.default_rng(7).random((N, 2)) * np.array([1.0, h])
        return project_to_triangle(pts)
    
    # --- Phase 2: Multi-start L-BFGS-B from diverse initial points ---
    # Generate diverse starting points
    starts = []
    
    # Symmetric grid layouts
    p_top = np.array([0.5, h])
    y3 = 0.75 * h
    p_r3 = np.array([[0.4, y3], [0.6, y3]])
    y2 = 0.5 * h
    p_r2 = np.array([[0.3, y2], [0.5, y2], [0.7, y2]])
    y1 = 0.25 * h
    x1s = np.linspace(0.15, 0.85, 5)
    p_r1 = np.array(list(zip(x1s, np.full(5, y1))))
    guess1 = np.vstack([p_top, p_r3, p_r2, p_r1])
    starts.append(guess1.flatten())
    
    # Jittered
    starts.append((guess1 + rng.normal(0, 0.05, guess1.shape)).flatten())
    
    # Different layout: 4-4-3
    y_a = 0.2 * h
    xa = np.linspace(0.1, 0.9, 4)
    y_b = 0.5 * h
    xb = np.linspace(0.2, 0.8, 4)
    y_c = 0.8 * h
    xc = np.linspace(0.35, 0.65, 3)
    guess2 = np.vstack([np.array(list(zip(xa, np.full(4, y_a)))),
                        np.array(list(zip(xb, np.full(4, y_b)))),
                        np.array(list(zip(xc, np.full(3, y_c))))])
    starts.append(guess2.flatten())
    starts.append((guess2 + rng.normal(0, 0.04, guess2.shape)).flatten())
    
    # Random starts
    for _ in range(8):
        rp = rng.random((N, 2)) * np.array([1.0, h])
        starts.append(project_to_triangle(rp).flatten())
    
    # Corner-inclusive starts
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
    midpoints = np.array([[0.5, 0.0], [0.75, h/2], [0.25, h/2]])
    center = np.array([[0.5, h/3]])
    # Add 4 more interior points
    extra = rng.random((4, 2)) * np.array([0.6, 0.5*h]) + np.array([0.2, 0.1*h])
    guess3 = np.vstack([corners, midpoints, center, extra])
    starts.append(project_to_triangle(guess3).flatten())
    
    # Run local refinement on each start
    for x0 in starts:
        if time.time() > deadline - 3.0:
            break
        area, pts = _local_refine(x0, deadline)
        if pts is not None and area > best_score:
            best_score = area
            best_points = pts
    
    # --- Phase 3: Fine-tune best solution with tighter L-BFGS-B ---
    if best_points is not None and time.time() < deadline - 2.0:
        x_best = best_points.flatten()
        try:
            res = minimize(objective_penalized, x_best, method='L-BFGS-B',
                           options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14})
            pts = project_to_triangle(res.x.reshape(N, 2))
            area = min_triangle_area(pts)
            if area > best_score:
                best_score = area
                best_points = pts
        except Exception:
            pass
    
    # --- Phase 4: If time remains, do a few more random L-BFGS-B restarts ---
    if best_points is not None and time.time() < deadline - 2.0:
        rng2 = np.random.default_rng(999)
        while time.time() < deadline - 1.0:
            rp = rng2.random((N, 2)) * np.array([1.0, h])
            x0 = project_to_triangle(rp).flatten()
            area, pts = _local_refine(x0, deadline)
            if pts is not None and area > best_score:
                best_score = area
                best_points = pts
    
    if best_points is None:
        # Fallback
        pts = np.random.default_rng(7).random((N, 2)) * np.array([1.0, h])
        best_points = project_to_triangle(pts)
    
    return best_points

# EVOLVE-BLOCK-END
