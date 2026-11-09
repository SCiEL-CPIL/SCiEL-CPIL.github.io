import numpy as np
from scipy.optimize import minimize, differential_evolution, linprog
import time
import itertools

N = 11
h = np.sqrt(3) / 2
sqrt3 = np.sqrt(3)
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, h])

# Precompute all triplets for vectorized area computation
_TRIPLETS = np.array(list(itertools.combinations(range(N), 3)), dtype=np.int32)  # shape (165, 3)

def bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian. bary: (N, 3)"""
    # p = b0*A + b1*B + b2*C, where A=(0,0), B=(1,0), C=(0.5, h)
    return np.column_stack([bary[:, 1] + 0.5 * bary[:, 2], h * bary[:, 2]])

def cart_to_bary(pts):
    """Convert Cartesian to barycentric. pts: (N, 2)"""
    # x = b1 + 0.5*b2, y = h*b2
    b2 = pts[:, 1] / h
    b1 = pts[:, 0] - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])

def project_to_triangle(p):
    """Project points to be inside the triangle."""
    p = p.copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, h)
    
    mask_ac = p[:, 1] > sqrt3 * p[:, 0]
    if np.any(mask_ac):
        px = p[mask_ac, 0]
        py = p[mask_ac, 1]
        dot = px + py * sqrt3
        px_proj = dot / 4.0
        py_proj = px_proj * sqrt3
        px_proj = np.clip(px_proj, 0.0, 0.5)
        py_proj = px_proj * sqrt3
        p[mask_ac, 0] = px_proj
        p[mask_ac, 1] = py_proj
    
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
    """Compute minimum triangle area from 11 points."""
    p_i = points[_TRIPLETS[:, 0]]
    p_j = points[_TRIPLETS[:, 1]]
    p_k = points[_TRIPLETS[:, 2]]
    
    v12 = p_j - p_i
    v13 = p_k - p_i
    
    cross = v12[:, 0] * v13[:, 1] - v12[:, 1] * v13[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))

def min_pairwise_dist(points):
    """Compute minimum pairwise distance."""
    dists = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            dists[i, j] = d
            dists[j, i] = d
    return float(np.min(dists))

def objective_bary(bary_flat):
    """Objective in barycentric coordinates: minimize negative min area."""
    bary = bary_flat.reshape(N, 3)
    # Normalize to ensure b0+b1+b2=1
    bary = bary / np.sum(bary, axis=1, keepdims=True)
    # Ensure non-negative
    bary = np.maximum(bary, 0)
    bary = bary / np.sum(bary, axis=1, keepdims=True)
    
    pts = bary_to_cart(bary)
    return -min_triangle_area(pts)

def objective_penalized(x):
    """Penalized objective for Cartesian coordinates."""
    points = x.reshape(N, 2)
    viol1 = np.maximum(0, -points[:, 1])
    viol2 = np.maximum(0, points[:, 1] - sqrt3 * points[:, 0])
    viol3 = np.maximum(0, points[:, 1] - sqrt3 * (1.0 - points[:, 0]))
    penalty = 50.0 * np.sum(viol1**2 + viol2**2 + viol3**2)
    min_a = min_triangle_area(points)
    return penalty - min_a

def _local_refine_cart(x0, deadline):
    """Local refinement in Cartesian space."""
    try:
        res = minimize(objective_penalized, x0, method='L-BFGS-B',
                       options={'maxiter': 300, 'gtol': 1e-9, 'ftol': 1e-13})
        pts = project_to_triangle(res.x.reshape(N, 2))
        area = min_triangle_area(pts)
        return area, pts
    except Exception:
        return -1, None

def _local_refine_bary(bary0, deadline):
    """Local refinement in barycentric space."""
    try:
        res = minimize(objective_bary, bary0, method='L-BFGS-B',
                       options={'maxiter': 300, 'gtol': 1e-9, 'ftol': 1e-13})
        bary = res.x.reshape(N, 3)
        bary = np.maximum(bary, 0)
        bary = bary / np.sum(bary, axis=1, keepdims=True)
        pts = bary_to_cart(bary)
        pts = project_to_triangle(pts)
        area = min_triangle_area(pts)
        return area, pts
    except Exception:
        return -1, None

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # --- Phase 0: Lattice-based initialization ---
    # Use a triangular lattice to get a good initial spread
    # For 11 points, try a 4-4-3 or 3-4-4 arrangement
    lattice_starts = []
    
    # 3-4-4 arrangement (rows from top)
    for row_scale in [0.95, 0.9, 0.85]:
        y_top = row_scale * h
        x_top = 0.5
        row1 = np.array([[x_top, y_top]])
        
        y_mid = 0.5 * row_scale * h
        x_mid = np.linspace(0.15, 0.85, 4) * row_scale + 0.5 * (1 - row_scale)
        row2 = np.array(list(zip(x_mid, np.full(4, y_mid))))
        
        y_bot = 0.15 * row_scale * h
        x_bot = np.linspace(0.1, 0.9, 4) * row_scale + 0.5 * (1 - row_scale)
        row3 = np.array(list(zip(x_bot, np.full(4, y_bot))))
        
        pts = np.vstack([row1, row2, row3])
        pts = project_to_triangle(pts)
        lattice_starts.append(pts.flatten())
    
    # 4-3-4 arrangement
    for row_scale in [0.95, 0.9]:
        y_top = row_scale * h
        x_top = np.linspace(0.2, 0.8, 4) * row_scale + 0.5 * (1 - row_scale)
        row1 = np.array(list(zip(x_top, np.full(4, y_top))))
        
        y_mid = 0.5 * row_scale * h
        x_mid = np.linspace(0.25, 0.75, 3) * row_scale + 0.5 * (1 - row_scale)
        row2 = np.array(list(zip(x_mid, np.full(3, y_mid))))
        
        y_bot = 0.15 * row_scale * h
        x_bot = np.linspace(0.2, 0.8, 4) * row_scale + 0.5 * (1 - row_scale)
        row3 = np.array(list(zip(x_bot, np.full(4, y_bot))))
        
        pts = np.vstack([row1, row2, row3])
        pts = project_to_triangle(pts)
        lattice_starts.append(pts.flatten())
    
    # Evaluate all lattice starts
    for x0 in lattice_starts:
        if time.time() > deadline:
            break
        area, pts = _local_refine_cart(x0, deadline)
        if pts is not None and area > best_score:
            best_score = area
            best_points = pts
    
    # --- Phase 1: Differential Evolution in Cartesian space ---
    bounds = [(0.0, 1.0)] * N + [(0.0, h)] * N
    
    try:
        de_res = differential_evolution(
            objective_penalized,
            bounds=bounds,
            seed=42,
            maxiter=25,
            popsize=12,
            tol=1e-5,
            mutation=(0.4, 0.8),
            recombination=0.8,
            init='sobol',
            polish=True,
            polish_options={'maxiter': 150, 'method': 'L-BFGS-B'}
        )
        if de_res.fun < 0:
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
        pts = np.random.default_rng(7).random((N, 2)) * np.array([1.0, h])
        return project_to_triangle(pts)
    
    # --- Phase 2: Barycentric optimization ---
    # Optimize in barycentric space where constraints are simpler
    if best_points is not None:
        bary_best = cart_to_bary(best_points)
        bary_best = np.maximum(bary_best, 0)
        bary_best = bary_best / np.sum(bary_best, axis=1, keepdims=True)
        
        area, pts = _local_refine_bary(bary_best.flatten(), deadline)
        if pts is not None and area > best_score:
            best_score = area
            best_points = pts
    
    # --- Phase 3: Multi-start refinement ---
    # Generate diverse starts and refine
    starts = []
    
    # Random starts
    for _ in range(12):
        rp = rng.random((N, 2)) * np.array([1.0, h])
        starts.append(project_to_triangle(rp).flatten())
    
    # Jittered lattice starts
    for x0 in lattice_starts[:4]:
        jittered = x0 + rng.normal(0, 0.03, len(x0))
        starts.append(project_to_triangle(jittered.reshape(N, 2)).flatten())
    
    # Corner-inclusive starts
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
    midpoints = np.array([[0.5, 0.0], [0.75, h/2], [0.25, h/2]])
    center = np.array([[0.5, h/3]])
    extra = rng.random((4, 2)) * np.array([0.5, 0.4*h]) + np.array([0.25, 0.1*h])
    guess = np.vstack([corners, midpoints, center, extra])
    starts.append(project_to_triangle(guess).flatten())
    
    for x0 in starts:
        if time.time() > deadline - 3.0:
            break
        area, pts = _local_refine_cart(x0, deadline)
        if pts is not None and area > best_score:
            best_score = area
            best_points = pts
    
    # --- Phase 4: Fine-tune best solution ---
    if best_points is not None and time.time() < deadline - 2.0:
        x_best = best_points.flatten()
        try:
            res = minimize(objective_penalized, x_best, method='L-BFGS-B',
                           options={'maxiter': 500, 'gtol': 1e-11, 'ftol': 1e-15})
            pts = project_to_triangle(res.x.reshape(N, 2))
            area = min_triangle_area(pts)
            if area > best_score:
                best_score = area
                best_points = pts
        except Exception:
            pass
        
        # Also try barycentric fine-tune
        if time.time() < deadline - 2.0:
            bary_best = cart_to_bary(best_points)
            bary_best = np.maximum(bary_best, 0)
            bary_best = bary_best / np.sum(bary_best, axis=1, keepdims=True)
            area, pts = _local_refine_bary(bary_best.flatten(), deadline)
            if pts is not None and area > best_score:
                best_score = area
                best_points = pts
    
    # --- Phase 5: Additional random restarts if time remains ---
    if best_points is not None and time.time() < deadline - 2.0:
        rng2 = np.random.default_rng(999)
        while time.time() < deadline - 1.0:
            rp = rng2.random((N, 2)) * np.array([1.0, h])
            x0 = project_to_triangle(rp).flatten()
            area, pts = _local_refine_cart(x0, deadline)
            if pts is not None and area > best_score:
                best_score = area
                best_points = pts
    
    if best_points is None:
        pts = np.random.default_rng(7).random((N, 2)) * np.array([1.0, h])
        best_points = project_to_triangle(pts)
    
    return best_points
