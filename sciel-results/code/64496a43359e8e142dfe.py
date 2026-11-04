"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 11
S3 = 3 ** 0.5

# Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
# Area of this triangle = sqrt(3)/4
TRI_AREA = S3 / 4.0

# Barycentric coordinates for the triangle
# A point (x, y) is inside if we can write it as alpha*A + beta*B + gamma*C with alpha+beta+gamma=1, all >= 0
# For triangle (0,0), (1,0), (0.5, h) where h = sqrt(3)/2:
# x = beta + 0.5*gamma
# y = h * gamma
# So gamma = y/h, beta = x - 0.5*gamma, alpha = 1 - beta - gamma
H = S3 / 2.0

def to_barycentric(pts):
    """Convert cartesian points to barycentric (alpha, beta, gamma) coordinates.
    Returns array of shape (n, 3) where each row sums to 1."""
    n = len(pts)
    bar = np.zeros((n, 3))
    bar[:, 2] = pts[:, 1] / H  # gamma
    bar[:, 1] = pts[:, 0] - 0.5 * bar[:, 2]  # beta
    bar[:, 0] = 1.0 - bar[:, 1] - bar[:, 2]  # alpha
    return bar

def from_barycentric(bar):
    """Convert barycentric coordinates back to cartesian."""
    pts = np.zeros((len(bar), 2))
    # A=(0,0), B=(1,0), C=(0.5, H)
    pts[:, 0] = bar[:, 1] + 0.5 * bar[:, 2]
    pts[:, 1] = H * bar[:, 2]
    return pts

def min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Area = 0.5 * |cross product|
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                if area < min_area:
                    min_area = area
    return min_area

def min_triangle_area_fast(pts):
    """Faster computation using vectorized operations for small n."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            # Vectorized over k
            diff_j = pts[j] - pts[i]
            diffs_k = pts - pts[i]
            # Cross product: (dx_j, dy_j) x (dx_k, dy_k)
            cross = diff_j[0] * diffs_k[:, 1] - diff_j[1] * diffs_k[:, 0]
            areas = 0.5 * np.abs(cross)
            # Exclude i and j themselves (area would be 0)
            areas[i] = np.inf
            areas[j] = np.inf
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def objective_neg_min_area(x):
    """Objective for minimization: negative of min triangle area (we maximize min area).
    x is in barycentric space, shape (n, 3), but we only use n*2 parameters (drop one barycentric coord per point)."""
    n = N
    # x has shape (n, 2): beta and gamma for each point (alpha = 1 - beta - gamma)
    bar = np.ones((n, 3))
    bar[:, 1] = x[:, 0]
    bar[:, 2] = x[:, 1]
    bar[:, 0] = 1.0 - bar[:, 1] - bar[:, 2]
    
    # Check constraints: all barycentric coords >= 0
    if np.any(bar < -1e-10):
        return 1e6  # Penalty for being outside
    
    # Clip small negatives
    bar = np.clip(bar, 0, None)
    bar = bar / bar.sum(axis=1, keepdims=True)
    
    pts = from_barycentric(bar)
    return -min_triangle_area_fast(pts)

def objective_slsqp(x):
    """Same as above but for SLSQP."""
    n = N
    bar = np.ones((n, 3))
    bar[:, 1] = x[:, 0]
    bar[:, 2] = x[:, 1]
    bar[:, 0] = 1.0 - bar[:, 1] - bar[:, 2]
    
    if np.any(bar < -1e-10):
        return 1e6
    
    bar = np.clip(bar, 0, None)
    bar = bar / bar.sum(axis=1, keepdims=True)
    
    pts = from_barycentric(bar)
    return -min_triangle_area_fast(pts)

def constraint_alpha(x):
    """alpha = 1 - beta - gamma >= 0"""
    n = N
    return 1.0 - x[:, 0] - x[:, 1]

def constraint_beta(x):
    """beta >= 0"""
    return x[:, 0]

def constraint_gamma(x):
    """gamma >= 0"""
    return x[:, 1]

def fps_init(rng, n_points=N):
    """Farthest Point Sampling initialization inside the triangle."""
    # Start with a random point
    pts = []
    u, v = rng.random(), rng.random()
    while u + v > 1.0:
        u, v = rng.random(), rng.random()
    p0 = np.array([u + 0.5 * v, H * v])
    pts.append(p0)
    
    min_dists = np.array([0.0])
    
    for _ in range(n_points - 1):
        # Generate many candidate points
        candidates = []
        for _ in range(5000):
            u, v = rng.random(), rng.random()
            if u + v <= 1.0:
                candidates.append(np.array([u + 0.5 * v, H * v]))
        candidates = np.array(candidates)
        
        # For each candidate, compute min distance to existing points
        best_candidate = None
        best_score = -np.inf
        for c in candidates:
            dists = np.sqrt(np.sum((pts - c)**2, axis=1))
            min_d = np.min(dists)
            if min_d > best_score:
                best_score = min_d
                best_candidate = c
        
        pts.append(best_candidate)
    
    return np.array(pts)

def random_init(rng, n_points=N):
    """Random initialization inside the triangle."""
    pts = []
    while len(pts) < n_points:
        u, v = rng.random(), rng.random()
        if u + v <= 1.0:
            pts.append(np.array([u + 0.5 * v, H * v]))
    return np.array(pts)

def grid_init(n_points=N):
    """Grid-like initialization for the triangle."""
    # Use a structured grid in barycentric coordinates
    # For 11 points, we can use a pattern
    # Let's use a simple approach: place points at vertices, midpoints of edges, and center
    pts = []
    # Vertices
    pts.append(np.array([0.0, 0.0]))
    pts.append(np.array([1.0, 0.0]))
    pts.append(np.array([0.5, H]))
    # Midpoints of edges
    pts.append(np.array([0.5, 0.0]))
    pts.append(np.array([0.75, H/2]))
    pts.append(np.array([0.25, H/2]))
    # Center
    pts.append(np.array([0.5, H/3]))
    # Additional points - use a finer subdivision
    # Points at 1/3 and 2/3 along edges from center
    for i in range(3):
        # This is getting complex; let's just add random-ish structured points
        pass
    
    # Add more points to reach n_points
    while len(pts) < n_points:
        u, v = np.random.random(), np.random.random()
        if u + v <= 1.0:
            p = np.array([u + 0.5 * v, H * v])
            # Check it's not too close to existing
            if all(np.linalg.norm(p - q) > 0.05 for q in pts):
                pts.append(p)
        if len(pts) >= n_points:
            break
    
    return np.array(pts[:n_points])

def optimize_from_initial(pts, method='SLSQP', maxiter=200):
    """Optimize from an initial configuration."""
    n = N
    # Convert to barycentric (beta, gamma) parameters
    bar = to_barycentric(pts)
    x0 = np.column_stack([bar[:, 1], bar[:, 2]]).flatten()
    
    # Bounds: beta in [0, 1], gamma in [0, 1]
    bounds = [(0, 1)] * (2 * n)
    
    # Constraints
    cons = []
    for i in range(n):
        # alpha_i = 1 - beta_i - gamma_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
        # beta_i >= 0 (already in bounds)
        # gamma_i >= 0 (already in bounds)
    
    try:
        result = minimize(
            objective_slsqp,
            x0,
            method=method,
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-8}
        )
        if result.success:
            x_opt = result.x.reshape(n, 2)
            bar_opt = np.ones((n, 3))
            bar_opt[:, 1] = x_opt[:, 0]
            bar_opt[:, 2] = x_opt[:, 1]
            bar_opt[:, 0] = 1.0 - bar_opt[:, 1] - bar_opt[:, 2]
            bar_opt = np.clip(bar_opt, 0, None)
            bar_opt = bar_opt / bar_opt.sum(axis=1, keepdims=True)
            pts_opt = from_barycentric(bar_opt)
            return pts_opt, -result.fun
        else:
            return pts, min_triangle_area_fast(pts)
    except Exception:
        return pts, min_triangle_area_fast(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_pts = None
    best_area = 0.0
    
    # Try multiple initializations and optimization methods
    rng = np.random.RandomState(42)
    
    # Different initial configurations
    inits = []
    
    # 1. FPS initialization
    try:
        pts_fps = fps_init(rng)
        inits.append(pts_fps)
    except Exception:
        pass
    
    # 2. Grid initialization
    try:
        pts_grid = grid_init()
        inits.append(pts_grid)
    except Exception:
        pass
    
    # 3. Random initializations
    for _ in range(5):
        try:
            pts_rand = random_init(rng)
            inits.append(pts_rand)
        except Exception:
            pass
    
    # 4. Symmetric initialization based on known good patterns
    # For Heilbronn triangles, good configurations often have symmetry
    # Let's try a configuration with 3-fold symmetry
    try:
        pts_sym = []
        # Center
        pts_sym.append(np.array([0.5, H/3]))
        # Three points at 1/3 height, equally spaced
        for angle in [0, 2*np.pi/3, 4*np.pi/3]:
            r = 0.2
            x = 0.5 + r * np.cos(angle)
            y = H/3 + r * np.sin(angle) * (H / 0.5)  # Adjust for aspect ratio
            # Check if inside triangle
            bar = to_barycentric(np.array([[x, y]]))[0]
            if np.all(bar >= -0.01):
                pts_sym.append(np.array([x, y]))
        # Three more points closer to vertices
        for angle in [np.pi/3, np.pi, 5*np.pi/3]:
            r = 0.25
            x = 0.5 + r * np.cos(angle)
            y = H/2 + r * np.sin(angle) * (H / 0.5)
            bar = to_barycentric(np.array([[x, y]]))[0]
            if np.all(bar >= -0.01):
                pts_sym.append(np.array([x, y]))
        # Fill remaining with random
        while len(pts_sym) < N:
            u, v = rng.random(), rng.random()
            if u + v <= 1.0:
                p = np.array([u + 0.5 * v, H * v])
                pts_sym.append(p)
        inits.append(np.array(pts_sym[:N]))
    except Exception:
        pass
    
    # Evaluate all initial configurations
    for pts in inits:
        if len(pts) != N:
            continue
        area = min_triangle_area_fast(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    # Optimize from the best initial configurations
    # Sort by initial area, best first
    init_areas = [(min_triangle_area_fast(p), i) for i, p in enumerate(inits) if len(p) == N]
    init_areas.sort(key=lambda x: -x[0])
    
    for area, idx in init_areas[:5]:  # Try top 5 initializations
        if time.time() > deadline - 5.0:
            break
        
        pts = inits[idx]
        try:
            pts_opt, area_opt = optimize_from_initial(pts, method='SLSQP', maxiter=100)
            if area_opt > best_area:
                best_area = area_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    # Try L-BFGS-B as well
    for area, idx in init_areas[:3]:
        if time.time() > deadline - 5.0:
            break
        
        pts = inits[idx]
        try:
            n = N
            bar = to_barycentric(pts)
            x0 = np.column_stack([bar[:, 1], bar[:, 2]]).flatten()
            bounds = [(0, 1)] * (2 * n)
            
            result = minimize(
                objective_neg_min_area,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            if result.fun < -1e-10:
                x_opt = result.x.reshape(n, 2)
                bar_opt = np.ones((n, 3))
                bar_opt[:, 1] = x_opt[:, 0]
                bar_opt[:, 2] = x_opt[:, 1]
                bar_opt[:, 0] = 1.0 - bar_opt[:, 1] - bar_opt[:, 2]
                bar_opt = np.clip(bar_opt, 0, None)
                bar_opt = bar_opt / bar_opt.sum(axis=1, keepdims=True)
                pts_opt = from_barycentric(bar_opt)
                area_opt = min_triangle_area_fast(pts_opt)
                if area_opt > best_area:
                    best_area = area_opt
                    best_pts = pts_opt.copy()
        except Exception:
            pass
    
    # Final local refinement with small perturbations
    if best_pts is not None and time.time() < deadline - 10.0:
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            # Perturb each point slightly
            pts_pert = best_pts.copy()
            for i in range(N):
                dx = rng.uniform(-0.02, 0.02)
                dy = rng.uniform(-0.02, 0.02)
                pts_pert[i] = best_pts[i] + np.array([dx, dy])
                # Project back to triangle if needed
                bar = to_barycentric(pts_pert[i:i+1])[0]
                if np.any(bar < -0.01):
                    # Scale back toward center
                    center = np.array([0.5, H/3])
                    pts_pert[i] = 0.8 * pts_pert[i] + 0.2 * center
                    bar = to_barycentric(pts_pert[i:i+1])[0]
                    if np.any(bar < -0.01):
                        pts_pert[i] = center
            
            area_pert = min_triangle_area_fast(pts_pert)
            if area_pert > best_area:
                best_area = area_pert
                best_pts = pts_pert.copy()
    
    if best_pts is None:
        # Fallback: return a valid configuration
        best_pts = random_init(np.random.RandomState(0))
    
    return best_pts
