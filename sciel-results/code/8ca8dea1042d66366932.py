"""Heilbronn triangles: maximize smallest triangle area among 11 points in unit equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)
TRI_AREA = S3 / 4.0  # area of (0,0),(1,0),(0.5,sqrt(3)/2)

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3 / 2.0])


def _inside_triangle(p, eps=1e-9):
    """Check if point p is inside or on the triangle."""
    def sign(a, b, c):
        return (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])
    d1 = sign(p, V0, V1)
    d2 = sign(p, V1, V2)
    d3 = sign(p, V2, V0)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _soft_min_area(pts, beta=20.0):
    """Smooth approximation of the minimum triangle area using soft-min."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                areas.append(area)
    areas = np.array(areas)
    # Soft minimum: -1/beta * log(sum(exp(-beta * x)))
    shifted = areas - areas.min()
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = areas.min() - np.log(np.sum(np.exp(-beta * (areas - areas.min())))) / beta
    return soft_min


def _objective_flat(x, beta=20.0):
    """Objective: negative soft-min area (to minimize)."""
    pts = x.reshape(-1, 2)
    # Penalty for points outside triangle
    penalty = 0.0
    for p in pts:
        if not _inside_triangle(p, eps=0.0):
            # Compute distance to triangle boundary (approximate)
            # Use barycentric coordinates
            v0 = V2 - V0
            v1 = V1 - V0
            v2 = p - V0
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d02 = np.dot(v0, v2)
            d11 = np.dot(v1, v1)
            d12 = np.dot(v1, v2)
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-15:
                v = (d11 * d02 - d01 * d12) / denom
                w = (d00 * d12 - d01 * d02) / denom
            else:
                v = (d11 * d02 - d01 * d12) / denom
                w = (d00 * d12 - d01 * d02) / denom
            if v < 0 or w < 0 or v + w > 1:
                penalty += 100.0 * max(0, -v) ** 2 + 100.0 * max(0, -w) ** 2 + 100.0 * max(0, v + w - 1) ** 2
    return -_soft_min_area(pts, beta) + penalty


def _random_init(rng, n=N):
    """Random points inside triangle with margin."""
    pts = []
    for _ in range(n):
        while True:
            u = rng.uniform(0.02, 0.98)
            v = rng.uniform(0.02, 0.98)
            if u + v < 0.96:
                break
        pts.append([u + 0.5 * v, S3 / 2.0 * v])
    return np.array(pts)


def _grid_init(n=N):
    """Structured grid-like initialization inside triangle."""
    pts = []
    # Place points on a triangular grid
    rows = 4
    cols = 4
    margin = 0.08
    for i in range(rows):
        for j in range(cols):
            if len(pts) >= n:
                break
            # Map to triangle coordinates
            t = (i + 0.5) / rows
            s = (j + 0.5) / (cols * (1 - t) + 1)
            x = margin + t * (1 - 2 * margin) * (1 - s) + s * (0.5 * (1 - 2 * margin))
            y = S3 / 2.0 * t
            if _inside_triangle(np.array([x, y]), eps=0.0):
                pts.append([x, y])
    # Fill remaining with random
    rng = np.random.RandomState(42)
    while len(pts) < n:
        u = rng.uniform(0.05, 0.95)
        v = rng.uniform(0.05, 0.95)
        if u + v < 0.9:
            p = [u + 0.5 * v, S3 / 2.0 * v]
            if _inside_triangle(np.array(p), eps=0.0):
                pts.append(p)
    return np.array(pts[:n])


def _corners_init(n=N):
    """Initialize near vertices and midpoints."""
    pts = []
    # Near vertices
    for v in [V0, V1, V2]:
        for dx in [-0.05, 0.0, 0.05]:
            for dy in [-0.05, 0.0, 0.05]:
                p = v + np.array([dx, dy])
                if _inside_triangle(p, eps=0.0) and len(pts) < n:
                    pts.append(p)
    # Fill with random
    rng = np.random.RandomState(123)
    while len(pts) < n:
        u = rng.uniform(0.05, 0.95)
        v = rng.uniform(0.05, 0.95)
        if u + v < 0.9:
            p = [u + 0.5 * v, S3 / 2.0 * v]
            if _inside_triangle(np.array(p), eps=0.0):
                pts.append(p)
    return np.array(pts[:n])


def _optimize_from(pts, deadline, beta_start=10.0, beta_end=50.0):
    """Multi-stage optimization from initial points."""
    x0 = pts.flatten()
    best_pts = pts.copy()
    best_area = _min_triangle_area(pts)
    
    # Stage 1: L-BFGS-B with moderate beta
    for beta in [beta_start, beta_start * 2, beta_end]:
        if time.time() > deadline:
            break
        try:
            res = minimize(_objective_flat, x0, args=(beta,), method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10})
            if res.success or res.fun < _objective_flat(x0, beta):
                new_pts = res.x.reshape(-1, 2)
                # Verify all points are inside
                valid = all(_inside_triangle(p, eps=1e-6) for p in new_pts)
                if valid:
                    area = _min_triangle_area(new_pts)
                    if area > best_area:
                        best_area = area
                        best_pts = new_pts.copy()
                        x0 = res.x
        except Exception:
            pass
    
    # Stage 2: Nelder-Mead polish
    if time.time() < deadline - 5:
        try:
            def nm_obj(x):
                pts = x.reshape(-1, 2)
                if not all(_inside_triangle(p, eps=1e-6) for p in pts):
                    return 1.0
                return -_min_triangle_area(pts)
            
            res = minimize(nm_obj, best_pts.flatten(), method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < nm_obj(best_pts.flatten()):
                new_pts = res.x.reshape(-1, 2)
                if all(_inside_triangle(p, eps=1e-6) for p in new_pts):
                    area = _min_triangle_area(new_pts)
                    if area > best_area:
                        best_area = area
                        best_pts = new_pts.copy()
        except Exception:
            pass
    
    return best_pts, best_area


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_pts = None
    best_area = 0.0
    
    # Multiple diverse initializations
    inits = []
    
    # 1. Random with different seeds
    for seed in [7, 42, 123, 999, 2024]:
        rng = np.random.RandomState(seed)
        pts = []
        for _ in range(N):
            while True:
                u = rng.uniform(0.02, 0.98)
                v = rng.uniform(0.02, 0.98)
                if u + v < 0.96:
                    break
            pts.append([u + 0.5 * v, S3 / 2.0 * v])
        inits.append(np.array(pts))
    
    # 2. Grid-like
    inits.append(_grid_init())
    
    # 3. Corners-based
    inits.append(_corners_init())
    
    # 4. Symmetric pattern
    pts_sym = []
    # Center
    pts_sym.append([0.5, S3 / 6.0])
    # 6 points around center at radius ~0.25
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        x = 0.5 + 0.25 * np.cos(angle)
        y = S3 / 6.0 + 0.25 * np.sin(angle)
        if _inside_triangle(np.array([x, y]), eps=0.0):
            pts_sym.append([x, y])
    # Fill remaining
    rng = np.random.RandomState(777)
    while len(pts_sym) < N:
        u = rng.uniform(0.05, 0.95)
        v = rng.uniform(0.05, 0.95)
        if u + v < 0.9:
            p = [u + 0.5 * v, S3 / 2.0 * v]
            if _inside_triangle(np.array(p), eps=0.0):
                pts_sym.append(p)
    inits.append(np.array(pts_sym[:N]))
    
    # Optimize from each initialization
    for i, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
        pts, area = _optimize_from(init_pts, deadline)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    # Final polish on best if time remains
    if best_pts is not None and time.time() < deadline - 10:
        try:
            def final_obj(x):
                pts = x.reshape(-1, 2)
                if not all(_inside_triangle(p, eps=1e-6) for p in pts):
                    return 1.0
                return -_min_triangle_area(pts)
            
            res = minimize(final_obj, best_pts.flatten(), method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < final_obj(best_pts.flatten()):
                new_pts = res.x.reshape(-1, 2)
                if all(_inside_triangle(p, eps=1e-6) for p in new_pts):
                    area = _min_triangle_area(new_pts)
                    if area > best_area:
                        best_area = area
                        best_pts = new_pts.copy()
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: return random
        rng = np.random.RandomState(7)
        pts = []
        for _ in range(N):
            while True:
                u = rng.uniform(0.02, 0.98)
                v = rng.uniform(0.02, 0.98)
                if u + v < 0.96:
                    break
            pts.append([u + 0.5 * v, S3 / 2.0 * v])
        return np.array(pts)
    
    return best_pts
