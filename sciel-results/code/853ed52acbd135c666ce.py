import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50
DEADLINE = time.time() + 140.0

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _angular_dist(a, b):
    dot = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.arccos(dot)

def _covering_radius(pts):
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
        facets = hull.simplices
    except Exception:
        grid = _dense_grid(200)
        d = np.min(_angular_dist(grid, pts.T), axis=1)
        return np.max(d)
    
    max_cr = 0.0
    for tri in facets:
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        ab = a - b
        ac = a - c
        n = np.cross(ab, ac)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        da = _angular_dist(n.reshape(1,3), a.reshape(1,3))[0]
        db = _angular_dist(n.reshape(1,3), b.reshape(1,3))[0]
        dc = _angular_dist(n.reshape(1,3), c.reshape(1,3))[0]
        cr = max(da, db, dc)
        if cr > max_cr:
            max_cr = cr
    return max_cr

def _dense_grid(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _soft_max_covering_radius(pts, beta=50.0, grid_size=100):
    pts = _normalize(pts)
    grid = _dense_grid(grid_size)
    d = _angular_dist(grid, pts.T)
    min_d = np.min(d, axis=1)
    shifted = min_d - np.max(min_d)
    smax = np.max(min_d) + np.log(np.sum(np.exp(beta * shifted))) / beta
    return smax

def _repulsion_relax(pts, steps=50, lr=0.01):
    pts = _normalize(pts).copy()
    for step in range(steps):
        if time.time() > DEADLINE - 20:
            break
        grad = np.zeros_like(pts)
        for i in range(N):
            for j in range(i+1, N):
                d = _angular_dist(pts[i].reshape(1,3), pts[j].reshape(1,3))[0]
                if d < 1e-8:
                    continue
                direction = pts[i] - pts[j]
                direction = direction - np.dot(direction, pts[i]) * pts[i]
                direction_norm = np.linalg.norm(direction)
                if direction_norm < 1e-12:
                    continue
                direction = direction / direction_norm
                force = 2.0 / (d**3) * direction
                grad[i] += force
                grad[j] -= force
        pts = pts - lr * grad
        pts = _normalize(pts)
    return pts

def _lloyd_iteration(pts, steps=20):
    pts = _normalize(pts).copy()
    for step in range(steps):
        if time.time() > DEADLINE - 15:
            break
        grid = _dense_grid(500)
        d = _angular_dist(grid, pts.T)
        labels = np.argmin(d, axis=1)
        new_pts = pts.copy()
        for i in range(N):
            mask = (labels == i)
            if np.sum(mask) > 0:
                centroid = np.mean(grid[mask], axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-12:
                    new_pts[i] = centroid / centroid_norm
        pts = _normalize(new_pts)
    return pts

def _facet_targeted_move(pts, steps=10):
    pts = _normalize(pts).copy()
    for step in range(steps):
        if time.time() > DEADLINE - 10:
            break
        try:
            hull = ConvexHull(pts)
            facets = hull.simplices
        except Exception:
            break
        crs = []
        for tri in facets:
            a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            ab = a - b
            ac = a - c
            n = np.cross(ab, ac)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                crs.append(0.0)
                continue
            n = n / n_norm
            da = _angular_dist(n.reshape(1,3), a.reshape(1,3))[0]
            db = _angular_dist(n.reshape(1,3), b.reshape(1,3))[0]
            dc = _angular_dist(n.reshape(1,3), c.reshape(1,3))[0]
            crs.append(max(da, db, dc))
        crs = np.array(crs)
        worst_idx = np.argmax(crs)
        worst_tri = facets[worst_idx]
        a, b, c = pts[worst_tri[0]], pts[worst_tri[1]], pts[worst_tri[2]]
        ab = a - b
        ac = a - c
        n = np.cross(ab, ac)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            break
        n = n / n_norm
        alpha = 0.1
        for idx in worst_tri:
            direction = n - pts[idx]
            direction = direction - np.dot(direction, pts[idx]) * pts[idx]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-12:
                pts[idx] = pts[idx] + alpha * direction / direction_norm
        pts = _normalize(pts)
    return pts

def construct():
    start_time = time.time()
    
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    pts = _repulsion_relax(pts, steps=50, lr=0.01)
    pts = _lloyd_iteration(pts, steps=20)
    pts = _facet_targeted_move(pts, steps=10)
    
    if time.time() < DEADLINE - 30:
        def objective(params):
            p = _normalize(params.reshape(N, 3))
            return _soft_max_covering_radius(p, beta=60.0, grid_size=80)
        
        try:
            result = minimize(
                objective,
                pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result.fun < _covering_radius(pts):
                pts = _normalize(result.x.reshape(N, 3))
        except Exception:
            pass
    
    pts = _normalize(pts)
    if not np.all(np.isfinite(pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        pts = _normalize(pts)
    
    return pts
