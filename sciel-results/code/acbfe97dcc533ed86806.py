# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

_DEADLINE = time.time() + 615.0

def _vertex_score(distances):
    d = np.sort(distances)
    if np.any(d < 0.05):
        return 0.0
    gaps = d[1:] - d[:-1]
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        sel = d[0:4]
    elif k == 17:
        sel = d[15:19]
    else:
        sel = d[k - 1:k + 3]
    ratio = np.maximum(sel / D, D / sel)
    raw = np.mean(ratio)
    return min(1.0, 1.0 / raw)

def _min_vertex_score(polygon):
    n = len(polygon)
    scores = []
    for i in range(n):
        dists = np.linalg.norm(polygon - polygon[i], axis=1)
        dists = np.delete(dists, i)
        scores.append(_vertex_score(dists))
    return float(np.min(scores))

def _is_valid(polygon, n=20):
    if not np.all(np.isfinite(polygon)):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-12:
                return False
    pts = np.vstack([polygon, polygon[:1]])
    edges = pts[1:] - pts[:-1]
    if np.any(np.linalg.norm(edges, axis=1) < 1e-12):
        return False
    cross = np.cross(edges[:-1], edges[1:])
    if np.any(np.abs(cross) < 1e-12):
        return False
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    dangles = np.diff(angles)
    dangles = np.append(dangles, angles[0] - angles[-1])
    dangles = np.arctan2(np.sin(dangles), np.cos(dangles))
    total = np.sum(dangles)
    if abs(abs(total) - 2 * np.pi) > 1e-6:
        return False
    return True

def _normalized(polygon):
    c = np.mean(polygon, axis=0)
    p = polygon - c
    m = np.mean(np.linalg.norm(p, axis=1))
    if m <= 0.01:
        return None
    return p / m

def _regular_polygon():
    a = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    return np.stack([np.cos(a), np.sin(a)], axis=1)

def _fourier_to_points(params):
    """Convert Fourier coefficients to 20 points.
    params = [a1, b1, a2, b2, a3, b3]
    r_i = 1 + a1*cos(i*2pi/20) + b1*sin(i*2pi/20) + a2*cos(2*i*2pi/20) + b2*sin(2*i*2pi/20) + a3*cos(3*i*2pi/20) + b3*sin(3*i*2pi/20)
    """
    n = 20
    i = np.arange(n)
    base_angles = 2 * np.pi * i / n
    r = np.ones(n)
    for k, (ak, bk) in enumerate(zip(params[0::2], params[1::2]), start=1):
        r += ak * np.cos(k * base_angles) + bk * np.sin(k * base_angles)
    r = np.maximum(r, 0.3)  # ensure positive radii
    pts = np.stack([r * np.cos(base_angles), r * np.sin(base_angles)], axis=1)
    return pts

def _objective(params):
    """Objective for differential_evolution: minimize negative min vertex score."""
    if time.time() > _DEADLINE - 5:
        return 10.0
    
    pts = _fourier_to_points(params)
    
    if not _is_valid(pts):
        return 10.0
    
    nm = _normalized(pts)
    if nm is None:
        return 10.0
    
    score = _min_vertex_score(nm)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    # Start with regular polygon
    reg = _regular_polygon()
    best = reg.copy()
    best_score = _min_vertex_score(reg)
    
    # Bounds for Fourier coefficients: keep radii in [0.3, 1.7]
    # r = 1 + sum of up to 3 harmonics, each coefficient bounded by 0.25
    # Max perturbation: 0.25 * 3 = 0.75, so r in [0.25, 1.75]
    # Use tighter bounds to maintain convexity
    bounds = [(-0.2, 0.2)] * 6  # a1, b1, a2, b2, a3, b3
    
    # Differential Evolution with global search
    try:
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=True,
            init='latinhypercube'
        )
        
        if result.fun < -best_score:
            pts = _fourier_to_points(result.x)
            if _is_valid(pts):
                nm = _normalized(pts)
                if nm is not None:
                    s = _min_vertex_score(nm)
                    if s > best_score:
                        best_score = s
                        best = nm
    except Exception:
        pass
    
    # Try a second DE run with different bounds (wider)
    if time.time() < _DEADLINE - 30:
        bounds2 = [(-0.3, 0.3)] * 6
        try:
            result2 = differential_evolution(
                _objective,
                bounds2,
                seed=123,
                maxiter=100,
                popsize=12,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=True,
                init='sobol'
            )
            
            if result2.fun < -best_score:
                pts = _fourier_to_points(result2.x)
                if _is_valid(pts):
                    nm = _normalized(pts)
                    if nm is not None:
                        s = _min_vertex_score(nm)
                        if s > best_score:
                            best_score = s
                            best = nm
        except Exception:
            pass
    
    # Third DE run with asymmetric bounds favoring certain harmonics
    if time.time() < _DEADLINE - 15:
        bounds3 = [(-0.25, 0.25), (-0.25, 0.25), (-0.15, 0.15), (-0.15, 0.15), (-0.1, 0.1), (-0.1, 0.1)]
        try:
            result3 = differential_evolution(
                _objective,
                bounds3,
                seed=777,
                maxiter=100,
                popsize=12,
                tol=1e-8,
                mutation=(0.4, 0.9),
                recombination=0.75,
                polish=True,
                init='latinhypercube'
            )
            
            if result3.fun < -best_score:
                pts = _fourier_to_points(result3.x)
                if _is_valid(pts):
                    nm = _normalized(pts)
                    if nm is not None:
                        s = _min_vertex_score(nm)
                        if s > best_score:
                            best_score = s
                            best = nm
        except Exception:
            pass
    
    return best

# EVOLVE-BLOCK-END
