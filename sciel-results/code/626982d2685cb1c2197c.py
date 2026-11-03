"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_DEADLINE = time.time() + 110.0  # safety margin vs 120 s kill


def _to_sphere(pts):
    """Normalize points to the unit sphere (robust to near-zero norms)."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _facet_circumradius(a, b, c):
    """Angular radius of the empty-cap circumcircle of spherical triangle (a,b,c)."""
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.pi
    n = n / nn
    best = np.pi
    best_p = None
    for s in (1.0, -1.0):
        p = s * n
        da = np.arccos(np.clip(np.dot(p, a), -1.0, 1.0))
        db = np.arccos(np.clip(np.dot(p, b), -1.0, 1.0))
        dc = np.arccos(np.clip(np.dot(p, c), -1.0, 1.0))
        if abs(da - db) < 1e-4 and abs(da - dc) < 1e-4:
            if da < best:
                best = da
                best_p = p
    return best, best_p


def _covering_radius(pts):
    """Covering radius (radians): max facet circumradius of the convex hull."""
    pts = _to_sphere(pts)
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    if np.min(d2[np.triu_indices(len(pts), k=1)]) < 1e-14:
        return np.pi
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    if len(hull.simplices) == 0:
        return np.pi
    s = hull.simplices
    maxr = 0.0
    for idx in s:
        r, _ = _facet_circumradius(pts[idx[0]], pts[idx[1]], pts[idx[2]])
        if r > maxr:
            maxr = r
    return maxr


def _fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_pts():
    v = np.random.randn(N, 3)
    return _to_sphere(v)


def _smooth_surrogate(pts, power=2):
    """Compute a smooth surrogate objective: sum of capped facet circumradii^power.
    
    This is differentiable w.r.t. point positions and can be minimized with gradient methods.
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 1e6, np.zeros_like(pts)
    
    if len(hull.simplices) == 0:
        return 1e6, np.zeros_like(pts)
    
    total = 0.0
    grad = np.zeros_like(pts)
    
    for idx in hull.simplices:
        a, b, c = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        r, best_p = _facet_circumradius(a, b, c)
        if best_p is None:
            continue
        
        # Add to surrogate with capping
        capped_r = min(r, 1.0)  # Cap at 1 radian for smoothness
        total += capped_r ** power
        
        # Approximate gradient: push points away from circumcenter
        for vi, v in [(idx[0], a), (idx[1], b), (idx[2], c)]:
            d = np.dot(v, best_p)
            denom = np.sqrt(np.maximum(1e-8, 1.0 - d * d))
            grad[vi] += -power * (capped_r ** (power - 1)) * best_p / denom
    
    return total, grad


def _refine_smooth(pts, max_iter=30):
    """Refine using smooth surrogate minimization with L-BFGS-B."""
    pts = _to_sphere(pts.copy())
    best_cr = _covering_radius(pts)
    best_pts = pts.copy()
    
    for restart in range(3):
        if time.time() > _DEADLINE - 10.0:
            break
        
        def objective(flat_x):
            x = flat_x.reshape(N, 3)
            x = _to_sphere(x)
            val, _ = _smooth_surrogate(x)
            return val
        
        def gradient(flat_x):
            x = flat_x.reshape(N, 3)
            x = _to_sphere(x)
            _, g = _smooth_surrogate(x)
            grad_flat = np.zeros_like(flat_x)
            for i in range(N):
                tangent = g[i] - np.dot(g[i], x[i]) * x[i]
                grad_flat[i * 3:(i + 1) * 3] = tangent
            return grad_flat
        
        x0 = pts.flatten()
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'gtol': 1e-6})
            if res.success or res.fun < 1e5:
                new_pts = _to_sphere(res.x.reshape(N, 3))
                new_cr = _covering_radius(new_pts)
                if new_cr < best_cr:
                    best_cr = new_cr
                    best_pts = new_pts
                    pts = new_pts
        except Exception:
            pass
    
    return best_pts, best_cr


def _worst_facet_projection(pts, n_iter=200, step_scale=0.05):
    """Geometric worst-facet projection: iteratively find the hull facet with
    the largest spherical circumradius and project its three vertices along
    the negative gradient direction (toward the circumcenter) to shrink it.
    
    This is a deterministic, non-stochastic geometric relaxation distinct from
    Lloyd's centroid averaging or global energy minimization.
    """
    pts = _to_sphere(pts.copy())
    best_cr = _covering_radius(pts)
    best_pts = pts.copy()
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 15.0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        if len(hull.simplices) == 0:
            break
        
        # Find the worst facet
        worst_idx = None
        worst_r = -1.0
        worst_p = None
        for idx in hull.simplices:
            r, p = _facet_circumradius(pts[idx[0]], pts[idx[1]], pts[idx[2]])
            if p is None:
                continue
            if r > worst_r:
                worst_r = r
                worst_idx = idx
                worst_p = p
        
        if worst_idx is None or worst_p is None:
            break
        
        # Project each of the three vertices toward the circumcenter
        # The gradient of arccos(v·p) w.r.t. v is -p/sqrt(1-(v·p)^2)
        # We want to move v to increase v·p (move toward p), so step = +grad direction
        # But we also need to stay on the sphere, so use tangent projection
        
        # Adaptive step size based on the facet radius
        step = step_scale * min(worst_r, 0.5)
        
        new_pts = pts.copy()
        for vi in worst_idx:
            v = pts[vi]
            d = np.dot(v, worst_p)
            denom = np.sqrt(np.maximum(1e-8, 1.0 - d * d))
            # Gradient direction (toward circumcenter)
            grad_dir = worst_p / denom
            # Project to tangent space
            tangent = grad_dir - np.dot(grad_dir, v) * v
            # Move along tangent
            new_pts[vi] = v + step * tangent
        
        new_pts = _to_sphere(new_pts)
        new_cr = _covering_radius(new_pts)
        
        if new_cr < best_cr:
            best_cr = new_cr
            best_pts = new_pts
            pts = new_pts
        else:
            # Reduce step size if no improvement
            step_scale *= 0.8
            if step_scale < 1e-6:
                break
    
    return best_pts, best_cr


def _final_polish(pts, n_iter=50):
    """High-precision final polish using worst-facet projection with very small steps."""
    pts = _to_sphere(pts.copy())
    best_cr = _covering_radius(pts)
    best_pts = pts.copy()
    
    step = 0.005
    for it in range(n_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        if len(hull.simplices) == 0:
            break
        
        worst_idx = None
        worst_r = -1.0
        worst_p = None
        for idx in hull.simplices:
            r, p = _facet_circumradius(pts[idx[0]], pts[idx[1]], pts[idx[2]])
            if p is None:
                continue
            if r > worst_r:
                worst_r = r
                worst_idx = idx
                worst_p = p
        
        if worst_idx is None or worst_p is None:
            break
        
        new_pts = pts.copy()
        for vi in worst_idx:
            v = pts[vi]
            d = np.dot(v, worst_p)
            denom = np.sqrt(np.maximum(1e-8, 1.0 - d * d))
            grad_dir = worst_p / denom
            tangent = grad_dir - np.dot(grad_dir, v) * v
            new_pts[vi] = v + step * tangent
        
        new_pts = _to_sphere(new_pts)
        new_cr = _covering_radius(new_pts)
        
        if new_cr < best_cr:
            best_cr = new_cr
            best_pts = new_pts
            pts = new_pts
        else:
            step *= 0.7
            if step < 1e-7:
                break
    
    return best_pts, best_cr


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 110.0
    rng = np.random.default_rng(12345)

    # Candidate starts: Fibonacci + random
    starts = [_fibonacci()]
    for _ in range(4):
        starts.append(_random_pts())

    best_pts = None
    best_cr = np.inf
    for st in starts:
        if time.time() > _DEADLINE - 30.0:
            break
        p, cr = _refine_smooth(st)
        if cr < best_cr:
            best_cr = cr
            best_pts = p

    if best_pts is None:
        best_pts = _fibonacci()
        best_cr = _covering_radius(best_pts)
    
    # Apply worst-facet projection refinement
    if time.time() < _DEADLINE - 20.0:
        p, cr = _worst_facet_projection(best_pts, n_iter=300, step_scale=0.08)
        if cr < best_cr:
            best_cr = cr
            best_pts = p
    
    # Final high-precision polish
    if time.time() < _DEADLINE - 5.0:
        p, cr = _final_polish(best_pts, n_iter=100)
        if cr < best_cr:
            best_cr = cr
            best_pts = p
    
    return _to_sphere(best_pts)
# EVOLVE-BLOCK-END
