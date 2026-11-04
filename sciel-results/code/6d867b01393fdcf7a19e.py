import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _facet_circumradii(pts, simplices):
    """Vectorized computation of circumradii for all facets.
    Returns array of radii (radians), circumcenter unit vectors, and validity mask."""
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    n = np.cross(a - b, a - c)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm > 1e-10
    n = n / np.maximum(n_norm[:, None], 1e-10)
    centroids = (a + b + c) / 3.0
    signs = np.where(np.einsum('ij,ij->i', n, centroids) < 0, -1.0, 1.0)
    n = n * signs[:, None]
    dots = np.clip(np.einsum('ij,ij->i', n, a), -1.0, 1.0)
    radii = np.arccos(dots)
    radii[~mask] = 0.0
    return radii, n, mask

def _covering_radius_rad(pts):
    """Exact covering radius in radians via convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        radii, _, _ = _facet_circumradii(pts, hull.simplices)
        return float(np.max(radii))
    except Exception:
        return np.pi

def _fibonacci_spiral(offset=0.0):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _electrostatic_refine(pts, n_iters=500, lr=0.01):
    """Repulsion-based refinement: each point is pushed away by all others via Coulomb-like force."""
    pts = pts.copy()
    for it in range(n_iters):
        D = pts @ pts.T
        chord2 = 2.0 - 2.0 * D
        np.fill_diagonal(chord2, 1.0)
        W = 1.0 / chord2
        np.fill_diagonal(W, 0.0)
        row_sum = W.sum(axis=1)
        forces = pts * row_sum[:, None] - (W @ pts)
        forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        forces = forces / (norms + 1e-12)
        pts = pts + lr * forces
        pts = _normalize(pts)
    return pts

def _smooth_max_objective(pts, beta=20.0):
    """Smooth surrogate for the max facet circumradius using log-sum-exp."""
    try:
        hull = ConvexHull(pts)
        radii, _, mask = _facet_circumradii(pts, hull.simplices)
        valid = radii[mask]
        if len(valid) == 0:
            return 0.0
        m = np.max(valid)
        # Log-sum-exp for numerical stability
        vals = beta * valid
        s = np.sum(np.exp(vals - np.max(vals)))
        return m + np.log(s) / beta
    except Exception:
        return np.pi

def _smooth_max_grad(pts, beta=20.0):
    """Approximate gradient of the smooth max objective using finite differences."""
    eps = 1e-6
    grad = np.zeros((N, 3))
    base_val = _smooth_max_objective(pts, beta)
    
    for i in range(N):
        for j in range(3):
            # Move point i slightly in direction j
            tangent = np.zeros(3)
            tangent[j] = 1.0
            tangent -= np.dot(tangent, pts[i]) * pts[i]
            tn = np.linalg.norm(tangent)
            if tn < 1e-10:
                continue
            tangent /= tn
            
            pts_plus = pts.copy()
            pts_plus[i] = _normalize(pts_plus[i] + eps * tangent)
            val_plus = _smooth_max_objective(pts_plus, beta)
            
            grad[i, j] = (val_plus - base_val) / eps
    
    return grad

def _lbfgs_polish(pts, max_time=10.0, max_iter=200):
    """Polish using L-BFGS-B with the smooth max objective."""
    start = time.time()
    best_pts = pts.copy()
    best_rad = _covering_radius_rad(best_pts)
    
    def objective(x_flat):
        if time.time() - start > max_time - 0.5:
            return best_rad
        new_pts = x_flat.reshape(N, 3)
        new_pts = _normalize(new_pts)
        if not np.all(np.isfinite(new_pts)):
            return np.pi
        return _smooth_max_objective(new_pts, beta=20.0)
    
    x0 = pts.flatten()
    
    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50
            }
        )
        
        if res.fun < best_rad:
            new_pts = _normalize(res.x.reshape(N, 3))
            new_rad = _covering_radius_rad(new_pts)
            if new_rad < best_rad:
                best_pts = new_pts
                best_rad = new_rad
    except Exception:
        pass
    
    return best_pts

def _nm_polish_top_facets(pts, max_time=5.0, k_facets=10, max_iter=100):
    """
    Polish the configuration by running Nelder-Mead on the vertices of the worst k facets.
    """
    start = time.time()
    best_pts = pts.copy()
    best_rad = _covering_radius_rad(best_pts)
    
    for _ in range(3):
        if time.time() - start > max_time - 1.0:
            break
            
        try:
            hull = ConvexHull(best_pts)
            radii, _, _ = _facet_circumradii(best_pts, hull.simplices)
            valid_idx = np.where(radii > 1e-10)[0]
            if len(valid_idx) == 0:
                break
            
            order = valid_idx[np.argsort(-radii[valid_idx])]
            top_k = min(k_facets, len(order))
            worst_facets = order[:top_k]
            
            vertex_indices = set()
            for f_idx in worst_facets:
                for v in hull.simplices[f_idx]:
                    vertex_indices.add(v)
            
            vertex_indices = list(vertex_indices)
            if not vertex_indices:
                break
                
            def objective(x_flat):
                new_pts = best_pts.copy()
                for i, idx in enumerate(vertex_indices):
                    new_pts[idx] = x_flat[i*3:(i+1)*3]
                new_pts = _normalize(new_pts)
                if not np.all(np.isfinite(new_pts)):
                    return np.pi
                return _covering_radius_rad(new_pts)
            
            x0 = np.array([best_pts[idx] for idx in vertex_indices]).flatten()
            
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'maxiter': max_iter,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            
            if res.fun < best_rad:
                new_pts = best_pts.copy()
                for i, idx in enumerate(vertex_indices):
                    new_pts[idx] = res.x[i*3:(i+1)*3]
                new_pts = _normalize(new_pts)
                new_rad = _covering_radius_rad(new_pts)
                if new_rad < best_rad:
                    best_pts = new_pts
                    best_rad = new_rad
                    
        except Exception:
            break
            
    return best_pts

def _lloyd_refine(pts, n_iters=200, step_init=0.03):
    """Voronoi-based Lloyd refinement: move each point toward the centroid of its Voronoi cell."""
    pts = pts.copy()
    step = step_init
    
    for it in range(n_iters):
        try:
            hull = ConvexHull(pts)
            radii, cc, mask = _facet_circumradii(pts, hull.simplices)
        except Exception:
            break
        
        if not np.any(mask):
            break
        
        # For each vertex, compute the weighted centroid of adjacent facet circumcenters
        # Weight by the radius of each facet
        new_pts = pts.copy()
        for v in range(N):
            # Find facets containing vertex v
            mask_v = np.any(hull.simplices == v, axis=1) & mask
            if not np.any(mask_v):
                continue
            
            facet_indices = np.where(mask_v)[0]
            weights = radii[facet_indices]
            centers = cc[facet_indices]
            
            # Weighted centroid
            w_sum = np.sum(weights)
            if w_sum < 1e-10:
                continue
            centroid = np.average(centers, axis=0, weights=weights)
            centroid = _normalize(centroid.reshape(1, 3))[0]
            
            # Move toward centroid
            new_pts[v] = _normalize(pts[v] + step * (centroid - pts[v]))
        
        pts = new_pts
        step *= 0.995
        if step < 0.001:
            step = 0.001
        
        # Check for improvement
        cur_rad = _covering_radius_rad(pts)
        if it % 10 == 0:
            pass  # Could add early stopping
    
    return pts

def construct():
    start_time = time.time()
    max_time = 140.0  # Safety margin before 150s kill

    def _time_left():
        return max_time - (time.time() - start_time)

    best_points = _fibonacci_spiral()
    best_rad = _covering_radius_rad(best_points)

    # Phase 0: Multiple initializations
    inits = []
    inits.append(_fibonacci_spiral(0.0))
    inits.append(_fibonacci_spiral(0.3))
    inits.append(_fibonacci_spiral(0.7))
    inits.append(_fibonacci_spiral(1.1))
    
    rng = np.random.RandomState(42)
    for _ in range(3):
        pts = rng.randn(N, 3)
        pts = _normalize(pts)
        inits.append(pts)
    
    # Electrostatic-refined from different Fibonacci offsets
    for offset in [0.0, 0.5]:
        inits.append(_electrostatic_refine(_fibonacci_spiral(offset), n_iters=200, lr=0.005))

    # Evaluate all initializations
    for init in inits:
        if _time_left() < 5.0:
            break
        rad = _covering_radius_rad(init)
        if rad < best_rad:
            best_rad = rad
            best_points = init.copy()

    # Phase 1: For each good initialization, do Lloyd + L-BFGS + NM Polish
    init_list = []
    for init in inits:
        rad = _covering_radius_rad(init)
        init_list.append((rad, init.copy()))
    init_list.sort(key=lambda x: x[0])
    top_inits = [x[1] for x in init_list[:4]]

    for init_idx, init_pts in enumerate(top_inits):
        if _time_left() < 20.0:
            break
        pts = init_pts.copy()
        cur_rad = _covering_radius_rad(pts)
        if cur_rad < best_rad:
            best_rad = cur_rad
            best_points = pts.copy()

        # Phase 1a: Lloyd refinement
        if _time_left() > 15.0:
            pts = _lloyd_refine(pts, n_iters=100, step_init=0.02)
            cur_rad = _covering_radius_rad(pts)
            if cur_rad < best_rad:
                best_rad = cur_rad
                best_points = pts.copy()

        # Phase 1b: L-BFGS polish with smooth objective
        if _time_left() > 10.0:
            alloc_time = min(_time_left() - 5.0, 15.0)
            if alloc_time > 3.0:
                pts = _lbfgs_polish(pts, max_time=alloc_time, max_iter=150)
                cur_rad = _covering_radius_rad(pts)
                if cur_rad < best_rad:
                    best_rad = cur_rad
                    best_points = pts.copy()

        # Phase 1c: Nelder-Mead polish on top facets
        if _time_left() > 8.0:
            alloc_time = min(_time_left() - 3.0, 10.0)
            if alloc_time > 2.0:
                pts = _nm_polish_top_facets(pts, max_time=alloc_time, k_facets=15, max_iter=150)
                cur_rad = _covering_radius_rad(pts)
                if cur_rad < best_rad:
                    best_rad = cur_rad
                    best_points = pts.copy()

    # Final high-precision polish on best_points
    if _time_left() > 5.0:
        pts = best_points.copy()
        
        # Try L-BFGS with tighter settings
        alloc_time = min(_time_left() - 2.0, 12.0)
        if alloc_time > 2.0:
            pts = _lbfgs_polish(pts, max_time=alloc_time, max_iter=200)
            cur_rad = _covering_radius_rad(pts)
            if cur_rad < best_rad:
                best_rad = cur_rad
                best_points = pts.copy()
        
        # Final NM polish
        if _time_left() > 3.0:
            alloc_time = min(_time_left() - 1.0, 8.0)
            if alloc_time > 1.0:
                pts = _nm_polish_top_facets(pts, max_time=alloc_time, k_facets=20, max_iter=200)
                cur_rad = _covering_radius_rad(pts)
                if cur_rad < best_rad:
                    best_rad = cur_rad
                    best_points = pts.copy()

    return _normalize(best_points)
