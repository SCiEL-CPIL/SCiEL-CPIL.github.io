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
    Returns array of radii (radians) and array of circumcenter unit vectors."""
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
        # Pairwise dot products
        D = pts @ pts.T
        # Force on i from j: proportional to (pts[i] - pts[j]) / |...|^2 but on sphere we use tangent component
        # Use inverse-square repulsion in chordal distance
        # chord^2 = 2 - 2*D
        chord2 = 2.0 - 2.0 * D
        np.fill_diagonal(chord2, 1.0)
        # Repulsion magnitude ~ 1/chord^2
        W = 1.0 / chord2
        np.fill_diagonal(W, 0.0)
        # Force = sum_j W_ij * (pts[i] - pts[j]) projected to tangent
        # = pts[i] * sum_j W_ij - sum_j W_ij * pts[j]
        row_sum = W.sum(axis=1)
        forces = pts * row_sum[:, None] - (W @ pts)
        # Project to tangent plane
        forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
        forces /= (np.linalg.norm(forces, axis=1, keepdims=True) + 1e-12)
        pts = pts + lr * forces
        pts = _normalize(pts)
    return pts

def _nm_polish_top_facets(pts, max_time=5.0, k_facets=10, max_iter=100):
    """
    Polish the configuration by running Nelder-Mead on the vertices of the worst k facets.
    This is inspired by the 'polish_largest_facets' from the inspiration.
    """
    start = time.time()
    best_pts = pts.copy()
    best_rad = _covering_radius_rad(best_pts)
    
    for _ in range(3): # A few restarts to handle topology changes
        if time.time() - start > max_time - 1.0:
            break
            
        try:
            hull = ConvexHull(best_pts)
            radii, _, _ = _facet_circumradii(best_pts, hull.simplices)
            valid_idx = np.where(radii > 1e-10)[0]
            if len(valid_idx) == 0:
                break
            
            # Sort facets by radius descending
            order = valid_idx[np.argsort(-radii[valid_idx])]
            # Take top k facets
            top_k = min(k_facets, len(order))
            worst_facets = order[:top_k]
            
            # Collect unique vertex indices from these facets
            vertex_indices = set()
            for f_idx in worst_facets:
                for v in hull.simplices[f_idx]:
                    vertex_indices.add(v)
            
            vertex_indices = list(vertex_indices)
            if not vertex_indices:
                break
                
            # Objective function for Nelder-Mead
            # We optimize the coordinates of the vertices in vertex_indices
            # The rest of the points are fixed.
            def objective(x_flat):
                new_pts = best_pts.copy()
                for i, idx in enumerate(vertex_indices):
                    new_pts[idx] = x_flat[i*3:(i+1)*3]
                
                # Normalize
                new_pts = _normalize(new_pts)
                
                # Check validity
                if not np.all(np.isfinite(new_pts)):
                    return np.pi
                
                # Compute covering radius
                rad = _covering_radius_rad(new_pts)
                return rad
            
            x0 = np.array([best_pts[idx] for idx in vertex_indices]).flatten()
            
            # Run Nelder-Mead
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
                
                # Verify improvement
                new_rad = _covering_radius_rad(new_pts)
                if new_rad < best_rad:
                    best_pts = new_pts
                    best_rad = new_rad
                    
        except Exception:
            # If hull fails or other error, break and return current best
            break
            
    return best_pts

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
    # Random
    rng = np.random.RandomState(42)
    for _ in range(2):
        pts = rng.randn(N, 3)
        pts = _normalize(pts)
        inits.append(pts)
    # Electrostatic-refined from Fibonacci
    inits.append(_electrostatic_refine(_fibonacci_spiral(0.0), n_iters=300, lr=0.005))

    # Evaluate all initializations
    for init in inits:
        if _time_left() < 5.0:
            break
        rad = _covering_radius_rad(init)
        if rad < best_rad:
            best_rad = rad
            best_points = init.copy()

    # Phase 1: For each good initialization, do Greedy Lloyd + NM Polish
    init_list = []
    for init in inits:
        rad = _covering_radius_rad(init)
        init_list.append((rad, init.copy()))
    init_list.sort(key=lambda x: x[0])
    # Take top 3 initializations
    top_inits = [x[1] for x in init_list[:3]]

    for init_idx, init_pts in enumerate(top_inits):
        if _time_left() < 15.0:
            break
        pts = init_pts.copy()
        cur_rad = _covering_radius_rad(pts)
        if cur_rad < best_rad:
            best_rad = cur_rad
            best_points = pts.copy()

        # Phase 1a: Greedy Lloyd (move worst facet vertices toward circumcenter)
        step = 0.02
        for it in range(300):
            if _time_left() < 10.0:
                break
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            radii, cc, mask = _facet_circumradii(pts, hull.simplices)
            if not np.any(mask):
                break
            # Find worst facet
            valid_idx = np.where(mask)[0]
            worst_local = valid_idx[np.argmax(radii[valid_idx])]
            worst_rad = radii[worst_local]
            if worst_rad < 0.15:  # ~8.6 degrees, good enough to stop Lloyd
                break
            simp = hull.simplices[worst_local]
            c = cc[worst_local]
            # Move vertices toward circumcenter
            for j in range(3):
                idx = simp[j]
                v = pts[idx]
                new_v = v + step * (c - v)
                new_v = new_v / np.linalg.norm(new_v)
                pts[idx] = new_v
            # Small random perturbation every 20 iters to escape local minima
            if it % 20 == 0:
                for _ in range(2):
                    idx = rng.randint(N)
                    tangent = rng.randn(3)
                    tangent -= np.dot(tangent, pts[idx]) * pts[idx]
                    tn = np.linalg.norm(tangent)
                    if tn > 1e-10:
                        tangent /= tn
                        new_v = pts[idx] + 0.005 * tangent
                        pts[idx] = new_v / np.linalg.norm(new_v)
            cur_rad = _covering_radius_rad(pts)
            if cur_rad < best_rad:
                best_rad = cur_rad
                best_points = pts.copy()
            # Decay step
            step *= 0.995
            if step < 0.001:
                step = 0.001

        # Phase 1b: Nelder-Mead Polish on Top Facets
        if _time_left() < 10.0:
            continue
        
        # Allocate remaining time for this polish, with a buffer
        alloc_time = min(_time_left() - 5.0, 20.0)
        if alloc_time < 2.0:
            continue
            
        pts = _nm_polish_top_facets(pts, max_time=alloc_time, k_facets=15, max_iter=200)
        cur_rad = _covering_radius_rad(pts)
        if cur_rad < best_rad:
            best_rad = cur_rad
            best_points = pts.copy()

    # Final high-precision polish on best_points
    if _time_left() > 5.0:
        pts = best_points.copy()
        # Use tighter tolerances and more vertices for the final push
        alloc_time = min(_time_left() - 2.0, 15.0)
        if alloc_time > 1.0:
            pts = _nm_polish_top_facets(pts, max_time=alloc_time, k_facets=20, max_iter=300)
            cur_rad = _covering_radius_rad(pts)
            if cur_rad < best_rad:
                best_rad = cur_rad
                best_points = pts.copy()

    return _normalize(best_points)
