import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _facet_data(pts):
    """Compute convex hull and return facet data: radii, circumcenters, simplices, valid mask."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    
    simplices = hull.simplices
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    # Normal vectors
    n = np.cross(a - b, a - c)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm > 1e-10
    n = n / np.maximum(n_norm[:, None], 1e-10)
    
    # Ensure normals point outward
    centroids = (a + b + c) / 3.0
    signs = np.where(np.einsum('ij,ij->i', n, centroids) < 0, -1.0, 1.0)
    n = n * signs[:, None]
    
    # Circumradii: cos(r) = n . a
    dots = np.clip(np.einsum('ij,ij->i', n, a), -1.0, 1.0)
    radii = np.arccos(dots)
    radii[~mask] = 0.0
    
    return radii, n, simplices, mask

def _covering_radius_deg(pts):
    """Exact covering radius in degrees via convex hull facet circumradii."""
    data = _facet_data(pts)
    if data is None:
        return 180.0
    radii, _, _, mask = data
    if not np.any(mask):
        return 180.0
    return float(np.degrees(np.max(radii[mask])))

def _covering_radius_rad(pts):
    data = _facet_data(pts)
    if data is None:
        return np.pi
    radii, _, _, mask = data
    if not np.any(mask):
        return np.pi
    return float(np.max(radii[mask]))

def _fibonacci_spiral(offset=0.0):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _electrostatic_refine(pts, n_iters=300, lr=0.01):
    """Repulsion-based refinement."""
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
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        forces = forces / (force_norms + 1e-12)
        current_lr = lr * (0.99 ** (it / 50))
        pts = pts + current_lr * forces
        pts = _normalize(pts)
    return pts

def _subgradient_of_max_facet(pts, worst_idx):
    """
    Compute the subgradient of the covering radius (max facet radius) w.r.t. each point.
    The covering radius is max_k r_k where r_k = arccos(n_k . a_k).
    Subgradient w.r.t. point i: if i is a vertex of facet k, the gradient is along the tangent direction.
    
    For a facet with vertices v0, v1, v2 and circumcenter c, radius r:
    d(r)/d(v_j) projected to tangent = -sin(r) * (c - v_j)_tangent / |c - v_j|_tangent * (1/sin(r)) ... 
    Actually, let's derive more carefully.
    
    r_k = arccos(c_k . v_j) for any vertex v_j of facet k.
    d(r_k)/d(v_j) = -1/sqrt(1 - (c_k.v_j)^2) * c_k = -1/sin(r_k) * c_k
    But v_j is constrained to the sphere, so we project: grad_tangent = (-c_k / sin(r_k)) - ((-c_k/sin(r_k)).v_j) v_j
    Since c_k . v_j = cos(r_k), the projection is:
    grad_tangent = -c_k/sin(r_k) + (cos(r_k)/sin(r_k)) v_j
    
    The subgradient of max_k r_k at the point where facet k is maximized is the gradient w.r.t. that facet.
    So the subgradient w.r.t. point i is the sum over all worst facets k that contain i of grad_tangent_k,i.
    """
    data = _facet_data(pts)
    if data is None:
        return np.zeros_like(pts)
    
    radii, cc, simplices, mask = data
    if not np.any(mask):
        return np.zeros_like(pts)
    
    valid_idx = np.where(mask)[0]
    # Get the worst facet(s) - use all facets within a small tolerance of the max
    max_rad = np.max(radii[valid_idx])
    tol = 1e-8
    worst_set = valid_idx[radii[valid_idx] > max_rad - tol]
    
    grad = np.zeros_like(pts)
    for k in worst_set:
        c = cc[k]
        r = radii[k]
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue
        for j in range(3):
            idx = simplices[k, j]
            v = pts[idx]
            # Tangent gradient: -c/sin(r) + (cos(r)/sin(r)) * v
            grad_tan = -c / sin_r + (np.cos(r) / sin_r) * v
            # Project to tangent plane at v
            grad_tan = grad_tan - np.dot(grad_tan, v) * v
            grad[idx] += grad_tan
    
    # Normalize the subgradient to have unit norm (for trust region)
    g_norm = np.linalg.norm(grad)
    if g_norm > 1e-10:
        grad /= g_norm
    return grad

def _trust_region_step(pts, subgrad, delta=0.1):
    """
    Take a trust region step on the sphere.
    We want to move in the direction of -subgrad (since we want to decrease the covering radius).
    The step is constrained to be within a trust region of radius delta on the sphere.
    """
    # Move in the direction of -subgrad, projected to tangent
    direction = -subgrad
    # Project to tangent plane at each point
    direction = direction - np.einsum('ij,ij->i', direction, pts)[:, None] * pts
    
    # Normalize direction
    dir_norms = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / np.maximum(dir_norms, 1e-12)
    
    # Take a step of size delta in the direction
    new_pts = pts + delta * direction
    new_pts = _normalize(new_pts)
    return new_pts

def _lloyd_step(pts, n_worst=3):
    """
    Lloyd-like step: move vertices of worst facets toward their circumcenters.
    This is a heuristic but often effective.
    """
    data = _facet_data(pts)
    if data is None:
        return pts.copy()
    
    radii, cc, simplices, mask = data
    if not np.any(mask):
        return pts.copy()
    
    valid_idx = np.where(mask)[0]
    top_k = min(n_worst, len(valid_idx))
    worst_locals = valid_idx[np.argsort(-radii[valid_idx])][:top_k]
    
    new_pts = pts.copy()
    moved = set()
    for worst_local in worst_locals:
        simp = simplices[worst_local]
        c = cc[worst_local]
        for j in range(3):
            idx = simp[j]
            if idx in moved:
                continue
            v = new_pts[idx]
            # Move toward circumcenter with a fixed step
            step = 0.02
            new_v = v + step * (c - v)
            new_v = new_v / np.linalg.norm(new_v)
            new_pts[idx] = new_v
            moved.add(idx)
    
    return new_pts

def construct():
    start_time = time.time()
    max_time = 145.0  # Safety margin before 150s kill
    
    def _time_left():
        return max_time - (time.time() - start_time)
    
    # Phase 0: Multiple initializations
    rng = np.random.RandomState(42)
    inits = []
    
    # 1. Fibonacci + Electrostatic
    fib1 = _fibonacci_spiral(0.0)
    inits.append(_electrostatic_refine(fib1, n_iters=300, lr=0.02))
    
    # 2. Fibonacci (offset) + Electrostatic
    fib2 = _fibonacci_spiral(0.5)
    inits.append(_electrostatic_refine(fib2, n_iters=300, lr=0.02))
    
    # 3. Random + Electrostatic
    rand1 = _normalize(rng.randn(N, 3))
    inits.append(_electrostatic_refine(rand1, n_iters=500, lr=0.02))
    
    # 4. Raw Fibonacci (fallback)
    inits.append(fib1)
    
    best_points = inits[0]
    best_rad = _covering_radius_deg(best_points)
    
    for init in inits:
        if _time_left() < 5.0:
            break
        rad = _covering_radius_deg(init)
        if rad < best_rad:
            best_rad = rad
            best_points = init.copy()
    
    # Phase 1: Trust region optimization with subgradient
    pts = best_points.copy()
    cur_rad = _covering_radius_deg(pts)
    
    trust_radius = 0.1
    min_trust_radius = 0.001
    max_trust_radius = 0.5
    
    for it in range(200):
        if _time_left() < 10.0:
            break
        
        # Compute subgradient of the max facet radius
        data = _facet_data(pts)
        if data is None:
            break
        radii, cc, simplices, mask = data
        if not np.any(mask):
            break
        
        valid_idx = np.where(mask)[0]
        max_rad = np.max(radii[valid_idx])
        if max_rad < 0.05:  # ~2.86 degrees, very good
            break
        
        # Get subgradient
        subgrad = _subgradient_of_max_facet(pts, valid_idx[np.argmax(radii[valid_idx])])
        
        # Take trust region step
        new_pts = _trust_region_step(pts, subgrad, delta=trust_radius)
        new_rad = _covering_radius_deg(new_pts)
        
        # Update trust region
        if new_rad < cur_rad - 1e-10:
            # Success: accept and possibly increase trust region
            pts = new_pts
            cur_rad = new_rad
            if cur_rad < best_rad:
                best_rad = cur_rad
                best_points = pts.copy()
            trust_radius = min(trust_radius * 1.5, max_trust_radius)
        else:
            # Failure: decrease trust region
            trust_radius = max(trust_radius * 0.5, min_trust_radius)
            if trust_radius < min_trust_radius * 1.1:
                # Try Lloyd step as fallback
                new_pts = _lloyd_step(pts, n_worst=3)
                new_rad = _covering_radius_deg(new_pts)
                if new_rad < cur_rad - 1e-10:
                    pts = new_pts
                    cur_rad = new_rad
                    if cur_rad < best_rad:
                        best_rad = cur_rad
                        best_points = pts.copy()
                    trust_radius = min_trust_radius
                else:
                    # Stagnation: break
                    break
    
    # Phase 2: Polish with small steps
    if _time_left() > 10.0:
        pts = best_points.copy()
        cur_rad = _covering_radius_deg(pts)
        
        for it in range(100):
            if _time_left() < 5.0:
                break
            
            data = _facet_data(pts)
            if data is None:
                break
            radii, cc, simplices, mask = data
            if not np.any(mask):
                break
            
            valid_idx = np.where(mask)[0]
            max_rad = np.max(radii[valid_idx])
            if max_rad < 0.05:
                break
            
            # Move vertices of worst facets toward circumcenters with small steps
            worst_locals = valid_idx[np.argsort(-radii[valid_idx])][:2]
            improved = False
            for worst_local in worst_locals:
                if _time_left() < 3.0:
                    break
                simp = simplices[worst_local]
                c = cc[worst_local]
                for j in range(3):
                    idx = simp[j]
                    v = pts[idx]
                    for s in [0.001, 0.0005, 0.0002, 0.0001]:
                        new_v = v + s * (c - v)
                        new_v = new_v / np.linalg.norm(new_v)
                        old_v = pts[idx].copy()
                        pts[idx] = new_v
                        new_rad = _covering_radius_deg(pts)
                        if new_rad < best_rad - 1e-10:
                            best_rad = new_rad
                            best_points = pts.copy()
                            improved = True
                            break
                        else:
                            pts[idx] = old_v
            
            if not improved and it > 50:
                break
    
    return _normalize(best_points)
