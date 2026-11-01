import numpy as np
import scipy.spatial
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, ConvexHull

N = 50
TIME_LIMIT = 145.0  # safety margin from 150s

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii (vectorized).
    Returns: hull, centers (array), radii (array), worst_idx
    """
    points = _normalize(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None, None
    
    facets = hull.simplices
    n_facets = len(facets)
    if n_facets == 0:
        return hull, np.zeros((0, 3)), np.zeros(0), None
    
    a = points[facets[:, 0]]  # (F, 3)
    b = points[facets[:, 1]]
    c = points[facets[:, 2]]
    
    # Vectorized circumcenter computation
    v1 = b - a
    v2 = c - a
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1)
    
    valid = norm_n > 1e-12
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    if np.any(valid):
        n_norm = n[valid] / norm_n[valid, None]
        
        # Solve for circumcenter: C = a + s*v1 + t*v2
        # where C is equidistant from a, b, c
        # |C-a|^2 = |C-b|^2 => 2*C.(b-a) = |b-a|^2
        # |C-a|^2 = |C-c|^2 => 2*C.(c-a) = |c-a|^2
        
        v1_v = v1[valid]
        v2_v = v2[valid]
        
        # 2*(s*v1 + t*v2).v1 = |v1|^2
        # 2*(s*v1 + t*v2).v2 = |v2|^2
        
        A = np.array([
            [np.einsum('ij,ij->i', v1_v, v1_v), np.einsum('ij,ij->i', v1_v, v2_v)],
            [np.einsum('ij,ij->i', v2_v, v1_v), np.einsum('ij,ij->i', v2_v, v2_v)]
        ])
        A = A.transpose(1, 0, 2)  # (F, 2, 2)
        
        rhs = np.array([
            0.5 * np.einsum('ij,ij->i', v1_v, v1_v),
            0.5 * np.einsum('ij,ij->i', v2_v, v2_v)
        ])
        rhs = rhs.transpose(1, 0)  # (F, 2)
        
        try:
            # Solve A @ [s, t]^T = rhs for each facet
            det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
            valid_det = np.abs(det) > 1e-14
            
            s = np.zeros(np.sum(valid))
            t = np.zeros(np.sum(valid))
            
            if np.any(valid_det):
                s[valid_det] = (rhs[valid_det, 0] * A[valid_det, 1, 1] - rhs[valid_det, 1] * A[valid_det, 0, 1]) / det[valid_det]
                t[valid_det] = (A[valid_det, 0, 0] * rhs[valid_det, 1] - A[valid_det, 1, 0] * rhs[valid_det, 0]) / det[valid_det]
            
            C_valid = a[valid] + s[:, None] * v1_v + t[:, None] * v2_v
            C_norm = np.linalg.norm(C_valid, axis=1, keepdims=True)
            C_norm = np.maximum(C_norm, 1e-15)
            centers_valid = C_valid / C_norm
            
            # Compute radii
            dots = np.einsum('ij,ij->i', centers_valid, a[valid])
            dots = np.clip(dots, -1.0, 1.0)
            radii_valid = np.arccos(dots)
            
            centers[valid] = centers_valid
            radii[valid] = radii_valid
        except Exception:
            pass
    
    if n_facets > 0 and np.any(radii > 0):
        worst_idx = np.argmax(radii)
    else:
        worst_idx = None
    
    return hull, centers, radii, worst_idx

def _compute_covering_radius(points):
    """Compute the covering radius (max facet circumradius)."""
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi
    return radii[worst_idx]

def _soft_max_radius_and_grad(points, beta=50.0):
    """
    Compute soft-max of facet radii and its gradient.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi, np.zeros_like(points)
    
    max_r = np.max(radii)
    shifted = beta * (radii - max_r)
    max_shifted = np.max(shifted)
    exp_terms = np.exp(shifted - max_shifted)
    sum_exp = np.sum(exp_terms)
    if sum_exp < 1e-300:
        return max_r, np.zeros_like(points)
    log_sum = max_shifted + np.log(sum_exp)
    soft_radius = max_r + log_sum / beta
    weights = exp_terms / sum_exp
    
    grads = np.zeros_like(points)
    n_facets = len(hull.simplices)
    
    for i in range(n_facets):
        w = weights[i]
        if w < 1e-15:
            continue
        r = radii[i]
        if r < 1e-12:
            continue
        
        C = centers[i]
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue
        
        # Gradient of arccos(dot(C, p)) w.r.t. p is -C/sin(r) projected to tangent space
        grad_contrib = -C / sin_r
        
        simplex = hull.simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            p = points[p_idx]
            # Project gradient to tangent space at p
            radial = np.dot(grad_contrib, p) * p
            tangential = grad_contrib - radial
            grads[p_idx] += w * tangential
    
    return soft_radius, grads

def _lbfgs_optimize(points, time_budget, beta=80.0, maxiter=1000):
    """L-BFGS-B optimization of soft-max covering radius."""
    start_time = time.time()
    
    def objective_and_grad(x):
        if time.time() - start_time > time_budget:
            raise Exception("timeout")
        pts = _normalize(x.reshape(N, 3))
        soft_r, grads = _soft_max_radius_and_grad(pts, beta=beta)
        return soft_r, grads.reshape(-1)
    
    x0 = points.reshape(-1)
    current_best = _compute_covering_radius(points)
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 100}
        )
        if result.fun < current_best:
            return _normalize(result.x.reshape(N, 3))
    except Exception:
        pass
    
    return points

def _local_polish(points, time_budget, beta=100.0):
    """
    Local polish using projected gradient descent on tangent space.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.02
    min_step = 1e-13
    decay = 0.85
    growth = 1.05
    
    while time.time() - start_time < time_budget:
        soft_radius, grads = _soft_max_radius_and_grad(current_points, beta=beta)
        
        true_radius = _compute_covering_radius(current_points)
        if true_radius < best_radius - 1e-14:
            best_radius = true_radius
            best_points = current_points.copy()
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-16:
            break
        
        direction = grads / grad_norm
        new_points = current_points + step_size * direction
        new_points = _normalize(new_points)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < true_radius - 1e-15:
            current_points = new_points
            step_size = min(step_size * growth, 0.1)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _repulsive_polish(points, time_budget):
    """
    Repulsive force based polish.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.012
    min_step = 1e-13
    decay = 0.85
    growth = 1.05
    
    while time.time() - start_time < time_budget:
        hull, centers, radii, worst_idx = _compute_facet_data(current_points)
        if hull is None:
            break
        
        true_radius = radii[worst_idx] if worst_idx is not None else np.pi
        if true_radius < best_radius - 1e-14:
            best_radius = true_radius
            best_points = current_points.copy()
        
        n = len(current_points)
        forces = np.zeros_like(current_points)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d_vec = current_points[i] - current_points[j]
                d = np.linalg.norm(d_vec)
                if d < 0.005:
                    d = 0.005
                d_hat = d_vec / d
                radial_i = np.dot(d_hat, current_points[i]) * current_points[i]
                tangential_i = d_hat - radial_i
                mag = 1.0 / (d ** 2)
                forces[i] += mag * tangential_i
                forces[j] -= mag * tangential_i
        
        # Pressure from worst facets
        if worst_idx is not None and len(centers) > 0:
            sorted_idx = np.argsort(radii)[::-1]
            n_worst = min(10, len(sorted_idx))
            max_r = np.max(radii)
            
            for idx in sorted_idx[:n_worst]:
                if radii[idx] < 1e-10:
                    continue
                C = centers[idx]
                r = radii[idx]
                sin_r = np.sin(r)
                if sin_r < 1e-10:
                    continue
                
                weight = (radii[idx] / max_r) ** 3
                
                for i in range(n):
                    dot_ic = np.dot(current_points[i], C)
                    ang_dist = np.arccos(np.clip(dot_ic, -1.0, 1.0))
                    if ang_dist > r + 0.3:
                        continue
                    
                    radial_C = dot_ic * current_points[i]
                    tangential_C = C - radial_C
                    tangential_C_norm = np.linalg.norm(tangential_C)
                    if tangential_C_norm < 1e-10:
                        continue
                    tangential_C = tangential_C / tangential_C_norm
                    
                    strength = weight * max(0, 1 - ang_dist / (r + 0.3))
                    forces[i] += strength * tangential_C
        
        force_norm = np.linalg.norm(forces)
        if force_norm < 1e-16:
            break
        
        direction = forces / force_norm
        new_points = current_points + step_size * direction
        new_points = _normalize(new_points)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < true_radius - 1e-15:
            current_points = new_points
            step_size = min(step_size * growth, 0.06)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _generate_seeds():
    """Generate diverse initial configurations."""
    seeds = []
    
    # 1. Fibonacci sphere
    j = np.arange(N) + 0.5
    z = 1.0 - 2.0 * j / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    seeds.append(fib)
    
    # 2. Random starts
    for seed_val in [42, 1, 2, 3, 7, 13, 99, 123, 555, 777, 2024, 31415, 9999, 4242]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts = _normalize(pts)
        seeds.append(pts)
    
    # 3. Icosahedron-based
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_verts = _normalize(ico_verts)
    rng_ico = np.random.default_rng(12345)
    extra = rng_ico.normal(size=(N - len(ico_verts), 3))
    extra = _normalize(extra)
    ico_seed = np.vstack([ico_verts, extra])
    seeds.append(ico_seed)
    
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = None
    best_radius = np.inf
    
    seeds = _generate_seeds()
    n_seeds = len(seeds)
    
    # Stage 1: L-BFGS optimization for each seed
    base_time_per_seed = 8.0
    
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 15:
            break
        
        if i < n_seeds - 1:
            time_budget = min(base_time_per_seed, remaining / max(1, n_seeds - i))
        else:
            time_budget = min(15.0, remaining)
        
        if time_budget < 4.0:
            break
        
        pts = _lbfgs_optimize(seed_pts, time_budget, beta=80.0, maxiter=800)
        
        # Quick local polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 12:
            pts, rad = _local_polish(pts, min(6.0, remaining - 8), beta=100.0)
        else:
            rad = _compute_covering_radius(pts)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    # Stage 2: Final polish on best candidate
    if best_points is not None:
        # Repulsive polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 25:
            budget = min(30.0, remaining - 12)
            if budget > 8:
                pts, rad = _repulsive_polish(best_points, budget)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Fine local polish with moderate beta
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 18:
            budget = min(20.0, remaining - 10)
            if budget > 5:
                pts, rad = _local_polish(best_points, budget, beta=300.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Very fine polish with high beta
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 12:
            budget = min(12.0, remaining - 6)
            if budget > 3:
                pts, rad = _local_polish(best_points, budget, beta=800.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Final ultra-fine polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 8:
            budget = min(6.0, remaining - 4)
            if budget > 2:
                pts, rad = _local_polish(best_points, budget, beta=2000.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    if best_points is None:
        best_points = seeds[0]
    
    # Ensure output is normalized
    best_points = _normalize(best_points)
    
    return best_points
