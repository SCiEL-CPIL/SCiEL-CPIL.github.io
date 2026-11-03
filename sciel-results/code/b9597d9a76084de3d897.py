N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-14] = 1.0
    return pts / norms


def _facet_circumradii(pts):
    """
    Compute circumradii (radians) for all triangular facets of the convex hull.
    Returns (radii_array, circumcenters_array, simplices_array, valid_mask).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.array([np.pi / 2]), np.array([[0, 0, 1]]), np.array([[0, 1, 2]]), np.array([True])
    
    simplices = hull.simplices
    n_facets = len(simplices)
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    valid = np.ones(n_facets, dtype=bool)
    
    for k in range(n_facets):
        p1, p2, p3 = pts[simplices[k, 0]], pts[simplices[k, 1]], pts[simplices[k, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            valid[k] = False
            radii[k] = 0.0
            continue
        n = n / nn
        # Ensure normal points outward
        if np.dot(n, p1) < 0:
            n = -n
        centers[k] = n
        dots = np.array([np.dot(n, p1), np.dot(n, p2), np.dot(n, p3)])
        dots = np.clip(dots, -1.0, 1.0)
        radii[k] = np.max(np.arccos(dots))
    
    return radii, centers, simplices, valid


def _covering_radius_rad(pts):
    radii, _, _, valid = _facet_circumradii(pts)
    if not np.any(valid):
        return np.pi / 2
    return float(np.max(radii[valid]))


def _covering_radius_deg(pts):
    return np.degrees(_covering_radius_rad(pts))


def _smooth_surrogate(pts, beta=10.0):
    """
    Smooth surrogate objective: log-sum-exp of squared facet radii.
    Returns (value, weights, radii, centers, simplices, valid).
    """
    radii, centers, simplices, valid = _facet_circumradii(pts)
    if not np.any(valid):
        return np.pi, None, None, None, None, None
    
    r_valid = radii[valid]
    c_valid = centers[valid]
    s_valid = simplices[valid]
    
    r2 = r_valid * r_valid
    m = np.max(r2)
    exps = np.exp((r2 - m) * beta)
    s = np.sum(exps)
    if s < 1e-300:
        weights = np.ones_like(exps) / len(exps)
    else:
        weights = exps / s
    
    surrogate = m + np.log(s) / beta
    return surrogate, weights, r_valid, c_valid, s_valid, valid


def _surrogate_gradient(pts, weights, simplices, centers, radii):
    """
    Gradient of the smooth surrogate w.r.t. points.
    For each facet, push its vertices toward the circumcenter.
    """
    grad = np.zeros_like(pts)
    for k in range(len(simplices)):
        w = weights[k]
        if w < 1e-8:
            continue
        c = centers[k]
        r_k = radii[k]
        cos_r = np.clip(np.dot(c, pts[simplices[k, 0]]), -1.0, 1.0)
        sin_r = np.sqrt(max(1.0 - cos_r * cos_r, 1e-12))
        
        for j in range(3):
            idx = simplices[k, j]
            v = pts[idx]
            # Direction from v to c, projected to tangent at v
            d = c - v
            d_t = d - np.dot(d, v) * v
            d_t_norm = np.linalg.norm(d_t)
            if d_t_norm < 1e-10:
                continue
            d_t = d_t / d_t_norm
            # Magnitude: proportional to r_k / sin(r_k) * w
            mag = 2.0 * r_k / sin_r * w
            grad[idx] -= mag * d_t
    
    # Project gradient to tangent plane
    grad = grad - np.einsum('ij,ij->i', grad, pts)[:, None] * pts
    return grad


def _electrostatic_refine(pts, n_iters=200, lr=0.01):
    """
    Repulsion-based refinement using Coulomb-like forces on the sphere.
    """
    pts = pts.copy()
    for _ in range(n_iters):
        D = pts @ pts.T
        chord2 = 2.0 - 2.0 * D
        np.fill_diagonal(chord2, 1.0)
        W = 1.0 / chord2
        np.fill_diagonal(W, 0.0)
        row_sum = W.sum(axis=1)
        forces = pts * row_sum[:, None] - (W @ pts)
        # Project to tangent plane
        forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        forces /= norms
        pts = pts + lr * forces
        pts = _normalize(pts)
    return pts


def _fibonacci_sphere(n, offset=0.0):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _lloyd_refine(pts, deadline, max_iter=500):
    """
    Greedy Lloyd refinement: move vertices of the worst facet toward its circumcenter.
    Combined with electrostatic repulsion to avoid clustering.
    """
    best_pts = pts.copy()
    best_cr = _covering_radius_rad(pts)
    
    step = 0.03
    no_improve = 0
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        radii, centers, simplices, valid = _facet_circumradii(pts)
        if not np.any(valid):
            break
        
        # Find worst facet
        valid_indices = np.where(valid)[0]
        worst_idx = valid_indices[np.argmax(radii[valid_indices])]
        worst_r = radii[worst_idx]
        worst_c = centers[worst_idx]
        worst_s = simplices[worst_idx]
        
        # Move vertices of worst facet toward circumcenter
        new_pts = pts.copy()
        for j in range(3):
            idx = worst_s[j]
            v = pts[idx]
            d = worst_c - v
            d_t = d - np.dot(d, v) * v
            d_t_norm = np.linalg.norm(d_t)
            if d_t_norm > 1e-10:
                new_pts[idx] = v + step * (d_t / d_t_norm)
        
        # Add small electrostatic repulsion to all points
        D = new_pts @ new_pts.T
        chord2 = 2.0 - 2.0 * D
        np.fill_diagonal(chord2, 1.0)
        W = 1.0 / chord2
        np.fill_diagonal(W, 0.0)
        row_sum = W.sum(axis=1)
        forces = new_pts * row_sum[:, None] - (W @ new_pts)
        forces = forces - np.einsum('ij,ij->i', forces, new_pts)[:, None] * new_pts
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        forces /= norms
        new_pts = new_pts + 0.005 * forces
        
        new_pts = _normalize(new_pts)
        new_cr = _covering_radius_rad(new_pts)
        
        if new_cr < best_cr - 1e-10:
            best_cr = new_cr
            best_pts = new_pts.copy()
            pts = new_pts
            no_improve = 0
            step = min(step * 1.01, 0.05)
        else:
            no_improve += 1
            step *= 0.95
            if no_improve >= 30:
                # Perturb to escape local minimum
                noise = np.random.randn(len(pts), 3) * 0.01
                pts = _normalize(best_pts + noise)
                no_improve = 0
                step = 0.03
    
    return best_pts


def _gradient_descent(pts, deadline, max_iter=300, lr=0.01):
    """
    Gradient descent on the smooth surrogate objective.
    """
    best_pts = pts.copy()
    best_val = _covering_radius_rad(pts)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        surrogate, weights, radii, centers, simplices, valid = _smooth_surrogate(pts)
        if weights is None:
            break
        
        grad = _surrogate_gradient(pts, weights, simplices, centers, radii)
        
        # Adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-8:
            break
        
        new_pts = pts - lr * (grad / grad_norm)
        new_pts = _normalize(new_pts)
        
        new_cr = _covering_radius_rad(new_pts)
        
        if new_cr < best_val - 1e-10:
            best_val = new_cr
            best_pts = new_pts.copy()
            pts = new_pts
            lr = min(lr * 1.02, 0.05)
        else:
            lr *= 0.9
            if lr < 1e-5:
                break
    
    return best_pts


def _lbfgs_polish(pts, deadline, max_iter=100):
    """
    Final polishing with L-BFGS-B using the smooth surrogate as objective.
    """
    def objective(x):
        p = _normalize(x.reshape(N, 3))
        val, _, _, _, _, _ = _smooth_surrogate(p)
        return val
    
    def jac(x):
        p = _normalize(x.reshape(N, 3))
        _, weights, radii, centers, simplices, valid = _smooth_surrogate(p)
        if weights is None:
            return np.zeros_like(x)
        grad = _surrogate_gradient(p, weights, simplices, centers, radii)
        return grad.flatten()
    
    x0 = pts.flatten()
    
    try:
        res = minimize(
            objective,
            x0,
            jac=jac,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-4}
        )
        return _normalize(res.x.reshape(N, 3))
    except Exception:
        return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Phase 1: Generate multiple diverse starting configurations
    starts = [
        _fibonacci_sphere(N, 0.0),
        _fibonacci_sphere(N, 0.3),
        _fibonacci_sphere(N, 0.7),
        _random_sphere(N, 1),
        _random_sphere(N, 2),
    ]
    
    # Electrostatic refinement from best Fibonacci
    starts.append(_electrostatic_refine(_fibonacci_sphere(N, 0.0), n_iters=100, lr=0.005))
    
    best_pts = starts[0].copy()
    best_cr = _covering_radius_rad(best_pts)
    
    # Evaluate all initializations
    for s in starts:
        cr = _covering_radius_rad(s)
        if cr < best_cr:
            best_cr = cr
            best_pts = s.copy()
    
    # Phase 2: For each of the top 3 initializations, do Lloyd + gradient descent
    scored_starts = []
    for s in starts:
        cr = _covering_radius_rad(s)
        scored_starts.append((cr, s.copy()))
    scored_starts.sort(key=lambda x: x[0])
    
    for i, (cr, s) in enumerate(scored_starts[:3]):
        if time.time() > deadline - 30:
            break
        
        sub_deadline = min(deadline - 15, start_time + 35 * (i + 1))
        
        try:
            # Lloyd refinement
            lloyd_pts = _lloyd_refine(s, sub_deadline, max_iter=300)
            lloyd_cr = _covering_radius_rad(lloyd_pts)
            
            if lloyd_cr < best_cr:
                best_cr = lloyd_cr
                best_pts = lloyd_pts.copy()
            
            # Gradient descent from Lloyd result
            if time.time() < deadline - 10:
                grad_pts = _gradient_descent(lloyd_pts, deadline - 10, max_iter=200, lr=0.01)
                grad_cr = _covering_radius_rad(grad_pts)
                
                if grad_cr < best_cr:
                    best_cr = grad_cr
                    best_pts = grad_pts.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B polishing
    if time.time() < deadline - 5:
        polished = _lbfgs_polish(best_pts, deadline - 2, max_iter=80)
        cr = _covering_radius_rad(polished)
        if cr < best_cr:
            best_cr = cr
            best_pts = polished
    
    # Final validation: ensure no coincident points, not in hemisphere
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            min_dist = min(min_dist, d)
    if min_dist < 1e-6:
        rng = np.random.default_rng(42)
        best_pts = _normalize(best_pts + rng.randn(N, 3) * 1e-4)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
