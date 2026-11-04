import numpy as np
from scipy.spatial import ConvexHull
import time
import scipy.optimize

N = 130

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _facet_circumradii(points, simplices):
    """
    Compute circumradii and circumcenters for all hull facets (vectorized).
    Returns:
        radii: (M,) array of angular circumradii
        centers: (M, 3) array of circumcenters on sphere
        mask: (M,) boolean array of valid facets
    """
    M = len(simplices)
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    norm_cross = np.linalg.norm(cross, axis=1)
    
    valid = norm_cross > 1e-14
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        c = cross[valid] / norm_cross[valid, np.newaxis]
        
        centroid = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dot_c_centroid = np.sum(c * centroid, axis=1)
        flip = dot_c_centroid < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        dot_val = np.clip(np.sum(c * p1[valid], axis=1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        radii[valid] = np.where(dist > np.pi / 2, np.pi - dist, dist)
    
    return radii, centers, valid

def _covering_radius_rad(points):
    """
    Compute the covering radius of a set of points on the unit sphere (in radians).
    """
    try:
        hull = ConvexHull(points)
        radii, centers, valid = _facet_circumradii(points, hull.simplices)
        if not np.any(valid):
            return np.pi
        return float(np.max(radii[valid]))
    except Exception:
        return np.pi

def _covering_radius_deg(points):
    return _covering_radius_rad(points) * 180.0 / np.pi

def _soft_max_objective_and_grad_flat(flat_points, beta=50.0):
    """
    Compute the soft-max (log-sum-exp) surrogate for the max circumradius
    and its gradient with respect to the flattened point positions.
    Returns:
        objective: float
        grad: flat array of shape (3*N,)
    """
    points = flat_points.reshape(N, 3)
    points = _normalize(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return 1.0, np.zeros(3 * N)
    
    radii, centers, valid = _facet_circumradii(points, hull.simplices)
    M = len(radii)
    
    if M == 0 or not np.any(valid):
        return 0.0, np.zeros(3 * N)
    
    r_valid = radii[valid]
    c_valid = centers[valid]
    f_valid = hull.simplices[valid]
    
    if len(r_valid) == 0:
        return 0.0, np.zeros(3 * N)
    
    r_max = np.max(r_valid)
    if r_max < 1e-15:
        return 0.0, np.zeros(3 * N)
    
    shifted = beta * (r_valid - r_max)
    # Numerical stability for log-sum-exp
    max_shifted = np.max(shifted)
    log_sum_exp = r_max + (max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))) / beta
    
    weights = np.exp(shifted - max_shifted) / np.sum(np.exp(shifted - max_shifted))
    
    grad = np.zeros(3 * N)
    
    for k in range(len(f_valid)):
        if weights[k] < 1e-10:
            continue
        
        tri = f_valid[k]
        c = c_valid[k]
        r_k = r_valid[k]
        
        if r_k < 1e-12:
            continue
        
        for idx in tri:
            p = points[idx]
            dot_cp = np.dot(c, p)
            sin_r = np.sqrt(max(0.0, 1.0 - dot_cp * dot_cp))
            if sin_r > 1e-12:
                # Gradient of r = arccos(c . p) w.r.t. p is -(c - (c.p)p) / sin(r)
                grad_p = -(c - dot_cp * p) / sin_r
                # Project to tangent plane
                grad_p -= np.dot(grad_p, p) * p
                # Add to flat gradient
                grad[idx * 3: idx * 3 + 3] += weights[k] * grad_p
    
    return float(log_sum_exp), grad

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    def time_left():
        return deadline - time.time()

    rng = np.random.default_rng(42)
    
    # Phase 0: Generate diverse initializations
    inits = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_pts = _normalize(fib_pts)
    
    # Perturbed Fibonacci
    fib_pert = fib_pts + rng.normal(0, 0.05, fib_pts.shape)
    fib_pert = _normalize(fib_pert)
    
    inits.append(fib_pts)
    inits.append(fib_pert)
    
    # 2. Random points
    for _ in range(5):
        pts = rng.normal(0, 1, (N, 3))
        pts = _normalize(pts)
        inits.append(pts)
    
    # 3. Electrostatic refinement of Fibonacci
    def electrostatic_refine(pts, n_iters=100, lr=0.01):
        pts = pts.copy()
        for _ in range(n_iters):
            D = pts @ pts.T
            chord2 = 2.0 - 2.0 * D
            np.fill_diagonal(chord2, 1.0)
            W = 1.0 / chord2
            np.fill_diagonal(W, 0.0)
            row_sum = W.sum(axis=1)
            forces = pts * row_sum[:, None] - (W @ pts)
            forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
            norms = np.linalg.norm(forces, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            forces /= norms
            pts = pts + lr * forces
            pts = _normalize(pts)
        return pts
    
    if time_left() > 10.0:
        elec_pts = electrostatic_refine(fib_pts, n_iters=200, lr=0.005)
        inits.append(elec_pts)
    
    # Evaluate initializations and pick the best
    best_points = None
    best_radius = np.pi
    
    for init in inits:
        if time_left() < 5.0:
            break
        rad = _covering_radius_rad(init)
        if rad < best_radius:
            best_radius = rad
            best_points = init.copy()
    
    if best_points is None:
        best_points = fib_pts
        best_radius = _covering_radius_rad(fib_pts)
    
    # Phase 1: Multi-start L-BFGS-B optimization on soft-max surrogate
    # We use a moderate beta to keep the surrogate smooth but close to the max
    beta = 50.0
    
    # We will try to optimize from the best initial point and a few random restarts
    # L-BFGS-B is robust and uses a quasi-Newton method
    
    # Define objective and gradient functions for scipy
    def objective_and_grad(flat_pts):
        return _soft_max_objective_and_grad_flat(flat_pts, beta)
    
    # Optimization settings
    max_iter = 50 # Limit iterations per restart to stay within time budget
    
    # Try optimizing from the current best
    if time_left() > 20.0:
        x0 = best_points.flatten()
        try:
            res = scipy.optimize.minimize(
                lambda x: objective_and_grad(x)[0],
                x0,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'gtol': 1e-5}
            )
            new_points = _normalize(res.x.reshape(N, 3))
            new_radius = _covering_radius_rad(new_points)
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points
        except Exception:
            pass
    
    # Try a few random restarts if time permits
    restarts = 5
    for i in range(restarts):
        if time_left() < 25.0:
            break
        
        # Generate a new random initialization
        x0_flat = rng.normal(0, 1, 3 * N)
        x0_pts = _normalize(x0_flat.reshape(N, 3))
        
        # Quick local polish with electrostatics to get a better start for L-BFGS
        if time_left() > 30.0:
             x0_pts = electrostatic_refine(x0_pts, n_iters=50, lr=0.01)
        
        x0 = x0_pts.flatten()
        try:
            res = scipy.optimize.minimize(
                lambda x: objective_and_grad(x)[0],
                x0,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'gtol': 1e-5}
            )
            new_points = _normalize(res.x.reshape(N, 3))
            new_radius = _covering_radius_rad(new_points)
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points
        except Exception:
            pass
    
    # Phase 2: High-precision polish using Nelder-Mead on the worst facets
    # This is a derivative-free method, structurally different from L-BFGS
    # We apply it to the best points found
    
    points = best_points.copy()
    polish_deadline = deadline - 5.0
    
    # Use a small number of Nelder-Mead steps on the tangent space?
    # Or just simple local search on the worst facet vertices.
    # Let's stick to a simple local search on the worst facet to avoid complexity.
    
    polish_step = 0.003
    for _ in range(100):
        if time.time() > polish_deadline:
            break
        
        try:
            hull = ConvexHull(points)
            radii, centers, valid = _facet_circumradii(points, hull.simplices)
            if not np.any(valid):
                break
        except Exception:
            break
        
        r_valid = radii[valid]
        c_valid = centers[valid]
        f_valid = hull.simplices[valid]
        
        max_idx = np.argmax(r_valid)
        max_r = r_valid[max_idx]
        worst_facet = f_valid[max_idx]
        worst_center = c_valid[max_idx]
        
        # Move points in the worst facet towards the circumcenter
        improved = False
        for idx in worst_facet:
            v = worst_center - points[idx]
            dot_vp = np.dot(v, points[idx])
            tangent_v = v - dot_vp * points[idx]
            norm_tv = np.linalg.norm(tangent_v)
            if norm_tv > 1e-12:
                points[idx] += polish_step * tangent_v / norm_tv
                improved = True
        
        if not improved:
            break
            
        points = _normalize(points)
        
        current_radius = _covering_radius_rad(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        else:
            polish_step *= 0.7
            if polish_step < 1e-7:
                break
            points = best_points.copy()
    
    return best_points
