import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 130

def _thomson_init(n, iters=5000, lr_start=0.01, lr_end=0.0001):
    """Initialize n points on unit sphere via Thomson (electrostatic) relaxation."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    for it in range(iters):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist[dist < 1e-10] = 1e-10
        
        inv_dist3 = 1.0 / (dist ** 3)
        inv_dist3[np.arange(n)[:, None] == np.arange(n)[None, :]] = 0.0
        
        forces = np.sum(diff * inv_dist3[:, :, np.newaxis], axis=1)
        
        t = it / iters
        lr = lr_start * (lr_end / lr_start) ** t
        
        pts = pts + lr * forces
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        pts = pts / norms
    
    return pts

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return pts / norms

def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_facets(points):
    """Compute exact covering radius using convex hull facets."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: grid-based estimate
        return _covering_radius_grid(points, 360, 720)
    
    max_cr = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]
        if len(verts) < 3:
            continue
        
        if len(verts) == 3:
            v1, v2, v3 = verts
            e1 = v2 - v1
            e2 = v3 - v1
            A = np.array([[np.dot(e1, e1), np.dot(e2, e1)],
                           [np.dot(e1, e2), np.dot(e2, e2)]])
            b = np.array([np.dot(e1, e1) / 2, np.dot(e2, e2) / 2])
            
            try:
                alpha, beta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            
            pc = v1 + alpha * e1 + beta * e2
            pc_norm = np.linalg.norm(pc)
            if pc_norm < 1e-12:
                continue
            
            cos_R = np.dot(v1, pc) / pc_norm
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R = np.arccos(cos_R)
            max_cr = max(max_cr, R)
        else:
            # For higher-dimensional facets, use the first 3 vertices as approximation
            # This is rare for 3D convex hulls of points in general position
            v1, v2, v3 = verts[0], verts[1], verts[2]
            e1 = v2 - v1
            e2 = v3 - v1
            A = np.array([[np.dot(e1, e1), np.dot(e2, e1)],
                           [np.dot(e1, e2), np.dot(e2, e2)]])
            b = np.array([np.dot(e1, e1) / 2, np.dot(e2, e2) / 2])
            
            try:
                alpha, beta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            
            pc = v1 + alpha * e1 + beta * e2
            pc_norm = np.linalg.norm(pc)
            if pc_norm < 1e-12:
                continue
            
            cos_R = np.dot(v1, pc) / pc_norm
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R = np.arccos(cos_R)
            max_cr = max(max_cr, R)
    
    return np.degrees(max_cr)

def _covering_radius_grid(points, n_theta=360, n_phi=720):
    """Fallback: estimate covering radius using a grid on the sphere."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)
    
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    max_dist = np.max(min_dists)
    
    return np.degrees(max_dist)

def _soft_covering_obj_grad(flat_pts, test_pts, beta):
    """
    Computes the soft covering radius and its gradient.
    J = (1/beta) * log( sum_y exp(beta * min_i d(y, p_i)) )
    """
    pts = _normalize(flat_pts.reshape(N, 3))
    
    # d(y, p_i) = arccos(y . p_i)
    dots = test_pts @ pts.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # (M, N)
    
    # min_i d(y, p_i)
    min_dists = np.min(dists, axis=1)  # (M,)
    
    # Soft max
    max_d = np.max(min_dists)
    exp_term = np.exp(beta * (min_dists - max_d))
    S = np.sum(exp_term)
    J = max_d + np.log(S) / beta
    
    # Weights w_y = exp(beta * (min_dists - max_d)) / S
    w = exp_term / S  # (M,)
    
    # Gradient of J wrt p_i
    # dJ/dp_i = sum_y w_y * d(min_d)/dp_i
    # min_d is determined by the nearest point p_k for each y.
    # Let k(y) be the index of the nearest point to y.
    # d(min_d)/dp_i = 0 if i != k(y)
    # d(min_d)/dp_{k(y)} = gradient of d(y, p_{k(y)}) wrt p_{k(y)}
    
    # Find nearest index for each y
    nearest_idx = np.argmin(dists, axis=1)  # (M,)
    
    # Gradient of distance d(y, p) wrt p:
    # g = (y - (y.p)p) / ||y - (y.p)p|| = (y - cos(d)p) / sin(d)
    # We only need this for the nearest point.
    
    # y_b: (M, 1, 3)
    # p_k: (M, 1, 3) - the nearest point for each y
    y_b = test_pts[:, np.newaxis, :]
    p_k = pts[nearest_idx, :]  # (M, 3)
    p_k_b = p_k[:, np.newaxis, :]
    
    dots_nearest = dots[np.arange(len(test_pts)), nearest_idx]  # (M,)
    cos_d = dots_nearest
    sin_d = np.sqrt(np.maximum(1e-10, 1.0 - cos_d * cos_d))
    
    # grad_d = (y - cos_d * p_k) / sin_d
    # shape (M, 1, 3)
    grad_d = (y_b - cos_d[:, np.newaxis, np.newaxis] * p_k_b) / sin_d[:, np.newaxis, np.newaxis]
    
    # Accumulate gradients for each point i
    # grad_pts[i] = sum_{y: k(y)=i} w_y * grad_d[y]
    grad_pts = np.zeros((N, 3))
    np.add.at(grad_pts, nearest_idx, w[:, np.newaxis] * grad_d)
    
    grad_flat = grad_pts.flatten()
    
    return J, grad_flat

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Phase 0: Thomson initialization
    pts = _thomson_init(N, iters=2000, lr_start=0.01, lr_end=0.0001)
    best_pts = pts.copy()
    best_score = _covering_radius_facets(best_pts)
    
    # Test points for soft objective
    # Use a dense grid to approximate the true covering radius
    n_test = 100000
    test_points = _fibonacci_spiral(n_test)
    
    # Phase 1: Coarse optimization with soft-soft objective
    # Start with low beta (smooth landscape)
    beta = 5.0
    
    def objective_coarse(flat_pts):
        val, _ = _soft_covering_obj_grad(flat_pts, test_points, beta)
        return val
        
    def grad_coarse(flat_pts):
        _, g = _soft_covering_obj_grad(flat_pts, test_points, beta)
        return g
        
    try:
        res = minimize(
            objective_coarse,
            grad_coarse,
            pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 100, 'gtol': 1e-5, 'maxfun': 200}
        )
        candidate = _normalize(res.x.reshape(N, 3))
        candidate_score = _covering_radius_facets(candidate)
        if candidate_score < best_score:
            best_score = candidate_score
            best_pts = candidate
    except Exception:
        pass
    
    if time.time() > deadline - 60:
        return best_pts
    
    # Phase 2: Fine optimization
    beta = 20.0
    
    def objective_fine(flat_pts):
        val, _ = _soft_covering_obj_grad(flat_pts, test_points, beta)
        return val
        
    def grad_fine(flat_pts):
        _, g = _soft_covering_obj_grad(flat_pts, test_points, beta)
        return g

    try:
        res2 = minimize(
            objective_fine,
            grad_fine,
            best_pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 100, 'gtol': 1e-6, 'maxfun': 200}
        )
        candidate = _normalize(res2.x.reshape(N, 3))
        candidate_score = _covering_radius_facets(candidate)
        if candidate_score < best_score:
            best_score = candidate_score
            best_pts = candidate
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        return best_pts
    
    # Phase 3: Very fine polish
    beta = 50.0
    
    def objective_vf(flat_pts):
        val, _ = _soft_covering_obj_grad(flat_pts, test_points, beta)
        return val
        
    def grad_vf(flat_pts):
        _, g = _soft_covering_obj_grad(flat_pts, test_points, beta)
        return g

    try:
        res3 = minimize(
            objective_vf,
            grad_vf,
            best_pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 100, 'gtol': 1e-7, 'maxfun': 200}
        )
        candidate = _normalize(res3.x.reshape(N, 3))
        candidate_score = _covering_radius_facets(candidate)
        if candidate_score < best_score:
            best_score = candidate_score
            best_pts = candidate
    except Exception:
        pass
    
    # Phase 4: Local facet-based refinement
    # Try to shrink the worst facet
    for _ in range(10):
        if time.time() > deadline - 5:
            break
            
        points = _normalize(best_pts)
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Find the facet with largest circumradius
        worst_simplex = None
        worst_R = 0.0
        for simplex in hull.simplices:
            verts = points[simplex]
            if len(verts) == 3:
                v1, v2, v3 = verts
                e1 = v2 - v1
                e2 = v3 - v1
                A = np.array([[np.dot(e1, e1), np.dot(e2, e1)],
                               [np.dot(e1, e2), np.dot(e2, e2)]])
                b = np.array([np.dot(e1, e1) / 2, np.dot(e2, e2) / 2])
                try:
                    alpha, beta_c = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue
                pc = v1 + alpha * e1 + beta_c * e2
                pc_norm = np.linalg.norm(pc)
                if pc_norm < 1e-12:
                    continue
                cos_R = np.dot(v1, pc) / pc_norm
                cos_R = np.clip(cos_R, -1.0, 1.0)
                R = np.arccos(cos_R)
                if R > worst_R:
                    worst_R = R
                    worst_simplex = simplex
        
        if worst_simplex is None:
            break
        
        # Try moving vertices of the worst facet towards the circumcenter
        # This shrinks the triangle
        verts_idx = worst_simplex
        verts = points[verts_idx]
        v1, v2, v3 = verts
        e1 = v2 - v1
        e2 = v3 - v1
        A = np.array([[np.dot(e1, e1), np.dot(e2, e1)],
                       [np.dot(e1, e2), np.dot(e2, e2)]])
        b = np.array([np.dot(e1, e1) / 2, np.dot(e2, e2) / 2])
        try:
            alpha, beta_c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        pc = v1 + alpha * e1 + beta_c * e2
        pc_norm = np.linalg.norm(pc)
        if pc_norm < 1e-12:
            continue
        pc_dir = pc / pc_norm
        
        improved = False
        # Try moving each vertex slightly towards pc_dir
        for idx in verts_idx:
            for step in [0.001, 0.005, 0.01, 0.02]:
                trial = best_pts.copy()
                trial[idx] = trial[idx] + step * (pc_dir - trial[idx])
                trial = _normalize(trial)
                
                trial_score = _covering_radius_facets(trial)
                if trial_score < best_score:
                    best_score = trial_score
                    best_pts = trial
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # Try moving a point NOT in the facet towards the circumcenter
            # This might create a new, smaller facet
            for idx in range(N):
                if idx in worst_simplex:
                    continue
                for step in [0.001, 0.005]:
                    trial = best_pts.copy()
                    trial[idx] = trial[idx] + step * (pc_dir - trial[idx])
                    trial = _normalize(trial)
                    
                    trial_score = _covering_radius_facets(trial)
                    if trial_score < best_score:
                        best_score = trial_score
                        best_pts = trial
                        improved = True
                        break
                
                if improved:
                    break
        
        if not improved:
            break
    
    return _normalize(best_pts)
