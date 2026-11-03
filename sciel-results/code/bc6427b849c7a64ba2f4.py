import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _compute_covering_radius(points):
    """Compute exact covering radius via convex hull facet circumradii (in radians)."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    faces = hull.simplices
    max_circ = 0.0
    
    for face in faces:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        d = np.dot(n, a)
        t = d / (n_norm ** 2)
        o = t * n
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-10:
            continue
        o = o / o_norm
        
        # Check if Voronoi vertex is inside the spherical triangle
        n_ab = np.cross(a, b)
        n_bc = np.cross(b, c)
        n_ca = np.cross(c, a)
        
        same_side_ab = np.dot(c, n_ab) * np.dot(o, n_ab) > 0
        same_side_bc = np.dot(a, n_bc) * np.dot(o, n_bc) > 0
        same_side_ca = np.dot(b, n_ca) * np.dot(o, n_ca) > 0
        
        if not (same_side_ab and same_side_bc and same_side_ca):
            # Circumcenter is outside the triangle
            ab_ang = np.arccos(np.clip(np.dot(a, b), -1, 1))
            bc_ang = np.arccos(np.clip(np.dot(b, c), -1, 1))
            ca_ang = np.arccos(np.clip(np.dot(c, a), -1, 1))
            circumradius = np.max([ab_ang, bc_ang, ca_ang]) / 2.0
        else:
            dots = np.array([np.dot(a, o), np.dot(b, o), np.dot(c, o)])
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots)
            circumradius = np.max(angles)
        
        if circumradius > max_circ:
            max_circ = circumradius
    
    return max_circ

def _compute_facet_data(points):
    """
    Compute circumradius and Riemannian gradients for each facet.
    Returns:
        radii: array of circumradii for each facet
        grads: list of lists of arrays, each of shape (3,) where grads[k][i] is the gradient for vertex i of facet k
        faces: array of indices
        centers: array of Voronoi vertices (circumcenters) for each facet
        valid: boolean array indicating if circumcenter is inside the triangle
    """
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.array([]), [], np.array([]), np.array([]), np.array([])
    
    faces = hull.simplices
    radii = []
    grads = []
    centers = []
    valid = []
    
    for face in faces:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        d = np.dot(n, a)
        t = d / (n_norm ** 2)
        o = t * n
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-10:
            continue
        o = o / o_norm
        
        # Check if Voronoi vertex is inside the spherical triangle
        n_ab = np.cross(a, b)
        n_bc = np.cross(b, c)
        n_ca = np.cross(c, a)
        
        same_side_ab = np.dot(c, n_ab) * np.dot(o, n_ab) > 0
        same_side_bc = np.dot(a, n_bc) * np.dot(o, n_bc) > 0
        same_side_ca = np.dot(b, n_ca) * np.dot(o, n_ca) > 0
        
        is_valid = same_side_ab and same_side_bc and same_side_ca
        valid.append(is_valid)
        centers.append(o)
        
        if is_valid:
            dots = np.array([np.dot(a, o), np.dot(b, o), np.dot(c, o)])
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots)
            circumradius = np.max(angles)
            
            # Compute gradients for each vertex
            facet_grads = []
            for i, v in enumerate([a, b, c]):
                sin_r = np.sqrt(max(0, 1 - dots[i]**2))
                if sin_r < 1e-10:
                    grad = np.zeros(3)
                else:
                    grad = -o / sin_r
                grad_tangent = grad - np.dot(grad, v) * v
                facet_grads.append(grad_tangent)
            
            grads.append(facet_grads)
        else:
            ab_ang = np.arccos(np.clip(np.dot(a, b), -1, 1))
            bc_ang = np.arccos(np.clip(np.dot(b, c), -1, 1))
            ca_ang = np.arccos(np.clip(np.dot(c, a), -1, 1))
            circumradius = np.max([ab_ang, bc_ang, ca_ang]) / 2.0
            
            facet_grads = [np.zeros(3), np.zeros(3), np.zeros(3)]
            if ab_ang == np.max([ab_ang, bc_ang, ca_ang]):
                facet_grads[0] = -a / max(ab_ang, 1e-10)
                facet_grads[1] = -b / max(ab_ang, 1e-10)
            elif bc_ang == np.max([ab_ang, bc_ang, ca_ang]):
                facet_grads[1] = -b / max(bc_ang, 1e-10)
                facet_grads[2] = -c / max(bc_ang, 1e-10)
            else:
                facet_grads[0] = -a / max(ca_ang, 1e-10)
                facet_grads[2] = -c / max(ca_ang, 1e-10)
            
            for i, v in enumerate([a, b, c]):
                facet_grads[i] = facet_grads[i] - np.dot(facet_grads[i], v) * v
            
            grads.append(facet_grads)
        
        radii.append(circumradius)
    
    return np.array(radii), grads, faces, np.array(centers), np.array(valid)

def _gradient_optimize(points, max_iters=500, deadline=None, beta=20.0):
    """
    Optimize using gradient descent with soft-max weighting on facet circumradii.
    Uses log-sum-exp for smooth max approximation.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    step = 0.05
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        radii, grads, faces, centers, valid = _compute_facet_data(points)
        if len(radii) == 0:
            break
        
        current_radius = np.max(radii)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Soft-max weights
        shifted = radii - np.max(radii)
        weights = np.exp(beta * shifted)
        total_weight = np.sum(weights)
        if total_weight < 1e-30:
            break
        weights /= total_weight
        
        # Focus on facets close to the max
        threshold = np.max(radii) - 0.01
        active_mask = radii >= threshold
        
        total_grad = np.zeros((N, 3))
        for k in range(len(radii)):
            if not active_mask[k]:
                continue
            w = weights[k]
            face = faces[k]
            facet_grads = grads[k]
            for i, idx in enumerate(face):
                total_grad[idx] += w * facet_grads[i]
                
        grad_norm = np.linalg.norm(total_grad)
        if grad_norm < 1e-15:
            break
        
        total_grad = total_grad / grad_norm
        
        # Backtracking line search
        alpha = step
        improved = False
        for _ in range(10):
            new_points = points + alpha * total_grad
            new_points = _normalize(new_points)
            new_radius = _compute_covering_radius(new_points)
            if new_radius < current_radius - 1e-10:
                points = new_points
                current_radius = new_radius
                improved = True
                break
            else:
                alpha *= 0.5
        
        if improved:
            step = min(alpha * 1.2, 0.1)
        else:
            step *= 0.7
            if step < 1e-8:
                break
        
        step = np.clip(step, 1e-8, 0.1)
        
    return best_points, best_radius

def _local_facet_optimize(points, max_iters=300, deadline=None):
    """
    Targeted optimization: only move the 3 vertices of the worst facet.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        radii, grads, faces, centers, valid = _compute_facet_data(points)
        if len(radii) == 0:
            break
        
        current_radius = np.max(radii)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Find the worst facet
        worst_idx = np.argmax(radii)
        worst_face = faces[worst_idx]
        worst_grads = grads[worst_idx]
        
        improved = False
        alpha = 0.05
        
        for _ in range(5):
            new_points = points.copy()
            for i, idx in enumerate(worst_face):
                new_points[idx] = new_points[idx] + alpha * worst_grads[i]
            new_points = _normalize(new_points)
            new_radius = _compute_covering_radius(new_points)
            if new_radius < current_radius - 1e-12:
                points = new_points
                current_radius = new_radius
                improved = True
                break
            else:
                alpha *= 0.5
        
        if not improved:
            o = centers[worst_idx]
            if valid[worst_idx]:
                for idx in worst_face:
                    v = points[idx]
                    away = v - np.dot(v, o) * o
                    away_norm = np.linalg.norm(away)
                    if away_norm > 1e-10:
                        away = away / away_norm
                        new_points = points.copy()
                        new_points[idx] = new_points[idx] + 0.02 * away
                        new_points = _normalize(new_points)
                        new_radius = _compute_covering_radius(new_points)
                        if new_radius < current_radius - 1e-12:
                            points = new_points
                            current_radius = new_radius
                            improved = True
                            break
            
            if not improved:
                break
    
    return best_points, best_radius

def _fibonacci_spiral(n, jitter=0.0, seed=0):
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    if jitter > 0:
        i = i + rng.randn(n) * jitter
        i = np.clip(i, 0.5, n - 0.5)
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(points)

def _force_directed_optimize(points, max_iters=200, deadline=None, lr_init=0.02):
    """
    Force-directed optimization using Coulomb repulsion.
    """
    P = points.copy()
    best_P = P.copy()
    best_radius = _compute_covering_radius(P)
    lr = lr_init
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        P = _normalize(P)
        
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist3_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist3_sq, 1.0)
        dist3 = np.sqrt(dist3_sq)
        
        eps = 1e-6
        inv_dist3 = 1.0 / (dist3 ** 3 + eps)
        np.fill_diagonal(inv_dist3, 0.0)
        
        F = (diff * inv_dist3[:, :, np.newaxis]).sum(axis=1)
        
        P_new = P + lr * F
        P_new = _normalize(P_new)
        
        if iteration % 5 == 0 or iteration == max_iters - 1:
            new_radius = _compute_covering_radius(P_new)
            if new_radius < best_radius:
                best_radius = new_radius
                best_P = P_new.copy()
                P = P_new
                lr *= 1.03
            else:
                lr *= 0.95
        
        if lr < 1e-6:
            break
            
    return best_P, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_radius = np.pi
    
    inits = [_fibonacci_spiral(N)]
    
    rng = np.random.RandomState(42)
    for _ in range(5):
        perturbed = inits[0].copy()
        noise = rng.randn(N, 3) * 0.08
        perturbed = perturbed + noise
        perturbed = _normalize(perturbed)
        inits.append(perturbed)
    
    for init in inits:
        if time.time() > deadline - 25.0:
            break
        
        current_points = init.copy()
        current_radius = _compute_covering_radius(current_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = current_points.copy()
        
        if time.time() < deadline - 35.0:
            points, radius = _force_directed_optimize(current_points, max_iters=100, deadline=deadline - 35.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        if time.time() < deadline - 15.0:
            points, radius = _gradient_optimize(current_points, max_iters=300, deadline=deadline - 15.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        if time.time() < deadline - 5.0:
            points, radius = _local_facet_optimize(current_points, max_iters=200, deadline=deadline - 5.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = current_points.copy()
    
    if best_points is not None and time.time() < deadline - 10.0:
        points, radius = _gradient_optimize(best_points, max_iters=200, deadline=deadline - 10.0)
        if radius < best_radius:
            best_points = points
            best_radius = radius
            
        points, radius = _local_facet_optimize(best_points, max_iters=100, deadline=deadline - 3.0)
        if radius < best_radius:
            best_points = points
            best_radius = radius
    
    if best_points is not None:
        best_points = _normalize(best_points)
    else:
        best_points = _fibonacci_spiral(N)
    
    return best_points
