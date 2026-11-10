import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _covering_radius(points):
    """
    Computes the covering radius in radians using the convex hull approach.
    """
    if len(points) < 3:
        return np.pi / 2
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi / 2

    max_radius = 0.0
    for face in hull.simplices:
        p0, p1, p2 = points[face[0]], points[face[1]], points[face[2]]
        
        e1 = p1 - p0
        e2 = p2 - p0
        e1e1 = np.dot(e1, e1)
        e1e2 = np.dot(e1, e2)
        e2e2 = np.dot(e2, e2)
        
        det = e1e1 * e2e2 - e1e2 * e1e2
        if abs(det) < 1e-12:
            centroid = (p0 + p1 + p2) / 3.0
            norm_c = np.linalg.norm(centroid)
            if norm_c < 1e-12:
                r_face = np.pi / 2
            else:
                centroid = centroid / norm_c
                d0 = np.arccos(np.clip(np.dot(centroid, p0), -1, 1))
                d1 = np.arccos(np.clip(np.dot(centroid, p1), -1, 1))
                d2 = np.arccos(np.clip(np.dot(centroid, p2), -1, 1))
                r_face = max(d0, d1, d2)
        else:
            u = (e1e1 * e2e2 - e1e2 * e2e2) / (2.0 * det)
            v = (e1e1 * e1e2 - e1e2 * e2e2) / (2.0 * det)
            O = p0 + u * e1 + v * e2
            norm_O = np.linalg.norm(O)
            if norm_O < 1e-12:
                centroid = (p0 + p1 + p2) / 3.0
                norm_c = np.linalg.norm(centroid)
                if norm_c < 1e-12:
                    r_face = np.pi / 2
                else:
                    centroid = centroid / norm_c
                    d0 = np.arccos(np.clip(np.dot(centroid, p0), -1, 1))
                    d1 = np.arccos(np.clip(np.dot(centroid, p1), -1, 1))
                    d2 = np.arccos(np.clip(np.dot(centroid, p2), -1, 1))
                    r_face = max(d0, d1, d2)
            else:
                C = O / norm_O
                d0 = np.arccos(np.clip(np.dot(C, p0), -1, 1))
                d1 = np.arccos(np.clip(np.dot(C, p1), -1, 1))
                d2 = np.arccos(np.clip(np.dot(C, p2), -1, 1))
                r_face = max(d0, d1, d2)
        
        if r_face > max_radius:
            max_radius = r_face
            
    return max_radius

def _repulsion_grad(points, power=6):
    """
    Gradient of the sum of 1/d^power repulsion potential.
    """
    N_pts = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq[dist_sq < 1e-12] = 1e-12
    
    mask = ~np.eye(N_pts, dtype=bool)
    
    coeff = power * (dist_sq ** (-(power + 2) / 2.0))
    coeff[~mask] = 0.0
    
    sum_coeff = np.sum(coeff, axis=1)
    term1 = sum_coeff[:, np.newaxis] * points
    term2 = coeff @ points
    
    grad = -term1 + term2
    
    dot_prod = np.sum(grad * points, axis=1, keepdims=True)
    grad_tan = grad - dot_prod * points
    
    return grad_tan

def _soft_max_covering_radius(points, temperature=0.01):
    """
    Compute a soft-max approximation of the covering radius for smooth optimization.
    """
    if len(points) < 3:
        return np.pi / 2
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi / 2

    face_radii = []
    for face in hull.simplices:
        p0, p1, p2 = points[face[0]], points[face[1]], points[face[2]]
        
        e1 = p1 - p0
        e2 = p2 - p0
        e1e1 = np.dot(e1, e1)
        e1e2 = np.dot(e1, e2)
        e2e2 = np.dot(e2, e2)
        
        det = e1e1 * e2e2 - e1e2 * e1e2
        if abs(det) < 1e-12:
            centroid = (p0 + p1 + p2) / 3.0
            norm_c = np.linalg.norm(centroid)
            if norm_c < 1e-12:
                r_face = np.pi / 2
            else:
                centroid = centroid / norm_c
                d0 = np.arccos(np.clip(np.dot(centroid, p0), -1, 1))
                d1 = np.arccos(np.clip(np.dot(centroid, p1), -1, 1))
                d2 = np.arccos(np.clip(np.dot(centroid, p2), -1, 1))
                r_face = max(d0, d1, d2)
        else:
            u = (e1e1 * e2e2 - e1e2 * e2e2) / (2.0 * det)
            v = (e1e1 * e1e2 - e1e2 * e2e2) / (2.0 * det)
            O = p0 + u * e1 + v * e2
            norm_O = np.linalg.norm(O)
            if norm_O < 1e-12:
                centroid = (p0 + p1 + p2) / 3.0
                norm_c = np.linalg.norm(centroid)
                if norm_c < 1e-12:
                    r_face = np.pi / 2
                else:
                    centroid = centroid / norm_c
                    d0 = np.arccos(np.clip(np.dot(centroid, p0), -1, 1))
                    d1 = np.arccos(np.clip(np.dot(centroid, p1), -1, 1))
                    d2 = np.arccos(np.clip(np.dot(centroid, p2), -1, 1))
                    r_face = max(d0, d1, d2)
            else:
                C = O / norm_O
                d0 = np.arccos(np.clip(np.dot(C, p0), -1, 1))
                d1 = np.arccos(np.clip(np.dot(C, p1), -1, 1))
                d2 = np.arccos(np.clip(np.dot(C, p2), -1, 1))
                r_face = max(d0, d1, d2)
        
        face_radii.append(r_face)
    
    face_radii = np.array(face_radii)
    if len(face_radii) == 0:
        return np.pi / 2
    
    # Soft-max approximation
    shifted = face_radii - np.max(face_radii)
    weights = np.exp(shifted / temperature)
    weights /= np.sum(weights)
    soft_max = np.sum(weights * face_radii)
    
    return soft_max

def _soft_max_grad(points, temperature=0.01):
    """
    Compute gradient of soft-max covering radius using finite differences.
    """
    if len(points) < 3:
        return np.zeros_like(points)
    
    eps = 1e-6
    grad = np.zeros_like(points)
    
    for i in range(len(points)):
        for j in range(3):
            orig = points[i, j]
            
            # Forward difference
            points[i, j] = orig + eps
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            points_normed = points / norms
            f_plus = _soft_max_covering_radius(points_normed, temperature)
            
            # Backward difference
            points[i, j] = orig - eps
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            points_normed = points / norms
            f_minus = _soft_max_covering_radius(points_normed, temperature)
            
            points[i, j] = orig
            
            grad[i, j] = (f_plus - f_minus) / (2 * eps)
    
    # Project onto tangent space
    dot_prod = np.sum(grad * points, axis=1, keepdims=True)
    grad_tan = grad - dot_prod * points
    
    return grad_tan

def _optimize_combined(points, deadline, use_soft_max=True):
    """
    Combined optimization using both repulsion gradient and soft-max covering radius gradient.
    """
    points = points.copy()
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    lr = 0.005
    min_lr = 1e-9
    decay = 0.998
    
    for step in range(8000):
        if time.time() > deadline:
            break
        
        # Compute repulsion gradient
        grad_rep = _repulsion_grad(points, power=6)
        
        # Optionally compute soft-max covering radius gradient
        if use_soft_max and step % 5 == 0:
            grad_soft = _soft_max_grad(points, temperature=0.01)
            grad = 0.3 * grad_rep + 0.7 * grad_soft
        else:
            grad = grad_rep
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        grad_dir = grad / grad_norm
        points_new = points + lr * grad_dir
        
        norms = np.linalg.norm(points_new, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        points_new = points_new / norms
        
        points = points_new
        lr *= decay
        
        if lr < min_lr:
            break
        
    return points

def _nelder_mead_worst(points, worst_indices, deadline, max_iter=200):
    """
    Apply Nelder-Mead optimization to the worst face vertices.
    """
    if len(worst_indices) < 2:
        return points
    
    points = points.copy()
    
    # Create simplex around current positions
    alpha = 0.1
    gamma = 0.5
    rho = 0.5
    
    # Initialize simplex
    simplex = [points.copy()]
    for idx in worst_indices:
        for dim in range(3):
            perturbation = points.copy()
            perturbation[idx, dim] += alpha
            norms = np.linalg.norm(perturbation, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            perturbation = perturbation / norms
            simplex.append(perturbation)
    
    # Evaluate initial simplex
    f_values = []
    for simplex_point in simplex:
        f_values.append(_covering_radius(simplex_point))
    
    best_idx = np.argmin(f_values)
    worst_idx = np.argmax(f_values)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Sort simplex
        order = np.argsort(f_values)
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        reflected = centroid + gamma * (centroid - simplex[-1])
        norms = np.linalg.norm(reflected, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        reflected = reflected / norms
        f_reflected = _covering_radius(reflected)
        
        if f_reflected < f_values[0]:
            # Expansion
            expanded = centroid + 2.0 * (reflected - centroid)
            norms = np.linalg.norm(expanded, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            expanded = expanded / norms
            f_expanded = _covering_radius(expanded)
            
            if f_expanded < f_reflected:
                simplex[-1] = expanded
                f_values[-1] = f_expanded
            else:
                simplex[-1] = reflected
                f_values[-1] = f_reflected
        elif f_reflected < f_values[-2]:
            simplex[-1] = reflected
            f_values[-1] = f_reflected
        else:
            # Contraction
            contracted = centroid + 0.5 * (simplex[-1] - centroid)
            norms = np.linalg.norm(contracted, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            contracted = contracted / norms
            f_contracted = _covering_radius(contracted)
            
            if f_contracted < f_values[-1]:
                simplex[-1] = contracted
                f_values[-1] = f_contracted
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    norms = np.linalg.norm(simplex[i], axis=1, keepdims=True)
                    norms[norms < 1e-15] = 1.0
                    simplex[i] = simplex[i] / norms
                    f_values[i] = _covering_radius(simplex[i])
        
        # Check convergence
        if max(f_values) - min(f_values) < 1e-10:
            break
    
    # Return best point
    best_idx = np.argmin(f_values)
    return simplex[best_idx]

def _get_worst_face_indices(points):
    """
    Get indices of points in the worst (largest radius) face.
    """
    if len(points) < 3:
        return []
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return []

    max_radius = 0.0
    worst_face = None
    for face in hull.simplices:
        p0, p1, p2 = points[face[0]], points[face[1]], points[face[2]]
        
        e1 = p1 - p0
        e2 = p2 - p0
        e1e1 = np.dot(e1, e1)
        e1e2 = np.dot(e1, e2)
        e2e2 = np.dot(e2, e2)
        
        det = e1e1 * e2e2 - e1e2 * e1e2
        if abs(det) < 1e-12:
            centroid = (p0 + p1 + p2) / 3.0
            norm_c = np.linalg.norm(centroid)
            if norm_c < 1e-12:
                r_face = np.pi / 2
            else:
                centroid = centroid / norm_c
                d0 = np.arccos(np.clip(np.dot(centroid, p0), -1, 1))
                d1 = np.arccos(np.clip(np.dot(centroid, p1), -1, 1))
                d2 = np.arccos(np.clip(np.dot(centroid, p2), -1, 1))
                r_face = max(d0, d1, d2)
        else:
            u = (e1e1 * e2e2 - e1e2 * e2e2) / (2.0 * det)
            v = (e1e1 * e1e2 - e1e2 * e2e2) / (2.0 * det)
            O = p0 + u * e1 + v * e2
            norm_O = np.linalg.norm(O)
            if norm_O < 1e-12:
                centroid = (p0 + p1 + p2) / 3.0
                norm_c = np.linalg.norm(centroid)
                if norm_c < 1e-12:
                    r_face = np.pi / 2
                else:
                    centroid = centroid / norm_c
                    d0 = np.arccos(np.clip(np.dot(centroid, p0), -1, 1))
                    d1 = np.arccos(np.clip(np.dot(centroid, p1), -1, 1))
                    d2 = np.arccos(np.clip(np.dot(centroid, p2), -1, 1))
                    r_face = max(d0, d1, d2)
            else:
                C = O / norm_O
                d0 = np.arccos(np.clip(np.dot(C, p0), -1, 1))
                d1 = np.arccos(np.clip(np.dot(C, p1), -1, 1))
                d2 = np.arccos(np.clip(np.dot(C, p2), -1, 1))
                r_face = max(d0, d1, d2)
        
        if r_face > max_radius:
            max_radius = r_face
            worst_face = face
    
    if worst_face is None:
        return []
    
    return list(worst_face)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_points = None
    best_radius = np.pi / 2.0
    
    # 1. Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    inits = [fib_points]
    
    # 2. Random initializations with different seeds
    for seed in [42, 123, 456, 789, 321]:
        rng = np.random.default_rng(seed)
        vecs = rng.normal(size=(N, 3))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        inits.append(vecs)
    
    # 3. Optimized initializations
    for init_pts in inits:
        if time.time() > deadline - 15.0:
            break
        
        # Phase 1: Combined optimization
        points = _optimize_combined(init_pts, deadline - 10.0, use_soft_max=True)
        radius = _covering_radius(points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    # Phase 2: Nelder-Mead on worst face
    if best_points is not None and time.time() < deadline - 10.0:
        points = best_points.copy()
        worst = _get_worst_face_indices(points)
        if len(worst) > 0:
            points = _nelder_mead_worst(points, worst, deadline - 5.0, max_iter=150)
            radius = _covering_radius(points)
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()
    
    # Phase 3: Fine polish with very small steps
    if best_points is not None and time.time() < deadline - 5.0:
        points = best_points.copy()
        
        # Ultra-fine polish
        lr = 0.0005
        for step in range(3000):
            if time.time() > deadline - 2.0:
                break
            grad = _repulsion_grad(points, power=10)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-16:
                break
            grad_dir = grad / grad_norm
            points = points + lr * grad_dir
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            points = points / norms
            lr *= 0.997
            
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()

    if best_points is None:
        return fib_points
        
    return best_points
