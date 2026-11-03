import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
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
        
        # Check if Voronoi vertex is valid (inside the spherical triangle)
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
    """
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.array([]), [], np.array([])
    
    faces = hull.simplices
    radii = []
    grads = []
    
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
        
        dots = np.array([np.dot(a, o), np.dot(b, o), np.dot(c, o)])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        circumradius = np.max(angles)
        
        radii.append(circumradius)
        
        # Compute gradients for each vertex
        # We use the approximation that o is fixed for the gradient w.r.t each vertex.
        # This is a common first-order approximation for local optimization.
        facet_grads = []
        for i, v in enumerate([a, b, c]):
            sin_r = np.sqrt(1 - dots[i]**2)
            if sin_r < 1e-10:
                grad = np.zeros(3)
            else:
                # Gradient of arccos(v . o) w.r.t v is -o / sin(r)
                grad = -o / sin_r
            
            # Project to tangent plane at v
            grad_tangent = grad - np.dot(grad, v) * v
            facet_grads.append(grad_tangent)
        
        grads.append(facet_grads)
    
    return np.array(radii), grads, faces

def _gradient_optimize(points, max_iters=300, deadline=None, beta=10.0):
    """
    Optimize using gradient descent with soft-max weighting on facet circumradii.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    step = 0.01
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        radii, grads, faces = _compute_facet_data(points)
        if len(radii) == 0:
            break
        
        current_radius = np.max(radii)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Soft-max weights
        # Shift for stability
        shifted = radii - np.max(radii)
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)
        
        # Aggregate gradients
        total_grad = np.zeros((N, 3))
        for k in range(len(radii)):
            w = weights[k]
            face = faces[k]
            facet_grads = grads[k]
            for i, idx in enumerate(face):
                total_grad[idx] += w * facet_grads[i]
                
        # Normalize gradients to prevent steps from being too large
        grad_norms = np.linalg.norm(total_grad, axis=1, keepdims=True)
        grad_norms[grad_norms < 1e-10] = 1.0
        total_grad = total_grad / grad_norms
        
        # Update points
        # Move in direction of negative gradient (since we want to minimize)
        new_points = points - step * total_grad
        new_points = _normalize(new_points)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < current_radius:
            points = new_points
            # Increase step slightly if we improved
            step *= 1.1
        else:
            # Decrease step
            step *= 0.8
            # If step is too small, break
            if step < 1e-6:
                break
        
        # Clamp step
        step = np.clip(step, 1e-5, 0.1)
        
    return best_points, best_radius

def _compute_facet_circumradii_and_centers(points):
    """
    For each point, compute the maximum circumradius among its adjacent facets,
    and the direction (toward the Voronoi vertex of the worst facet) to move.
    """
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.ones(N) * np.pi, np.zeros((N, 3))
    
    faces = hull.simplices
    max_circ_per_point = np.zeros(N)
    move_dirs = np.zeros((N, 3))
    
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
        
        dots = np.array([np.dot(a, o), np.dot(b, o), np.dot(c, o)])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        circumradius = np.max(angles)
        
        for idx, angle in zip(face, angles):
            if angle >= max_circ_per_point[idx]:
                max_circ_per_point[idx] = angle
                move_dirs[idx] = o
    
    norms = np.linalg.norm(move_dirs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    move_dirs = move_dirs / norms
    
    return max_circ_per_point, move_dirs

def _facet_based_optimize(points, max_iters=200, deadline=None):
    """
    Lloyd-type optimization: move each point toward the Voronoi vertex of its worst facet.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        max_circ, move_dirs = _compute_facet_circumradii_and_centers(points)
        current_radius = np.max(max_circ)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Adaptive step size
        step = 0.1 * np.sqrt(current_radius)
        step = np.clip(step, 0.001, 0.1)
        
        # Move points toward their worst facet Voronoi vertices
        new_points = points + step * move_dirs
        new_points = _normalize(new_points)
        
        # Check if this improved things; if not, try smaller step
        new_radius = _compute_covering_radius(new_points)
        if new_radius < current_radius:
            points = new_points
        else:
            # Try half step
            new_points = points + 0.5 * step * move_dirs
            new_points = _normalize(new_points)
            new_radius = _compute_covering_radius(new_points)
            if new_radius < current_radius:
                points = new_points
            else:
                # Try quarter step
                new_points = points + 0.25 * step * move_dirs
                new_points = _normalize(new_points)
                new_radius = _compute_covering_radius(new_points)
                if new_radius < current_radius:
                    points = new_points
                else:
                    break  # No improvement possible with this step
    
    return best_points, best_radius

def _nelder_mead_polish(points, max_iters=50, deadline=None):
    """
    Polish the configuration using Nelder-Mead on the exact covering radius.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    def objective(x):
        if deadline is not None and time.time() > deadline:
            return best_radius
        pts = x.reshape(N, 3)
        pts = _normalize(pts)
        return _compute_covering_radius(pts)
    
    try:
        res = minimize(
            objective,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        if res.fun < best_radius:
            new_points = res.x.reshape(N, 3)
            new_points = _normalize(new_points)
            new_radius = _compute_covering_radius(new_points)
            if new_radius < best_radius:
                best_points = new_points
                best_radius = new_radius
    except Exception:
        pass
    
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

def _spectral_force_optimize(points, max_iters=100, deadline=None, lr_init=0.05):
    """
    Fast, vectorized optimization using a spectral/force proxy.
    Avoids ConvexHull for speed.
    Uses Coulomb repulsion to spread points.
    """
    P = points.copy()
    best_P = P.copy()
    best_radius = _compute_covering_radius(P)
    lr = lr_init
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        P = _normalize(P)
        
        # Compute pairwise differences
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist3_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist3_sq, 1.0)
        dist3 = np.sqrt(dist3_sq)
        
        # Coulomb force: F_i = sum_j (P[i] - P[j]) / dist^3
        inv_dist3 = 1.0 / (dist3 ** 3)
        inv_dist3[dist3 < 1e-6] = 0.0
        F_coulomb = (diff * inv_dist3[:, :, np.newaxis]).sum(axis=1)
        
        # Update
        P_new = P + lr * F_coulomb
        P_new = _normalize(P_new)
        
        # Check improvement (using exact radius, but this is expensive. 
        # In a real scenario, we might use a proxy or check less frequently.
        # For this "fast" phase, we check every few iterations or use a cheaper metric.
        # Here we check every 10 iterations to save time, or just trust the force direction.
        # Let's check every 10 iterations.
        if iteration % 10 == 0 or iteration == max_iters - 1:
            new_radius = _compute_covering_radius(P_new)
            if new_radius < best_radius:
                best_radius = new_radius
                best_P = P_new.copy()
                P = P_new
                lr *= 1.05 # Increase step if improving
            else:
                P = P_new # Keep moving even if not immediate improvement? 
                # Or revert? Let's revert if it's worse than current best.
                # Actually, let's just keep P as P_new but track best_P.
                # If it gets too bad, we might stop.
                lr *= 0.9
        
        if lr < 1e-5:
            break
            
    return best_P, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Multiple restarts with different initializations
    best_points = None
    best_radius = np.pi
    
    # Initial configurations to try
    inits = [_fibonacci_spiral(N)]
    
    # Add some perturbed versions
    rng = np.random.RandomState(42)
    for _ in range(3):
        perturbed = inits[0].copy()
        noise = rng.randn(N, 3) * 0.05
        perturbed = perturbed + noise
        perturbed = _normalize(perturbed)
        inits.append(perturbed)
    
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        
        current_points = init.copy()
        current_radius = _compute_covering_radius(current_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = current_points.copy()
        
        # Phase 1: Fast Spectral/Force optimization (No ConvexHull)
        if time.time() < deadline - 40.0:
            points, radius = _spectral_force_optimize(current_points, max_iters=50, deadline=deadline - 40.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        # Phase 2: Facet-based Lloyd optimization
        if time.time() < deadline - 25.0:
            points, radius = _facet_based_optimize(current_points, max_iters=100, deadline=deadline - 25.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        # Phase 3: Gradient-based optimization
        if time.time() < deadline - 15.0:
            points, radius = _gradient_optimize(current_points, max_iters=200, deadline=deadline - 15.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
                
        # Phase 4: Nelder-Mead polish
        if time.time() < deadline - 5.0:
            points, radius = _nelder_mead_polish(current_points, max_iters=50, deadline=deadline - 5.0)
            if radius < current_radius:
                current_points = points
                current_radius = radius
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = current_points.copy()
    
    # Final polish on the best configuration
    if best_points is not None and time.time() < deadline - 5.0:
        points, radius = _gradient_optimize(best_points, max_iters=100, deadline=deadline - 5.0)
        if radius < best_radius:
            best_points = points
            best_radius = radius
            
        points, radius = _nelder_mead_polish(best_points, max_iters=30, deadline=deadline - 2.0)
        if radius < best_radius:
            best_points = points
            best_radius = radius
    
    # Ensure output is normalized
    if best_points is not None:
        best_points = _normalize(best_points)
    else:
        best_points = _fibonacci_spiral(N)
    
    return best_points
