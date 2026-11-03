import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _facet_circumradii(points):
    """Compute circumradii of all hull facets. Returns array of radii in radians."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.array([np.pi])
    
    radii = []
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        n = np.cross(v1 - v2, v1 - v3)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            radii.append(0.0)
            continue
        c = n / norm_n
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
        rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        radii.append(rad)
    return np.array(radii)

def _covering_radius(points):
    """Compute the covering radius in radians."""
    radii = _facet_circumradii(points)
    return np.max(radii) if len(radii) > 0 else np.pi

def _fibonacci_spiral():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _electrostatic_energy(points):
    """Compute electrostatic-like repulsion energy to spread points."""
    # Use a soft repulsion to avoid singularity
    min_pair_dist = 0.05
    energy = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.arccos(np.clip(np.dot(points[i], points[j]), -1.0, 1.0))
            d = max(d, min_pair_dist)
            energy += 1.0 / (d * d)
    return energy

def _smooth_max(x, beta=50.0):
    """Smooth maximum using log-sum-exp."""
    if len(x) == 0:
        return 0.0
    max_x = np.max(x)
    if max_x < -50:
        return max_x
    shifted = (x - max_x) * beta
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    return max_x + np.log(np.sum(np.exp(shifted)) / len(x)) / beta

def _surrogate_objective(points, beta=50.0):
    """
    Smooth surrogate: smooth max of facet circumradii.
    This is differentiable and avoids the nonsmoothness of max.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    radii = []
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        n = np.cross(v1 - v2, v1 - v3)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        c = n / norm_n
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
        rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        radii.append(rad)
    
    if not radii:
        return np.pi
    
    radii_arr = np.array(radii)
    return _smooth_max(radii_arr, beta)

def _gradient_step(points, step_size=0.01, beta=50.0):
    """
    Perform a gradient step on the smooth surrogate.
    We compute the gradient numerically by perturbing each point.
    """
    n_points = len(points)
    gradient = np.zeros_like(points)
    
    eps = 0.001
    base_obj = _surrogate_objective(points, beta)
    
    for i in range(n_points):
        for d in range(3):
            # Perturb in direction d
            old_val = points[i, d]
            
            # Tangent direction
            tangent = np.zeros(3)
            tangent[d] = 1.0
            tangent -= np.dot(tangent, points[i]) * points[i]
            norm_t = np.linalg.norm(tangent)
            if norm_t < 1e-10:
                continue
            tangent /= norm_t
            
            points[i] = points[i] + eps * tangent
            norm = np.linalg.norm(points[i])
            if norm > 1e-10:
                points[i] = points[i] / norm
            
            new_obj = _surrogate_objective(points, beta)
            
            # Restore
            points[i, d] = old_val
            # Re-normalize the point
            norm = np.linalg.norm(points[i])
            if norm > 1e-10:
                points[i] = points[i] / norm
            
            gradient[i, d] = (new_obj - base_obj) / eps
    
    # Project gradient to tangent space
    for i in range(n_points):
        gradient[i] -= np.dot(gradient[i], points[i]) * points[i]
    
    return gradient

def construct():
    start_time = time.time()
    max_time = 145.0  # Safety margin before 150s kill
    
    # Initialize with Fibonacci spiral
    points = _fibonacci_spiral()
    best_points = points.copy()
    best_rad = _covering_radius(best_points)
    
    # Phase 1: Greedy Lloyd-style optimization
    step_size = 0.03
    for it in range(5000):
        if time.time() - start_time > max_time:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Find the facet with the largest circumradius
        max_facet_rad = 0.0
        worst_circumcenter = None
        worst_indices = None
        
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            n = np.cross(v1 - v2, v1 - v3)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            c = n / norm_n
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
            rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
            
            if rad > max_facet_rad:
                max_facet_rad = rad
                worst_circumcenter = c
                worst_indices = simplex
        
        if worst_indices is None:
            break
        
        # Move vertices of worst facet toward circumcenter
        for j in range(3):
            v_idx = worst_indices[j]
            v = points[v_idx]
            new_v = v + step_size * (worst_circumcenter - v)
            norm = np.linalg.norm(new_v)
            if norm > 1e-10:
                points[v_idx] = new_v / norm
        
        # Also apply small random perturbations to escape local minima
        if it % 50 == 0:
            for _ in range(3):
                idx = np.random.randint(N)
                # Random direction on tangent plane
                tangent = np.random.randn(3)
                tangent -= np.dot(tangent, points[idx]) * points[idx]
                norm_t = np.linalg.norm(tangent)
                if norm_t > 1e-10:
                    tangent /= norm_t
                    new_v = points[idx] + 0.01 * tangent
                    norm = np.linalg.norm(new_v)
                    if norm > 1e-10:
                        points[idx] = new_v / norm
        
        current_rad = _covering_radius(points)
        if current_rad < best_rad:
            best_rad = current_rad
            best_points = points.copy()
    
    # Phase 2: Gradient-based polishing on the smooth surrogate
    if time.time() - start_time < max_time:
        points = best_points.copy()
        
        # Adaptive step size for gradient descent
        grad_step = 0.005
        for grad_it in range(200):
            if time.time() - start_time > max_time:
                break
            
            grad = _gradient_step(points, step_size=0.001, beta=50.0)
            
            # Check if gradient is small
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-6:
                break
            
            # Take step
            new_points = points - grad_step * grad
            
            # Re-normalize all points
            for i in range(N):
                norm = np.linalg.norm(new_points[i])
                if norm > 1e-10:
                    new_points[i] = new_points[i] / norm
            
            new_rad = _covering_radius(new_points)
            if new_rad < best_rad:
                best_rad = new_rad
                best_points = new_points.copy()
                points = new_points.copy()
            else:
                # Reduce step size
                grad_step *= 0.8
    
    # Phase 3: Fine-tuning with smaller steps
    if time.time() - start_time < max_time:
        points = best_points.copy()
        
        for fine_it in range(100):
            if time.time() - start_time > max_time:
                break
            
            try:
                hull = ConvexHull(points)
            except Exception:
                break
            
            # Get all facet radii and find worst
            max_facet_rad = 0.0
            worst_circumcenter = None
            worst_indices = None
            
            for simplex in hull.simplices:
                v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                n = np.cross(v1 - v2, v1 - v3)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                c = n / norm_n
                centroid = (v1 + v2 + v3) / 3.0
                if np.dot(c, centroid) < 0:
                    c = -c
                rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
                
                if rad > max_facet_rad:
                    max_facet_rad = rad
                    worst_circumcenter = c
                    worst_indices = simplex
            
            if worst_indices is None:
                break
            
            # Move vertices of worst facet toward circumcenter with smaller step
            for j in range(3):
                v_idx = worst_indices[j]
                v = points[v_idx]
                new_v = v + 0.01 * (worst_circumcenter - v)
                norm = np.linalg.norm(new_v)
                if norm > 1e-10:
                    points[v_idx] = new_v / norm
            
            current_rad = _covering_radius(points)
            if current_rad < best_rad:
                best_rad = current_rad
                best_points = points.copy()
    
    return best_points
