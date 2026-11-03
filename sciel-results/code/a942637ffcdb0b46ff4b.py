import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _facet_circumradii(points):
    """Compute circumradii of all convex hull facets in radians."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        radii = np.zeros(len(facets))
        
        for i, tri in enumerate(facets):
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                radii[i] = 0.0
                continue
                
            c = cross / norm_cross
            if np.dot(c, p1) < 0:
                c = -c
                
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            radii[i] = np.arccos(dot_val)
            
        return radii, facets
    except Exception:
        return np.array([np.pi / 2]), np.array([[]])

def _compute_covering_radius(points):
    """Compute covering radius in degrees."""
    radii, _ = _facet_circumradii(points)
    return np.degrees(np.max(radii)) if len(radii) > 0 else np.pi / 2

def _soft_max(x, beta=100.0):
    """Soft maximum using log-sum-exp."""
    x_max = np.max(x)
    if x_max > 1e6:
        return x_max
    shifted = x - x_max
    return x_max + np.log(np.sum(np.exp(beta * shifted))) / beta

def _soft_max_grad(x, beta=100.0):
    """Gradient of soft max."""
    x_max = np.max(x)
    shifted = x - x_max
    exp_shifted = np.exp(beta * shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    return weights

def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(points)

def _gradient_descent_smooth(points, max_iters=500, lr_init=0.01, beta=50.0, deadline=None):
    """Gradient descent on soft-max surrogate for covering radius."""
    points = points.copy()
    lr = lr_init
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        radii, facets = _facet_circumradii(points)
        if len(radii) == 0:
            break
            
        # Soft max of radii
        soft_r = _soft_max(radii, beta)
        weights = _soft_max_grad(radii, beta)
        
        # Compute gradient w.r.t. each point
        grad = np.zeros_like(points)
        
        for i, tri in enumerate(facets):
            if weights[i] < 1e-8:
                continue
                
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            if np.dot(c, p1) < 0:
                c = -c
                
            # Gradient of arccos(c.p1) w.r.t. p1
            # d/dp1 arccos(c.p1) = -c / sqrt(1 - (c.p1)^2)
            dot_val = np.dot(c, p1)
            denom = np.sqrt(max(1e-12, 1.0 - dot_val * dot_val))
            
            # The gradient of the circumradius w.r.t. each vertex
            # For vertex p1: -c / denom (projected to tangent plane)
            g1 = -c / denom
            g1 = g1 - np.dot(g1, p1) * p1  # Project to tangent plane
            
            # Similarly for p2 and p3
            dot_val2 = np.dot(c, p2)
            denom2 = np.sqrt(max(1e-12, 1.0 - dot_val2 * dot_val2))
            g2 = -c / denom2
            g2 = g2 - np.dot(g2, p2) * p2
            
            dot_val3 = np.dot(c, p3)
            denom3 = np.sqrt(max(1e-12, 1.0 - dot_val3 * dot_val3))
            g3 = -c / denom3
            g3 = g3 - np.dot(g3, p3) * p3
            
            # Weight by soft-max weight
            w = weights[i]
            grad[tri[0]] += w * g1
            grad[tri[1]] += w * g2
            grad[tri[2]] += w * g3
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
            
        # Update points
        points = points - lr * grad
        points = _normalize(points)
        
        # Decay learning rate
        lr *= 0.995
        
    return points

def _greedy_facet_shrink(points, deadline=None):
    """Greedy facet shrinking: move worst facet vertices toward circumcenter."""
    points = points.copy()
    step_size = 0.02
    min_step_size = 1e-4
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    while time.time() < deadline:
        radii, facets = _facet_circumradii(points)
        if len(radii) == 0:
            break
            
        worst_idx = np.argmax(radii)
        worst_radius = radii[worst_idx]
        tri = facets[worst_idx]
        
        if worst_radius < 1e-6:
            break
            
        # Get circumcenter
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            break
            
        c = cross / norm_cross
        if np.dot(c, p1) < 0:
            c = -c
        
        # Store previous positions
        prev = [points[i].copy() for i in tri]
        
        # Move vertices toward circumcenter
        for v_idx in tri:
            p = points[v_idx]
            direction = c - p
            norm_dir = np.linalg.norm(direction)
            if norm_dir < 1e-12:
                continue
            direction /= norm_dir
            points[v_idx] = points[v_idx] + step_size * direction
            points[v_idx] /= np.linalg.norm(points[v_idx])
        
        current_radius = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            step_size *= 0.98
        else:
            # Revert
            for i, v_idx in enumerate(tri):
                points[v_idx] = prev[i]
            step_size *= 0.9
            
            if step_size < min_step_size:
                # Try random perturbation
                for v_idx in tri:
                    noise = np.random.normal(0, 0.01, 3)
                    points[v_idx] = points[v_idx] + noise
                    points[v_idx] /= np.linalg.norm(points[v_idx])
                step_size = 0.01
                
                current_radius = _compute_covering_radius(points)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
                else:
                    for i, v_idx in enumerate(tri):
                        points[v_idx] = prev[i]
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Phase 1: Fibonacci spiral initialization
    points = _fibonacci_spiral(N)
    best_radius = _compute_covering_radius(points)
    best_points = points.copy()
    
    # Phase 2: Smooth gradient descent on soft-max surrogate
    if time.time() < deadline - 30:
        points_smooth = _gradient_descent_smooth(points, max_iters=300, lr_init=0.005, beta=30.0, deadline=deadline - 30)
        r_smooth = _compute_covering_radius(points_smooth)
        if r_smooth < best_radius:
            best_radius = r_smooth
            best_points = points_smooth.copy()
    
    # Phase 3: Greedy facet shrinking
    if time.time() < deadline - 10:
        points_greedy, r_greedy = _greedy_facet_shrink(best_points, deadline=deadline - 10)
        if r_greedy < best_radius:
            best_radius = r_greedy
            best_points = points_greedy.copy()
    
    # Phase 4: Final smooth polish
    if time.time() < deadline - 2:
        points_final = _gradient_descent_smooth(best_points, max_iters=200, lr_init=0.002, beta=50.0, deadline=deadline - 2)
        r_final = _compute_covering_radius(points_final)
        if r_final < best_radius:
            best_radius = r_final
            best_points = points_final.copy()
    
    return best_points
