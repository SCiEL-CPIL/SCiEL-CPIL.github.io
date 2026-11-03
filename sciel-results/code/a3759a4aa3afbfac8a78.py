"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _covering_radius_and_grad(points, temp=0.5):
    """Compute smooth covering radius and gradient w.r.t. points.
    
    The covering radius is the max over facets of the angular distance
    from the facet center to the facet plane. We use a smooth max (log-sum-exp)
    over facet circumradii to get gradients that balance all facets.
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Degenerate: return a large penalty
        return np.pi, np.zeros_like(points)
    
    facets = hull.simplices  # (M, 3) for 3D
    m = len(facets)
    
    # For each facet, compute the circumradius (angular distance from sphere center to farthest point)
    # The facet plane: normal = cross(p1-p0, p2-p0), normalized
    # The cap center is along the normal from the origin
    # Circumradius = angle between normal and any vertex = arccos(|dot(n, v)|) where n is unit normal
    
    # Compute normals for each facet
    p0 = points[facets[:, 0]]  # (m, 3)
    p1 = points[facets[:, 1]]
    p2 = points[facets[:, 2]]
    
    e1 = p1 - p0
    e2 = p2 - p0
    normals = np.cross(e1, e2)  # (m, 3)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    normals = normals / norms  # (m, 3)
    
    # Ensure normals point outward (away from origin)
    # The facet center (cap center) is in the direction of the normal from origin
    # For a convex hull containing origin, normals should point outward
    # Check: dot(normal, p0) should be positive if normal points outward
    dots = np.sum(normals * p0, axis=1)
    # If dot is negative, flip normal
    flip = dots < 0
    normals[flip] *= -1
    
    # Circumradius for each facet: angle between normal and a vertex
    # cos(angle) = dot(normal, vertex) / (|normal| * |vertex|) = dot(normal, vertex) since both unit
    cos_angles = np.sum(normals * p0, axis=1)  # (m,)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    circumradii = np.arccos(cos_angles)  # (m,)
    
    # Smooth max using log-sum-exp
    # soft_max(x) = (1/tau) * log(sum(exp(tau * x)))
    tau = 1.0 / temp  # temperature parameter
    scaled = tau * circumradii
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    smooth_max = log_sum_exp / tau
    
    # Gradient of smooth max w.r.t. circumradii
    # d/d(circumradii) [ (1/tau) * log(sum(exp(tau * c))) ]
    # = exp(tau * c) / sum(exp(tau * c))
    weights = np.exp(scaled - max_scaled)
    weights = weights / np.sum(weights)  # (m,)
    
    # Now compute gradient w.r.t. points
    # circumradius_i = arccos(dot(n_i, p0_i))
    # d(circumradius_i)/d(p0_i) = -n_i / sqrt(1 - cos^2) = -n_i / sin(angle)
    # But we also need d(n_i)/d(points) which is complex.
    
    # Alternative: use automatic differentiation via finite differences for the gradient
    # This is simpler and more robust. Let's compute gradient numerically.
    
    grad = np.zeros_like(points)
    eps = 1e-7
    
    for i in range(n):
        for j in range(3):
            orig = points[i, j]
            points[i, j] = orig + eps
            try:
                hull2 = ConvexHull(points)
                facets2 = hull2.simplices
                p0_2 = points[facets2[:, 0]]
                p1_2 = points[facets2[:, 1]]
                p2_2 = points[facets2[:, 2]]
                e1_2 = p1_2 - p0_2
                e2_2 = p2_2 - p0_2
                normals_2 = np.cross(e1_2, e2_2)
                norms_2 = np.linalg.norm(normals_2, axis=1, keepdims=True)
                norms_2 = np.maximum(norms_2, 1e-15)
                normals_2 = normals_2 / norms_2
                dots_2 = np.sum(normals_2 * p0_2, axis=1)
                flip_2 = dots_2 < 0
                normals_2[flip_2] *= -1
                cos_angles_2 = np.sum(normals_2 * p0_2, axis=1)
                cos_angles_2 = np.clip(cos_angles_2, -1.0, 1.0)
                circumradii_2 = np.arccos(cos_angles_2)
                
                scaled_2 = tau * circumradii_2
                max_scaled_2 = np.max(scaled_2)
                log_sum_exp_2 = max_scaled_2 + np.log(np.sum(np.exp(scaled_2 - max_scaled_2)))
                smooth_max_2 = log_sum_exp_2 / tau
            except Exception:
                smooth_max_2 = smooth_max
            points[i, j] = orig
            grad[i, j] = (smooth_max_2 - smooth_max) / eps
    
    return smooth_max, grad


def _analytic_covering_radius(points):
    """Compute exact covering radius from convex hull facets."""
    points = np.asarray(points, dtype=np.float64)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    facets = hull.simplices
    p0 = points[facets[:, 0]]
    p1 = points[facets[:, 1]]
    p2 = points[facets[:, 2]]
    
    e1 = p1 - p0
    e2 = p2 - p0
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    normals = normals / norms
    
    dots = np.sum(normals * p0, axis=1)
    flip = dots < 0
    normals[flip] *= -1
    
    cos_angles = np.sum(normals * p0, axis=1)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    circumradii = np.arccos(cos_angles)
    
    return np.max(circumradii)


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_sphere(n, offset=0.0):
    """Generate points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_radius = np.pi
    
    # Try multiple restarts with different initializations
    n_restarts = 5
    max_iters = 200
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Different initializations
        if restart == 0:
            points = _fibonacci_sphere(N)
        elif restart == 1:
            points = _fibonacci_sphere(N, offset=0.3)
        elif restart == 2:
            points = _fibonacci_sphere(N, offset=0.7)
        else:
            # Add small random perturbation
            points = _fibonacci_sphere(N)
            rng = np.random.default_rng(restart * 1000 + 42)
            points += rng.normal(0, 0.01, size=points.shape)
            points = _normalize(points)
        
        # Adam optimizer parameters
        lr = 0.05
        beta1 = 0.9
        beta2 = 0.999
        eps_adam = 1e-8
        m = np.zeros_like(points)  # First moment
        v = np.zeros_like(points)  # Second moment
        t = 0
        
        # Anneal temperature
        temp_start = 2.0
        temp_end = 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            
            # Anneal temperature
            progress = it / max(max_iters - 1, 1)
            temp = temp_start * (temp_end / temp_start) ** progress
            
            # Compute objective and gradient
            obj, grad = _covering_radius_and_grad(points, temp=temp)
            
            # Check convergence
            if it > 50 and obj > best_radius + 1e-10:
                # Not improving, reduce learning rate
                lr *= 0.95
            
            # Adam update
            t += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad * grad
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Update points
            points = points - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
            
            # Project back to unit sphere
            points = _normalize(points)
            
            # Track best
            if obj < best_radius:
                best_radius = obj
                best_points = points.copy()
            
            # Early stopping if converged
            if obj < 0.198:  # ~11.3 degrees in radians
                break
        
        # After each restart, do a final polish with lower temperature
        if best_points is not None and time.time() < deadline - 5.0:
            points = best_points.copy()
            m = np.zeros_like(points)
            v = np.zeros_like(points)
            t = 0
            lr = 0.01
            
            for it in range(100):
                if time.time() > deadline - 5.0:
                    break
                
                temp = 0.05  # Very low temperature for precise max
                obj, grad = _covering_radius_and_grad(points, temp=temp)
                
                t += 1
                m = 0.9 * m + 0.1 * grad
                v = 0.999 * v + 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                
                points = points - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
                points = _normalize(points)
                
                if obj < best_radius:
                    best_radius = obj
                    best_points = points.copy()
                
                if obj < 0.197:
                    break
    
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    # Final normalization to ensure unit vectors
    best_points = _normalize(best_points)
    
    # Verify no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(best_points[i] - best_points[j])
            if dist < 1e-10:
                # Separate them slightly
                best_points[j] += 1e-6 * np.random.randn(3)
                best_points[j] = _normalize(best_points[j])
    
    return best_points
# EVOLVE-BLOCK-END
