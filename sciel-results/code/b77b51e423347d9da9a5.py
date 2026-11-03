import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius in radians.
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest point in the set. This is equal to the maximum circumradius
    of the facets of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # Invalid hull, return large value
        
    max_radius = 0.0
    simplices = hull.simplices
    
    for simplex in simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        # Compute normal vector
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-10:
            continue
            
        n = n / n_norm
        
        # Ensure normal points outward (away from origin)
        # The origin is at (0,0,0). The facet plane is n . x = d.
        # Since points are on unit sphere, d = n . p1.
        # If n . p1 < 0, the normal points towards the origin, so flip it.
        if np.dot(n, p1) < 0:
            n = -n
            
        # The angular distance from the origin to the plane is not the radius.
        # The "hole" center is the point on the sphere in the direction of n.
        # The angular radius of the cap is the angle between n and any vertex.
        # Due to numerical errors, angles might differ slightly.
        angle = np.arccos(np.clip(np.dot(n, p1), -1.0, 1.0))
        
        if angle > max_radius:
            max_radius = angle
             
    return max_radius

def _find_hole_and_facet(points):
    """
    Find the normal vector of the facet with the largest circumradius.
    Returns (normal_vector, max_angle, facet_indices)
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, 0.0, None
        
    max_angle = 0.0
    best_n = None
    best_indices = None
    
    simplices = hull.simplices
    
    for simplex in simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-10:
            continue
            
        n = n / n_norm
        
        if np.dot(n, p1) < 0:
            n = -n
            
        angle = np.arccos(np.clip(np.dot(n, p1), -1.0, 1.0))
        
        if angle > max_angle:
            max_angle = angle
            best_n = n
            best_indices = simplex

    return best_n, max_angle, best_indices

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    best_X = X.copy()
    best_radius = _compute_covering_radius(X)
    
    start_time = time.time()
    time_limit = 140  # Leave 10s margin for 150s limit
    
    eta = 0.05
    alpha_att = 5.0  # Attraction decay
    p_rep = 2.0      # Repulsion power
    
    # We will iterate, trying to reduce the covering radius
    # Strategy: Move points towards the "hole" (normal of largest facet)
    # while repelling each other to maintain spread.
    
    iteration = 0
    while time.time() - start_time < time_limit:
        n_hole, current_radius, facet_indices = _find_hole_and_facet(X)
        
        if n_hole is None:
            break
            
        # Calculate forces
        forces = np.zeros_like(X)
        
        # 1. Repulsion force (Tammes-like)
        # F_rep_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^(p+1)
        # This is the gradient of the potential -1/||x_i - x_j||^p
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = X[i] - X[j]
                dist = np.linalg.norm(d)
                if dist < 1e-8:
                    # Random jitter if too close
                    forces[i] += np.random.normal(0, 0.1, 3)
                    continue
                # Gradient of 1/dist^p is -p * dist^(-p-2) * d
                # We want to maximize distance, so force is + p * dist^(-p-2) * d
                # Or simply d / dist^(p+1)
                forces[i] += d / (dist ** (p_rep + 1))
                
        # 2. Attraction to the hole
        # We want to shrink the largest cap.
        # Move points towards n_hole.
        # Weight by proximity to the hole to avoid moving far points too much.
        for i in range(N):
            d = X[i] - n_hole
            dist = np.linalg.norm(d)
            # Weight is high for points near the hole
            w = np.exp(-alpha_att * dist**2)
            # Force is towards n_hole, so -d
            forces[i] += w * (-d)
            
        # Update positions
        X_new = X + eta * forces
        
        # Normalize to unit sphere
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms < 1e-10] = 1.0
        X_new = X_new / norms
        
        # Check improvement
        new_radius = _compute_covering_radius(X_new)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_X = X_new.copy()
            X = X_new
            # Reduce step size slightly as we get closer?
            # Or keep it constant for now.
            eta *= 0.999
        else:
            # No improvement. Try smaller step or random restart?
            # For now, reduce step size.
            eta *= 0.9
            if eta < 1e-5:
                # Try a random perturbation of best_X
                X_perturbed = best_X + np.random.normal(0, 0.1, (N, 3))
                X_perturbed = X_perturbed / np.linalg.norm(X_perturbed, axis=1, keepdims=True)
                r_perturbed = _compute_covering_radius(X_perturbed)
                if r_perturbed < best_radius:
                    best_radius = r_perturbed
                    best_X = X_perturbed
                    X = X_perturbed
                    eta = 0.05
                else:
                    break
        iteration += 1
        
    # Final check: ensure valid output
    # Check for coincident points
    # Check if in hemisphere (should be fine with repulsion)
    
    return best_X
