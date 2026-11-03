# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull

N = 130

def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def compute_facet_data(points):
    """
    Computes circumcenters and radii for all triangular facets of the convex hull.
    Returns:
    simplices: array of indices
    centers: array of circumcenter vectors (unit length)
    radii: array of angular radii
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None

    simplices = hull.simplices
    m = len(simplices)
    centers = np.zeros((m, 3))
    radii = np.zeros(m)
    
    for i, simplex in enumerate(simplices):
        a, b, c = points[simplex]
        v1 = a - b
        v2 = b - c
        normal = np.cross(v1, v2)
        norm_n = np.linalg.norm(normal)
        if norm_n < 1e-10:
            radii[i] = np.pi # Mark as bad/degenerate
            centers[i] = (a+b+c)/3.0
            continue
        
        o = normal / norm_n
        centroid = (a + b + c) / 3.0
        if np.dot(o, centroid) < 0:
            o = -o
            
        centers[i] = o
        cos_r = np.clip(np.dot(o, a), -1.0, 1.0)
        radii[i] = np.arccos(cos_r)
        
    return simplices, centers, radii

def construct():
    # Initial configuration
    points = fibonacci_sphere(N)
    points = normalize(points)
    
    # Parameters for Nesterov Momentum Local Search
    steps = 3000
    momentum = 0.9
    learning_rate = 0.05
    active_epsilon = 0.01 # Radians. Facets within this of max are active.
    
    velocity = np.zeros_like(points)
    
    # Pre-allocate for forces
    forces = np.zeros_like(points)
    
    for step in range(steps):
        # Decay learning rate
        lr = learning_rate * (1.0 - step / steps) * 0.5 + 0.001
        
        # Compute hull data
        simplices, centers, radii = compute_facet_data(points)
        if simplices is None:
            # Hull failed, perturb points slightly
            points += 0.01 * np.random.randn(*points.shape)
            points = normalize(points)
            continue
            
        max_radius = np.max(radii)
        
        # Identify active facets
        active_mask = radii > (max_radius - active_epsilon)
        
        # Reset forces
        forces.fill(0.0)
        
        # Accumulate forces from active facets
        for i in range(len(simplices)):
            if not active_mask[i]:
                continue
            
            simplex = simplices[i]
            center = centers[i]
            
            # For each vertex in this active facet, add a force pushing it towards the center
            # This shrinks the circumradius of the facet locally.
            for idx in simplex:
                # Force direction: from point to center, projected onto tangent plane
                diff = center - points[idx]
                # Project diff onto tangent plane at points[idx]
                # Tangent plane normal is points[idx]
                dot_p = np.dot(diff, points[idx])
                tangent_diff = diff - dot_p * points[idx]
                
                forces[idx] += tangent_diff
        
        # Normalize forces to prevent large jumps if many facets active
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms < 1e-10, 1.0, force_norms)
        forces = forces / force_norms
        
        # Update velocity (Nesterov)
        velocity = momentum * velocity + lr * forces
        
        # Update points
        points = points + velocity
        points = normalize(points)
        
        # Optional: Check for coincidence and fix
        # If two points are too close, push them apart
        # This is a simple O(N^2) check, N=130 is small enough
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(points[i] - points[j])
                if dist < 0.05: # Too close
                    # Push apart
                    mid = (points[i] + points[j]) / 2.0
                    mid /= np.linalg.norm(mid)
                    perp1 = points[i] - np.dot(points[i], mid) * mid
                    perp2 = points[j] - np.dot(points[j], mid) * mid
                    if np.linalg.norm(perp1) < 1e-6:
                        perp1 = np.cross(points[i], mid)
                    if np.linalg.norm(perp2) < 1e-6:
                        perp2 = np.cross(points[j], mid)
                    
                    points[i] = points[i] + 0.01 * perp1
                    points[j] = points[j] - 0.01 * perp2
                    points[i] /= np.linalg.norm(points[i])
                    points[j] /= np.linalg.norm(points[j])

    return points
# EVOLVE-BLOCK-END
