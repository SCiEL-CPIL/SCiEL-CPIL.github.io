import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_P = P.copy()
    best_radius = np.inf
    
    # Helper to compute covering radius (max circumradius of hull facets)
    def compute_covering_radius(points):
        try:
            hull = ConvexHull(points)
            max_rad = 0.0
            # Each facet is a triangle (or more, but usually triangles for generic points)
            for simplex in hull.simplices:
                v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                # Circumcenter of the spherical triangle
                # The normal to the plane of v1, v2, v3
                n = np.cross(v2 - v1, v3 - v1)
                if np.linalg.norm(n) < 1e-10:
                    continue
                c = n / np.linalg.norm(n)
                # Distance from c to v1 is the circumradius
                # dot product c.v1 = cos(angle)
                cos_d = np.clip(np.dot(c, v1), -1.0, 1.0)
                d = np.arccos(cos_d)
                if d > max_rad:
                    max_rad = d
            return max_rad
        except Exception:
            return np.inf

    # Initial radius
    current_radius = compute_covering_radius(P)
    if current_radius < best_radius:
        best_radius = current_radius
        best_P = P.copy()

    alpha = 0.05  # Step size
    iterations = 0
    
    while time.time() - start_time < time_limit:
        # Compute gradients
        try:
            hull = ConvexHull(P)
        except Exception:
            break
            
        gradients = np.zeros_like(P)
        
        # For each facet, find circumcenter and add to gradients of its vertices
        for simplex in hull.simplices:
            v1, v2, v3 = P[simplex[0]], P[simplex[1]], P[simplex[2]]
            n = np.cross(v2 - v1, v3 - v1)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            c = n / norm_n
            
            # Vector from vertex to circumcenter
            # We want to move vertices towards c to shrink the triangle
            vec1 = c - v1
            vec2 = c - v2
            vec3 = c - v3
            
            gradients[simplex[0]] += vec1
            gradients[simplex[1]] += vec2
            gradients[simplex[2]] += vec3
        
        # Update points
        P_new = P + alpha * gradients
        
        # Normalize to sphere
        norms = np.linalg.norm(P_new, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        P_new = P_new / norms
        
        # Check for collisions and separate if necessary
        # Simple check: if any two points are very close, push them apart
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(P_new[i] - P_new[j])
                if dist < 0.01:  # Threshold
                    # Push apart
                    mid = (P_new[i] + P_new[j]) / 2
                    mid = mid / np.linalg.norm(mid)
                    perp = np.cross(P_new[i], mid)
                    perp = perp / np.linalg.norm(perp)
                    P_new[i] -= 0.01 * perp
                    P_new[j] += 0.01 * perp
                    P_new[i] = P_new[i] / np.linalg.norm(P_new[i])
                    P_new[j] = P_new[j] / np.linalg.norm(P_new[j])
        
        P = P_new
        current_radius = compute_covering_radius(P)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_P = P.copy()
            
        iterations += 1
        
        # If no improvement for a while, reduce step size or break?
        # For now, just run until time limit.
        
    return best_P
