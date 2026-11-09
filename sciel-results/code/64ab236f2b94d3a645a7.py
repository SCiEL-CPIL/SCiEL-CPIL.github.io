import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    time_limit = 100  # Leave 20s safety margin for 120s limit
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = np.inf

    # Helper function to calculate covering radius
    def calc_covering_radius(pts):
        try:
            hull = ConvexHull(pts)
            # hull.simplices gives indices of points forming each triangular facet
            simplices = hull.simplices
            circumcenters = []
            max_dist = 0.0
            for s in simplices:
                v1, v2, v3 = pts[s[0]], pts[s[1]], pts[s[2]]
                # Calculate circumcenter of triangle v1, v2, v3 on the sphere
                # The circumcenter is the intersection of the sphere with the normal to the plane
                # Normal vector
                n = np.cross(v2 - v1, v3 - v1)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n = n / n_norm
                # Check orientation: normal should point outward
                # Dot product of normal with any vertex should be positive (since origin is inside)
                if np.dot(n, v1) < 0:
                    n = -n
                
                # Circumcenter C satisfies C = k * n
                # |C - v1|^2 = |C|^2 - 2 C.v1 + |v1|^2
                # 1 - 2 k n.v1 + 1 = 1 - 1 + 1 = 1 (since |C|=1, |v1|=1)
                # 1 - 2 k (n.v1) = 0  => k = 1 / (2 * n.v1)
                dot_val = np.dot(n, v1)
                if abs(dot_val) < 1e-10:
                    continue
                k = 1.0 / (2.0 * dot_val)
                c = k * n
                # Normalize to ensure it's on the sphere (numerical stability)
                c = c / np.linalg.norm(c)
                
                # Distance from circumcenter to nearest vertex
                # It is equidistant to v1, v2, v3, so just check v1
                dist = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
                if dist > max_dist:
                    max_dist = dist
                    
            return max_dist
        except Exception:
            return np.inf

    # Initial radius
    current_radius = calc_covering_radius(points)
    if current_radius < best_radius:
        best_radius = current_radius
        best_points = points.copy()

    # Optimization Loop
    # We use a momentum-based update towards Voronoi centroids
    # To be fast, we do a limited number of iterations
    
    momentum = np.zeros_like(points)
    lr = 0.05 # Learning rate
    iterations = 0
    max_iters = 20000
    
    while time.time() - start_time < time_limit and iterations < max_iters:
        iterations += 1
        
        try:
            hull = ConvexHull(points)
            simplices = hull.simplices
            
            # Calculate target for each point (average of adjacent circumcenters)
            targets = np.zeros_like(points)
            counts = np.zeros(N)
            
            for s in simplices:
                v1, v2, v3 = points[s[0]], points[s[1]], points[s[2]]
                n = np.cross(v2 - v1, v3 - v1)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n = n / n_norm
                if np.dot(n, v1) < 0:
                    n = -n
                dot_val = np.dot(n, v1)
                if abs(dot_val) < 1e-10:
                    continue
                k = 1.0 / (2.0 * dot_val)
                c = k * n
                c = c / np.linalg.norm(c)
                
                # Add this circumcenter to the targets of the 3 vertices
                targets[s[0]] += c
                targets[s[1]] += c
                targets[s[2]] += c
                counts[s[0]] += 1
                counts[s[1]] += 1
                counts[s[2]] += 1
            
            # Average targets
            for j in range(N):
                if counts[j] > 0:
                    targets[j] /= counts[j]
                else:
                    targets[j] = points[j]
            
            # Update points with momentum
            # Move towards target
            diff = targets - points
            # Normalize diff to get direction on tangent plane? 
            # Simple Euclidean update followed by normalization works well for sphere
            momentum = 0.5 * momentum + lr * diff
            points += momentum
            
            # Normalize
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
        except Exception:
            # If hull fails, break or reset
            break

        # Evaluate
        current_radius = calc_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Add small noise to escape local minima
        # Only if we are stuck or periodically
        if iterations % 100 == 0:
            noise = np.random.normal(0, 0.01, points.shape)
            points += noise
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms

    # Final check
    # Return the best configuration found
    return best_points
