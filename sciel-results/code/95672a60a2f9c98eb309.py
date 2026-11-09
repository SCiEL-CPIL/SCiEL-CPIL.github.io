import numpy as np
from scipy.spatial import ConvexHull

def construct():
    N = 130
    
    # 1. Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Optimization Loop
    num_iters = 200
    base_lr = 0.05
    
    for it in range(num_iters):
        # Learning rate decay
        lr = base_lr * (0.99 ** it)
        
        # Compute Convex Hull to get Delaunay triangulation
        try:
            hull = ConvexHull(pts)
            facets = hull.simplices
        except Exception:
            # Fallback if hull fails (e.g., coplanar points), skip iteration or jitter
            pts += 1e-5 * np.random.randn(N, 3)
            norms = np.linalg.norm(pts, axis=1)
            pts = pts / norms[:, np.newaxis]
            continue
            
        # Calculate circumcenter and radius for each facet
        num_facets = len(facets)
        circumcenters = np.zeros((num_facets, 3))
        circumradii = np.zeros(num_facets)
        
        for k in range(num_facets):
            idx = facets[k]
            A = pts[idx[0]]
            B = pts[idx[1]]
            C = pts[idx[2]]
            
            # Vectors
            AB = B - A
            BC = C - B
            # Normal to the plane of the triangle (chord plane)
            # The circumcenter of the spherical triangle is perpendicular to the chords
            # So it is parallel to (B-A) x (C-B) ? 
            # Let's check: O . (B-A) = 0 and O . (C-B) = 0
            # So O is parallel to (B-A) x (C-B)
            V = np.cross(AB, BC)
            norm_V = np.linalg.norm(V)
            if norm_V < 1e-10:
                circumcenters[k] = (A+B+C)/3
                circumradii[k] = 0
                continue
            O = V / norm_V
            
            # Ensure O is on the same side as the triangle centroid
            centroid = (A + B + C) / 3.0
            if np.dot(O, centroid) < 0:
                O = -O
                V = -V # Not strictly needed if we just use O
                
            circumcenters[k] = O
            # Circumradius is angle between O and A
            # cos(theta) = O . A
            cos_theta = np.dot(O, A)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            circumradii[k] = np.arccos(cos_theta)
            
        # Determine target movements
        # We want to move points towards the circumcenters of the "worst" facets
        # Weight by the square of the radius to emphasize larger holes
        weights = circumradii ** 2
        
        # Accumulate target directions for each point
        target_vecs = np.zeros((N, 3))
        target_weights = np.zeros(N)
        
        for k in range(num_facets):
            w = weights[k]
            if w < 1e-6:
                continue
            O = circumcenters[k]
            idx = facets[k]
            for v in idx:
                target_vecs[v] += w * O
                target_weights[v] += w
                
        # Update points
        for i in range(N):
            if target_weights[i] > 1e-6:
                # Normalize the accumulated vector to get direction
                dir_vec = target_vecs[i] / np.linalg.norm(target_vecs[i])
                # Move point towards direction
                pts[i] = pts[i] + lr * dir_vec
                # Normalize
                pts[i] = pts[i] / np.linalg.norm(pts[i])
            else:
                # If no facets contribute (shouldn't happen), keep as is
                pass
                
        # Global normalization
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, np.newaxis]
        
    return pts
