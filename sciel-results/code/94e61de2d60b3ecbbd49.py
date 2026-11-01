import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def construct():
        # Initial Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        best_pts = pts.copy()
        best_radius = float('inf')
        
        start_time = time.time()
        deadline = start_time + 110.0  # 10s margin
        
        # Parameters
        eta = 0.05
        gamma = 1.0
        alpha = 10.0 # For exponential weighting if needed
        
        for step in range(2000):
            if time.time() > deadline:
                break
                
            # Compute Convex Hull
            try:
                hull = ConvexHull(pts)
                facets = hull.simplices
            except Exception:
                # If hull fails, maybe points are coplanar or something
                break
                
            # Compute forces
            forces = np.zeros_like(pts)
            
            # Repulsive forces between all pairs
            # This is O(N^2)
            for i in range(N):
                for j in range(N):
                    if i == j: continue
                    diff = pts[i] - pts[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < 1e-12:
                        # Add small random noise
                        forces[i] += np.random.normal(0, 1e-6, 3)
                    else:
                        forces[i] += diff / dist_sq
                        
            # Attractive forces from facet circumcenters
            # For each facet, compute circumcenter
            for facet in facets:
                a = pts[facet[0]]
                b = pts[facet[1]]
                c = pts[facet[2]]
                
                # Circumcenter is parallel to (a-b) x (b-c)
                v1 = a - b
                v2 = b - c
                n = np.cross(v1, v2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-12:
                    continue
                c_center = n / norm_n
                # Ensure c_center is on the correct side (dot product with a is positive)
                if np.dot(c_center, a) < 0:
                    c_center = -c_center
                    
                # Calculate circumradius
                cos_r = np.dot(c_center, a)
                cos_r = np.clip(cos_r, -1.0, 1.0)
                r_facet = np.arccos(cos_r)
                
                # Force towards circumcenter
                # Weight by r_facet to prioritize larger facets
                weight = r_facet
                for idx in facet:
                    forces[idx] += gamma * weight * (c_center - pts[idx])
                    
            # Update points
            pts += eta * forces
            
            # Normalize
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            pts = pts / norms
            
            # Evaluate covering radius
            # Approximate by max facet circumradius
            current_radius = 0.0
            for facet in facets:
                a = pts[facet[0]]
                b = pts[facet[1]]
                c = pts[facet[2]]
                v1 = a - b
                v2 = b - c
                n = np.cross(v1, v2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-12:
                    continue
                c_center = n / norm_n
                if np.dot(c_center, a) < 0:
                    c_center = -c_center
                cos_r = np.clip(np.dot(c_center, a), -1.0, 1.0)
                r_facet = np.arccos(cos_r)
                if r_facet > current_radius:
                    current_radius = r_facet
                    
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
                
            # Annealing: decrease eta?
            eta *= 0.999
            
        # Final verification/evaluation of best_pts
        # To be safe, we should do a more accurate evaluation of best_pts
        # But the max facet circumradius is the covering radius for the hull facets.
        # Are there other points that are worse?
        # The covering radius is the max over ALL points on the sphere.
        # The maxima occur at the circumcenters of the convex hull facets.
        # So evaluating the max facet circumradius is sufficient.
        
        return best_pts
