import numpy as np
from scipy.spatial import ConvexHull

def construct():
       N = 50
       # Initial Fibonacci spiral
       i = np.arange(N) + 0.5
       z = 1.0 - 2.0 * i / N
       phi = np.pi * (1.0 + np.sqrt(5.0)) * i
       r = np.sqrt(1.0 - z * z)
       points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
       
       # Local optimization using Facet Push
       # We run for a fixed number of iterations or until convergence
       
       prev_max_rad = np.inf
       
       for iteration in range(500):
           try:
               hull = ConvexHull(points)
           except Exception:
               break
               
           facets = hull.simplices
           radii = np.zeros(len(facets))
           centers = np.zeros((len(facets), 3))
           
           for k, tri in enumerate(facets):
               a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
               n = np.cross(b - a, c - a)
               norm_n = np.linalg.norm(n)
               if norm_n < 1e-10:
                   radii[k] = 0
                   centers[k] = a
                   continue
               n = n / norm_n
               if np.dot(n, a) < 0:
                   u = -n
               else:
                   u = n
               centers[k] = u
               dot_val = np.clip(np.dot(u, a), -1.0, 1.0)
               radii[k] = np.arccos(dot_val)
               
           current_max_rad = np.max(radii)
           
           # If no improvement, break
           if current_max_rad > prev_max_rad * 1.01 and iteration > 50:
               # Allow some increase to escape local minima?
               # For now, just break if it gets significantly worse
               pass 
           
           prev_max_rad = current_max_rad
           
           # Update points
           # For each point, sum up the influence of its incident facets
           # We need to map facets to points
           point_influence = np.zeros((N, 3))
           point_weight = np.zeros(N)
           
           for k, tri in enumerate(facets):
               # Weight by radius? Or just 1?
               # Using radius makes it focus on the worst facets
               w = radii[k]
               for idx in tri:
                   point_influence[idx] += w * centers[k]
                   point_weight[idx] += w
                   
           # Move points
           for i in range(N):
               if point_weight[i] > 1e-10:
                   target = point_influence[i] / point_weight[i]
                   # Normalize target to unit sphere?
                   # The center is already on the sphere.
                   # The weighted average of unit vectors is not necessarily unit length.
                   # We should move towards the normalized target.
                   norm_t = np.linalg.norm(target)
                   if norm_t > 1e-10:
                       target /= norm_t
                       
                       # Step size
                       step = 0.1
                       points[i] = (1 - step) * points[i] + step * target
                       points[i] /= np.linalg.norm(points[i])
                       
       return points
