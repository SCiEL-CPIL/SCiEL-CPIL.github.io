"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120
    
    # Define compact cluster arrangements for 11 hexagons
    # Using axial coordinates on triangular lattice
    # Lattice vectors: a=(sqrt(3),0), b=(sqrt(3)/2, 3/2)
    sqrt3 = np.sqrt(3)
    
    # Several candidate cluster shapes (in axial coords (q,r))
    # Convert to cartesian: x = sqrt3*(q + r/2), y = 1.5*r
    def axial_to_cartesian(coords):
        pts = []
        for q, r in coords:
            x = sqrt3 * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)
    
    # Cluster 1: 3-4-3-1 arrangement
    cluster1 = [
        (0, 0), (1, 0), (-1, 0),
        (0, 1), (1, 1), (-1, 1), (0, -1),
        (0, 2), (1, 2), (-1, 2),
        (0, -2)
    ]
    
    # Cluster 2: more symmetric 11-hexagon cluster
    # Center + 6 neighbors + 4 more
    cluster2 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    
    # Cluster 3: 2-3-3-2-1
    cluster3 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3),
        (0, 4)
    ]
    
    # Cluster 4: compact 4-3-4
    cluster4 = [
        (0, 0), (1, 0), (-1, 0), (2, 0),
        (0, 1), (1, 1), (-1, 1),
        (0, 2), (1, 2), (-1, 2), (2, 2)
    ]
    
    clusters = [cluster1, cluster2, cluster3, cluster4]
    
    best_side = float('inf')
    best_config = None
    
    for cluster in clusters:
        if time.time() > deadline - 5:
            break
        
        pts = axial_to_cartesian(cluster)
        
        # Center the cluster
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        
        # For a given rotation angle theta of the enclosing hexagon,
        # compute the minimum side length
        def min_side_for_angle(theta, pts):
            # 6 outward normals of the enclosing hexagon
            # Hexagon oriented at angle theta has normals at theta + k*60 degrees
            normals = []
            for k in range(6):
                angle = theta + k * np.pi / 3
                normals.append(np.array([np.cos(angle), np.sin(angle)]))
            normals = np.array(normals)
            
            # For each normal direction, compute max and min projection
            projections = pts @ normals.T  # shape (n_points, 6)
            max_proj = np.max(projections, axis=0)  # shape (6,)
            min_proj = np.min(projections, axis=0)  # shape (6,)
            
            # The enclosing hexagon centered at origin with side s contains pts
            # if for each normal n_i: max_proj[i] - 0 <= s*sqrt3/2 and 0 - min_proj[i] <= s*sqrt3/2
            # So s*sqrt3/2 >= max(max_proj[i], -min_proj[i]) for all i
            # s >= max_i(max(max_proj[i], -min_proj[i])) / (sqrt3/2)
            extents = np.maximum(max_proj, -min_proj)
            s = np.max(extents) / (sqrt3 / 2.0)
            return s
        
        # Optimize over theta
        # The function is periodic with period pi/3 (60 degrees) due to hexagon symmetry
        # So we only need to search theta in [0, pi/3)
        
        best_theta = 0
        best_s = float('inf')
        
        # Coarse search
        n_coarse = 360
        thetas = np.linspace(0, np.pi/3, n_coarse, endpoint=False)
        for theta in thetas:
            if time.time() > deadline - 5:
                break
            s = min_side_for_angle(theta, pts_centered)
            if s < best_s:
                best_s = s
                best_theta = theta
        
        # Fine search around best
        if time.time() < deadline - 5:
            n_fine = 100
            thetas_fine = np.linspace(best_theta - 0.05, best_theta + 0.05, n_fine)
            for theta in thetas_fine:
                if time.time() > deadline - 5:
                    break
                s = min_side_for_angle(theta, pts_centered)
                if s < best_s:
                    best_s = s
                    best_theta = theta
        
        if best_s < best_side:
            best_side = best_s
            best_config = (cluster, best_theta, centroid)
    
    if best_config is None:
        # Fallback: return seed-like result
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    cluster, best_theta, centroid = best_config
    pts = axial_to_cartesian(cluster)
    pts_centered = pts - centroid
    
    # The inner hexagons are at positions pts_centered, all with angle 0
    # The outer hexagon is centered at origin, with side best_side, rotated by best_theta
    inner = [[float(x), float(y), 0.0] for x, y in pts_centered]
    outer_center = (0.0, 0.0)
    outer_side = float(best_side)
    outer_angle = float(np.degrees(best_theta))
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END

