import numpy as np
import time
from scipy.optimize import minimize_scalar

def hexagon_packing_12():
    start_time = time.time()
    deadline = 620.0

    # Axial to cartesian for hexagonal grid
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Generate candidate clusters
    def get_clusters():
        clusters = []
        
        # Cluster 1: Cap (7 central + 5 consecutive on ring 2)
        cap_coords = [
            (0,0), 
            (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
            (2,0), (1,1), (0,2), (-1,2), (-2,2)
        ]
        clusters.append(cap_coords)
        
        # Cluster 2: Star (7 central + 5 alternating on ring 2)
        star_coords = [
            (0,0), 
            (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
            (2,0), (0,2), (-2,2), (-2,0), (0,-2)
        ]
        clusters.append(star_coords)
        
        # Cluster 3: Another variant (7 central + 5 forming a different shape)
        # Maybe (2,0), (1,1), (0,2), (-1,1) is already in ring 1? No.
        # Let's try a "T" shape or similar?
        # For now, Cap and Star are the most likely candidates.
        
        return clusters

    def compute_s_for_cluster(centers_local, alpha_rad):
        """
        Compute the minimal side length s for a rigid cluster of hexagons
        with centers at centers_local, rotated by alpha_rad.
        """
        # Get all vertices of all hexagons
        all_vertices = []
        for cx, cy in centers_local:
            angles = np.arange(6) * np.pi / 3
            # Vertices relative to center
            vx = np.cos(angles)
            vy = np.sin(angles)
            # Rotate vertices by alpha
            cos_a = np.cos(alpha_rad)
            sin_a = np.sin(alpha_rad)
            rx = cos_a * vx - sin_a * vy
            ry = sin_a * vx + cos_a * vy
            # Translate to center (which is also rotated)
            # Note: centers_local are in local frame. We rotate the whole cluster.
            # So center also rotates.
            cx_rot = cos_a * cx - sin_a * cy
            cy_rot = sin_a * cx + cos_a * cy
            all_vertices.append(np.column_stack((cx_rot + rx, cy_rot + ry)))
        
        all_vertices = np.vstack(all_vertices)
        
        # Normals of the outer hexagon (fixed orientation)
        # Vertices of outer hexagon are at 0, 60, ...
        # Normals are at 30, 90, ...
        normal_angles = np.pi / 6 + np.arange(6) * np.pi / 3
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # Project vertices onto normals
        projections = all_vertices @ normals.T
        widths = np.max(projections, axis=0) - np.min(projections, axis=0)
        
        max_width = np.max(widths)
        s = max_width / np.sqrt(3)
        return s

    def optimize_cluster(cluster_coords):
        # Convert to cartesian local coordinates
        centers_local = np.array([axial_to_cartesian(q, r) for q, r in cluster_coords])
        
        # Center the cluster at the origin
        centroid = np.mean(centers_local, axis=0)
        centers_local -= centroid
        
        # Optimize rotation alpha in [0, pi/3]
        res = minimize_scalar(
            lambda alpha: compute_s_for_cluster(centers_local, alpha),
            bounds=(0, np.pi / 3),
            method='bounded'
        )
        
        if res.success:
            best_alpha = res.x
            best_s = res.fun
            return best_s, best_alpha, centers_local
        else:
            return None, None, None

    best_s = float('inf')
    best_config = None
    
    clusters = get_clusters()
    
    for cluster in clusters:
        if time.time() > deadline:
            break
            
        s, alpha, centers_local = optimize_cluster(cluster)
        
        if s is not None and s < best_s:
            best_s = s
            best_config = (alpha, centers_local)

    # If no config found, fallback
    if best_config is None:
        # Fallback to a known valid (though maybe not optimal) configuration
        # Just use the seed's fallback or a simple grid
        inner_hex_data = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0], [-1.25, -2.17, 0],
            [1.25, -2.17, 0], [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0], [0, -4, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    alpha, centers_local = best_config
    
    # Rotate centers to global frame
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    centers_global = np.zeros_like(centers_local)
    centers_global[:, 0] = cos_a * centers_local[:, 0] - sin_a * centers_local[:, 1]
    centers_global[:, 1] = sin_a * centers_local[:, 0] + cos_a * centers_local[:, 1]
    
    # The angle of each inner hexagon is alpha (in degrees)
    angle_deg = np.rad2deg(alpha)
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i] = [centers_global[i, 0], centers_global[i, 1], angle_deg]
        
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, best_s
