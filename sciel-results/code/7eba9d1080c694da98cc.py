import numpy as np
from scipy.spatial import ConvexHull
import itertools
import time

N = 11

def get_hex_vertices(center, side, angle_deg):
    angle = np.deg2rad(angle_deg)
    angles = angle + np.linspace(0, 2*np.pi, 7)[:-1]
    x = center[0] + side * np.cos(angles)
    y = center[1] + side * np.sin(angles)
    return np.column_stack((x, y))

def min_enclosing_hexagon(points):
    # points: Nx2 array
    # Compute convex hull
    if len(points) < 3:
        # Handle degenerate case
        center = np.mean(points, axis=0)
        # Max distance from center
        dists = np.linalg.norm(points - center, axis=1)
        r = np.max(dists)
        side = 2.0 * r / np.sqrt(3)
        return center, side, 0.0
        
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    except:
        hull_points = points
        
    best_r = float('inf')
    best_angle = 0
    best_center = np.mean(hull_points, axis=0)
    
    angles = np.linspace(0, np.pi/3, 50) # Reduced samples
    for theta in angles:
        normals = []
        for k in range(6):
            alpha = theta + k * np.pi/3
            normals.append(np.array([np.cos(alpha), np.sin(alpha)]))
            
        projections = hull_points @ np.array(normals).T
        widths = np.max(projections, axis=0) - np.min(projections, axis=0)
        r = np.max(widths) / 2.0
        if r < best_r:
            best_r = r
            best_angle = theta
            # Center estimation
            idx_max_width = np.argmax(widths)
            n = normals[idx_max_width]
            min_val = np.min(projections[:, idx_max_width])
            max_val = np.max(projections[:, idx_max_width])
            # Project hull points onto n
            # The center should be at the midpoint of the min and max projections
            # But this is only for this direction.
            # A better center is the centroid of the hull points
            best_center = np.mean(hull_points, axis=0)
            
    side = 2.0 * best_r / np.sqrt(3)
    angle_deg = np.rad2deg(best_angle)
    return best_center, side, angle_deg

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    s = np.sqrt(3) * 1.001 # Gap factor
    centers = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = s * (q + r/2.0)
            y = s * (r * np.sqrt(3)/2.0)
            dist = np.sqrt(x*x + y*y)
            if dist < 2.5 * s:
                centers.append((x, y))
                
    best_score = float('inf')
    best_config = None
    
    indices = list(range(len(centers)))
    
    # Try all combinations if fast enough, else random
    # 19 choose 11 is 75582.
    # With convex hull and 50 angles, it might be fast.
    
    for combo in itertools.combinations(indices, 11):
        if time.time() > deadline:
            break
            
        selected_centers = [centers[i] for i in combo]
        
        all_vertices = []
        for c in selected_centers:
            verts = get_hex_vertices(c, 1.0, 0.0)
            all_vertices.extend(verts)
            
        points = np.array(all_vertices)
        
        center, side, angle = min_enclosing_hexagon(points)
        
        if side < best_score:
            best_score = side
            best_config = (selected_centers, center, side, angle)
            
    if best_config is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    selected_centers, center, side, angle = best_config
    
    inner = [[c[0], c[1], 0.0] for c in selected_centers]
    
    return inner, (center[0], center[1]), side, angle
