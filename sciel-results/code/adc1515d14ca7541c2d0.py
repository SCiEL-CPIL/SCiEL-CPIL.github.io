import numpy as np
import math

def hex_vertices(center, side=1.0, angle=0.0):
    """
    Returns vertices of a regular hexagon.
    Center (x, y), side length, rotation angle in radians.
    Vertices are at distance 'side' from center.
    """
    cx, cy = center
    vertices = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        vertices.append((x, y))
    return vertices

def min_container_side(vertices, n_angles=720):
    """
    Calculate the minimum side length of a regular hexagon container
    centered at origin that contains all given vertices.
    The container can be rotated.
    We sample rotations of the container.
    """
    vertices = np.array(vertices)
    # Projection onto 6 normals of a hexagon with rotation theta
    # Normals are at theta + k * pi/3
    # Apothem a = max projection
    # Side L = 2 * a / sqrt(3)
    
    min_L = float('inf')
    for i in range(n_angles):
        theta = i * 2 * math.pi / n_angles
        # Normals
        normals = []
        for k in range(6):
            ang = theta + k * math.pi / 3
            normals.append(np.array([math.cos(ang), math.sin(ang)]))
        
        max_proj = 0
        for n in normals:
            projs = np.dot(vertices, n)
            # We need to contain the whole hexagon, so we need max positive projection
            # Since the container is centered at origin, and vertices can be negative,
            # we just need the max absolute projection?
            # No, a hexagon centered at origin contains points p if |p . n_k| <= a for all 6 normals?
            # Yes, for a centrally symmetric convex polygon.
            # The condition is max_k max_v (v . n_k) <= a.
            # Since the polygon is symmetric, max_v (v . n_k) = max_v |v . n_k| is not quite right if origin is not inside?
            # But origin is the center of the container. The inner shapes are around origin.
            # So we need max_v (v . n_k) <= a AND min_v (v . n_k) >= -a.
            # This is equivalent to max_v |v . n_k| <= a.
            max_abs_proj = np.max(np.abs(projs))
            if max_abs_proj > max_proj:
                max_proj = max_abs_proj
        
        L = 2 * max_proj / math.sqrt(3)
        if L < min_L:
            min_L = L
            
    return min_L

def generate_cluster_11():
    """
    Generates centers for 11 hexagons in a compact cluster.
    Base: 1 center + 6 neighbors (flower).
    Add: 4 consecutive from the next ring.
    """
    centers = []
    # Center
    centers.append((0.0, 0.0))
    
    # Ring 1 (6 hexagons)
    # Distance sqrt(3)
    r1 = math.sqrt(3)
    for i in range(6):
        ang = i * math.pi / 3 + math.pi / 6 # Offset to align with lattice?
        # Let's use standard hexagonal lattice directions.
        # Neighbors are at 30, 90, 150, 210, 270, 330?
        # If hexagon has flat top, neighbors are at 30, 90...
        # Let's just use the vectors.
        x = r1 * math.cos(ang)
        y = r1 * math.sin(ang)
        centers.append((x, y))
        
    # Ring 2 (12 hexagons)
    # We need to identify the 12 centers.
    # In a hexagonal lattice, the next ring is at distance 2*sqrt(3)?
    # Let's generate all lattice points with distance <= 2*sqrt(3) + epsilon.
    # Lattice vectors:
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 1.5)
    # Centers are m*v1 + n*v2.
    # Distance from origin: sqrt(3) * sqrt(m^2 + m*n + n^2)? No.
    # Let's just generate a grid and filter by distance.
    
    ring2_centers = []
    # m, n range -2 to 2
    for m in range(-2, 3):
        for n in range(-2, 3):
            # Skip origin and ring 1
            if m == 0 and n == 0: continue
            
            x = m * math.sqrt(3) + n * math.sqrt(3) / 2
            y = n * 1.5
            
            dist = math.sqrt(x*x + y*y)
            
            # Ring 1 distance is sqrt(3) ~ 1.732
            # Ring 2 distance is 2*sqrt(3) ~ 3.464?
            # Let's check (1,1): x = 1.5*sqrt(3), y = 1.5. dist = sqrt(6.75 + 2.25) = 3.
            # (2,0): x = 2*sqrt(3), y = 0. dist = 2*sqrt(3) ~ 3.464.
            # (1,-1): x = 0.5*sqrt(3), y = -1.5. dist = sqrt(0.75 + 2.25) = sqrt(3) ~ 1.732. This is ring 1.
            # So ring 1 points are those with dist ~ sqrt(3).
            # Ring 2 points are those with dist > 2.0 and < 4.0?
            
            if 2.0 < dist < 4.0:
                ring2_centers.append((x, y))
                
    # Sort ring2 centers by angle to pick consecutive ones
    ring2_centers.sort(key=lambda p: math.atan2(p[1], p[0]))
    
    # Pick 4 consecutive
    best_L = float('inf')
    best_config = None
    best_theta = 0.0
    
    # There are 12 centers in ring 2.
    # We try all 12 starting indices.
    for start in range(12):
        added = []
        for i in range(4):
            idx = (start + i) % 12
            added.append(ring2_centers[idx])
            
        all_centers = centers + added
        
        # Collect all vertices
        all_vertices = []
        for c in all_centers:
            verts = hex_vertices(c, 1.0, 0.0)
            all_vertices.extend(verts)
            
        # Find min container side
        L = min_container_side(all_vertices, 1440)
        
        if L < best_L:
            best_L = L
            best_config = all_centers
            # We need the optimal theta for this config to report?
            # The return interface asks for inner rows (x, y, angle_degrees) and outer (x, y, side, angle).
            # We can just return the centers with angle 0.
            # The evaluator will check if they fit in the returned outer hexagon.
            # We should return the outer hexagon with the optimal rotation.
            # So we need to find the optimal theta for this config.
            
            # Re-run min_container_side but return theta
            # Let's modify min_container_side to return L and theta
            
    return best_config, best_L

def min_container_side_with_theta(vertices, n_angles=720):
    vertices = np.array(vertices)
    min_L = float('inf')
    best_theta = 0.0
    for i in range(n_angles):
        theta = i * 2 * math.pi / n_angles
        normals = []
        for k in range(6):
            ang = theta + k * math.pi / 3
            normals.append(np.array([math.cos(ang), math.sin(ang)]))
        
        max_proj = 0
        for n in normals:
            projs = np.dot(vertices, n)
            max_abs_proj = np.max(np.abs(projs))
            if max_abs_proj > max_proj:
                max_proj = max_abs_proj
        
        L = 2 * max_proj / math.sqrt(3)
        if L < min_L:
            min_L = L
            best_theta = theta
            
    return min_L, best_theta

def construct():
    # Generate the best configuration
    # We will hardcode the best result from a quick search or run the search here.
    # Since we have 120s, we can run the search.
    
    centers_base = [(0.0, 0.0)]
    r1 = math.sqrt(3)
    for i in range(6):
        ang = i * math.pi / 3 + math.pi / 6
        x = r1 * math.cos(ang)
        y = r1 * math.sin(ang)
        centers_base.append((x, y))
        
    ring2_centers = []
    for m in range(-2, 3):
        for n in range(-2, 3):
            if m == 0 and n == 0: continue
            x = m * math.sqrt(3) + n * math.sqrt(3) / 2
            y = n * 1.5
            dist = math.sqrt(x*x + y*y)
            if 2.0 < dist < 4.0:
                ring2_centers.append((x, y))
                
    ring2_centers.sort(key=lambda p: math.atan2(p[1], p[0]))
    
    best_L = float('inf')
    best_config = None
    best_outer_theta = 0.0
    
    for start in range(12):
        added = []
        for i in range(4):
            idx = (start + i) % 12
            added.append(ring2_centers[idx])
            
        all_centers = centers_base + added
        
        all_vertices = []
        for c in all_centers:
            verts = hex_vertices(c, 1.0, 0.0)
            all_vertices.extend(verts)
            
        L, theta = min_container_side_with_theta(all_vertices, 1440)
        
        if L < best_L:
            best_L = L
            best_config = all_centers
            best_outer_theta = theta
            
    # Format output
    inner = [[c[0], c[1], 0.0] for c in best_config]
    outer_center = (0.0, 0.0)
    outer_side = best_L
    outer_angle = math.degrees(best_outer_theta)
    
    return inner, outer_center, outer_side, outer_angle
