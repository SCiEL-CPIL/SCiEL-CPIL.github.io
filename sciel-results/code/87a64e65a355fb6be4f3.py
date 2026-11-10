# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize_scalar
from itertools import product

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side s, angle angle_deg."""
    angles = np.radians(angle_deg + 60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def get_edge_normals(verts):
    """Get outward-pointing edge normals for a hexagon given its 6 vertices."""
    normals = []
    center = np.mean(verts, axis=0)
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        edge = p2 - p1
        # Perpendicular to edge
        n = np.array([-edge[1], edge[0]])
        mid = (p1 + p2) / 2
        # Ensure outward pointing
        if np.dot(n, mid - center) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)

def min_gap_between_hexes(verts1, verts2):
    """
    Compute the minimum gap between two hexagons based on the 12 edge normals.
    Returns the maximum violation (0 if valid separation > 1e-6).
    """
    n1 = get_edge_normals(verts1)
    n2 = get_edge_normals(verts2)
    all_normals = np.vstack([n1, n2])
    
    proj1 = verts1 @ all_normals.T
    proj2 = verts2 @ all_normals.T
    
    min1 = np.min(proj1, axis=0)
    max1 = np.max(proj1, axis=0)
    min2 = np.min(proj2, axis=0)
    max2 = np.max(proj2, axis=0)
    
    # Overlap 1: Hex1 is to the 'left' of Hex2 (max1 < min2 is good)
    # Violation if max1 >= min2 - 1e-6
    viol1 = np.maximum(0, max1 - min2 + 1e-6)
    # Overlap 2: Hex2 is to the 'left' of Hex1
    viol2 = np.maximum(0, max2 - min1 + 1e-6)
    
    total_viol = np.maximum(viol1, viol2)
    max_violation = np.max(total_viol)
    
    return max_violation

def calculate_min_side_fast(inner_hex_data, outer_angle_deg):
    """
    Calculates the minimum side length S of an outer regular hexagon centered at origin,
    with given angle, that contains all inner hexagons.
    Vectorized calculation.
    """
    outer_angle_rad = np.radians(outer_angle_deg)
    # Normals of the outer hexagon (apothem directions)
    # The apothem direction is perpendicular to the side.
    # Vertices are at angles a + 60*i.
    # Side i is between vertex i and i+1.
    # Midpoint angle is a + 30 + 60*i.
    normal_angles = np.radians(outer_angle_deg + 30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Precompute base vertices for unit hexagon
    base_angles = np.radians(60 * np.arange(6))
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    max_proj = 0.0
    
    # inner_hex_data: (11, 3) -> x, y, angle_deg
    # We need to check all 66 vertices (11 * 6)
    
    # For each inner hex, we can rotate the base vertices and translate.
    # To be efficient, we can precompute rotated base vertices for each unique angle.
    # But angles might be different for each hex.
    
    # Let's just loop over the 11 hexagons but vectorize the 6 vertices per hex.
    for i in range(inner_hex_data.shape[0]):
        x, y, ang_deg = inner_hex_data[i]
        ang_rad = np.radians(ang_deg)
        c, s = np.cos(ang_rad), np.sin(ang_rad)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate base vertices
        rot_verts = base_verts @ R.T
        # Translate
        verts = rot_verts + np.array([x, y])
        
        projs = verts @ normals.T
        max_p = np.max(projs)
        if max_p > max_proj:
            max_proj = max_p
            
    # Apothem = S * sqrt(3) / 2
    # S = 2 * Apothem / sqrt(3)
    side = 2.0 * max_proj / math.sqrt(3)
    
    return side

def is_valid_configuration(inner_hex_data, outer_hex_data, outer_side, tol=1e-6):
    """Check if the configuration is valid."""
    outer_cx, outer_cy, outer_angle = outer_hex_data
    
    # Outer vertices
    outer_verts = hexagon_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    
    base_angles = np.radians(60 * np.arange(6))
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    # 1. Containment
    for i in range(inner_hex_data.shape[0]):
        x, y, ang_deg = inner_hex_data[i]
        ang_rad = np.radians(ang_deg)
        c, s = np.cos(ang_rad), np.sin(ang_rad)
        R = np.array([[c, -s], [s, c]])
        verts = (base_verts @ R.T) + np.array([x, y])
        
        for v in verts:
            for j in range(6):
                p1 = outer_verts[j]
                p2 = outer_verts[(j+1)%6]
                e = p2 - p1
                vp = v - p1
                cross = e[0]*vp[1] - e[1]*vp[0]
                if cross < -tol:
                    return False
                    
    # 2. Separation
    # Precompute vertices for all inner hexes to avoid recomputation
    all_verts = []
    for i in range(inner_hex_data.shape[0]):
        x, y, ang_deg = inner_hex_data[i]
        verts = hexagon_vertices(x, y, 1.0, ang_deg)
        all_verts.append(verts)
        
    for i in range(11):
        for j in range(i+1, 11):
            viol = min_gap_between_hexes(all_verts[i], all_verts[j])
            if viol > tol:
                return False
                
    return True

def generate_configurations():
    """
    Generate several candidate rigid configurations.
    Returns a list of arrays of shape (11, 3) with x, y, angle_deg.
    """
    configs = []
    h = math.sqrt(3) / 2
    
    # Config 1: 4-3-4 Grid (Honeycomb)
    # Rows:
    # 4 hexes at y=0: x = 0, 3, 6, 9
    # 3 hexes at y=2h: x = 1.5, 4.5, 7.5
    # 4 hexes at y=4h: x = 0, 3, 6, 9
    # Total 11.
    coords = []
    for i in range(4):
        coords.append([i*3.0, 0.0, 0.0])
    for i in range(3):
        coords.append([1.5 + i*3.0, 2*h, 0.0])
    for i in range(4):
        coords.append([i*3.0, 4*h, 0.0])
        
    # Center it
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    for c in coords:
        c[0] -= cx
        c[1] -= cy
    configs.append(np.array(coords))
    
    # Config 2: 1-6-4 (Flower with tail)
    # Center at 0,0
    # 6 around it
    # 4 more attached to one side?
    # Let's try a more compact "hexagon of hexagons" shape minus corners?
    # A hexagon of side 2 has 1+6+12=19 hexes. Too many.
    # Side 1 has 7 hexes.
    # We need 11.
    # 7 (side 1) + 4 attached to one side.
    # The 4 attached can be in a line or a 2x2 block?
    # Let's try attaching 4 in a line to one of the 6 neighbors.
    
    coords = [[0,0,0]]
    # 6 neighbors
    for i in range(6):
        ang = np.radians(60*i)
        x = 2*h * np.cos(ang + np.pi/6) # Distance to neighbor center is 2h? No, distance is 2*apothem = sqrt(3).
        # Wait, distance between centers of touching hexes is 2 * apothem = sqrt(3).
        # Apothem = sqrt(3)/2.
        # So distance is sqrt(3).
        d = math.sqrt(3)
        x = d * np.cos(ang)
        y = d * np.sin(ang)
        coords.append([x, y, 0.0])
        
    # Add 4 more.
    # Let's add them extending from one of the neighbors.
    # Neighbor at angle 0 is at (sqrt(3), 0).
    # Add hexes at (2*sqrt(3), 0), (3*sqrt(3), 0)... this makes it long.
    # Maybe add them in a 2x2 block?
    # Let's stick to the 4-3-4 grid which is quite compact.
    # Another option: 3-4-3-1? No.
    
    # Let's try a "diamond" or other symmetric shapes.
    # The 4-3-4 grid is rectangular.
    # Maybe rotate it 30 degrees?
    
    return configs

def optimize_rotation(config, outer_angle=0.0):
    """
    Optimize the global rotation of the rigid cluster to minimize the enclosing hexagon side.
    """
    def objective(theta):
        # Rotate the cluster
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        new_config = config.copy()
        # Rotate positions
        positions = new_config[:, :2]
        new_positions = positions @ R.T
        new_config[:, :2] = new_positions
        # Rotate angles
        new_config[:, 2] = config[:, 2] + np.degrees(theta)
        
        return calculate_min_side_fast(new_config, outer_angle)
        
    # Optimize theta in [0, pi/3) due to 6-fold symmetry of outer hex?
    # Actually, the cluster symmetry might be lower.
    # Let's scan 0 to 180 degrees.
    res = minimize_scalar(objective, bounds=(0, math.pi/3), method='bounded', options={'xatol': 1e-8})
    
    if res.success:
        theta = res.x
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        new_config = config.copy()
        positions = new_config[:, :2]
        new_positions = positions @ R.T
        new_config[:, :2] = new_positions
        new_config[:, 2] = config[:, 2] + np.degrees(theta)
        
        return new_config, res.fun
    else:
        return config, float('inf')

def hexagon_packing_11():
    start_time = time.time()
    deadline = 630.0
    safety_margin = 10.0

    best_side = float('inf')
    best_inner = None
    best_outer = None

    configs = generate_configurations()
    
    # Try different outer angles as well?
    # The problem allows optimizing the outer hexagon angle too.
    # But relative angle is what matters.
    # We can fix outer angle to 0 and rotate the cluster.
    
    for config in configs:
        if time.time() - start_time > deadline - safety_margin:
            break
            
        # Optimize rotation of cluster relative to outer hex (fixed at 0)
        new_config, side = optimize_rotation(config, outer_angle=0.0)
        
        # Check validity
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        
        if side < best_side:
            # Verify validity before updating best
            if is_valid_configuration(new_config, outer_hex_data, side):
                best_side = side
                best_inner = new_config
                best_outer = outer_hex_data
                
    # If no valid config found, use the first one scaled up slightly?
    # Or just return the best found even if slightly invalid? No, must be valid.
    # The 4-3-4 grid is usually valid.
    
    if best_inner is None:
        # Fallback
        config = configs[0]
        new_config, side = optimize_rotation(config, outer_angle=0.0)
        best_inner = new_config
        best_outer = np.array([0.0, 0.0, 0.0])
        best_side = side
        
    return best_inner, best_outer, best_side

# EVOLVE-BLOCK-END
