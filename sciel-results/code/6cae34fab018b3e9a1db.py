import numpy as np
import random
import math
import time

N = 12

# Precompute vertices of a unit regular hexagon (side length 1)
# Vertices at angles 0, 60, 120, 180, 240, 300
HEX_VERTS = np.array([
    [math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)
])

# Normals to the edges of a unit hexagon (pointing outward)
# Edge 0 is between vertex 0 and 1. Midpoint angle is 30. Normal is 30.
HEX_NORMALS = np.array([
    [math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 360, 60)
])

def get_hex_vertices(center, angle_deg):
    """Get vertices of a hexagon of side 1 centered at 'center' with rotation 'angle_deg'."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return center + (R @ HEX_VERTS.T).T

def get_hex_normals(angle_deg):
    """Get outward normals of a hexagon with rotation 'angle_deg'."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R @ HEX_NORMALS.T

def project_points(points, normal):
    """Project points onto a normal vector, return min and max."""
    dots = points @ normal
    return np.min(dots), np.max(dots)

def check_overlap(v1, n1, v2, n2):
    """
    Check if two hexagons overlap using SAT.
    v1, v2: vertices (6x2)
    n1, n2: normals (6x2)
    Returns overlap depth (positive if overlapping, 0 if disjoint).
    """
    all_normals = np.vstack([n1, n2])
    min_overlap = float('inf')
    
    for normal in all_normals:
        min1, max1 = project_points(v1, normal)
        min2, max2 = project_points(v2, normal)
        
        # Check for separation
        if max1 <= min2 or max2 <= min1:
            return 0.0 # Disjoint
        
        # Calculate overlap depth on this axis
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    return min_overlap

def calculate_min_enclosing_hexagon_side(centers, angles, verts_cache=None):
    """
    Calculate the minimum side length S of a regular hexagon centered at origin 
    (rotation 0) that encloses all given hexagons.
    """
    max_proj = 0.0
    outer_normals = HEX_NORMALS # Since outer rotation is fixed to 0
    
    for i in range(N):
        if verts_cache is not None:
            verts = verts_cache[i]
        else:
            verts = get_hex_vertices(centers[i], angles[i])
            
        dots = verts @ outer_normals.T
        max_proj = max(max_proj, np.max(dots))
        
    return 2.0 * max_proj / math.sqrt(3)

def construct():
    random.seed(42)
    
    # Initial positions: Hexagonal packing pattern
    # 1 center, 6 around, 5 more?
    # Let's try a simple grid and let SA optimize.
    positions = []
    angles = []
    
    # Start with a loose grid
    for i in range(N):
        x = (i % 4) * 2.5
        y = (i // 4) * 2.5
        # Add some randomness
        x += random.uniform(-0.5, 0.5)
        y += random.uniform(-0.5, 0.5)
        positions.append([x, y])
        angles.append(random.uniform(0, 60))
        
    positions = np.array(positions)
    angles = np.array(angles)
    
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_energy = float('inf')
    
    current_positions = positions.copy()
    current_angles = angles.copy()
    
    def calc_energy(pos, ang):
        # Precompute vertices and normals
        v_list = []
        n_list = []
        for i in range(N):
            v = get_hex_vertices(pos[i], ang[i])
            n = get_hex_normals(ang[i])
            v_list.append(v)
            n_list.append(n)
            
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < 2.1: # Only check if close
                    ov = check_overlap(v_list[i], n_list[i], v_list[j], n_list[j])
                    if ov > 0:
                        overlap_pen += ov * ov * 50.0
                        
        S = calculate_min_enclosing_hexagon_side(pos, ang, v_list)
        
        return S + overlap_pen, v_list
    
    current_energy, _ = calc_energy(current_positions, current_angles)
    best_energy = current_energy
    
    start_time = time.time()
    time_limit = 90.0
    T = 2.0
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        idx = random.randint(0, N-1)
        
        step = T * 0.3
        dx = random.gauss(0, step)
        dy = random.gauss(0, step)
        dtheta = random.gauss(0, step * 5)
        
        new_pos = current_positions.copy()
        new_ang = current_angles.copy()
        
        new_pos[idx][0] += dx
        new_pos[idx][1] += dy
        new_ang[idx] = (new_ang[idx] + dtheta) % 60.0
        
        new_energy, _ = calc_energy(new_pos, new_ang)
        
        delta = new_energy - current_energy
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_positions = new_pos
            current_angles = new_ang
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
                best_angles = current_angles.copy()
                
        T *= cooling_rate
        if T < 0.05:
            T = 0.05

    # Final calculation of side length
    final_side = calculate_min_enclosing_hexagon_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
        
    return inner, (0.0, 0.0), final_side, 0.0
