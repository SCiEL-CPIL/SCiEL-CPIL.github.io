# EVOLVE-BLOCK-START
import time
import numpy as np

def _hex_vertices(cx, cy, s, a_deg):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angs = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])

def _minkowski_diff_area(v1, v2):
    """
    Computes the area of the Minkowski difference of two convex polygons v1 and v2.
    If the area is 0, they are separated (or touching). If > 0, they intersect.
    """
    # Edge vectors for v1
    e1 = np.roll(v1, -1, axis=0) - v1
    # Edge vectors for v2 (negated for difference A - B)
    e2 = -(np.roll(v2, -1, axis=0) - v2)
    
    # Angles of edge vectors
    ang1 = np.arctan2(e1[:, 1], e1[:, 0])
    ang2 = np.arctan2(e2[:, 1], e2[:, 0])
    
    # Sort edges by angle
    idx1 = np.argsort(ang1)
    idx2 = np.argsort(ang2)
    e1_sorted = e1[idx1]
    e2_sorted = e2[idx2]
    
    # Merge edges in angular order
    e_merged = np.empty((12, 2))
    i, j = 0, 0
    for k in range(12):
        if i < 6 and (j >= 6 or ang1[idx1[i]] <= ang2[idx2[j]]):
            e_merged[k] = e1_sorted[i]
            i += 1
        else:
            e_merged[k] = e2_sorted[j]
            j += 1
            
    # Reconstruct polygon vertices from edge vectors
    verts = np.zeros((12, 2))
    verts[0] = 0
    for i in range(1, 12):
        verts[i] = verts[i-1] + e_merged[i-1]
        
    # Shoelace formula for area
    x = verts[:, 0]
    y = verts[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    
    return area

def _compute_outer_hex(hexagons):
    """
    Compute the minimal enclosing regular hexagon side length and optimal rotation.
    Returns (side_length, angle_deg).
    """
    all_verts = []
    for (cx, cy, ang) in hexagons:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    best_s = np.inf
    best_a = 0.0
    
    # Scan rotation angles
    for a_deg in np.linspace(0, 60, 61):
        a = np.radians(a_deg)
        edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
        normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
        
        proj = all_verts @ normals.T
        max_proj = proj.max(axis=0)
        min_proj = proj.min(axis=0)
        
        required_apothem = np.maximum(max_proj, -min_proj)
        apothem = required_apothem.max()
        s = apothem / (np.sqrt(3) / 2)
        
        if s < best_s:
            best_s = s
            best_a = a_deg
            
    return best_s, best_a

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 590.0 # Safety margin
    
    # Initial state: Center + 6 ring + 4 outer ring
    hexes = np.zeros((11, 3))
    # Center
    hexes[0] = [0.0, 0.0, 0.0]
    # Ring 1 (6 hexes)
    r1 = np.sqrt(3)
    for i in range(6):
        ang = 60.0 * i
        rad = np.radians(ang)
        hexes[i+1, 0] = r1 * np.cos(rad)
        hexes[i+1, 1] = r1 * np.sin(rad)
        hexes[i+1, 2] = 0.0
    # Ring 2 (4 hexes)
    r2 = 3.5
    for i in range(4):
        ang = 90.0 * i
        rad = np.radians(ang)
        hexes[i+7, 0] = r2 * np.cos(rad)
        hexes[i+7, 1] = r2 * np.sin(rad)
        hexes[i+7, 2] = 0.0
    
    current_hexes = hexes.copy()
    best_hexes = hexes.copy()
    
    # Calculate initial best
    best_s, best_a = _compute_outer_hex(best_hexes)
    # Verify separation for best
    verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in best_hexes]
    sep_viol = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            area = _minkowski_diff_area(verts_list[i], verts_list[j])
            if area > 1e-6:
                sep_viol += area
                
    current_s, _ = _compute_outer_hex(current_hexes)
    current_sep_viol = sep_viol
    current_cost = current_s + (100 * current_sep_viol if current_sep_viol > 0 else 0)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        # Generate a perturbation
        new_hexes = current_hexes.copy()
        idx = np.random.randint(0, 11)
        
        # Move position
        new_hexes[idx, 0] += np.random.normal(0, 0.2)
        new_hexes[idx, 1] += np.random.normal(0, 0.2)
        
        # Rotate
        new_hexes[idx, 2] += np.random.normal(0, 10.0)
        
        # Compute cost
        verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in new_hexes]
        
        new_sep_viol = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                area = _minkowski_diff_area(verts_list[i], verts_list[j])
                if area > 1e-6:
                    new_sep_viol += area
        
        new_s, _ = _compute_outer_hex(new_hexes)
        
        if new_sep_viol == 0:
            new_cost = new_s
        else:
            new_cost = new_s + 100 * new_sep_viol
            
        # Acceptance
        delta = new_cost - current_cost
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_hexes = new_hexes
            current_s = new_s
            current_sep_viol = new_sep_viol
            current_cost = new_cost
            
            # Update best if valid and better
            if new_sep_viol == 0 and new_s < best_s:
                best_s = new_s
                best_hexes = new_hexes.copy()
                
        T *= cooling
        
    # Final check on best_hexes
    # If best_hexes is invalid, we have a problem.
    # But SA should find a valid one.
    # If not, we fall back to the initial configuration.
    verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in best_hexes]
    valid = True
    for i in range(11):
        for j in range(i+1, 11):
            if _minkowski_diff_area(verts_list[i], verts_list[j]) > 1e-6:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to initial
        best_hexes = hexes.copy()
        best_s, best_a = _compute_outer_hex(best_hexes)
    else:
        best_s, best_a = _compute_outer_hex(best_hexes)
        
    outer_hex_data = np.array([0.0, 0.0, best_a])
    
    return best_hexes, outer_hex_data, float(best_s)

# EVOLVE-BLOCK-END
