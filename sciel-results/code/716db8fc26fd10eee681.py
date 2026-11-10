# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def get_edge_normals(vertices):
    """Return 6 outward edge normal unit vectors for a CCW hexagon."""
    n = len(vertices)
    normals = np.empty((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        length = np.linalg.norm(edge)
        if length == 0:
            normals[i] = [1.0, 0.0]
        else:
            normals[i] = [edge[1], -edge[0]] / length
    return normals

def validate_fast(inner_data, outer_data, side):
    """
    Fast validation.
    Returns True if valid, False otherwise.
    """
    if not (np.all(np.isfinite(inner_data)) and np.all(np.isfinite(outer_data)) and np.isfinite(side)):
        return False
    if side < math.sqrt(11) - 1e-9:
        return False

    inner_hexes = []
    for i in range(11):
        x, y, a = inner_data[i]
        verts = hex_vertices(x, y, 1.0, a)
        inner_hexes.append(verts)
    
    outer_x, outer_y, outer_a = outer_data
    outer_verts = hex_vertices(outer_x, outer_y, side, outer_a)
    
    # Precompute axes for inner hexagons
    axes_list = [get_edge_normals(h) for h in inner_hexes]
    
    # 1. Check inner-inner separation
    for i in range(11):
        for j in range(i + 1, 11):
            v1 = inner_hexes[i]
            v2 = inner_hexes[j]
            axes = np.vstack([axes_list[i], axes_list[j]])
            
            # Project v1 and v2 onto all axes
            # v1: (6, 2), axes: (12, 2)
            # proj1: (6, 12)
            proj1 = v1.T @ axes.T
            p1_min = np.min(proj1, axis=0)
            p1_max = np.max(proj1, axis=0)
            
            proj2 = v2.T @ axes.T
            p2_min = np.min(proj2, axis=0)
            p2_max = np.max(proj2, axis=0)
            
            # Check separation on any axis
            sep1 = p2_min - p1_max
            sep2 = p1_min - p2_max
            
            if np.any(sep1 > 1e-6) or np.any(sep2 > 1e-6):
                continue # Separated
            
            # If not separated, it's invalid
            return False

    # 2. Check inner inside outer
    outer_edges = np.empty((6, 2))
    for k in range(6):
        p1 = outer_verts[k]
        p2 = outer_verts[(k+1)%6]
        outer_edges[k] = p2 - p1
    
    all_inner_verts = np.vstack(inner_hexes) # (66, 2)
    
    for k in range(6):
        p1 = outer_verts[k]
        v1 = outer_edges[k]
        v2 = all_inner_verts - p1
        
        cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
        
        if np.any(cross < -1e-6):
            return False

    return True

def hexagon_packing_11():
    start_time = time.time()
    deadline = 620.0
    
    # Seed: 7 hexes in a hexagon (side 2) + 4 more.
    # s3 = sqrt(3)
    s3 = math.sqrt(3)
    
    # Centers for 7 hexes (side 2)
    centers = [
        (0, 0),
        (s3, 0),
        (s3/2, 1.5),
        (-s3/2, 1.5),
        (-s3, 0),
        (-s3/2, -1.5),
        (s3/2, -1.5),
    ]
    
    # Add 4 more.
    # Place them at the "corners" of the next ring?
    # Angles 0, 60, 120, 180, 240, 300 at distance 2*s3?
    # That would be 6 more. We only need 4.
    # Let's pick 4: 0, 120, 240, and maybe 60?
    # Or just spread them out.
    
    # Let's try adding them at distance 2*s3 * 0.9 to keep them inside?
    # No, they must not overlap.
    # Distance between center (0,0) and (2*s3, 0) is 2*s3.
    # Distance between (s3, 0) and (2*s3, 0) is s3. They touch.
    # So (2*s3, 0) is a valid neighbor to (s3, 0).
    
    # Let's add:
    # (2*s3, 0)
    # (s3, 1.5*sqrt(3))? No.
    # Let's use axial coordinates for the 4th ring?
    
    # Simpler: Just use the previous valid seed (side 8) and optimize.
    # It's safer.
    
    inner_seed = np.array([
        [0, 0, 0],  
        [-2.5, 0, 0],  
        [2.5, 0, 0],  
        [-1.25, 2.17, 0],  
        [1.25, 2.17, 0],  
        [-1.25, -2.17, 0],  
        [1.25, -2.17, 0],  
        [-3.75, 2.17, 0],  
        [3.75, 2.17, 0],  
        [-3.75, -2.17, 0],  
        [3.75, -2.17, 0],  
    ])
    
    best_inner = inner_seed.copy()
    best_outer = np.array([0.0, 0.0, 0.0])
    best_side = 8.0
    
    # Verify seed
    if not validate_fast(best_inner, best_outer, best_side):
        # Fallback to very large
        best_side = 20.0
        for i in range(11):
            best_inner[i, 0] = i * 2.0
            best_inner[i, 1] = 0.0
            best_inner[i, 2] = 0.0

    # Optimization: Hill Climbing on positions to minimize max_radius
    def max_radius(inner_data):
        max_r = 0
        for i in range(11):
            x, y, a = inner_data[i]
            verts = hex_vertices(x, y, 1.0, a)
            r = np.max(np.linalg.norm(verts, axis=1))
            if r > max_r:
                max_r = r
        return max_r

    current_params = best_inner.flatten().copy()
    current_max_r = max_radius(best_inner)
    
    # Check if we can already improve side
    # side must be at least max_radius?
    # For a hexagon centered at 0,0 with vertices at 0, 60...
    # The distance to vertex is side.
    # The distance to edge is side * sqrt(3)/2.
    # If max_radius is the distance to the furthest vertex of an inner hex,
    # we need the outer hex to contain it.
    # If the inner vertex is at angle 0, distance R.
    # Outer hex vertex at angle 0 is at distance side.
    # So side >= R is necessary if the vertex aligns with an outer vertex.
    # If it aligns with an edge, side >= R / (sqrt(3)/2).
    # To be safe, let's use side = max_radius.
    # And check validity.
    
    test_side = current_max_r
    if validate_fast(best_inner, best_outer, test_side):
        best_side = test_side
    else:
        # Try slightly larger
        test_side = current_max_r * 1.05
        if validate_fast(best_inner, best_outer, test_side):
            best_side = test_side

    # Hill Climb
    step_size = 0.1
    for iteration in range(50000):
        if time.time() - start_time > deadline:
            break
            
        idx = np.random.randint(0, 33)
        delta = (np.random.rand() - 0.5) * 2 * step_size
        new_params = current_params.copy()
        new_params[idx] += delta
        
        new_inner = np.zeros((11, 3))
        for i in range(11):
            new_inner[i, 0] = new_params[3*i]
            new_inner[i, 1] = new_params[3*i + 1]
            new_inner[i, 2] = new_params[3*i + 2]
            
        new_max_r = max_radius(new_inner)
        
        # Accept if better
        if new_max_r < current_max_r:
            current_params = new_params
            current_max_r = new_max_r
            
            # Check validity with new max_r as side
            if validate_fast(new_inner, best_outer, current_max_r):
                best_inner = new_inner.copy()
                best_side = current_max_r
                if iteration % 5000 == 0 and step_size > 0.005:
                    step_size *= 0.95
        elif np.random.rand() < 0.05:
            current_params = new_params
            current_max_r = new_max_r
            if validate_fast(new_inner, best_outer, current_max_r):
                best_inner = new_inner.copy()
                best_side = current_max_r

    # Final check
    if not validate_fast(best_inner, best_outer, best_side):
        best_inner = inner_seed
        best_outer = np.array([0.0, 0.0, 0.0])
        best_side = 8.0
        
    return best_inner, best_outer, best_side

# EVOLVE-BLOCK-END
