import numpy as np
import time
from scipy.optimize import minimize_scalar

def construct():
    # Deadline guard
    start_time = time.time()
    DEADLINE = 140.0

    # 1. Define the fixed topology of 11 hexagons
    # Center at (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3)
    # Ring 2: 4 hexagons at distance 3.0 (specifically, the ones that fit in the "holes" or extending the lattice)
    # Standard hexagonal packing of 11:
    # (0,0)
    # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    # Ring 2 positions for 11 hexagons usually form a larger hexagon with side length 2 units of the lattice.
    # The 11 hexagons are:
    # Center
    # 6 neighbors
    # 4 more neighbors? No, a hexagon of side 2 (in terms of small hexagons) has 1 + 6 + 12 = 19.
    # We only have 11.
    # Let's pick the 11 that form a compact cluster.
    # The prompt says "11 unit hexagons".
    # A common compact cluster of 11 is the center, 6 neighbors, and 4 of the next ring.
    # Which 4? To be symmetric, we can't pick just 4 arbitrarily without breaking 6-fold symmetry if we want to optimize rotation.
    # However, the "Strategy" says "fixing the adjacency structure... as a specific hexagonal cluster (one central, six in a ring, four in the second ring)".
    # If we fix the structure, we break the 6-fold symmetry of the inner cluster relative to the outer, unless the 4 in the second ring are placed symmetrically?
    # 4 items in a ring of 12 possible spots cannot have 6-fold symmetry.
    # So the inner cluster has lower symmetry. The rotation angle optimization will find the best alignment.
    
    # Let's define the 11 positions.
    # Use axial coordinates (q, r) for hex grid, convert to Cartesian.
    # Flat-top hexagons:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    positions_rel = []
    # Center
    positions_rel.append((0.0, 0.0))
    
    # Ring 1 (q, r)
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for q, r in ring1:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        positions_rel.append((x, y))
        
    # Ring 2: We need 4 more.
    # The ring 2 has 12 spots.
    # To make the cluster as compact as possible, we should pick spots that are "closer" or fill gaps.
    # Actually, any 4 distinct spots from ring 2 will work.
    # Let's pick 4 that are roughly symmetric to avoid huge imbalance, e.g., at angles 0, 90, 180, 270 relative to center?
    # Ring 2 axial coords:
    # (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-2), (2,-1)
    # Let's pick (2,0), (0,2), (-2,0), (0,-2) -> This forms a cross shape.
    # Cartesian for (2,0): x=2sqrt(3), y=0
    # (0,2): x=sqrt(3), y=3
    # (-2,0): x=-2sqrt(3), y=0
    # (0,-2): x=-sqrt(3), y=-3
    # This set is symmetric under 180 deg rotation, but not 60.
    
    ring2_selected = [(2, 0), (0, 2), (-2, 0), (0, -2)]
    for q, r in ring2_selected:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        positions_rel.append((x, y))
        
    positions_rel = np.array(positions_rel)
    
    # Vertices of a unit flat-top hexagon centered at origin
    # Angles: 0, 60, 120, 180, 240, 300 ?
    # Flat top means top and bottom are flat.
    # Vertices at angles 30, 90, 150, 210, 270, 330.
    # Wait, standard "flat top" usually means the top edge is horizontal.
    # If top edge is horizontal, the vertices are at angles 0, 60, 120, 180, 240, 300? No.
    # Let's check the seed program's definition: "angle 30 degrees (flat-top)... Vertices at angles 30, 90, 150, 210, 270, 330".
    # This corresponds to a pointy-top hexagon if 0 is right.
    # Let's stick to the seed's definition to be safe, or just use a consistent definition.
    # Seed: "inner.append([x, y, 30.0])" -> angle 30.
    # In the seed, `compute_side_length` used max_x offset sqrt(3)/2 and max_y offset 1.0.
    # This implies vertices at (sqrt(3)/2, 1/2), (0, 1), etc.
    # This is a "pointy top" hexagon if y is up?
    # Let's use the vertices that match the seed's implicit geometry.
    # Seed vertices:
    v0 = (np.sqrt(3)/2, 0.5)
    v1 = (0, 1)
    v2 = (-np.sqrt(3)/2, 0.5)
    v3 = (-np.sqrt(3)/2, -0.5)
    v4 = (0, -1)
    v5 = (np.sqrt(3)/2, -0.5)
    vertices = np.array([v0, v1, v2, v3, v4, v5])
    
    # Outer hexagon normals
    # The outer hexagon is also regular. We can rotate the inner cluster.
    # The outer hexagon is fixed in orientation (or we can rotate it, equivalent to rotating inner).
    # Let's assume outer hexagon has flat top/bottom? Or pointy?
    # The seed returns `side, 30.0` as the last element. This is the angle of the outer hexagon.
    # If we rotate the inner cluster by `theta`, it's equivalent to keeping inner fixed and rotating outer by `-theta`.
    # We want to minimize the side length S.
    
    def calc_side(theta):
        # Rotate inner positions by theta
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        rotated_pos = positions_rel @ R.T # (11, 2)
        
        max_proj = 0.0
        # Check 6 normal directions of the outer hexagon.
        # If the outer hexagon has angle 0 (flat top?), the normals are at 0, 60, ...
        # Let's assume the outer hexagon's orientation is fixed such that its normals are at 0, 60, 120, 180, 240, 300 degrees.
        # This corresponds to a "pointy left/right" or "flat top/bottom" depending on definition.
        # If normals are at 0 (x-axis), the face is vertical.
        # Let's just use these 6 directions.
        
        for k in range(6):
            angle = np.deg2rad(k * 60)
            nx = np.cos(angle)
            ny = np.sin(angle)
            n = np.array([nx, ny])
            
            # For each inner hexagon, calculate max projection of its vertices
            # proj = (pos + v) . n
            # Max over all i, v
            # proj = pos_i . n + v . n
            # Max v . n is constant for a given n.
            # Max pos_i . n is max over i.
            
            # Precompute max v . n for this direction
            v_dots = vertices @ n
            max_v_dot = np.max(v_dots)
            
            # Max pos_i . n
            pos_dots = rotated_pos @ n
            max_pos_dot = np.max(pos_dots)
            
            total_max = max_pos_dot + max_v_dot
            if total_max > max_proj:
                max_proj = total_max
                
        # Apothem of outer hexagon with side S is S * sqrt(3) / 2
        # We need max_proj <= S * sqrt(3) / 2
        # S >= max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / np.sqrt(3.0)
    
    # Optimize theta in [0, pi/3] due to 60-degree symmetry of the outer hexagon
    # The inner cluster has lower symmetry, so we scan the full 60 degrees.
    
    result = minimize_scalar(calc_side, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-8})
    
    if time.time() - start_time > DEADLINE:
         # Fallback
         theta = 0.0
    else:
         theta = result.x
         
    # Calculate final side
    side = calc_side(theta)
    
    # Add small margin
    side += 1e-8
    
    # Construct inner list
    # Rotate positions by theta
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    final_pos = positions_rel @ R.T
    
    inner = []
    for x, y in final_pos:
        # The angle of the inner hexagons.
        # In the seed, they were all 30.0.
        # Since we rotated the *positions*, did we rotate the *hexagons* themselves?
        # The strategy says "inner hexagon may translate and rotate".
        # If we just translate the positions and keep the hexagons axis-aligned (angle 30), 
        # we are not using the full rotational freedom of the *individual* hexagons, 
        # but we are rotating the *cluster*.
        # Is it better to rotate the individual hexagons too?
        # If we rotate the cluster, the relative orientations of the hexagons to each other stay the same.
        # If we keep them all at angle 30, they are all parallel.
        # In a hexagonal packing, parallel hexagons touch perfectly.
        # So keeping them parallel (angle 30) is correct for the lattice structure.
        # The "rotation" we optimized is the rotation of the entire lattice relative to the container.
        inner.append([x, y, 30.0])
        
    return inner, (0.0, 0.0), side, 30.0
