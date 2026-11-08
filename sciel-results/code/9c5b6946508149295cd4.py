import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the 11 hexagon centers in a hexagonal lattice
    # Basis vectors for hex grid with neighbor distance sqrt(3)
    # u = (sqrt(3), 0), v = (sqrt(3)/2, 3/2)
    sqrt3 = np.sqrt(3)
    u = np.array([sqrt3, 0.0])
    v = np.array([sqrt3/2.0, 3.0/2.0])
    
    # Grid coordinates (i, j) -> Cartesian i*u + j*v
    # 1 center
    coords = [(0, 0)]
    # 6 neighbors (Ring 1)
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    coords.extend(neighbors)
    # 4 from Ring 2
    # Ring 2 coords: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-1,-1), (0,-2), (1,-2), (2,-1)
    # Pick 4 to form a compact shape. Let's pick (2,0), (1,1), (0,2), (-1,2)
    ring2_subset = [(2, 0), (1, 1), (0, 2), (-1, 2)]
    coords.extend(ring2_subset)
    
    # Ensure we have exactly N=11
    assert len(coords) == 11
    
    centers = []
    for i, j in coords:
        c = i * u + j * v
        centers.append(c)
        
    # Generate vertices for all inner hexagons
    # Unit hexagon vertices relative to center
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    rel_vertices = np.array([np.cos(angles), np.sin(angles)]).T
    
    all_vertices = []
    for c in centers:
        verts = c + rel_vertices
        all_vertices.append(verts)
        
    all_vertices = np.vstack(all_vertices)
    
    # Find optimal outer hexagon side length S
    # S(theta) = (1/sqrt(3)) * max_k width_k(theta)
    # width_k(theta) = max(p . n_k) - min(p . n_k)
    # n_k are normals at theta + k*pi/3
    
    best_S = float('inf')
    best_theta = 0.0
    
    # Sample theta from 0 to pi/3 (60 deg)
    # The function is periodic with pi/3?
    # Width function is not necessarily periodic with pi/3, but the set of normals is.
    # We just need to check 0 to pi/3.
    
    # Use a fine grid
    thetas = np.linspace(0, np.pi/3, 1800)
    
    for theta in thetas:
        max_width = 0.0
        for k in range(6):
            angle = theta + k * np.pi / 3.0
            n = np.array([np.cos(angle), np.sin(angle)])
            projections = all_vertices @ n
            w = np.max(projections) - np.min(projections)
            if w > max_width:
                max_width = w
        S = max_width / sqrt3
        if S < best_S:
            best_S = S
            best_theta = theta
            
    # Construct the return values
    # inner: list of [x, y, angle_degrees]
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    # Outer center
    # The center c that minimizes the max projection is the midpoint of the min/max projections?
    # For each normal n_k, the optimal center projection is (max+min)/2.
    # We need a center that works for all k.
    # Since we calculated S based on widths, we can just pick the centroid of the vertices?
    # Or better, calculate the center that centers the bounding box in all 6 directions.
    # Actually, the score only depends on S. The center can be anything that makes it valid.
    # The evaluator likely checks if vertices are inside.
    # We should provide a center that ensures validity.
    
    # Let's compute the optimal center for the best_theta
    theta = best_theta
    center = np.zeros(2)
    for k in range(6):
        angle = theta + k * np.pi / 3.0
        n = np.array([np.cos(angle), np.sin(angle)])
        projections = all_vertices @ n
        p_min = np.min(projections)
        p_max = np.max(projections)
        # Center projection should be (p_min + p_max) / 2
        # We have 6 constraints on center.
        # We can just average the implied center projections?
        # Or use the centroid of the vertices as a safe guess, then verify?
        # Given the symmetry of the cluster, the centroid is likely good.
        pass
        
    # Simpler: Just return the centroid of the inner centers as the outer center.
    # And S calculated.
    # Is it guaranteed to be valid?
    # The width calculation assumes we can center the hexagon optimally.
    # If we return a center that doesn't align with the optimal centering for each axis,
    # the required S might be larger.
    # To be safe, let's find the center that minimizes the max deviation.
    
    # Let's just use the centroid of all vertices as the center.
    center = np.mean(all_vertices, axis=0)
    
    # Verify S with this center
    # If it's invalid, we might need to adjust.
    # But the width-based S is the theoretical minimum for *some* center.
    # If we pick the wrong center, we might need a larger S.
    # Let's calculate the required S for the specific center we return.
    
    required_S = 0.0
    for k in range(6):
        angle = theta + k * np.pi / 3.0
        n = np.array([np.cos(angle), np.sin(angle)])
        projections = (all_vertices - center) @ n
        # Max absolute projection
        m = np.max(np.abs(projections))
        # Condition: m <= S * sqrt(3) / 2
        S_req = 2 * m / sqrt3
        if S_req > required_S:
            required_S = S_req
            
    # Use the calculated required_S to be safe
    final_S = required_S
    
    return inner, (center[0], center[1]), final_S, np.degrees(theta)

# EVOLVE-BLOCK-END
