# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner, outer, s).
    """
    # Use a dense hexagonal cluster to minimize the bounding hexagon side length.
    # The cluster consists of 12 points on a hexagonal lattice.
    # We define the centers in axial coordinates (i, j) on the honeycomb lattice.
    # Neighbors in axial coords: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1).
    # Distance between adjacent lattice points is 1 (in units of hexagon side length? 
    # Wait, for hexagons of side 1, the distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    # Let's re-verify the geometry.
    # Hexagon side d=1.
    # Apothem a = d * sqrt(3)/2.
    # Distance between centers of two touching hexagons sharing an edge is 2*a = sqrt(3).
    # So the "lattice" distance for non-overlapping hexagons must be at least sqrt(3).
    
    # Let's define a lattice where the basis vectors have length sqrt(3) and angle 60 degrees.
    # Basis vector 1: v1 = (sqrt(3), 0)
    # Basis vector 2: v2 = (sqrt(3)/2, 1.5)
    # Length of v2: sqrt(3/4 + 2.25) = sqrt(0.75 + 2.25) = sqrt(3). Correct.
    # Dot product: 3/2. Cosine = (3/2) / 3 = 0.5 -> 60 degrees. Correct.
    
    # We need 12 centers.
    # Let's try a compact shape.
    # A hexagon of radius 1 (1 center + 6 neighbors) has 7 points.
    # Adding the next layer gives 7 + 12 = 19 points. Too many.
    # So we take a subset of the 19-point cluster or a different shape.
    # A 12-point cluster could be a "hexagon of radius 1" (7 points) + 5 points from the next layer.
    # Or a rectangular patch.
    # Let's try a "lozenge" or parallelogram shape in the lattice.
    # A 3x4 grid in axial coords?
    # i from 0 to 2, j from 0 to 3?
    # Points: (i, j) for i in 0..2, j in 0..3. Total 3*4=12.
    # Let's check if this fits in a small hexagon.
    
    sqrt3 = np.sqrt(3)
    v1 = np.array([sqrt3, 0])
    v2 = np.array([sqrt3/2, 1.5])
    
    centers = []
    for i in range(3):
        for j in range(4):
            c = i * v1 + j * v2
            centers.append(c)
            
    centers = np.array(centers)
    
    # Center the cluster at the origin
    centroid = np.mean(centers, axis=0)
    centers -= centroid
    
    # Now determine the minimum side length S of the outer hexagon.
    # The outer hexagon is regular, centered at origin, side length S.
    # Its apothem is S * sqrt(3)/2.
    # The inner hexagons have side 1, apothem sqrt(3)/2.
    # For an inner hexagon at center c, its farthest vertex from the origin in any direction u 
    # is |proj_u(c)| + 1 (since max proj of hexagon vertices from center is side length 1? 
    # No, max distance from center to vertex is 1. Max projection in direction u is 1 if u aligns with vertex, sqrt(3)/2 if u aligns with edge normal.
    # Actually, the support function of a unit hexagon in direction u is:
    # h(u) = max_{v in hex} v . u.
    # If u is a normal to an edge, h(u) = apothem = sqrt(3)/2.
    # If u is a direction to a vertex, h(u) = circumradius = 1.
    # The outer hexagon has support function H(u) = S * h_outer(u).
    # For a regular hexagon of side S, h_outer(u) is:
    # S * sqrt(3)/2 if u is normal to an edge, S if u is direction to a vertex.
    # We need c + h_inner(u) <= H(u) for all u? No.
    # The condition is: for all directions u, max_{v in inner_i} v . u <= max_{w in outer} w . u.
    # max_{v in inner_i} v . u = c_i . u + h_inner(u).
    # So we need c_i . u + h_inner(u) <= H_outer(u) for all i and u.
    # This is equivalent to: for all u, max_i (c_i . u + h_inner(u)) <= H_outer(u).
    # Let M(u) = max_i c_i . u.
    # Then M(u) + h_inner(u) <= H_outer(u).
    
    # We can optimize S by rotating the outer hexagon.
    # Let's assume the outer hexagon is aligned such that its edges are parallel to the lattice vectors?
    # Or just try to find the minimal S for a fixed orientation of the outer hexagon (e.g., flat top).
    # Flat top outer hexagon: normals at 0, 60, 120, 180, 240, 300 degrees?
    # Vertices at 0, 60, ... degrees. Normals at 30, 90, ... degrees?
    # Let's use the property that the bounding hexagon side S is determined by the max projection of the combined shape.
    
    # Simpler approach: 
    # The outer hexagon must contain all inner hexagons.
    # This is equivalent to finding the smallest S such that there exists a translation/rotation 
    # (here rotation of outer) such that all inner hexagons are inside.
    # Since we centered the cluster, we can try to compute the required S for a given orientation of the outer hexagon.
    
    # Let's try to compute S by checking the distance from the origin to the farthest point of the union of inner hexagons, 
    # in the directions normal to the outer hexagon edges.
    # If the outer hexagon has side S and is centered at origin, its inradius is R_in = S * sqrt(3)/2.
    # The distance from the origin to the edge in direction n (unit normal) is R_in.
    # So we need: max_i (c_i . n + h_inner(n)) <= R_in = S * sqrt(3)/2.
    # Thus S >= (2 / sqrt(3)) * max_n [ max_i c_i . n + h_inner(n) ].
    # We want to minimize S over the orientation of the outer hexagon.
    # Let the outer hexagon be rotated by angle theta.
    # The normals are at theta + 30 + k*60 degrees.
    # h_inner(n) depends on the angle of n relative to the inner hexagon orientation.
    # The inner hexagons are all axis-aligned (angle 0).
    # So h_inner(n) is periodic with period 60 degrees.
    # Let phi be the angle of n.
    # h_inner(phi) = sqrt(3)/2 / cos(phi - 30k) ... no.
    # For a unit hexagon with flat top (vertex at 0 deg? no, standard angle 0 usually means vertex at 0? 
    # Task says: vertices at cx + d cos(a + 60i). If a=0, vertex at 1,0. So pointy top? 
    # Vertices at 0, 60, 120, 180, 240, 300.
    # Normals are at 30, 90, 150, ...
    # h_inner(phi) = 1 / cos(phi) if phi in [-30, 30]? No.
    # The support function of a regular hexagon with circumradius 1 (side 1) is:
    # h(phi) = cos(phi - 30k) / cos(30) ? 
    # At phi=0 (vertex), h=1. cos(0)/cos(30) = 1/0.866 = 1.15. Wrong.
    # h(phi) = 1 / cos(phi - k*60) ?
    # At phi=0, h=1. At phi=30, h=1/cos(30)=1.15. But apothem is sqrt(3)/2 = 0.866.
    # Wait. Apothem is the distance to the edge. The support function in the normal direction is the apothem.
    # So h_inner(phi) = apothem / cos(delta) where delta is angle from normal.
    # Normals are at 30, 90, ...
    # Let delta = min |phi - (30 + k*60)|.
    # h_inner(phi) = (sqrt(3)/2) / cos(delta).
    
    # So for a given outer orientation theta, the normals are at n_k = theta + 30 + k*60.
    # We need S >= (2/sqrt(3)) * max_k [ max_i c_i . n_k + h_inner(n_k) ].
    # We can scan theta from 0 to 30 degrees (symmetry) to find the minimum S.
    
    # Let's implement this.
    
    def h_inner(phi):
        # phi in degrees
        # Reduce phi to range [-30, 30] relative to nearest normal (30, 90, ...)
        # Normal angles: 30, 90, 150, ...
        # Let's find the nearest multiple of 60, add 30.
        # Actually, just modulo 60.
        phi_mod = phi % 60
        # Distance from nearest normal (which is at 30 deg in the 0-60 range)
        delta = abs(phi_mod - 30)
        # h = (sqrt(3)/2) / cos(delta_rad)
        return (sqrt3 / 2) / np.cos(np.radians(delta))
    
    best_S = float('inf')
    best_theta = 0
    
    # Scan theta from 0 to 30 degrees
    for theta_deg in np.linspace(0, 30, 61):
        theta = np.radians(theta_deg)
        S_candidate = 0
        for k in range(6):
            # Normal direction
            n_angle = theta + np.radians(30 + k * 60)
            n_vec = np.array([np.cos(n_angle), np.sin(n_angle)])
            
            # Max projection of centers
            projs = centers @ n_vec
            max_proj = np.max(projs)
            
            # Support of inner hexagon in this direction
            hi = h_inner(np.degrees(n_angle))
            
            required_R_in = max_proj + hi
            # S = R_in / (sqrt(3)/2) = 2 * R_in / sqrt(3)
            S_k = 2 * required_R_in / sqrt3
            if S_k > S_candidate:
                S_candidate = S_k
        
        if S_candidate < best_S:
            best_S = S_candidate
            best_theta = theta_deg
            
    # Construct the result
    # Inner hexagons: centers, angle 0
    inner_hex_data = np.column_stack((centers, np.zeros(12)))
    
    # Outer hexagon: center (0,0), angle best_theta, side best_S
    outer_hex_data = np.array([0, 0, best_theta])
    
    return inner_hex_data, outer_hex_data, best_S


# EVOLVE-BLOCK-END
