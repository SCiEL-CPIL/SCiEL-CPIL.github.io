import numpy as np
from itertools import combinations

N = 11

def construct():
    # Lattice parameters
    # Triangular lattice with spacing sqrt(3)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # Generate central 7 points (q, r)
    # q^2 + qr + r^2 <= 1
    central_points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if q*q + q*r + r*r <= 1:
                central_points.append((q, r))
                
    # Generate next ring points (q^2 + qr + r^2 == 3)
    next_ring_points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if q*q + q*r + r*r == 3:
                next_ring_points.append((q, r))
                
    # Convert to Cartesian
    def to_cartesian(q, r):
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        return (x, y)
        
    central_cart = [to_cartesian(q, r) for q, r in central_points]
    next_ring_cart = [to_cartesian(q, r) for q, r in next_ring_points]
    
    # We need 4 more points from next_ring_cart
    # There are 6 points in next_ring_cart
    best_L = float('inf')
    best_config = None
    best_angle = 0.0
    
    # Function to calculate minimal L for a set of centers
    def calc_min_L(centers):
        # Get all vertices of the 11 hexagons
        # Hexagon side 1, angle 0
        # Vertices at angles 0, 60, ... 300
        # Distance from center to vertex is 1
        vertices = []
        for cx, cy in centers:
            for i in range(6):
                theta = np.deg2rad(60 * i)
                vx = cx + np.cos(theta)
                vy = cy + np.sin(theta)
                vertices.append((vx, vy))
        
        vertices = np.array(vertices)
        
        # Normals of the outer hexagon (angle 0)
        # Vertices at 0, 60...
        # Normals at 30, 90, 150, 210, 270, 330
        normals = []
        for i in range(6):
            theta = np.deg2rad(30 + 60 * i)
            normals.append((np.cos(theta), np.sin(theta)))
        normals = np.array(normals)
        
        # Minimize L over rotation phi
        # L(phi) = (2/sqrt(3)) * max( dot(rotate(p, -phi), normals) )
        # We scan phi from 0 to 60 degrees
        min_L_val = float('inf')
        best_phi = 0.0
        
        # Scan with step 0.1 degrees
        for phi_deg in np.arange(0, 60, 0.1):
            phi = np.deg2rad(phi_deg)
            # Rotation matrix for -phi
            # p' = R(-phi) p
            # dot(p', n) = dot(p, R(phi) n)
            # So we can rotate the normals by phi
            c = np.cos(phi)
            s = np.sin(phi)
            # Rotate normals
            nx = normals[:, 0] * c - normals[:, 1] * s
            ny = normals[:, 0] * s + normals[:, 1] * c
            rotated_normals = np.column_stack((nx, ny))
            
            # Calculate max dot product
            # vertices shape (M, 2), rotated_normals shape (6, 2)
            # We want max over k of max over p of dot(p, n_k)
            # This is max_k (max_p dot(p, n_k))
            # Which is max_k (rotated_normals[k] dot vertices.T).max()
            
            dots = vertices @ rotated_normals.T # Shape (M, 6)
            max_dots_per_normal = dots.max(axis=0) # Shape (6,)
            max_dot = max_dots_per_normal.max()
            
            L_val = (2.0 / np.sqrt(3)) * max_dot
            if L_val < min_L_val:
                min_L_val = L_val
                best_phi = phi_deg
                
        return min_L_val, best_phi

    # Iterate through combinations of 4 points from the 6 next-ring points
    # To speed up, we can just try all 15 combos
    for combo in combinations(range(len(next_ring_cart)), 4):
        extra_points = [next_ring_cart[i] for i in combo]
        all_centers = central_cart + extra_points
        
        L, phi = calc_min_L(all_centers)
        
        if L < best_L:
            best_L = L
            best_config = all_centers
            best_angle = phi

    # Format output
    inner = []
    for cx, cy in best_config:
        inner.append([cx, cy, 0.0])
        
    # Outer hexagon
    # Center (0,0)
    # Side best_L
    # Angle best_angle
    return inner, (0.0, 0.0), best_L, best_angle

# EVOLVE-BLOCK-END
