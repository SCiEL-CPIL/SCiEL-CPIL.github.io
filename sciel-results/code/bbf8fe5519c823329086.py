import numpy as np
import itertools
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Axial to Cartesian conversion for pointy-top hexagons
    # q, r -> x, y
    # x = 3/2 * q
    # y = sqrt(3)/2 * (q + 2r)
    # Distance between neighbors is sqrt(3)
    
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = math.sqrt(3) / 2 * (q + 2 * r)
        return x, y

    # Generate the 19 hexagons in the 3-ring cluster (1 + 6 + 12)
    # Center
    center_axial = (0, 0)
    
    # Ring 1
    ring1_axial = [
        (1, 0), (0, 1), (-1, 1),
        (-1, 0), (0, -1), (1, -1)
    ]
    
    # Ring 2
    # Neighbors of Ring 1
    ring2_axial_set = set()
    for q, r in ring1_axial:
        # 6 neighbors of (q,r)
        neighbors = [
            (q+1, r), (q+1, r-1), (q, r-1),
            (q-1, r), (q-1, r+1), (q, r+1)
        ]
        for nq, nr in neighbors:
            if (nq, nr) != (0,0) and (nq, nr) not in ring1_axial:
                ring2_axial_set.add((nq, nr))
                
    ring2_axial = list(ring2_axial_set)
    # There should be 12 hexagons in ring 2
    
    # Base cluster: Center + Ring 1 (7 hexagons)
    base_axial = [center_axial] + ring1_axial
    
    # We need to choose 5 from ring2_axial
    best_side = float('inf')
    best_cluster = None
    best_theta = 0.0
    
    # Pre-compute Cartesian coordinates for base and ring2
    base_cart = [axial_to_cartesian(q, r) for q, r in base_axial]
    ring2_cart = [axial_to_cartesian(q, r) for q, r in ring2_axial]
    
    # Indices of ring2
    # We will iterate over combinations of 5 indices from 0 to 11
    
    # To speed up, we can pre-compute the points for each combination?
    # No, just build the list each time. 792 * 12 is small.
    
    # Define the 6 normal vectors for the outer hexagon (fixed orientation for calculation)
    # Normals at 0, 60, 120, 180, 240, 300 degrees
    # Actually, we rotate the points, so normals are fixed.
    # Normal k: angle = k * 60 degrees
    # n_k = (cos(k*pi/3), sin(k*pi/3))
    
    normals = []
    for k in range(6):
        angle = k * math.pi / 3
        normals.append((math.cos(angle), math.sin(angle)))
        
    # Function to compute minimal apothem for a set of points at rotation theta
    def min_apothem(points, theta):
        # Rotate points by theta
        c = math.cos(theta)
        s = math.sin(theta)
        max_proj = 0.0
        for x, y in points:
            rx = x * c - y * s
            ry = x * s + y * c
            for nx, ny in normals:
                proj = abs(rx * nx + ry * ny)
                if proj > max_proj:
                    max_proj = proj
        return max_proj

    # Scan theta
    # The function is periodic with period 60 degrees (pi/3)
    # We can scan 0 to pi/3
    # Step size: 1 degree = pi/180 ~ 0.017 rad
    # 180 steps
    
    theta_steps = 180
    theta_range = math.pi / 3
    
    for comb in itertools.combinations(range(12), 5):
        # Build the cluster of 12 points
        # Base 7 + selected 5
        cluster_cart = base_cart.copy()
        for idx in comb:
            cluster_cart.append(ring2_cart[idx])
            
        # Find best theta
        min_apo = float('inf')
        best_th = 0.0
        for i in range(theta_steps + 1):
            theta = i * theta_range / theta_steps
            apo = min_apothem(cluster_cart, theta)
            if apo < min_apo:
                min_apo = apo
                best_th = theta
        
        # Side length S' = 2 * apothem / sqrt(3)
        side_prime = 2 * min_apo / math.sqrt(3)
        side_total = side_prime + 1.0
        
        if side_total < best_side:
            best_side = side_total
            best_cluster = cluster_cart
            best_theta = best_th
            
        # Time check
        if time.time() - start_time > 100:
            break

    # If no cluster found (should not happen), fallback
    if best_cluster is None:
        # Fallback to seed-like behavior but better
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    # Construct the return value
    # Outer hexagon center: (0,0)
    # Outer hexagon side: best_side
    # Outer hexagon angle: 0.0 (since we rotated the cluster relative to fixed normals)
    
    # Inner hexagons:
    # Positions are best_cluster rotated by best_theta
    # Angles are 0.0 (aligned with outer)
    
    c = math.cos(best_theta)
    s = math.sin(best_theta)
    
    inner = []
    for x, y in best_cluster:
        rx = x * c - y * s
        ry = x * s + y * c
        inner.append([rx, ry, 0.0])
        
    center = [0.0, 0.0]
    side = best_side
    angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
