import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # 1. Generate hexagonal grid coordinates (q, r)
    # We need N cells.
    # Generate a sufficient number of cells around the origin.
    # Max radius R in grid coords.
    # Area ~ R^2. N ~ 3R^2. So R ~ sqrt(N/3).
    # Add margin.
    R = int(np.ceil(np.sqrt(N / 3.0))) + 2
    
    coords = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            # Hex grid constraint: |q+r| <= R is not strictly needed if we just take closest,
            # but we can just generate a square block and filter by distance.
            # Distance in grid units (approx): sqrt(3*q^2 + 3*q*r + r^2) is not quite right for Euclidean.
            # Let's just compute Euclidean distance of the Cartesian position.
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            dist = np.sqrt(x*x + y*y)
            coords.append((dist, q, r, x, y))
            
    # Sort by distance from origin
    coords.sort(key=lambda x: x[0])
    
    # Take the N closest
    selected = coords[:N]
    
    # Extract positions
    positions = np.array([(c[3], c[4]) for c in selected])
    
    # 2. Center the cluster at the origin
    centroid = np.mean(positions, axis=0)
    positions = positions - centroid
    
    # 3. Calculate minimal outer side length S
    # Inner hexagons have side 1, angle 0.
    # Vertices of an inner hexagon at (x, y) with angle 0:
    # (x + cos(k*pi/3), y + sin(k*pi/3)) for k=0..5
    
    # Outer hexagon: center (0,0), side S, angle 0.
    # Condition for point (px, py) to be inside:
    # max_k (px * cos(theta_k) + py * sin(theta_k)) <= S * sqrt(3)/2
    # where theta_k = pi/6 + k*pi/3
    
    thetas = [np.pi/6 + k * np.pi/3 for k in range(6)]
    cos_thetas = [np.cos(t) for t in thetas]
    sin_thetas = [np.sin(t) for t in thetas]
    
    max_S = 0.0
    
    for i in range(N):
        px, py = positions[i]
        # Vertices of inner hexagon i
        for j in range(6):
            angle = j * np.pi / 3.0
            vx = px + np.cos(angle)
            vy = py + np.sin(angle)
            
            # Calculate required S for this vertex
            # S_req = max_k (2/sqrt(3) * (vx*cos_k + vy*sin_k))
            proj = [vx * ck + vy * sk for ck, sk in zip(cos_thetas, sin_thetas)]
            s_req = (2.0 / np.sqrt(3.0)) * np.max(proj)
            
            if s_req > max_S:
                max_S = s_req
                
    # Safety margin / time check
    elapsed = time.time() - start_time
    if elapsed > 110.0:
        # Return best found so far (which is the current one)
        pass
        
    # Return: inner list, center, side, angle
    # inner: list of [x, y, angle_degrees]
    inner = [[float(p[0]), float(p[1]), 0.0] for p in positions]
    center = (0.0, 0.0)
    side = float(max_S)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
