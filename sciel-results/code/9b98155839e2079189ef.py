import numpy as np
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Lattice vectors for flat-top hexagons with side 1
    # Distance between centers is sqrt(3)
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 1.5)
    # Let's use a grid of indices (i, j)
    # x = sqrt(3) * (i + j/2)
    # y = 1.5 * j
    
    # We will search for a compact cluster of 11 points.
    # We generate points in a hexagonal grid of radius 2 (which has 19 points)
    # and select 11 that are "connected" and compact.
    # To keep it simple and fast, we will try a few specific patterns.
    
    sqrt3 = math.sqrt(3)
    
    # Pattern 1: A "bar" of 11 is bad.
    # Pattern 2: Compact hexagonal-ish.
    # Let's generate all subsets of size 11 from a 3x3x3 hex grid? 
    # That's C(19, 11) which is large but manageable if we filter by connectivity?
    # Actually, C(19,11) = 75582. For each, we check distance constraints and compute S.
    # 75k * 11 * 6 * 30 (angles) = ~150 million ops. Might be too slow in Python.
    
    # Better: Hardcode a few good candidates or use a greedy heuristic.
    # Let's try to build a cluster greedily starting from center.
    
    def get_points(i, j):
        x = sqrt3 * (i + j/2.0)
        y = 1.5 * j
        return (x, y)
    
    # Generate candidate indices in a hexagon of radius 2
    candidates = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            # Check if (i,j) is in hexagon of radius 2
            # Distance in hex grid is max(|i|, |j|, |i+j|) ? No, that's for axial coords with 60 deg.
            # Our coords are offset.
            # Let's just generate a square grid and filter by distance to origin < some bound.
            p = get_points(i, j)
            dist = math.hypot(p[0], p[1])
            if dist < 5.0: # Radius 2 hexagon has max dist ~ 3.46
                candidates.append((i, j, p))
                
    # We need to pick 11 points from candidates such that pairwise distance >= sqrt(3) - epsilon.
    # And they should be compact.
    
    # Let's try a specific known good packing for N=11.
    # A 2-3-3-3 arrangement?
    # Let's manually define a compact set of indices.
    # Center (0,0).
    # Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 7 points.
    # Add 4 more:
    # (2,0), (1,1), (0,2), (-1,2)?
    # Let's check distances.
    
    # Instead of guessing, let's run a small local search on a predefined "seed" cluster.
    # Seed: The 7-hex cluster.
    # Indices:
    idx_7 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # We want to add 4 more.
    # Available neighbors of the cluster:
    # (2,0), (2,-1), (1,2), (0,2), (-1,2), (-2,1), (-2,0), (-2,-1), (-1,-2), (0,-2), (1,-2), (2,-2)
    # Let's try to add them in a way that minimizes the bounding box.
    
    # We will try all combinations of 4 from the 12 outer neighbors?
    # C(12,4) = 495. This is very fast.
    
    outer_neighbors = [
        (2,0), (2,-1), (1,2), (0,2), (-1,2), (-2,1),
        (-2,0), (-2,-1), (-1,-2), (0,-2), (1,-2), (2,-2)
    ]
    
    import itertools
    
    best_S = float('inf')
    best_config = None
    best_theta = 0.0
    
    # Function to calculate minimal S for a set of points and angle
    def calc_S(points, theta_rad):
        # points: list of (x,y) centered at origin
        # theta_rad: rotation of outer hexagon
        # Normals of outer hexagon
        # Outer hexagon vertices are at angles theta + k*60
        # Normals are at angles theta + 30 + k*60
        normals = []
        for k in range(6):
            angle = theta_rad + math.pi/6 + k * math.pi/3
            normals.append((math.cos(angle), math.sin(angle)))
        
        max_proj = 0.0
        for p in points:
            for n in normals:
                proj = abs(p[0]*n[0] + p[1]*n[1])
                if proj > max_proj:
                    max_proj = proj
        # S * sqrt(3)/2 >= max_proj
        return (2.0 / sqrt3) * max_proj

    # Iterate over all 4-subsets of outer neighbors
    for combo in itertools.combinations(outer_neighbors, 4):
        # Combine with base 7
        all_indices = idx_7 + list(combo)
        
        # Check for overlaps (distance < sqrt(3))
        valid = True
        pts = [get_points(i, j) for i, j in all_indices]
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                if dx*dx + dy*dy < (sqrt3*sqrt3) - 1e-9: # sqrt(3)^2 = 3
                    valid = False
                    break
            if not valid: break
            
        if not valid: continue
        
        # Center the points
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        centered_pts = [(p[0]-cx, p[1]-cy) for p in pts]
        
        # Find optimal theta
        min_S_for_config = float('inf')
        best_theta_for_config = 0.0
        # Search theta from 0 to 30 degrees (pi/6)
        for t in range(0, 16): # 0 to 15 deg, step 1 deg
            theta_rad = math.radians(t)
            S = calc_S(centered_pts, theta_rad)
            if S < min_S_for_config:
                min_S_for_config = S
                best_theta_for_config = theta_rad
        
        if min_S_for_config < best_S:
            best_S = min_S_for_config
            best_config = centered_pts
            best_theta = best_theta_for_config
            
    # If no valid config found (shouldn't happen), fallback
    if best_config is None:
        # Fallback to simple line
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Format output
    # inner: list of [x, y, angle_degrees]
    # All inner hexagons are aligned (angle 0)
    inner = [[p[0], p[1], 0.0] for p in best_config]
    
    # Outer center is (0,0)
    # Side length is best_S
    # Angle is best_theta in degrees
    outer_angle_deg = math.degrees(best_theta)
    
    return inner, (0.0, 0.0), best_S, outer_angle_deg
# EVOLVE-BLOCK-END
