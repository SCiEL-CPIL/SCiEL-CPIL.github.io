import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    gap = 1.001  # Small gap to ensure strict disjointness
    
    # Hexagonal lattice positions for 11 hexagons
    # Center + 6 in first ring + 4 in second ring
    d = 2 * np.sqrt(3)  # Distance between adjacent hexagon centers
    
    # First ring: 6 hexagons at 60 degree intervals
    first_ring = []
    for i in range(6):
        angle = np.radians(60 * i)
        first_ring.append([d * np.cos(angle), d * np.sin(angle)])
    
    # Second ring: 12 possible positions at distance d*sqrt(3)
    d2 = d * np.sqrt(3)  # = 6
    second_ring_positions = []
    for i in range(12):
        angle = np.radians(30 + 30 * i)
        second_ring_positions.append([d2 * np.cos(angle), d2 * np.sin(angle)])
    
    # Try different subsets of 4 from the 12 second ring positions
    # To keep it efficient, try symmetric subsets
    
    best_side = float('inf')
    best_arrangement = None
    best_outer_angle = 0.0
    best_center = (0.0, 0.0)
    
    # The base arrangement: center + 6 first ring + 4 second ring
    # Try different choices of 4 from 12 second ring positions
    # For symmetry, try patterns like {0,3,6,9}, {1,4,7,10}, etc.
    
    from itertools import combinations
    
    # Precompute all points for the base arrangement (center + first ring)
    base_points = [(0.0, 0.0)] + first_ring
    
    # Try different symmetric subsets of 4 from 12
    # Symmetric choices: indices {i, i+3, i+6, i+9} for i in range(4)
    # or {i, i+2, i+4, i+6, i+8, i+10} subsets...
    
    # Let's try: for each starting offset, take every 3rd position
    candidate_subsets = []
    for offset in range(12):
        subset = [second_ring_positions[(offset + 3*j) % 12] for j in range(4)]
        candidate_subsets.append(subset)
    
    # Also try other patterns
    for offset in range(12):
        # Take positions at offset, offset+2, offset+6, offset+8
        subset = [second_ring_positions[(offset + j) % 12] for j in [0, 2, 6, 8]]
        candidate_subsets.append(subset)
    
    for second_ring_subset in candidate_subsets:
        if time.time() > deadline - 10:
            break
        
        all_points = base_points + second_ring_subset
        
        # Compute minimal enclosing hexagon
        # A regular hexagon can be represented by its center, side length, and rotation
        # For a given rotation, the side length needed is determined by the max extent
        
        # The 6 normal directions of a regular hexagon with rotation theta are at
        # angles theta + 60*k for k=0..5
        # The support function in direction n is max over points of dot(point, n)
        # For a hexagon centered at c with side s and rotation theta, the distance
        # from center to each side is s*sqrt(3)/2
        # So we need: for all 6 directions n_k, max_i dot(p_i - c, n_k) <= s*sqrt(3)/2
        # and min_i dot(p_i - c, n_k) >= -s*sqrt(3)/2
        # So s >= 2/sqrt(3) * max over k of max_i |dot(p_i - c, n_k)|
        
        # To minimize s, we can choose c to minimize the max extent
        # For a fixed rotation theta, the optimal c is the center of the minimal
        # enclosing circle... actually it's more complex.
        
        # Let's use a simpler approach: for a given rotation, compute the bounding
        # box in the 6 hexagonal directions, and the side length is determined by
        # the maximum half-width.
        
        # Try several rotations
        min_s_for_rotation = float('inf')
        best_c_for_rotation = (0.0, 0.0)
        
        for theta_deg in range(0, 60, 5):
            theta = np.radians(theta_deg)
            # 6 normal directions
            normals = []
            for k in range(6):
                angle = theta + np.radians(60 * k)
                normals.append([np.cos(angle), np.sin(angle)])
            
            # For each point, compute projection onto each normal
            # The hexagon must contain all points, so for each normal direction,
            # the max projection and min projection determine the extent
            max_proj = [0.0] * 6
            min_proj = [0.0] * 6
            
            for px, py in all_points:
                for k in range(6):
                    proj = px * normals[k][0] + py * normals[k][1]
                    max_proj[k] = max(max_proj[k], proj)
                    min_proj[k] = min(min_proj[k], proj)
            
            # The half-width in direction k is (max_proj[k] - min_proj[k]) / 2
            # But we can shift the center to minimize the max half-width
            # Optimal center: for each direction k, the center projection should be
            # (max_proj[k] + min_proj[k]) / 2 to minimize the half-width in that direction
            # But we need a single center that works for all directions.
            
            # Simple approach: use the centroid as center
            cx = sum(p[0] for p in all_points) / len(all_points)
            cy = sum(p[1] for p in all_points) / len(all_points)
            
            # Compute half-widths with this center
            half_widths = []
            for k in range(6):
                max_h = max((px - cx) * normals[k][0] + (py - cy) * normals[k][1] for px, py in all_points)
                min_h = min((px - cx) * normals[k][0] + (py - cy) * normals[k][1] for px, py in all_points)
                half_widths.append(max(abs(max_h), abs(min_h)))
            
            # The side length s satisfies: s * sqrt(3)/2 = max half_width
            # So s = 2/sqrt(3) * max(half_widths)
            s_candidate = (2.0 / np.sqrt(3)) * max(half_widths)
            
            if s_candidate < min_s_for_rotation:
                min_s_for_rotation = s_candidate
                best_c_for_rotation = (cx, cy)
        
        if min_s_for_rotation < best_side:
            best_side = min_s_for_rotation
            best_arrangement = all_points
            best_center = best_c_for_rotation
    
    # If no good arrangement found, use a simple linear arrangement
    if best_arrangement is None:
        # Linear chain
        best_arrangement = [(d * gap * i, 0.0) for i in range(-(N//2), N - N//2)]
        best_center = (0.0, 0.0)
        # Compute side length for linear chain
        # The chain extends from -5*d to 5*d in x, so length = 10*d
        # For a hexagon containing this, we need to find minimal enclosing hexagon
        # This is complex, so use a rough estimate
        best_side = 10 * d / np.sqrt(3) * 2 / 2  # rough estimate
    
    # Apply gap factor to all positions
    inner = []
    for px, py in best_arrangement:
        # Scale positions by gap factor to ensure disjointness
        inner.append([px * gap, py * gap, 0.0])
    
    # The side length also needs to account for the gap
    # Actually, the gap factor scales the positions, so the enclosing hexagon
    # should also be scaled
    side_length = best_side * gap
    
    return inner, best_center, side_length, 0.0
# EVOLVE-BLOCK-END
