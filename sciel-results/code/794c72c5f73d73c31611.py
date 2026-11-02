"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def hex_vertices(cx, cy, angle_deg, side=1.0):
        """Return 6 vertices of a regular hexagon with given center, angle, side length."""
        angle = np.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle + k * np.pi / 3
            verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
        return np.array(verts)
    
    def hex_inradius(side):
        return side * np.sqrt(3) / 2
    
    def point_in_hex(px, py, center_x, center_y, side, angle_deg):
        """Check if point (px,py) is inside hexagon with given params."""
        angle = np.radians(angle_deg)
        dx = px - center_x
        dy = py - center_y
        # Rotate point to hexagon's local frame
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        # Hexagon with side s has inradius r = s*sqrt(3)/2
        # It contains points where for all 6 edge normals, dot <= r
        r = hex_inradius(side)
        for k in range(6):
            theta = k * np.pi / 3
            nx = np.cos(theta)
            ny = np.sin(theta)
            if lx * nx + ly * ny > r + 1e-9:
                return False
        return True
    
    def min_outer_side(inner_positions, inner_angles, outer_center=(0.0, 0.0), outer_angle=0.0):
        """Find minimum outer hexagon side length containing all inner hexagons."""
        # Collect all vertices of all inner hexagons
        all_verts = []
        for i, (pos, ang) in enumerate(zip(inner_positions, inner_angles)):
            verts = hex_vertices(pos[0], pos[1], ang)
            all_verts.extend(verts.tolist())
        all_verts = np.array(all_verts)
        
        # For a given outer hexagon (center, angle), find min side that contains all points
        # The constraint: for each point p, for each of 6 edge normals n_k of outer hex,
        #   dot(p - center, n_k) <= side * sqrt(3)/2
        # So side >= max_k dot(p - center, n_k) * 2/sqrt(3) for all p
        # side >= max over all p and all k of dot(p - center, n_k) * 2/sqrt(3)
        
        ox, oy = outer_center
        angle = np.radians(outer_angle)
        
        max_proj = 0.0
        for k in range(6):
            theta = angle + k * np.pi / 3
            nx = np.cos(theta)
            ny = np.sin(theta)
            # Project all vertices onto this normal
            projs = (all_verts[:, 0] - ox) * nx + (all_verts[:, 1] - oy) * ny
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        
        side = max_proj * 2.0 / np.sqrt(3)
        return max(side, 1.0)  # At least side 1
    
    def evaluate_config(positions, angles, outer_center=(0.0, 0.0), outer_angle=0.0):
        """Evaluate a configuration, return outer side length."""
        return min_outer_side(positions, angles, outer_center, outer_angle)
    
    best_score = float('inf')
    best_result = None
    
    # Strategy: Try hexagonal lattice arrangements
    # Hex lattice directions: 60 degrees apart
    hex_dirs = []
    for k in range(6):
        angle = k * np.pi / 3
        hex_dirs.append((np.cos(angle), np.sin(angle)))
    
    # Generate candidate configurations
    # Configuration 1: Central hex + 6 surrounding + 4 in second ring
    # The 6 surrounding are at distance 2 in the 6 hex directions
    # For 11 total: 1 + 6 + 4
    
    # Try different outer rotations and center offsets
    for outer_angle in [0.0, 30.0, 60.0]:
        for cx in [0.0, 0.1, -0.1]:
            for cy in [0.0, 0.1, -0.1]:
                # Config: 1 center, 6 at distance 2, 4 more
                positions = [[0.0, 0.0]]
                angles = [0.0]
                
                # 6 surrounding
                for k in range(6):
                    positions.append([2.0 * hex_dirs[k][0], 2.0 * hex_dirs[k][1]])
                    angles.append(0.0)
                
                # 4 more: place at distance 4 in alternating directions
                # Or better: fill gaps in second ring
                # Second ring positions in hex lattice: distance 4 or 2*sqrt(3) etc.
                # Let's try: 4 hexagons at the "corner" positions of second ring
                # In hex lattice, second ring has 12 positions. Pick 4 that minimize bounding box.
                # Positions at angles 0, 60, 120, 180 at distance 4? That's 4.
                for k in [0, 1, 3, 4]:  # 4 positions
                    positions.append([4.0 * hex_dirs[k][0], 4.0 * hex_dirs[k][1]])
                    angles.append(0.0)
                
                score = evaluate_config(positions, angles, (cx, cy), outer_angle)
                if score < best_score:
                    best_score = score
                    best_result = (positions, angles, (cx, cy), outer_angle)
    
    # Also try a more compact arrangement
    # Hexagonal number: 1, 7, 19... For 11, it's 1+6+4
    # Try rotating the inner hexagons to pack tighter
    
    # Try with inner hexagons rotated by 30 degrees
    for outer_angle in [0.0, 30.0]:
        positions = [[0.0, 0.0]]
        angles = [30.0]
        for k in range(6):
            positions.append([2.0 * hex_dirs[k][0], 2.0 * hex_dirs[k][1]])
            angles.append(30.0)
        for k in [0, 1, 3, 4]:
            positions.append([4.0 * hex_dirs[k][0], 4.0 * hex_dirs[k][1]])
            angles.append(30.0)
        
        score = evaluate_config(positions, angles, (0.0, 0.0), outer_angle)
        if score < best_score:
            best_score = score
            best_result = (positions, angles, (0.0, 0.0), outer_angle)
    
    # Try a linear arrangement as fallback
    positions = [[2.0 * k, 0.0] for k in range(-5, 6)]
    angles = [0.0] * 11
    score = evaluate_config(positions, angles, (0.0, 0.0), 0.0)
    if score < best_score:
        best_score = score
        best_result = (positions, angles, (0.0, 0.0), 0.0)
    
    # Time check
    if time.time() > deadline - 5.0:
        pass  # Return what we have
    
    if best_result is None:
        # Fallback: simple linear
        positions = [[2.0 * k, 0.0] for k in range(-5, 6)]
        angles = [0.0] * 11
        return [[p[0], p[1], a] for p, a in zip(positions, angles)], (0.0, 0.0), 11.0, 0.0
    
    positions, angles, center, outer_angle = best_result
    inner = [[p[0], p[1], a] for p, a in zip(positions, angles)]
    return inner, center, best_score, outer_angle
# EVOLVE-BLOCK-END
