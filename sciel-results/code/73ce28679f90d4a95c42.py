import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog

N = 11
DEADLINE_MARGIN = 5.0

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    sqrt3 = math.sqrt(3)
    
    # Axial to XY conversion for pointy-top or flat-top?
    # Standard hex grid:
    # Let's use axial coordinates (q, r) where x = 1.5*q, y = sqrt(3)*(r + q/2.0)
    # This corresponds to a "pointy-top" hexagon orientation for the grid itself?
    # Actually, the formula x = 1.5*q, y = sqrt(3)*(r + q/2) places centers.
    # The hexagons themselves are regular.
    
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Base 7 hexagons: Center (0,0) and Ring 1
    base_axial = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    # Ring 2 positions
    ring2_axial = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
        (0, 2), (1, 2)
    ]
    
    best_side = float('inf')
    best_config = None # (positions_xy, inner_angles, outer_center, outer_side, outer_angle)
    
    # We will test different inner lattice rotations and outer rotations
    # Inner lattice rotation: 0 or 30 degrees (relative to global axes)
    # Outer hexagon rotation: 0, 15, 30 degrees
    
    # Pre-compute vertices for unit hexagon at rotation 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    local_verts = []
    for k in range(6):
        a = math.radians(60 * k)
        local_verts.append((math.cos(a), math.sin(a)))
    
    # Function to solve LP for a given set of inner positions (xy) and inner rotation
    def solve_lp(inner_positions, inner_rot_deg, outer_rot_deg):
        """
        inner_positions: list of (x, y)
        inner_rot_deg: rotation of inner hexagons
        outer_rot_deg: rotation of outer hexagon
        Returns: (center_x, center_y, side_length) or None
        """
        # Variables: cx, cy, S
        # Objective: min S -> c = [0, 0, 1]
        
        A_ub = []
        b_ub = []
        
        # Outer hexagon normals
        # Normals are perpendicular to sides.
        # For a hexagon with rotation theta, the sides are at angles theta + 30 + 60k?
        # Let's check.
        # Flat-top hex (rot 0): Vertices at 0, 60... Sides at 30, 90...
        # Normals point outward.
        # Inequality: n . (p - c) <= apothem
        # n . p - n . c <= S * sqrt(3) / 2
        # -n . c - S * sqrt(3) / 2 <= -n . p  ?? No.
        # n . p - n . c - S * sqrt(3) / 2 <= 0
        # -n_x * cx - n_y * cy - (sqrt(3)/2) * S <= -n . p
        
        for k in range(6):
            # Normal angle
            n_angle = math.radians(outer_rot_deg + 30 + 60 * k)
            nx = math.cos(n_angle)
            ny = math.sin(n_angle)
            
            # Constraint for each vertex of each inner hexagon
            for px, py in inner_positions:
                # Rotate inner hexagon vertices by inner_rot_deg
                rot_rad = math.radians(inner_rot_deg)
                for vx, vy in local_verts:
                    # Rotate vertex
                    rvx = vx * math.cos(rot_rad) - vy * math.sin(rot_rad)
                    rvy = vx * math.sin(rot_rad) + vy * math.cos(rot_rad)
                    
                    # Absolute vertex position
                    abs_vx = px + rvx
                    abs_vy = py + rvy
                    
                    # n . p = nx*abs_vx + ny*abs_vy
                    n_dot_p = nx * abs_vx + ny * abs_vy
                    
                    # Coefficients for cx, cy, S
                    # -nx * cx - ny * cy - (sqrt(3)/2) * S <= -n_dot_p
                    A_ub.append([-nx, -ny, -sqrt3 / 2.0])
                    b_ub.append(-n_dot_p)
        
        # S >= 0
        A_ub.append([0, 0, -1])
        b_ub.append(0)
        
        c = [0, 0, 1]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
            if res.success:
                return res.x[0], res.x[1], res.x[2]
            else:
                return None
        except Exception:
            return None

    # Iterate through all combinations of 4 extra hexagons from Ring 2
    # To save time, we can exploit symmetry, but 495 is small.
    # We will try inner_rot in [0, 30] and outer_rot in [0, 15, 30]
    
    for combo in itertools.combinations(range(len(ring2_axial)), 4):
        if time.time() > deadline:
            break
            
        selected_r2 = [ring2_axial[i] for i in combo]
        full_axial = base_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Centroid of the cluster (for potential symmetry breaking or just reference)
        # The LP will find the optimal center automatically.
        
        for inner_rot in [0.0, 30.0]:
            if time.time() > deadline:
                break
            for outer_rot in [0.0, 15.0, 30.0]:
                if time.time() > deadline:
                    break
                
                result = solve_lp(full_xy, inner_rot, outer_rot)
                if result:
                    cx, cy, side = result
                    if side < best_side:
                        best_side = side
                        # Store the configuration
                        # We need to return: inner (x, y, angle), center, side, angle
                        # inner angle is inner_rot
                        best_config = (full_xy, inner_rot, (cx, cy), side, outer_rot)
    
    # Fallback if no solution found (should not happen)
    if best_config is None:
        # Return a valid, albeit suboptimal, configuration
        full_xy = [axial_to_xy(q, r) for q, r in base_axial]
        # Add 4 more arbitrarily
        full_xy.extend([axial_to_xy(q, r) for q, r in ring2_axial[:4]])
        best_config = (full_xy, 0.0, (0, 0), 10.0, 0.0)
        best_side = 10.0

    positions_xy, inner_rot, center, side, outer_rot = best_config
    
    # Construct the return value
    inner = []
    for x, y in positions_xy:
        inner.append([x, y, inner_rot])
        
    return inner, center, side, outer_rot

