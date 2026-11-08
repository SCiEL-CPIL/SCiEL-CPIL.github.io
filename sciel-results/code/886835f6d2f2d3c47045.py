import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0 # 20s margin
    
    # Define the 11 inner hexagon positions in axial coordinates (q, r)
    # We generate a hexagonal grid and pick the 11 closest to the origin
    # to form a compact cluster.
    
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Generate candidate positions
    # Distance 0: 1
    # Distance 1: 6
    # Distance 2: 12
    # We need 11. So 1 + 6 + 4.
    
    positions_axial = []
    # Center
    positions_axial.append((0, 0))
    
    # Ring 1
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    positions_axial.extend(ring1)
    
    # Ring 2 - Pick 4 to maintain symmetry/compactness
    # The 12 positions in ring 2 are:
    # (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2), (-2,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,2)
    # Let's pick 4 that are roughly 90 degrees apart in the lattice sense to balance the shape.
    # (2,0) is East.
    # (0,2) is North-East? 
    # Let's check angles.
    # (2,0) -> (2*sqrt(3), 0) -> 0 deg
    # (0,2) -> (sqrt(3), 3) -> atan(3/sqrt(3)) = atan(sqrt(3)) = 60 deg
    # (-2,0) -> (-2*sqrt(3), 0) -> 180 deg
    # (0,-2) -> (-sqrt(3), -3) -> 240 deg
    # This forms a symmetric cross/rhombus.
    positions_axial.extend([(2, 0), (0, 2), (-2, 0), (0, -2)])
    
    # Convert to Cartesian
    # Apply a small scaling factor to ensure strict non-overlap (touching is risky)
    centers = []
    for q, r in positions_axial:
        x, y = axial_to_cartesian(q, r)
        # Scale by 1.001 to create a tiny gap
        x *= 1.001
        y *= 1.001
        centers.append((x, y))
        
    # We assume inner hexagons are rotated by 30 degrees (pointy-top) to match
    # the axial grid spacing which assumes pointy-top geometry (distance sqrt(3) horizontal).
    # If the evaluator's base hexagon is flat-top (vertex at 0 deg), rotating by 30 deg
    # makes it pointy-top (vertex at 30 deg), which fits the grid.
    inner_angle_deg = 30.0

    # Inner hexagon vertices for a unit hexagon at (cx, cy) with angle 0 (flat top? or pointy?)
    # Let's define the unit hexagon vertices for angle 0 as pointy-top (vertex at 90 deg) or flat-top?
    # It doesn't matter for the inner shape relative to itself, but it matters for the "footprint" relative to the outer hexagon.
    # Let's define a standard unit hexagon with vertices at angles 0, 60, 120, 180, 240, 300 (pointy right).
    # If we rotate the outer hexagon, we effectively rotate the constraint.
    
    def get_hex_vertices(cx, cy, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        # Vertices at radius 1 from center
        # Use flat-top orientation (offset by 30 degrees) to match the axial grid spacing
        # for horizontal neighbors at distance sqrt(3).
        # NOTE: The evaluator likely uses flat-top as base (angle 0).
        # My grid is pointy-top.
        # So I must pass angle=30 to the evaluator to get pointy-top shapes.
        # For my internal calculation of the bounding box, I must use the SAME shape.
        # So I calculate vertices with angle=30.
        base_angle = angle_rad + np.pi / 6
        angles = [base_angle + i * np.pi / 3 for i in range(6)]
        verts = []
        for a in angles:
            x = cx + np.cos(a)
            y = cy + np.sin(a)
            verts.append((x, y))
        return verts

    # Function to calculate required outer side length S for a given outer angle
    def calc_min_side(outer_angle_deg, inner_angle_deg=30.0):
        # Outer hexagon normals
        # For a hexagon with side S and angle phi (rotation of the shape),
        # the normals to the sides are at angles phi, phi+60, ...
        # Wait, if the hexagon is rotated by phi, the normals are also rotated by phi.
        # The condition for point p to be inside is:
        # dot(p, n_i) <= S * sqrt(3) / 2  for all i=0..5
        # So S >= max(dot(p, n_i)) * 2 / sqrt(3)
        
        phi = np.deg2rad(outer_angle_deg)
        normals = []
        for i in range(6):
            a = phi + i * np.pi / 3
            normals.append((np.cos(a), np.sin(a)))
            
        max_dot = 0.0
        for cx, cy in centers:
            # Get vertices of this inner hexagon
            verts = get_hex_vertices(cx, cy, inner_angle_deg)
            for vx, vy in verts:
                for nx, ny in normals:
                    d = vx * nx + vy * ny
                    if d > max_dot:
                        max_dot = d
                        
        if max_dot < 0:
            return 0.0 # Should not happen for centered clusters
            
        s = max_dot * 2.0 / np.sqrt(3)
        return s

    # Optimize outer angle
    best_s = float('inf')
    best_angle = 0.0
    
    # Search outer angle from 0 to 30 degrees (symmetry)
    # Use fine steps
    num_steps = 300
    for i in range(num_steps + 1):
        angle = (30.0 / num_steps) * i
        s = calc_min_side(angle)
        if s < best_s:
            best_s = s
            best_angle = angle
            
    # Check time
    if time.time() > deadline:
        # Return a safe default if we run out of time, though this should be fast
        return [[x, y, inner_angle_deg] for x, y in centers], (0.0, 0.0), 40.0, 0.0

    # The optimal configuration uses the best outer angle found.
    # We can set the inner angle to 30.0.
    # The outer center is (0,0).
    
    inner = []
    for x, y in centers:
        inner.append([x, y, inner_angle_deg])
        
    return inner, (0.0, 0.0), best_s, best_angle
