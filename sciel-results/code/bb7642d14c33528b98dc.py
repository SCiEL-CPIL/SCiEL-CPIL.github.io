"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Define the 11 unit hexagon centers in a hexagonal close-packed arrangement.
    # Coordinates are in the "axial" coordinate system scaled for pointy-top hexagons.
    # For a regular hexagon of side length 1:
    # Width (point to point) = 2
    # Height (flat to flat) = sqrt(3)
    # Horizontal distance between adjacent columns = sqrt(3)
    # Vertical distance between adjacent rows = 1.5
    
    # We arrange them in a "flower" pattern or a compact cluster.
    # A common compact shape for 11 is a central hexagon surrounded by 6, 
    # and then 4 more added to make a larger hexagon shape (side length 2 in hex-grid units? No, side 2 has 7. Side 3 has 19).
    # 11 is not a perfect hex number (1, 7, 19, 37...).
    # Let's use a 3-3-3-2 arrangement or similar.
    # Let's try to pack them as tightly as possible.
    
    # Arrangement:
    # Row 0 (y=0): x = -1.5*sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3), 1.5*sqrt(3)  (4 hexes)
    # Row 1 (y=1.5): x = -sqrt(3), 0, sqrt(3)                               (3 hexes)
    # Row 2 (y=3.0): x = -1.5*sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3), 1.5*sqrt(3) (4 hexes)
    # Total: 4+3+4 = 11.
    # This forms a parallelogram-like shape.
    
    # Let's try a more symmetric "hexagon-ish" shape.
    # Center at (0,0).
    # Ring 0: (0,0) -> 1
    # Ring 1: 6 hexes around it.
    # Total 7.
    # Need 4 more.
    # Add them to the right side to extend it?
    # Or add them symmetrically.
    
    # Let's stick to the 3-3-3-2 or 4-3-4 grid which is easy to bound.
    # Actually, the 4-3-4 arrangement above has:
    # Topmost y: 3.0 + 0.5 (top of hex) = 3.5
    # Bottommost y: 0.0 - 0.5 (bottom of hex) = -0.5
    # Leftmost x: -1.5*sqrt(3) - 0.5 (left of hex) = -1.5*1.732 - 0.5 = -3.098
    # Rightmost x: 1.5*sqrt(3) + 0.5 = 3.098
    
    # Let's compute the minimal containing regular hexagon side length S.
    # The outer hexagon is regular, centered at (0,0), rotated by angle theta.
    # We want to minimize S such that all inner hexagons are inside.
    
    # For a regular hexagon with side S, the distance from center to vertex is S.
    # The distance from center to edge (apothem) is S * sqrt(3) / 2.
    
    # The condition for a point P to be inside a regular hexagon (centered at origin, rotated by theta) is:
    # max over 6 directions of (P dot u_i) <= S * sqrt(3) / 2
    # where u_i are unit vectors pointing to the midpoints of the sides.
    # The angles of the side normals for a flat-top hexagon (theta=0) are 0, 60, 120, 180, 240, 300.
    # If we rotate the outer hexagon by theta, the normals are at theta, theta+60, etc.
    
    # We can optimize theta to minimize the required S.
    # S(theta) = (2 / sqrt(3)) * max_i max_p (p dot u_i(theta))
    
    # Let's generate the centers for the 4-3-4 arrangement.
    sqrt3 = math.sqrt(3)
    centers = []
    # Row 0 (y=0)
    for k in range(4):
        x = -1.5 * sqrt3 + k * sqrt3
        centers.append((x, 0.0))
    # Row 1 (y=1.5)
    for k in range(3):
        x = -sqrt3 + k * sqrt3
        centers.append((x, 1.5))
    # Row 2 (y=3.0)
    for k in range(4):
        x = -1.5 * sqrt3 + k * sqrt3
        centers.append((x, 3.0))
        
    # Re-center the cluster to (0,0) for symmetry in optimization?
    # The current cluster is not centered at (0,0).
    # x range: -2.598 to 2.598. Center x = 0.
    # y range: 0 to 3.0. Center y = 1.5.
    # So let's shift all centers by (0, -1.5).
    centers = [(x, y - 1.5) for x, y in centers]
    
    # Now centers are roughly symmetric around (0,0).
    # Let's verify bounds.
    # Top y: 3.0 - 1.5 = 1.5. Plus 0.5 (hex height) = 2.0.
    # Bottom y: 0.0 - 1.5 = -1.5. Minus 0.5 = -2.0.
    # Left x: -1.5*sqrt(3) - 0.5 = -3.098.
    # Right x: 1.5*sqrt(3) + 0.5 = 3.098.
    
    # We need to find theta and S.
    # S(theta) = (2/sqrt(3)) * M(theta)
    # where M(theta) = max over all vertices of all inner hexagons of (v dot u_i(theta))
    # Actually, it's easier to check the support function.
    # For a convex polygon P, h_P(u) = max_{p in P} p dot u.
    # The inner hexagons are convex. The union is not convex, but the condition "all inside" is equivalent to "each one inside".
    # So we need max_{h in hexes} max_{v in h} (v dot u_i) <= S * sqrt(3) / 2 for all i.
    
    # Let's precompute the vertices of each inner hexagon.
    # A unit hexagon centered at (cx, cy) with angle 0 (flat top? pointy top?).
    # Standard pointy-top hexagon vertices (angle 0):
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # Wait, the task says "unit regular hexagons". Side length 1.
    # The seed program used angle 0.
    # Let's assume standard orientation for inner hexagons (can be rotated, but let's start with 0).
    # The seed program returned angle 0 for inner hexes.
    
    # Let's define the 6 vertices of a unit hexagon at origin, pointy-top.
    # Angles: 0, 60, 120, 180, 240, 300 degrees.
    # v_k = (cos(k*pi/3), sin(k*pi/3))
    base_verts = [(math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)) for k in range(6)]
    
    all_verts = []
    for cx, cy in centers:
        for vx, vy in base_verts:
            all_verts.append((cx + vx, cy + vy))
            
    # Now, for a given outer rotation theta, the side normals are at angles theta + i*pi/3.
    # We need to find the maximum projection of all_verts onto these 6 directions.
    # Let R(theta) be the max projection.
    # S = 2 * R(theta) / sqrt(3).
    # We want to minimize S, so minimize R(theta).
    
    # R(theta) is a periodic function with period pi/3 (since the hexagon has 6-fold symmetry, but the set of inner hexes might break this).
    # Actually, the set of inner hexes has reflection symmetry across y-axis and x-axis?
    # The 4-3-4 cluster shifted to center:
    # x coords: -2.598, -1.732, -0.866, 0, 0.866, 1.732, 2.598 (with y offsets).
    # It is symmetric across x=0.
    # It is NOT symmetric across y=0 because the rows are at y=-1.5, 0, 1.5?
    # Row 0 (original y=0) -> shifted y=-1.5. 4 hexes.
    # Row 1 (original y=1.5) -> shifted y=0. 3 hexes.
    # Row 2 (original y=3.0) -> shifted y=1.5. 4 hexes.
    # So it is symmetric across y=0 as well!
    # The shape is symmetric across both axes.
    # Therefore, the optimal theta is likely 0 or pi/6 (aligned or anti-aligned).
    # Let's test theta = 0 and theta = pi/6.
    
    def calc_side_length(theta):
        max_proj = 0.0
        # 6 normal directions
        for i in range(6):
            angle = theta + i * math.pi / 3
            ux = math.cos(angle)
            uy = math.sin(angle)
            # Max projection of all vertices
            for vx, vy in all_verts:
                proj = vx * ux + vy * uy
                if proj > max_proj:
                    max_proj = proj
        return (2.0 / sqrt3) * max_proj

    # Try a few angles to be safe, though symmetry suggests 0 or pi/6.
    best_s = float('inf')
    best_theta = 0.0
    
    # Sample angles
    for t in range(0, 6):
        theta = t * math.pi / 18
        s = calc_side_length(theta)
        if s < best_s:
            best_s = s
            best_theta = theta
            
    # Refine around best_theta
    for _ in range(20):
        # Simple local search
        found_improvement = False
        for dt in [-0.01, 0.01, -0.001, 0.001]:
            new_theta = best_theta + dt
            s = calc_side_length(new_theta)
            if s < best_s:
                best_s = s
                best_theta = new_theta
                found_improvement = True
        if not found_improvement:
            break
        if time.time() > deadline:
            break

    # Construct the return values
    # inner: list of [x, y, angle_degrees]
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    # outer center: (0.0, 0.0)
    # outer side: best_s
    # outer angle: best_theta in degrees
    outer_angle_deg = math.degrees(best_theta)
    
    return inner, (0.0, 0.0), best_s, outer_angle_deg
# EVOLVE-BLOCK-END
