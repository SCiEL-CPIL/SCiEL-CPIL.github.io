import numpy as np
import math

def construct():
    # Deadline guard
    import time
    start_time = time.time()
    deadline = start_time + 10.0 # 10s is plenty for this small calculation
    
    # Helper to calculate minimal enclosing hexagon side length
    def calc_side(coords):
        if not coords:
            return 0.0
        # coords: list of (x, y)
        # Normals for flat-top hexagon: 90, 30, 150 degrees
        # Wait, flat top means top is flat. Normal to top is 90.
        # Normal to top-right side is 30?
        # Let's check.
        # Vertices at 0, 60, 120, 180, 240, 300.
        # Sides are between vertices.
        # Side 1: 0 to 60. Midpoint angle 30. Normal is 30+90=120?
        # Let's use the property that width in direction theta is max dot - min dot.
        # For a regular hexagon of side S, width in normal direction is S*sqrt(3).
        # The 3 unique normal directions are 0, 60, 120?
        # No, the normals are perpendicular to the sides.
        # If vertices are at 0, 60..., sides are at 30, 90, 150...
        # So normals are at 30, 90, 150.
        angles = [math.radians(30), math.radians(90), math.radians(150)]
        max_width = 0.0
        for ang in angles:
            ux = math.cos(ang)
            uy = math.sin(ang)
            projections = [x*ux + y*uy for x, y in coords]
            w = max(projections) - min(projections)
            max_width = max(max_width, w)
        return max_width / math.sqrt(3)

    # Candidate 1: 19-hex cluster, remove 6 vertices and 2 edges
    # Generate 19 hexes
    coords_19 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q) <= 2 and abs(r) <= 2 and abs(q+r) <= 2:
                # Axial to Cartesian
                x = math.sqrt(3) * (q + r/2.0)
                y = 1.5 * r
                coords_19.append((x, y))
    
    # Identify vertices (max projection in normal dirs)
    # We want to remove points that contribute most to the width.
    # Simple heuristic: remove points with largest distance from origin?
    # All outer ring points have same distance.
    # Let's just remove the 6 "corners" of the hexagon.
    # Corners are at (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2) in axial?
    # Let's check axial coords of corners.
    # (2,0) -> x=sqrt(3)*2, y=0.
    # (0,2) -> x=sqrt(3)*1, y=3.
    # These are the 6 vertices.
    corners = [(2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)]
    
    # Create set of all 19 coords
    all_coords_set = set([(q,r) for q in range(-2,3) for r in range(-2,3) if abs(q)<=2 and abs(r)<=2 and abs(q+r)<=2])
    
    # Remove corners
    kept_axial = all_coords_set - set(corners)
    # We have 19 - 6 = 13 points.
    # We need to remove 2 more.
    # Remove the ones with next largest projection?
    # Or just arbitrary.
    # Let's remove (1,1) and (-1,-1)?
    # (1,1) is an edge hex.
    kept_axial.discard((1,1))
    kept_axial.discard((-1,-1))
    
    # Convert to Cartesian
    coords_c1 = []
    for q, r in kept_axial:
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        coords_c1.append((x, y))
    
    side_c1 = calc_side(coords_c1)
    
    # Candidate 2: 3x4 Parallelogram minus 1
    # q in [0,2], r in [0,3]
    coords_p = []
    for q in range(3):
        for r in range(4):
            x = math.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            coords_p.append((x, y))
    # Remove one, e.g., (2,3) which is a corner
    coords_p.pop() # Removes last one (2,3)
    
    side_c2 = calc_side(coords_p)
    
    # Choose best
    if side_c1 < side_c2:
        best_coords = coords_c1
        best_side = side_c1
    else:
        best_coords = coords_p
        best_side = side_c2
        
    # Center is centroid
    cx = sum(x for x,y in best_coords) / len(best_coords)
    cy = sum(y for x,y in best_coords) / len(best_coords)
    
    # Return inner hexes relative to center?
    # The interface expects inner rows (x, y, angle).
    # And outer center (x, y).
    # If I return inner coords as absolute, and outer center as (cx, cy),
    # the evaluator will place them.
    # I should return inner coords relative to (0,0) and set outer center to (0,0)?
    # Or return inner coords as calculated and outer center as centroid?
    # The score is the outer side length.
    # The position doesn't matter for the score, only the side length and validity.
    # Validity requires inner hexes to be inside outer.
    # My calc_side ensures they fit in a hex of that side centered at origin (if coords are centered).
    # My coords are not centered at origin.
    # I should center them.
    
    centered_coords = [(x - cx, y - cy) for x, y in best_coords]
    
    # Re-calc side to be sure
    final_side = calc_side(centered_coords)
    
    inner = [[x, y, 0.0] for x, y in centered_coords]
    center = (0.0, 0.0)
    side = final_side
    angle = 0.0
    
    return inner, center, side, angle
