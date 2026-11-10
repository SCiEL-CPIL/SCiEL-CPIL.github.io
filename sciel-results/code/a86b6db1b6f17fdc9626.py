# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float representing the side length of the outer hexagon
    """
    # Place 12 unit hexagons in a compact hexagonal arrangement
    # Using a 3-4-3 pattern (rows of 3, 4, 3 hexagons)
    # Hexagon spacing: distance between centers of touching hexagons = 2 * apothem = sqrt(3)
    # For flat-top hexagons, horizontal spacing = 1.5, vertical spacing = sqrt(3)
    
    # Let's use pointy-top hexagons (angle 90 degrees) for easier grid placement
    # Actually, let's use flat-top (angle 0) and arrange properly
    
    # For flat-top hexagons with side 1:
    # Width = 2, Height = sqrt(3)
    # Horizontal distance between centers in same row = 1.5 * 2 = 3? No...
    # Actually for flat-top: width = 2*s, height = sqrt(3)*s
    # Adjacent in row: dx = 1.5*s*2? Let me recalculate.
    
    # For a regular hexagon with side s, flat-top:
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # Width (horizontal) = 2s, Height (vertical) = sqrt(3)*s
    # Distance between centers of two adjacent flat-top hexagons sharing a vertical edge:
    # The shared edge is vertical, length s. The centers are separated by sqrt(3)*s horizontally.
    
    # Let me use a different approach: place hexagons on a triangular lattice
    # For hexagons of side 1, the center-to-center distance for touching hexagons is 2*apothem = sqrt(3)
    
    # Arrange in a hexagonal cluster:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    
    # For pointy-top hexagons (rotated 90 deg from flat-top):
    # Actually, let me just use flat-top and compute positions
    
    # Flat-top hexagon, side 1:
    # Apothem = sqrt(3)/2
    # Distance between centers of adjacent hexagons (sharing an edge) = 2 * apothem = sqrt(3)
    
    # For a row of hexagons sharing vertical edges:
    # Horizontal spacing = sqrt(3) (since they share a vertical edge of length 1,
    # and the centers are separated by the apothem on each side)
    
    # Wait, let me think more carefully. For flat-top hexagons:
    # The hexagon has vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # So width = 2, height = sqrt(3)
    # Two hexagons sharing the right edge of one and left edge of another:
    # The right edge goes from (0.5, sqrt(3)/2) to (0.5, -sqrt(3)/2)
    # The next hexagon's left edge is the same, so its center is at (1.5, 0) relative to first at (0,0)?
    # No: first center at (0,0), right edge at x=0.5. Second hexagon's left edge at x=0.5, 
    # its leftmost vertex at x=0.5-1=-0.5? That doesn't work.
    
    # Let me reconsider. For flat-top hexagon centered at origin:
    # Vertices: (1,0), (0.5, h), (-0.5, h), (-1,0), (-0.5, -h), (0.5, -h) where h = sqrt(3)/2
    # The rightmost point is at x=1, leftmost at x=-1
    # The right edge connects (0.5, h) and (0.5, -h) - this is a vertical edge at x=0.5
    # For two hexagons to share this edge, the second hexagon's left edge must be at x=0.5
    # Second hexagon's left edge is at its center_x - 0.5 (since left edge is at center_x - 0.5 for the vertical edge)
    # Wait no: left edge connects (-0.5, h) and (-0.5, -h) relative to its center, so at center_x - 0.5
    # So if first center at (0,0), right edge at x=0.5
    # Second center at (cx, 0), left edge at cx - 0.5
    # For sharing: cx - 0.5 = 0.5 => cx = 1.0
    # But then they overlap! The first hexagon extends to x=1, second starts at x=0.5... 
    # Actually they share the edge from x=0.5, so first occupies x in [-1, 1], second occupies x in [0.5, 1.5]
    # They share the region x in [0.5, 1]? No, they share only the edge at x=0.5.
    # First hexagon: leftmost x=-1, rightmost x=1
    # Second hexagon centered at (1,0): leftmost x=0, rightmost x=2
    # They share the edge at x=0.5? First's right edge at x=0.5, second's left edge at x=0.5 (since 1-0.5=0.5)
    # Yes! So horizontal spacing = 1.0 for flat-top hexagons sharing vertical edges.
    
    # But wait, that means they touch but the distance between centers is 1.0, not sqrt(3).
    # Let me verify: for flat-top hexagon side 1, the distance between centers of two hexagons
    # sharing a vertical edge: the shared edge is vertical, length 1. The centers are on a horizontal line.
    # The apothem (distance from center to midpoint of edge) = sqrt(3)/2 for the horizontal edges,
    # but for vertical edges, the distance from center to the vertical edge is 0.5 (half the width to the edge).
    # So center-to-center = 0.5 + 0.5 = 1.0. Yes.
    
    # For hexagons sharing a slanted edge (e.g., top-right edge):
    # The top-right edge connects (1,0) and (0.5, h). Midpoint at (0.75, h/2).
    # The direction from center to this midpoint is (0.75, h/2), length = sqrt(0.75^2 + (h/2)^2) = sqrt(9/16 + 3/16) = sqrt(12/16) = sqrt(3)/2.
    # So center-to-center for sharing a slanted edge = 2 * sqrt(3)/2 = sqrt(3).
    
    # So for flat-top hexagons:
    # - Sharing vertical edge: center distance = 1.0 (horizontal)
    # - Sharing slanted edge: center distance = sqrt(3) (at 60-degree angles)
    
    # For a hexagonal cluster, I'll use a triangular lattice arrangement.
    # Let me place them in rows where each row is offset.
    
    # Actually, the most compact arrangement for 12 hexagons is a hexagonal number pattern.
    # Hexagonal numbers: 1, 7, 19, 37... 12 is not a hexagonal number.
    # Closest: 7 (center + 6 around) or 19 (center + 6 + 12 around).
    # For 12, I'll use a 3-4-3 arrangement or similar.
    
    # Let me try a different approach: use optimization to find the best arrangement.
    # But first, let me try a simple compact arrangement.
    
    # Place 12 hexagons in a 3x4 grid-like pattern with hexagonal offsets
    # Using flat-top hexagons, angle = 0
    
    # Row 0 (y = 0): x = 0, 1, 2, 3 (4 hexagons)
    # Row 1 (y = h): x = 0.5, 1.5, 2.5 (3 hexagons) where h = sqrt(3)/2 * 2 = sqrt(3)? 
    # No, for flat-top, vertical spacing between rows that share slanted edges:
    # If row 0 at y=0, row 1 shares slanted edges, the y-offset is h = sqrt(3)/2 * ... 
    # The top vertex of row 0 hexagons is at y = h = sqrt(3)/2
    # The bottom vertex of row 1 hexagons should be at the same y to share the slanted edge.
    # Row 1 hexagon's bottom vertex at y = y1 - h
    # For sharing: y1 - h = h => y1 = 2h = sqrt(3)
    # And x-offset: row 1 hexagons are shifted by 0.5 in x
    
    # So:
    # Row 0 (y=0): x = 0, 1, 2, 3
    # Row 1 (y=sqrt(3)): x = 0.5, 1.5, 2.5
    # Row 2 (y=2*sqrt(3)): x = 0, 1, 2, 3
    
    # That's 4+3+4 = 11. Need one more.
    # Let me do 3-4-3:
    # Row 0 (y=0): x = 0.5, 1.5, 2.5 (3 hexagons)
    # Row 1 (y=sqrt(3)): x = 0, 1, 2, 3 (4 hexagons)
    # Row 2 (y=2*sqrt(3)): x = 0.5, 1.5, 2.5 (3 hexagons)
    
    sqrt3 = np.sqrt(3)
    inner_positions = []
    
    # Row 0: 3 hexagons at y=0
    for i in range(3):
        inner_positions.append([0.5 + i, 0, 0])
    
    # Row 1: 4 hexagons at y=sqrt(3)
    for i in range(4):
        inner_positions.append([i, sqrt3, 0])
    
    # Row 2: 3 hexagons at y=2*sqrt(3)
    for i in range(3):
        inner_positions.append([0.5 + i, 2*sqrt3, 0])
    
    inner_hex_data = np.array(inner_positions)
    
    # Now compute the minimal enclosing regular hexagon
    # The outer hexagon should be centered at the centroid of the inner hexagons
    # and have an orientation that minimizes its side length
    
    # Centroid of inner hexagons
    cx_mean = np.mean(inner_hex_data[:, 0])
    cy_mean = np.mean(inner_hex_data[:, 1])
    
    # For a regular hexagon with center (cx, cy), side s, angle a:
    # Vertices at (cx + s*cos(a + 60*i), cy + s*sin(a + 60*i)) for i=0..5
    # The hexagon contains point (px, py) if for all edges, the point is on the correct side.
    
    # To find minimal s for a given orientation a and center (cx, cy):
    # For each inner hexagon, compute the maximum "radius" in the 6 directions of the outer hexagon's edges.
    
    # The outer hexagon's edges have normals at angles a + 30 + 60*i (for i=0..5)
    # For a point to be inside, its projection onto each normal must be <= s * apothem_outer
    # where apothem_outer = s * sqrt(3)/2
    
    # Actually, for a regular hexagon with side s, the distance from center to each edge (apothem) = s * sqrt(3)/2
    # The hexagon is the intersection of 6 half-planes.
    
    # For a given orientation a, the 6 edge normals are at angles: a + 30 + 60*i for i=0..5
    # (since edges are perpendicular to these directions)
    
    # For each inner hexagon, I need to ensure all its vertices are inside the outer hexagon.
    # But it's sufficient to check the 6 vertices of each inner hexagon.
    
    # For a given outer hexagon (center, angle, side), check if all inner vertices are inside.
    # Then minimize side length.
    
    # Let me define a function to compute the minimum side length for given center and angle
    
    def min_outer_side(cx, cy, angle_deg):
        """Compute minimum side length of outer hexagon with given center and angle to contain all inner hexagons."""
        angle_rad = np.radians(angle_deg)
        
        # The 6 edge normals of the outer hexagon
        # For a regular hexagon with angle a (first vertex at angle a), 
        # the edges are between consecutive vertices.
        # Edge i connects vertex i and vertex i+1.
        # The normal to edge i points outward.
        
        # Vertices at angles: a, a+60, a+120, a+180, a+240, a+300
        # Edge 0: from vertex 0 to vertex 1. The outward normal is at angle a + 30 (bisector of the two vertex angles, pointing outward)
        # Actually, for a regular hexagon, the outward normal to edge i is at angle a + 30 + 60*i
        
        normals = []
        for i in range(6):
            normal_angle = angle_rad + np.pi/6 + i * np.pi/3  # a + 30 + 60*i
            normals.append(np.array([np.cos(normal_angle), np.sin(normal_angle)]))
        
        # For each inner hexagon, compute all 6 vertices
        max_projection = 0
        for ix, iy, iangle in inner_hex_data:
            iangle_rad = np.radians(iangle)
            # Vertices of inner hexagon
            for j in range(6):
                vx = ix + 1.0 * np.cos(iangle_rad + j * np.pi/3)
                vy = iy + 1.0 * np.sin(iangle_rad + j * np.pi/3)
                
                # Project onto each normal
                for normal in normals:
                    proj = (vx - cx) * normal[0] + (vy - cy) * normal[1]
                    if proj > max_projection:
                        max_projection = proj
        
        # The apothem of the outer hexagon must be at least max_projection
        # apothem = s * sqrt(3)/2
        # So s = max_projection / (sqrt(3)/2) = 2 * max_projection / sqrt(3)
        s = 2.0 * max_projection / sqrt3
        
        return max(s, np.sqrt(12))  # Ensure s >= sqrt(12)
    
    # Optimize over center and angle
    # Start with centroid and angle 0
    def objective(params):
        cx, cy, angle = params
        return min_outer_side(cx, cy, angle)
    
    # Initial guess
    x0 = [cx_mean, cy_mean, 0.0]
    
    # Use Nelder-Mead optimization
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    cx_opt, cy_opt, angle_opt = result.x
    s_opt = min_outer_side(cx_opt, cy_opt, angle_opt)
    
    # Ensure s is at least sqrt(12)
    s_opt = max(s_opt, np.sqrt(12))
    
    outer_hex_data = np.array([cx_opt, cy_opt, angle_opt])
    
    return inner_hex_data, outer_hex_data, s_opt


# EVOLVE-BLOCK-END
