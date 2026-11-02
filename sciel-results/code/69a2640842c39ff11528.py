"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Honeycomb cluster of 12 unit hexagons
    # Positions in honeycomb lattice (axial coordinates converted to Cartesian)
    # Each hexagon has side length 1, so center-to-center distance is sqrt(3)
    
    # Define 12 positions in a compact honeycomb arrangement
    # Use axial coordinates (q, r) for hex grid
    # Convert to Cartesian: x = sqrt(3) * (q + r/2), y = 1.5 * r
    # But for regular hexagons of side 1, the lattice spacing is:
    #   horizontal: sqrt(3)
    #   vertical offset: 1.5
    
    # A compact 12-hexagon cluster:
    # Row 0 (r=-2): q = -1, 0  -> 2 hexes
    # Row 1 (r=-1): q = -2, -1, 0, 1  -> 4 hexes  
    # Row 2 (r=0):  q = -1, 0, 1  -> 3 hexes
    # Row 3 (r=1):  q = -1, 0  -> 2 hexes
    # Row 4 (r=2):  q = 0  -> 1 hex
    # Total: 2+4+3+2+1 = 12
    
    axial_coords = [
        (-1, -2), (0, -2),
        (-2, -1), (-1, -1), (0, -1), (1, -1),
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1),
        (0, 2)
    ]
    
    # Convert axial to Cartesian for unit hexagons
    # For regular hexagons with flat-top orientation (angle=0):
    #   x = sqrt(3) * (q + r/2)
    #   y = 1.5 * r
    sqrt3 = np.sqrt(3)
    positions = []
    for q, r in axial_coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        positions.append((x, y))
    
    # Center the cluster
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    positions = [(x - cx, y - cy) for x, y in positions]
    
    # For each inner hexagon, compute the 6 vertices
    # A regular hexagon of side 1 centered at (px, py) with rotation 0 (flat-top):
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees from center
    # Vertex at angle theta: (px + cos(theta), py + sin(theta))
    
    def hex_vertices(px, py, angle_deg=0.0):
        angle_rad = np.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + np.radians(60 * i)
            vx = px + np.cos(theta)
            vy = py + np.sin(theta)
            verts.append((vx, vy))
        return verts
    
    # All vertices of all inner hexagons (all with angle 0 for now)
    all_vertices = []
    for px, py in positions:
        verts = hex_vertices(px, py, 0.0)
        all_vertices.extend(verts)
    
    all_vertices = np.array(all_vertices)
    
    # For an outer hexagon with center (0,0), side length s, rotation phi:
    # A point (x, y) is inside if for all 6 directions, the projection is <= s * sqrt(3)/2
    # The 6 normal directions for a flat-top hexagon rotated by phi are at angles:
    #   phi + 0, phi + 60, phi + 120, phi + 180, phi + 240, phi + 300
    # But actually, for a regular hexagon, the constraint is:
    #   |dot(point - center, normal_i)| <= s * sqrt(3)/2 for each of 6 normals
    # where normal_i are the outward normals to the 6 edges
    
    # For a flat-top hexagon (vertex at angle 0), the edge normals are at angles:
    #   30, 90, 150, 210, 270, 330 degrees (i.e., 30 + 60*i)
    # When rotated by phi, normals are at phi + 30 + 60*i
    
    def min_side_length(phi):
        """Given outer hexagon rotation phi (radians), find min side length to contain all vertices."""
        # The 6 edge normal directions
        normal_angles = np.radians(30) + phi + np.radians(60) * np.arange(6)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Project all vertices onto each normal
        # For each normal, we need max |dot(v, normal)| <= s * sqrt(3)/2
        # So s >= max_i |dot(v, normal_i)| / (sqrt(3)/2)
        
        max_proj = 0
        for normal in normals:
            projections = all_vertices @ normal
            max_abs = np.max(np.abs(projections))
            if max_abs > max_proj:
                max_proj = max_abs
        
        return max_proj / (np.sqrt(3) / 2)
    
    # Minimize over phi in [0, pi/3) due to hexagonal symmetry
    result = minimize_scalar(min_side_length, bounds=(0, np.pi / 3), method='bounded', 
                             options={'xatol': 1e-8})
    optimal_phi = result.x
    optimal_side = result.fun
    
    # Build the inner hexagon list
    inner = []
    for px, py in positions:
        inner.append([px, py, 0.0])  # angle in degrees
    
    # Outer hexagon centered at origin
    outer_center = (0.0, 0.0)
    outer_side = float(optimal_side)
    outer_angle = float(np.degrees(optimal_phi))
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
