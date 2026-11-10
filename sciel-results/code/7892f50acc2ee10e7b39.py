# EVOLVE-BLOCK-START
import numpy as np
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin
    
    n = 12
    
    # Use a compact hexagonal lattice arrangement of 12 unit hexagons
    # This is a 3-2-3-2-2 pattern or similar compact arrangement
    # Let's use a known valid configuration: center + 6 around + 5 more
    # Actually, let's use a rectangular-ish grid that we know works
    
    # Compact arrangement: 4 columns
    # Row 0 (top):    4 hexagons
    # Row 1:          3 hexagons (offset)
    # Row 2:          3 hexagons (offset)
    # Row 3 (bottom): 2 hexagons
    # Total: 4+3+3+2 = 12
    
    # For unit regular hexagons (side 1), the horizontal distance between adjacent columns is sqrt(3)
    # The vertical distance between rows is 1.5
    # Offset rows shift by sqrt(3)/2 horizontally
    
    sqrt3 = np.sqrt(3)
    col_spacing = sqrt3
    row_spacing = 1.5
    offset = sqrt3 / 2
    
    # Define positions for 12 hexagons in a compact grid
    # Using a 3-4-3-2 pattern or similar
    positions = []
    
    # Let me use a simpler approach: 3 rows of 4, but staggered
    # Row 0: 4 hexagons at x = 0, sqrt3, 2*sqrt3, 3*sqrt3
    # Row 1: 4 hexagons at x = sqrt3/2, 3*sqrt3/2, 5*sqrt3/2, 7*sqrt3/2
    # Row 2: 4 hexagons at x = 0, sqrt3, 2*sqrt3, 3*sqrt3
    # That's 12 total
    
    # Actually, let me use a known valid packing: the optimal 12-hexagon packing
    # For simplicity, use a 3x4 grid with proper staggering
    
    # Row 0 (y=0): 4 hexagons
    for i in range(4):
        positions.append([i * sqrt3, 0, 0])
    
    # Row 1 (y=-1.5): 4 hexagons, offset by sqrt3/2
    for i in range(4):
        positions.append([i * sqrt3 + sqrt3/2, -1.5, 0])
    
    # Row 2 (y=-3.0): 4 hexagons
    for i in range(4):
        positions.append([i * sqrt3, -3.0, 0])
    
    inner_hex_data = np.array(positions)
    
    # Center the arrangement
    cx_mean = np.mean(inner_hex_data[:, 0])
    cy_mean = np.mean(inner_hex_data[:, 1])
    inner_hex_data[:, 0] -= cx_mean
    inner_hex_data[:, 1] -= cy_mean
    
    def compute_vertices(cx, cy, d, angle_deg):
        """Compute vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
        angles = np.radians(angle_deg) + np.radians(60 * np.arange(6))
        return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])
    
    def find_min_outer_hex(inner_params, outer_angle_deg):
        """Find minimum side length s for outer hexagon at given angle containing all inner hexagons."""
        # For a regular hexagon with side s centered at origin, the inradius is s * sqrt(3)/2
        # But we need to check all vertices of inner hexagons
        
        # The constraint: for each outer edge, all inner vertices must be on the correct side
        # For a regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300 (offset by outer_angle)
        # The edges are between consecutive vertices
        
        # For a given outer angle, the normal directions to the edges are at angles:
        # outer_angle + 30, outer_angle + 90, outer_angle + 150, outer_angle + 210, outer_angle + 270, outer_angle + 330
        # The distance from center to each edge (inradius) is s * cos(30) = s * sqrt(3)/2
        
        # For each inner vertex v, the constraint is:
        # dot(v, normal_i) <= s * sqrt(3)/2 for all 6 edge normals
        
        # So s >= max_i (dot(v, normal_i) / (sqrt(3)/2)) for all vertices v and all 6 normals
        
        outer_angle_rad = np.radians(outer_angle_deg)
        # Edge normal directions (perpendicular to edges, pointing outward)
        # For a regular hexagon, edge normals are at angles: outer_angle + 30 + 60*k
        normal_angles = outer_angle_rad + np.radians(30) + np.radians(60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Compute all inner vertices
        all_vertices = []
        for i in range(len(inner_params)):
            cx, cy, angle = inner_params[i]
            v = compute_vertices(cx, cy, 1.0, angle)
            all_vertices.append(v)
        
        all_vertices = np.vstack(all_vertices)
        
        # For each normal, find the max projection
        min_s = 0
        for normal in normals:
            projections = all_vertices @ normal
            max_proj = np.max(projections)
            # s * sqrt(3)/2 >= max_proj => s >= max_proj * 2/sqrt(3)
            required_s = max_proj * 2 / sqrt3
            min_s = max(min_s, required_s)
        
        # Also need to ensure s is at least the minimum valid value
        min_s = max(min_s, np.sqrt(12))
        
        return min_s
    
    # Try different outer hexagon angles to find the best fit
    best_s = float('inf')
    best_outer_angle = 0
    
    for oa in range(0, 60, 1):  # Try angles from 0 to 59 degrees
        s = find_min_outer_hex(inner_hex_data, oa)
        if s < best_s:
            best_s = s
            best_outer_angle = oa
    
    # Set outer hexagon centered at origin with best angle
    outer_hex_data = np.array([0.0, 0.0, float(best_outer_angle)])
    outer_hex_side_length = float(best_s)
    
    # Ensure angles are in [0, 360)
    inner_hex_data[:, 2] = inner_hex_data[:, 2] % 360
    outer_hex_data[2] = outer_hex_data[2] % 360
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
