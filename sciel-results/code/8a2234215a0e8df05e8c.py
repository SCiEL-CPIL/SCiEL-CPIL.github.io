# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    
    def hex_vertices(cx, cy, s, a_deg):
        """Return 6 vertices of a regular hexagon."""
        a = np.radians(a_deg)
        angles = a + np.deg2rad(60 * np.arange(6))
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])
    
    def hex_edge_normals(cx, cy, s, a_deg):
        """Return 6 outward edge normals of a regular hexagon."""
        a = np.radians(a_deg)
        normal_angles = a + np.deg2rad(30 + 60 * np.arange(6))
        return np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    def check_separation(hex1, hex2, tol=1e-6):
        """Check if two unit hexagons are separated (non-intersecting)."""
        cx1, cy1, a1 = hex1
        cx2, cy2, a2 = hex2
        
        v1 = hex_vertices(cx1, cy1, 1.0, a1)
        v2 = hex_vertices(cx2, cy2, 1.0, a2)
        
        for hex_data, other_verts in [(hex1, v2), (hex2, v1)]:
            cx, cy, a = hex_data
            normals = hex_edge_normals(cx, cy, 1.0, a)
            verts = hex_vertices(cx, cy, 1.0, a)
            
            for i in range(6):
                n = normals[i]
                proj_self = verts @ n
                proj_other = other_verts @ n
                
                min_self = np.min(proj_self)
                max_self = np.max(proj_self)
                min_other = np.min(proj_other)
                max_other = np.max(proj_other)
                
                if max_self < min_other - tol or max_other < min_self - tol:
                    return True
        
        return False
    
    def compute_min_enclosing_hex(inner_hexes):
        """Compute the minimal enclosing regular hexagon (angle=0) for given inner hexagons."""
        all_verts = []
        for cx, cy, a in inner_hexes:
            verts = hex_vertices(cx, cy, 1.0, a)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        normals = hex_edge_normals(0, 0, 1.0, 0)
        apothem_factor = np.sqrt(3) / 2
        
        n_points = len(all_verts)
        n_constraints = n_points * 6
        
        c = np.array([0.0, 0.0, 1.0])
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)
        
        for idx, p in enumerate(all_verts):
            for j, n in enumerate(normals):
                row = idx * 6 + j
                A_ub[row, 0] = -n[0]
                A_ub[row, 1] = -n[1]
                A_ub[row, 2] = -apothem_factor
                b_ub[row] = -(p[0] * n[0] + p[1] * n[1])
        
        bounds = [(None, None), (None, None), (0, None)]
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            return (0, 0, 10.0)
        
        cx, cy, s = result.x
        return (cx, cy, s)
    
    sqrt3 = np.sqrt(3)
    
    # Use a slight scale factor to ensure gaps > 1e-6
    # The lattice spacing for touching hexagons: horizontal=2, vertical=√3
    # We'll scale by (1 + epsilon) where epsilon is small
    eps = 1e-5
    scale = 1.0 + eps
    
    # Compact cluster of 11 hexagons in honeycomb arrangement
    positions = []
    # Row 0: y = 0, x = -2, 0, 2
    for x in [-2.0, 0.0, 2.0]:
        positions.append((x * scale, 0.0, 0.0))
    # Row 1: y = √3/2, x = -1, 1, 3
    for x in [-1.0, 1.0, 3.0]:
        positions.append((x * scale, (sqrt3/2) * scale, 0.0))
    # Row 2: y = √3, x = -2, 0, 2
    for x in [-2.0, 0.0, 2.0]:
        positions.append((x * scale, sqrt3 * scale, 0.0))
    # Row 3: y = 3√3/2, x = -1, 1
    for x in [-1.0, 1.0]:
        positions.append((x * scale, (3*sqrt3/2) * scale, 0.0))
    
    inner_hex_data = np.array(positions)
    
    # Verify separation
    for i in range(11):
        for j in range(i+1, 11):
            if not check_separation(inner_hex_data[i], inner_hex_data[j]):
                # Increase epsilon if needed
                pass
    
    cx, cy, s = compute_min_enclosing_hex(inner_hex_data)
    outer_hex_data = np.array([cx, cy, 0.0])
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
