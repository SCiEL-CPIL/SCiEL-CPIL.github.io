"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side length, rotation."""
    theta = np.radians(angle_deg)
    angles = theta + np.pi/6 + np.arange(6) * np.pi/3  # flat-top: vertices at 30,90,150,...
    # Actually for flat-top, vertices are at 0, 60, 120, 180, 240, 300 degrees
    angles = theta + np.arange(6) * np.pi/3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _hex_inradius(side, angle_deg):
    """Inradius (apothem) of hexagon - distance from center to midpoint of edge."""
    return side * np.sqrt(3) / 2

def _point_in_hexagon(px, py, hx, hy, hside, hangle_deg):
    """Check if point (px,py) is inside hexagon centered at (hx,hy) with side hside, rotated hangle_deg."""
    # Transform to hexagon's local frame
    theta = np.radians(hangle_deg)
    dx = px - hx
    dy = py - hy
    # Rotate by -theta
    cx = np.cos(-theta)
    sx = np.sin(-theta)
    lx = cx * dx - sx * dy
    ly = sx * dx + cx * dy
    # In local frame (flat-top), hexagon has vertices at angles 0,60,120,180,240,300
    # The hexagon is defined by |x| <= side, |y| <= side*sqrt(3)/2, and |x|/side + |y|/(side*sqrt(3)/2) <= 1 for the "corner" regions
    # More precisely: for a flat-top hexagon with side s:
    # -x <= x <= s, and -s*sqrt(3)/2 <= y <= s*sqrt(3)/2, with the constraint that
    # for |x| > s/2: |y| <= s*sqrt(3)/2 * (1 - |x|/s) ... no
    # Let me use the proper inequalities. A regular hexagon (flat-top, side s) centered at origin:
    # The six edges are at distances s*sqrt(3)/2 from center, at angles 30, 90, 150, 210, 270, 330
    # Edge normals at angles: 30, 90, 150, 210, 270, 330 degrees
    # Inequality: dot(p, n_i) <= s*sqrt(3)/2 for all 6 edges
    edge_angles = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
    dists = normals @ np.array([lx, ly])
    max_dist = hside * np.sqrt(3) / 2
    return np.all(dists <= max_dist + 1e-9)

def _max_vertex_distance(cx, cy, side, angle_deg):
    """For a hexagon, the farthest any vertex can be from center is 'side' (circumradius)."""
    return side

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    
    # Initial configuration: 12 hexagons in a hexagonal lattice pattern
    # 4 columns x 3 rows of flat-top hexagons, side=1, angle=0
    # Hex lattice: col c, row r
    # x = c * 1.5, y = r * sqrt(3) + (sqrt(3)/2 if c odd else 0)
    
    sqrt3 = np.sqrt(3)
    inner = []
    cols = 4
    rows = 3
    # Use a scale factor > 1.0 to ensure disjointness (avoid touching/overlap due to float precision or strict evaluator)
    scale = 1.05
    for c in range(cols):
        for r in range(rows):
            x = c * 1.5 * scale
            y = (r * sqrt3 + (sqrt3/2 if c % 2 == 1 else 0)) * scale
            inner.append([x, y, 0.0])
    
    # Center the configuration
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    cx0 = (min(xs) + max(xs)) / 2
    cy0 = (min(ys) + max(ys)) / 2
    for i in range(len(inner)):
        inner[i][0] -= cx0
        inner[i][1] -= cy0
    
    # Outer hexagon parameters: center (ox, oy), side s, angle
    # We'll optimize these
    # Initial guess
    ox, oy, s, oangle = 0.0, 0.0, 4.0, 0.0
    
    def get_all_inner_vertices(inner_list):
        """Get all 72 vertices (12 hexagons x 6 vertices)."""
        verts = []
        for i, (px, py, pangle) in enumerate(inner_list):
            v = _hex_vertices(px, py, 1.0, pangle)
            verts.append(v)
        return np.vstack(verts)
    
    def cost(params):
        """Cost = outer side length, with penalty if any inner vertex is outside."""
        nonlocal deadline
        if time.time() > deadline:
            return 1e10
        ox_p, oy_p, s_p, oangle_p = params
        if s_p < 0.5:
            return 1e10
        # Get all inner vertices
        all_v = get_all_inner_vertices(inner)
        # Check if all vertices are inside the outer hexagon
        # Outer hexagon: center (ox_p, oy_p), side s_p, angle oangle_p
        theta = np.radians(oangle_p)
        dx = all_v[:, 0] - ox_p
        dy = all_v[:, 1] - oy_p
        cx_ = np.cos(-theta)
        sx_ = np.sin(-theta)
        lx = cx_ * dx - sx_ * dy
        ly = sx_ * dx + cx_ * dy
        edge_angles = np.radians(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
        dists = np.column_stack([lx, ly]) @ normals.T  # shape (72, 6)
        max_dist = s_p * np.sqrt(3) / 2
        violations = np.max(np.maximum(dists - max_dist, 0))
        # Also check inner hexagons don't overlap (optional, but required)
        # For now, just handle containment
        return s_p + 100 * violations
    
    # Optimize outer hexagon parameters
    x0 = np.array([ox, oy, s, oangle])
    result = minimize(cost, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
    
    if time.time() > deadline:
        # Return initial estimate
        return inner, (ox, oy), s, oangle
    
    ox, oy, s, oangle = result.x
    s = max(s, 0.5)
    
    # Verify and if violations, increase s slightly
    all_v = get_all_inner_vertices(inner)
    theta = np.radians(oangle)
    dx = all_v[:, 0] - ox
    dy = all_v[:, 1] - oy
    cx_ = np.cos(-theta)
    sx_ = np.sin(-theta)
    lx = cx_ * dx - sx_ * dy
    ly = sx_ * dx + cx_ * dy
    edge_angles = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
    dists = np.column_stack([lx, ly]) @ normals.T
    max_dist = s * np.sqrt(3) / 2
    max_viol = np.max(dists) - max_dist
    if max_viol > 1e-6:
        s += max_viol / (np.sqrt(3) / 2) + 0.05
    # Add a small buffer to ensure strict containment
    s += 0.01
    
    return inner, (ox, oy), s, oangle
# EVOLVE-BLOCK-END
