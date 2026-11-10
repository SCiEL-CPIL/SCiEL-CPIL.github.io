# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _enclosing_hexagon_side(points, n_angles=360):
    """Find minimal side length of a regular hexagon (any rotation, centered at origin after shifting points to centroid) containing all points."""
    # We will center the outer hexagon at the centroid of points for simplicity,
    # but allow shifting? The task allows any outer hexagon position. We'll center at centroid.
    centroid = points.mean(axis=0)
    pts = points - centroid
    best_s = np.inf
    best_ang = 0.0
    for ang_deg in np.linspace(0, 60, n_angles, endpoint=False):
        a = np.deg2rad(ang_deg)
        # For a regular hexagon centered at origin with rotation a and side s,
        # the support function in direction theta is s * max over vertices of dot(vertex, dir)/s ... 
        # Actually, the distance from center to each edge (apothem) = s * sqrt(3)/2.
        # A point p is inside iff for all 6 edge normals n_k, dot(p, n_k) <= apothem.
        # Edge normals are at angles a + 60*i + 90? Let's define: vertices at angle a+60i, 
        # edge between vertex i and i+1 has outward normal at angle a + 60i + 30.
        normals = a + np.deg2rad(60 * np.arange(6) + 30)
        n_mat = np.column_stack([np.cos(normals), np.sin(normals)])  # (6,2)
        # max projection of pts onto each normal
        projections = pts @ n_mat.T  # (N,6)
        max_proj = projections.max(axis=0)  # (6,)
        # need apothem >= max(max_proj) => s*sqrt(3)/2 >= max_proj_max => s >= 2*max_proj_max/sqrt(3)
        required_apothem = max_proj.max()
        s = 2 * required_apothem / np.sqrt(3)
        if s < best_s:
            best_s = s
            best_ang = ang_deg
    return best_s, best_ang, centroid


def hexagon_packing_12():
    # Build a compact hexagonal cluster of 12 unit hexagons.
    # Hex grid coordinates: use axial (q, r). Center-to-center distance for edge-sharing = sqrt(3).
    # Directions in axial: (1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)
    # Convert axial to cartesian: x = sqrt(3)*(q + r/2), y = 3/2 * r  (for pointy-top? ) 
    # Let's use flat-top: x = 3/2*q, y = sqrt(3)*(r + q/2)
    # Actually for flat-top hexagons, neighbors are at angles 0,60,120,180,240,300 with distance sqrt(3).
    # Let me just place them manually in a known compact 12 arrangement.
    
    # Arrangement: two rows of 3 (top and bottom) offset, plus middle row of 6? 
    # Better: use the "hexagon of side 2" = 7 hexes, plus 5 more on the outer ring.
    # Hex of side 2 (in unit hexagons) has 1+6=7. Add 5 more adjacent to outer vertices.
    
    # Let me generate all hexes in a hexagonal region of radius 2 (side 2): that's 19, too many.
    # Radius 1 (side 1): 7 hexes. We need 12, so 7 + 5.
    
    # Place 7 in center (radius 1), then 5 more at specific outer positions.
    # Axial coords for radius 1: (0,0), (1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)
    # Add 5 more at radius 2 positions: (2,0),(2,-1),(0,-2),(-2,0),(-2,2) ... pick 5.
    
    axial = [
        (0,0),
        (1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1),
        (2,0),(2,-1),(0,-2),(-2,0),(-2,2),
    ]
    # That's 12. Let's use flat-top conversion: 
    # For flat-top: x = 1.5 * q, y = sqrt(3) * (r + 0.5*q)
    # Center-to-center distance between (0,0) and (1,0): x diff = 1.5, y diff = sqrt(3)*0.5 = 0.866. 
    # Distance = sqrt(1.5^2 + 0.866^2) = sqrt(2.25+0.75)=sqrt(3)=1.732. Good, matches sqrt(3).
    
    centers = []
    for q, r in axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * q)
        centers.append((x, y))
    centers = np.array(centers)
    
    # All inner hexagons have angle 0 (flat-top), side 1.
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = centers[:, 0]
    inner_hex_data[:, 1] = centers[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    # Collect all vertices of all inner hexagons
    all_verts = []
    for i in range(12):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, 1.0, 0.0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Find minimal enclosing regular hexagon
    s, ang, centroid = _enclosing_hexagon_side(all_verts, n_angles=720)
    
    # Refine with optimization
    def obj(params):
        cx, cy, a_deg = params
        a = np.deg2rad(a_deg)
        normals = a + np.deg2rad(60 * np.arange(6) + 30)
        n_mat = np.column_stack([np.cos(normals), np.sin(normals)])
        pts = all_verts - np.array([cx, cy])
        projections = pts @ n_mat.T
        max_proj = projections.max(axis=0)
        required_apothem = max_proj.max()
        s_val = 2 * required_apothem / np.sqrt(3)
        return s_val
    
    x0 = [centroid[0], centroid[1], ang]
    res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol':1e-10, 'fatol':1e-12})
    cx, cy, a_deg = res.x
    s = res.fun
    
    outer_hex_data = np.array([cx, cy, a_deg])
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
