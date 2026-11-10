# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 580.0  # safety margin before the 630s wall-clock kill

# ---------- geometry helpers ----------

def _hex_vertices(cx, cy, s, a_deg):
    """(6,2) vertex array of a regular hexagon (center, side, angle_deg)."""
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])

def _edge_normals(vertices):
    """(6,2) outward-pointing unit edge normals of a convex polygon."""
    n = len(vertices)
    centroid = vertices.mean(axis=0)
    norms = np.empty((n, 2))
    for i in range(n):
        edge = vertices[(i + 1) % n] - vertices[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-15:
            normal = np.array([1.0, 0.0])
        else:
            normal /= norm
        mid = 0.5 * (vertices[i] + vertices[(i + 1) % n])
        if np.dot(normal, mid - centroid) < 0.0:
            normal = -normal
        norms[i] = normal
    return norms

def _min_gap(v1, v2, n1, n2):
    """Minimum separation of two convex hexagons via SAT (12 edge normals).
    Positive => separated, 0 => touching/overlapping."""
    best = np.inf
    # Combine normals for efficiency
    all_norms = np.vstack((n1, n2))
    for n in all_norms:
        a_proj = v1 @ n
        b_proj = v2 @ n
        if a_proj.max() < b_proj.min():
            g = b_proj.min() - a_proj.max()
        elif b_proj.max() < a_proj.min():
            g = a_proj.min() - b_proj.max()
        else:
            g = 0.0
        if g < best:
            best = g
    return best

def _outer_side_for(inner_verts, outer_center, outer_angle):
    """Smallest side of a regular hexagon (center, angle) that encloses all
    inner vertices. Uses the support function."""
    a = np.deg2rad(outer_angle)
    # Outward normal directions of the outer hexagon
    th = a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(th), np.sin(th)])
    
    # Shift vertices by outer center
    verts_shifted = inner_verts - outer_center
    max_proj = (verts_shifted @ normals.T).max(axis=0)
    # Side length s satisfies: s * sqrt(3)/2 = max_proj
    return float((2.0 / np.sqrt(3.0)) * max_proj.max())

def _validity(inner, outer_center, outer_angle, side):
    """Return True if the packing is valid (separated + contained + finite)."""
    if not np.all(np.isfinite(inner)):
        return False
    if side < np.sqrt(11.0) - 1e-9:
        return False
    
    verts = [_hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner]
    
    # Check separation
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            v1, v2 = verts[i], verts[j]
            n1, n2 = _edge_normals(v1), _edge_normals(v2)
            if _min_gap(v1, v2, n1, n2) <= 1e-6:
                return False
                
    # Check containment
    ov = _hex_vertices(outer_center[0], outer_center[1], side, outer_angle)
    for v in verts:
        for k in range(6):
            p1, p2 = ov[k], ov[(k + 1) % 6]
            if np.cross(p2 - p1, v - p1) < -1e-6:
                return False
    return True

# ---------- Lattice representation ----------

# Lattice basis vectors for a triangular lattice of touching unit hexagons
# Distance between centers is sqrt(3)
_D = np.sqrt(3.0)
_B1 = np.array([_D, 0.0])
_B2 = np.array([_D * 0.5, _D * np.sqrt(3.0) / 2.0])

# Indices in lattice coordinates (x, y) for the 11 hexagons
# Center: (0,0)
# Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
# Ring 2 (Corners): (2,0), (1,1), (-1,2), (2,-1)
_LATTICE_COORDS = np.array([
    [0, 0],
    [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
    [2, 0], [1, 1], [-1, 2], [2, -1]
])

def _lattice_to_positions(scale):
    """Convert lattice scale to (11, 2) center positions."""
    pos = np.zeros((11, 2))
    for i in range(11):
        lx, ly = _LATTICE_COORDS[i]
        pos[i] = (lx * _B1 + ly * _B2) * scale
    return pos

def _objective(params):
    """
    Objective function for optimization.
    params: [alpha_deg, beta_deg, scale]
    alpha_deg: Angle of outer hexagon
    beta_deg: Angle of inner hexagons
    scale: Scale factor for lattice (1.0 is touching)
    """
    alpha_deg, beta_deg, scale = params
    
    # Ensure scale is at least 1.0 to allow for gap
    if scale < 1.0:
        scale = 1.0

    centers = _lattice_to_positions(scale)
    
    # Build inner hexagons
    inner = np.zeros((11, 3))
    for i in range(11):
        inner[i, 0] = centers[i, 0]
        inner[i, 1] = centers[i, 1]
        inner[i, 2] = beta_deg

    # Calculate outer side
    all_verts = np.vstack([_hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner])
    side = _outer_side_for(all_verts, np.array([0.0, 0.0]), alpha_deg)
    
    # Calculate minimum gap
    min_gap_val = np.inf
    verts_list = [_hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner]
    for i in range(11):
        for j in range(i + 1, 11):
            v1, v2 = verts_list[i], verts_list[j]
            n1, n2 = _edge_normals(v1), _edge_normals(v2)
            g = _min_gap(v1, v2, n1, n2)
            if g < min_gap_val:
                min_gap_val = g

    # Penalty if gap is too small
    if min_gap_val <= 1e-6:
        penalty = (1e-6 - min_gap_val) * 100.0
    else:
        penalty = 0.0

    return side + penalty

def hexagon_packing_11():
    oc0 = np.array([0.0, 0.0])
    
    # Initial guess:
    # Alpha (outer) = 0
    # Beta (inner) = 0
    # Scale = 1.0 (touching, will be penalized and pushed up)
    x0 = np.array([0.0, 0.0, 1.0])
    
    best_inner = None
    best_side = np.inf
    best_ang = 0.0
    best_params = x0.copy()
    
    # Try multiple starting points for alpha/beta to avoid local minima
    # Due to symmetry, 0,0 is likely the global minimum, but let's be safe.
    start_points = [
        [0.0, 0.0, 1.0],
        [30.0, 0.0, 1.0],
        [0.0, 30.0, 1.0],
        [15.0, 15.0, 1.0],
        [45.0, 0.0, 1.0],
    ]
    
    for x0 in start_points:
        if time.time() - _START_TIME > _TIME_LIMIT - 30.0:
            break
        try:
            res = minimize(_objective, x0, method="Nelder-Mead",
                           options={"maxiter": 2000, "xatol": 1e-10, "fatol": 1e-12, "adaptive": True})
            alpha, beta, scale = res.x
            
            # Construct the packing
            centers = _lattice_to_positions(scale)
            inner = np.zeros((11, 3))
            for i in range(11):
                inner[i, 0] = centers[i, 0]
                inner[i, 1] = centers[i, 1]
                inner[i, 2] = beta

            all_verts = np.vstack([_hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner])
            side = _outer_side_for(all_verts, oc0, alpha)
            
            if _validity(inner, oc0, alpha, side):
                if side < best_side:
                    best_inner = inner.copy()
                    best_side = side
                    best_ang = alpha
                    best_params = res.x.copy()
        except Exception:
            continue

    # Fallback if optimization failed or no valid result found
    if best_inner is None:
        # Use the raw lattice packing, scaled slightly to ensure gap
        # We need to find a scale that gives gap > 1e-6
        # Approximation: gap scales linearly with (scale - 1)
        # For touching hexagons, the "distance" is 0.
        # We just pick a slightly larger scale.
        scale = 1.001 
        centers = _lattice_to_positions(scale)
        inner = np.zeros((11, 3))
        for i in range(11):
            inner[i, 0] = centers[i, 0]
            inner[i, 1] = centers[i, 1]
            inner[i, 2] = 0.0
        
        all_verts = np.vstack([_hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner])
        side = _outer_side_for(all_verts, oc0, 0.0)
        
        # If this fallback is invalid (unlikely with scale 1.001), try even larger
        if not _validity(inner, oc0, 0.0, side):
             scale = 1.01
             centers = _lattice_to_positions(scale)
             for i in range(11):
                 inner[i, 0] = centers[i, 0]
                 inner[i, 1] = centers[i, 1]
             all_verts = np.vstack([_hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner])
             side = _outer_side_for(all_verts, oc0, 0.0)
        
        best_inner = inner
        best_side = side
        best_ang = 0.0

    return best_inner, np.array([oc0[0], oc0[1], best_ang]), float(best_side)

# EVOLVE-BLOCK-END
