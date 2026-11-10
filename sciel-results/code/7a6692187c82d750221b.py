# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, angle a_deg."""
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _check_validity(inner, outer, side):
    """Check all validity constraints. Returns (valid, violation_msg)."""
    # All finite
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer)) and np.isfinite(side)):
        return False, "non-finite values"
    if side < np.sqrt(11) - 1e-12:
        return False, "side too small"
    
    # Compute inner hexagon vertices
    inner_verts = []
    for i in range(11):
        x, y, ang = inner[i]
        v = _hex_vertices(x, y, 1.0, ang)
        inner_verts.append(v)
    
    # Check edge lengths within 1e-8 of 1
    for v in inner_verts:
        for j in range(6):
            d = np.linalg.norm(v[(j+1)%6] - v[j])
            if abs(d - 1.0) > 1e-8:
                return False, f"edge length {d} not 1"
    
    # Outer hexagon
    ox, oy, oang = outer
    outer_verts = _hex_vertices(ox, oy, side, oang)
    
    # Shift coordinates by outer center
    shift = np.array([ox, oy])
    shifted_inner = inner.copy()
    shifted_inner[:, 0] -= ox
    shifted_inner[:, 1] -= oy
    
    # Re-check edge lengths after shift (should be same, but verify)
    for i in range(11):
        x, y, ang = shifted_inner[i]
        v = _hex_vertices(x, y, 1.0, ang)
        for j in range(6):
            d = np.linalg.norm(v[(j+1)%6] - v[j])
            if abs(d - 1.0) > 1e-8:
                return False, f"shifted edge length {d} not 1"
    
    # Check inner hexagons are separated (no overlap)
    # For each pair, check on all 12 edge normals (6 from each hex)
    for i in range(11):
        for j in range(i+1, 11):
            vi = inner_verts[i]
            vj = inner_verts[j]
            # Get edge normals for both hexagons
            normals_i = []
            for k in range(6):
                e = vi[(k+1)%6] - vi[k]
                n = np.array([-e[1], e[0]])
                n = n / np.linalg.norm(n)
                normals_i.append(n)
            normals_j = []
            for k in range(6):
                e = vj[(k+1)%6] - vj[k]
                n = np.array([-e[1], e[0]])
                n = n / np.linalg.norm(n)
                normals_j.append(n)
            
            all_normals = normals_i + normals_j
            
            separated = False
            for n in all_normals:
                # Project vertices of hex i onto normal
                proj_i = vi @ n
                proj_j = vj @ n
                min_i = proj_i.min()
                max_i = proj_i.max()
                min_j = proj_j.min()
                max_j = proj_j.max()
                # Check if intervals are apart by more than 1e-6
                gap = max(min_j - max_i, min_i - max_j)
                if gap > 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False, f"inner hexagons {i} and {j} overlap"
    
    # Check all inner vertices are inside the outer hexagon
    # For each outer edge from p1 to p2, cross(p2-p1, p-p1) >= -1e-6
    for iv in inner_verts:
        for p in iv:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                edge = p2 - p1
                vec = p - p1
                cross = edge[0]*vec[1] - edge[1]*vec[0]
                if cross < -1e-6:
                    return False, "inner vertex outside outer hexagon"
    
    return True, "valid"

def _objective(params):
    """Objective: minimize outer side length. params = [x1,y1,a1,...,x11,y11,a11, ox,oy,oang, side]"""
    # Extract parameters
    inner = params[:33].reshape(11, 3)
    ox, oy, oang = params[33:36]
    side = params[36]
    
    # Compute outer vertices
    outer_verts = _hex_vertices(ox, oy, side, oang)
    
    # Compute inner vertices
    inner_verts = []
    for i in range(11):
        x, y, ang = inner[i]
        v = _hex_vertices(x, y, 1.0, ang)
        inner_verts.append(v)
    
    # Penalty for vertices outside outer hexagon
    penalty = 0.0
    for iv in inner_verts:
        for p in iv:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                edge = p2 - p1
                vec = p - p1
                cross = edge[0]*vec[1] - edge[1]*vec[0]
                if cross < 0:
                    penalty += (-cross) ** 2
    
    # Penalty for overlapping inner hexagons
    for i in range(11):
        for j in range(i+1, 11):
            vi = inner_verts[i]
            vj = inner_verts[j]
            # Simple distance-based penalty (approximate)
            min_dist = np.inf
            for p1 in vi:
                for p2 in vj:
                    d = np.linalg.norm(p1 - p2)
                    min_dist = min(min_dist, d)
            if min_dist < 0.01:
                penalty += (0.01 - min_dist) ** 2 * 100
    
    return side + penalty * 1000

def _make_initial_config():
    """Create a good initial configuration for 11 hexagons."""
    # Hexagonal lattice positions for 11 hexagons
    # Center + 6 around + 4 more
    # Using axial coordinates converted to Cartesian
    
    # Standard hex grid: pointy-top or flat-top
    # Let's use flat-top hexagons (angle 0 means vertex at right)
    
    # For unit hexagons in a hexagonal packing:
    # Horizontal spacing: 1.5 (for flat-top) or √3 (for pointy-top)
    # Let's use the arrangement that minimizes bounding hexagon
    
    # Optimal known arrangement: 1 center, 6 in ring, 4 in second ring
    # Positions in a hex grid (axial coords q,r) -> Cartesian
    
    # For flat-top hexagons with side 1:
    # x = 1.5 * q
    # y = √3 * (r + q/2)
    
    sqrt3 = np.sqrt(3)
    
    # Axial coordinates for 11 hexagons (centered at origin)
    # Ring 0: (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: 4 hexagons (selected to be compact)
    
    positions_axial = [
        (0, 0),       # center
        (1, 0),       # ring 1
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        # Ring 2 - pick 4 to make it compact
        (2, 0),
        (1, 1),
        (-1, 2),
        (-2, 1),
    ]
    
    inner = np.zeros((11, 3))
    for i, (q, r) in enumerate(positions_axial):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2)
        inner[i] = [x, y, 0]
    
    # Estimate outer hexagon size
    # Find bounding box of all inner hexagon vertices
    all_verts = []
    for i in range(11):
        x, y, ang = inner[i]
        v = _hex_vertices(x, y, 1.0, ang)
        all_verts.append(v)
    
    all_verts = np.vstack(all_verts)
    x_min, y_min = all_verts.min(axis=0)
    x_max, y_max = all_verts.max(axis=0)
    
    # Center of bounding box
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    
    # Estimate side length needed
    # For a regular hexagon containing these points, we need to find minimum side
    # Approximate: the max distance from center to any vertex, divided by cos(30°) for flat-top
    max_dist = np.max(np.linalg.norm(all_verts - np.array([cx, cy]), axis=1))
    # For a regular hexagon with flat-top orientation, the apothem (inradius) = side * √3/2
    # The max distance from center to boundary depends on angle
    # Conservative estimate: side >= max_dist / cos(30°) for worst case
    est_side = max_dist / (np.sqrt(3)/2) + 0.1
    
    outer = np.array([cx, cy, 0])
    side = est_side
    
    return inner, outer, side

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = 600  # seconds
    
    # Start with initial configuration
    inner, outer, side = _make_initial_config()
    
    best_inner = inner.copy()
    best_outer = outer.copy()
    best_side = side
    best_valid, _ = _check_validity(best_inner, best_outer, best_side)
    
    if best_valid:
        best_score = 1.0 / best_side
    
    # Try optimization
    # Parameters: all inner (x,y,angle) + outer (x,y,angle) + side
    # We'll fix outer at origin for simplicity and optimize inner positions + side
    
    # Actually, let's use a simpler approach: optimize inner positions with outer fixed
    # Then compute the minimum enclosing hexagon
    
    def _min_enclosing_hex_side(verts):
        """Find minimum side length of a regular hexagon (any rotation) enclosing all vertices."""
        # For each rotation angle, compute the side length needed
        best_side = np.inf
        for a_deg in np.linspace(0, 60, 13):  # 13 angles (0 to 60 in 5° steps)
            a = np.radians(a_deg)
            # For a hexagon with this orientation, the constraints are:
            # For each of the 6 edge normals, the max projection must be <= apothem
            # apothem = side * √3/2, so side = 2 * max_projection / √3
            max_proj = 0
            for k in range(6):
                theta = a + np.radians(60) * k
                # Normal to edge k points outward at angle theta + 30°
                n_angle = theta + np.radians(30)
                n = np.array([np.cos(n_angle), np.sin(n_angle)])
                proj = verts @ n
                max_proj = max(max_proj, proj.max())
            side_needed = 2 * max_proj / np.sqrt(3)
            best_side = min(best_side, side_needed)
        return best_side
    
    # Optimize: minimize side length by adjusting inner hexagon positions
    def _obj(params):
        inner_opt = params.reshape(11, 3)
        # Compute all vertices
        all_verts = []
        for i in range(11):
            x, y, ang = inner_opt[i]
            v = _hex_vertices(x, y, 1.0, ang)
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Compute min enclosing hex side
        s = _min_enclosing_hex_side(all_verts)
        
        # Penalty for overlaps
        penalty = 0.0
        inner_verts_list = all_verts.reshape(11, 6, 2)
        for i in range(11):
            for j in range(i+1, 11):
                vi = inner_verts_list[i]
                vj = inner_verts_list[j]
                # Check minimum distance between any two vertices
                min_d = np.inf
                for p1 in vi:
                    for p2 in vj:
                        d = np.linalg.norm(p1 - p2)
                        min_d = min(min_d, d)
                if min_d < 0.05:
                    penalty += (0.05 - min_d) ** 2 * 1000
        return s + penalty
    
    # Initial params
    x0 = inner.flatten()
    
    # Run optimization
    try:
        result = minimize(_obj, x0, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        if time.time() - start_time < deadline - 30:
            best_params = result.x
            best_inner_opt = best_params.reshape(11, 3)
            
            # Compute min enclosing hexagon
            all_verts = []
            for i in range(11):
                x, y, ang = best_inner_opt[i]
                v = _hex_vertices(x, y, 1.0, ang)
                all_verts.append(v)
            all_verts = np.vstack(all_verts)
            
            # Find best rotation for enclosing hexagon
            best_s = np.inf
            best_a = 0
            for a_deg in np.linspace(0, 60, 61):
                a = np.radians(a_deg)
                max_proj = 0
                for k in range(6):
                    theta = a + np.radians(60) * k
                    n_angle = theta + np.radians(30)
                    n = np.array([np.cos(n_angle), np.sin(n_angle)])
                    proj = all_verts @ n
                    max_proj = max(max_proj, proj.max())
                s_needed = 2 * max_proj / np.sqrt(3)
                if s_needed < best_s:
                    best_s = s_needed
                    best_a = a_deg
            
            # Center the outer hexagon
            cx = all_verts[:, 0].mean()
            cy = all_verts[:, 1].mean()
            
            # Shift inner hexagons so outer is at origin
            best_inner_opt[:, 0] -= cx
            best_inner_opt[:, 1] -= cy
            
            outer_opt = np.array([0, 0, best_a])
            side_opt = best_s
            
            valid, msg = _check_validity(best_inner_opt, outer_opt, side_opt)
            if valid:
                best_inner = best_inner_opt
                best_outer = outer_opt
                best_side = side_opt
                best_score = 1.0 / best_side
    except Exception:
        pass
    
    # If still not valid, try a different approach
    if not best_valid or best_score < 1.0:
        # Use a known good configuration
        # 11 hexagons in a hexagonal arrangement
        sqrt3 = np.sqrt(3)
        # More compact arrangement
        inner = np.array([
            [0, 0, 0],
            [1.5, 0, 0],
            [0.75, 1.299, 0],
            [-0.75, 1.299, 0],
            [-1.5, 0, 0],
            [-0.75, -1.299, 0],
            [0.75, -1.299, 0],
            [3.0, 0, 0],
            [2.25, 1.299, 0],
            [-2.25, 1.299, 0],
            [-3.0, 0, 0],
        ])
        outer = np.array([0, 0, 0])
        # Compute required side
        all_verts = []
        for i in range(11):
            x, y, ang = inner[i]
            v = _hex_vertices(x, y, 1.0, ang)
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        max_dist = np.max(np.linalg.norm(all_verts, axis=1))
        side = max_dist / (np.sqrt(3)/2) + 0.01
        
        valid, msg = _check_validity(inner, outer, side)
        if valid:
            best_inner = inner
            best_outer = outer
            best_side = side
    
    # Final validity check
    valid, msg = _check_validity(best_inner, best_outer, best_side)
    if not valid:
        # Fallback: use very large hexagon
        best_inner = np.array([
            [0, 0, 0],
            [2.5, 0, 0],
            [-2.5, 0, 0],
            [1.25, 2.17, 0],
            [-1.25, 2.17, 0],
            [1.25, -2.17, 0],
            [-1.25, -2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, -2.17, 0],
            [-3.75, -2.17, 0],
        ], dtype=float)
        best_outer = np.array([0, 0, 0.0])
        best_side = 8.0
    
    return best_inner, best_outer, best_side

# EVOLVE-BLOCK-END
