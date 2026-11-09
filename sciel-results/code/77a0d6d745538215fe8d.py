# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog, differential_evolution, minimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with side d and angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _hex_edge_normals(angle_deg):
    """Return 6 unit edge normals for a hexagon with given angle."""
    a = np.radians(angle_deg)
    # Edge normals are perpendicular to edges
    # For a regular hexagon, edge directions are at angles a+30+60i degrees
    # Normals are at angles a+60i degrees (pointing outward)
    normal_angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])


def _min_outer_side_analytic(inner_centers, inner_angles, rot_deg):
    """
    Compute minimum side length s of outer hexagon (rotated by rot_deg) containing all inner hexagons.
    Uses the fact that for a regular hexagon centered at (ox,oy) with side s and rotation rot,
    a point (x,y) is inside iff for all 6 edge directions:
      n_i . (x - c) <= s * sqrt(3)/2
    where n_i are the outward edge normals and c = (ox, oy).
    
    The optimal center is the centroid of the projection extremes.
    """
    rot = np.radians(rot_deg)
    
    # All inner vertices
    all_verts = np.zeros((72, 2))
    for i in range(12):
        cx, cy = inner_centers[i]
        aang = inner_angles[i]
        verts = _hex_vertices(cx, cy, 1, aang)
        all_verts[6*i:6*(i+1)] = verts
    
    sqrt3 = np.sqrt(3)
    
    # For each of 6 edge directions, compute the required range
    # Edge normal direction i: angle = rot + 60*i + 30 degrees (this is the outward normal)
    # Wait, let me reconsider. The vertices are at angles rot + 60i.
    # The edges are between consecutive vertices.
    # The outward normal to edge from vertex i to vertex i+1 points in direction rot + 60*i + 30.
    
    s_min = 0
    ox = 0
    oy = 0
    
    for i in range(6):
        # Outward normal direction for edge i
        n_angle = rot + np.radians(60 * i + 30)
        nx = np.cos(n_angle)
        ny = np.sin(n_angle)
        
        # Project all vertices onto this normal
        projections = all_verts[:, 0] * nx + all_verts[:, 1] * ny
        p_min = projections.min()
        p_max = projections.max()
        
        # For the outer hexagon centered at (ox, oy), the constraint is:
        # n . (v - c) <= s * sqrt(3)/2 for all vertices v
        # This gives: n.v - n.c <= s*sqrt(3)/2
        # So: s >= (n.v - n.c) * 2/sqrt(3) for all v
        # The tightest constraint is for max(n.v) - n.c
        # Also: n.c - n.v <= s*sqrt(3)/2 => n.c - min(n.v) <= s*sqrt(3)/2
        # So: s >= (n.c - min(n.v)) * 2/sqrt(3)
        
        # To minimize s, we want to choose n.c = (p_min + p_max) / 2
        # Then s >= (p_max - p_min) / 2 * 2/sqrt(3) = (p_max - p_min) / sqrt(3)
        
        required_s = (p_max - p_min) / sqrt3
        s_min = max(s_min, required_s)
        
        # Center offset in this direction
        center_proj = (p_min + p_max) / 2
    
    # Now we need to find (ox, oy) such that for all 6 directions,
    # the center projection matches the required center.
    # This is an overdetermined system. Use least squares.
    
    # Build the system: for each direction i, n_i . c = center_proj_i
    A = np.zeros((6, 2))
    b = np.zeros(6)
    
    for i in range(6):
        n_angle = rot + np.radians(60 * i + 30)
        A[i, 0] = np.cos(n_angle)
        A[i, 1] = np.sin(n_angle)
        
        # Recompute center projection
        nx = A[i, 0]
        ny = A[i, 1]
        projections = all_verts[:, 0] * nx + all_verts[:, 1] * ny
        b[i] = (projections.min() + projections.max()) / 2
    
    # Solve least squares
    c, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    ox, oy = c
    
    # Verify: recompute s with this center
    s_final = 0
    for i in range(6):
        n_angle = rot + np.radians(60 * i + 30)
        nx = np.cos(n_angle)
        ny = np.sin(n_angle)
        
        # Distance from center to each vertex in the normal direction
        d_proj = all_verts[:, 0] * nx + all_verts[:, 1] * ny - (ox * nx + oy * ny)
        s_final = max(s_final, np.abs(d_proj).max() * 2 / sqrt3)
    
    return s_final, (ox, oy)


def _check_separation_analytic(center1, angle1, center2, angle2, eps=1e-6):
    """
    Check if two unit hexagons are separated using the separating axis theorem.
    For two regular hexagons, the potential separating axes are the edge normals of both.
    """
    d = center2 - center1
    
    # Get edge normals for both hexagons
    normals1 = _hex_edge_normals(angle1)
    normals2 = _hex_edge_normals(angle2)
    
    all_normals = np.vstack([normals1, normals2])
    
    # For each normal, project both hexagons and check for separation
    # A unit hexagon with center c and angle a has vertices at c + cos(a+60i), sin(a+60i)
    # The projection of a unit hexagon onto a unit normal n is:
    # [n.c - max(n.v), n.c + max(n.v)] where max(n.v) = max over vertices of n.(vertex - c)
    # For a unit regular hexagon, the support function in direction n is:
    # h(n) = max_i (cos(n_angle - (a + 60i)))
    # This equals 1 if n is aligned with a vertex direction, and sqrt(3)/2 if aligned with an edge normal
    
    # Actually, for a regular hexagon with side 1 (circumradius 1), the support function
    # in direction making angle theta with a vertex direction is:
    # h(theta) = cos(theta) for |theta| <= 30 deg, then it's the support of the edge
    
    # Simpler: just compute projections of vertices
    verts1 = _hex_vertices(center1[0], center1[1], 1, angle1)
    verts2 = _hex_vertices(center2[0], center2[1], 1, angle2)
    
    for n in all_normals:
        proj1 = verts1 @ n
        proj2 = verts2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 + eps < min2 or max2 + eps < min1:
            return True  # Separated
    
    return False  # Not separated (overlapping or touching)


def _validate_packing_fast(inner_hex_data, outer_hex_data, s, eps=1e-6):
    """Fast validation of packing."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    ox, oy, oang = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, s, oang)
    
    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    # Check separation between inner hexagons
    for i in range(12):
        for j in range(i+1, 12):
            if not _check_separation_analytic(
                inner_hex_data[i, :2], inner_hex_data[i, 2],
                inner_hex_data[j, :2], inner_hex_data[j, 2], eps
            ):
                return False
    
    # Check all inner vertices are inside outer hexagon
    for i in range(12):
        cx, cy, aang = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1, aang)
        for v in verts:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True


def _lattice_centers(e1, e2, lattice_points):
    """Compute centers from lattice basis vectors and integer lattice points."""
    centers = np.zeros((len(lattice_points), 2))
    for i, (a, b) in enumerate(lattice_points):
        centers[i] = a * e1 + b * e2
    return centers


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    sqrt3 = np.sqrt(3)
    
    # Lattice points for 12 hexagons in a compact arrangement
    lattice_points = np.array([
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        (-1, 2),
        (-2, 1),
        (1, -2),
        (2, -1),
        (0, 2),
    ])
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    def try_packing(centers, inner_angles, rot_deg):
        nonlocal best_s, best_inner, best_outer
        s, center = _min_outer_side_analytic(centers, inner_angles, rot_deg)
        if s < best_s:
            inner_hex_data = np.column_stack([centers, inner_angles])
            outer_hex_data = np.array([center[0], center[1], rot_deg])
            if _validate_packing_fast(inner_hex_data, outer_hex_data, s):
                best_s = s
                best_inner = inner_hex_data
                best_outer = outer_hex_data
    
    # Strategy 1: Standard hexagonal lattice with fine rotation search
    e1_std = np.array([2.0, 0.0])
    e2_std = np.array([1.0, sqrt3])
    centers_std = _lattice_centers(e1_std, e2_std, lattice_points)
    inner_angles_std = np.zeros(12)
    
    for rot_deg in np.linspace(0, 60, 121):
        if time.time() > deadline - 60:
            break
        try_packing(centers_std, inner_angles_std, rot_deg)
    
    # Strategy 2: Slightly expanded lattice for strict separation
    if time.time() < deadline - 60:
        for scale in [1.001, 1.002, 1.003, 1.005]:
            if time.time() > deadline - 60:
                break
            e1_exp = np.array([2.0 * scale, 0.0])
            e2_exp = np.array([1.0 * scale, sqrt3 * scale])
            centers_exp = _lattice_centers(e1_exp, e2_exp, lattice_points)
            inner_angles_exp = np.zeros(12)
            
            for rot_deg in np.linspace(0, 60, 61):
                if time.time() > deadline - 60:
                    break
                try_packing(centers_exp, inner_angles_exp, rot_deg)
    
    # Strategy 3: Optimize lattice parameters using Nelder-Mead
    if time.time() < deadline - 120:
        def objective(params):
            """Minimize outer hexagon side length given lattice parameters."""
            if time.time() > deadline - 90:
                return 1e10
            
            e1_len, e2_len, angle_between = params
            # Convert to basis vectors
            e1 = np.array([e1_len, 0.0])
            angle_rad = np.radians(angle_between)
            e2 = np.array([e2_len * np.cos(angle_rad), e2_len * np.sin(angle_rad)])
            
            centers = _lattice_centers(e1, e2, lattice_points)
            inner_angles = np.zeros(12)
            
            # Find best rotation
            best_local_s = 1e10
            for rot_deg in np.linspace(0, 60, 13):
                s, _ = _min_outer_side_analytic(centers, inner_angles, rot_deg)
                best_local_s = min(best_local_s, s)
            
            # Check if valid
            if best_local_s >= 1e9:
                return 1e10
            
            # Quick validity check: ensure no overlaps
            for i in range(12):
                for j in range(i+1, 12):
                    if not _check_separation_analytic(
                        centers[i], inner_angles[i],
                        centers[j], inner_angles[j], 1e-7
                    ):
                        return 1e10
            
            return best_local_s
        
        # Initial guess: standard hexagonal lattice
        x0 = [2.0, 2.0, 60.0]
        bounds = [(1.9, 2.5), (1.9, 2.5), (50, 70)]
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.fun < 1e9:
                e1_len, e2_len, angle_between = res.x
                e1_opt = np.array([e1_len, 0.0])
                angle_rad = np.radians(angle_between)
                e2_opt = np.array([e2_len * np.cos(angle_rad), e2_len * np.sin(angle_rad)])
                centers_opt = _lattice_centers(e1_opt, e2_opt, lattice_points)
                inner_angles_opt = np.zeros(12)
                
                # Fine rotation search for this configuration
                for rot_deg in np.linspace(0, 60, 61):
                    if time.time() > deadline - 30:
                        break
                    try_packing(centers_opt, inner_angles_opt, rot_deg)
        except Exception:
            pass
    
    # Strategy 4: Try different lattice point arrangements
    if time.time() < deadline - 30:
        # Alternative: more symmetric arrangement
        lattice_points_alt = np.array([
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (-1, 2), (-2, 1), (2, -1), (1, -2), (-2, 2),
        ])
        
        for scale in [1.0, 1.002]:
            if time.time() > deadline - 30:
                break
            e1_alt = np.array([2.0 * scale, 0.0])
            e2_alt = np.array([1.0 * scale, sqrt3 * scale])
            centers_alt = _lattice_centers(e1_alt, e2_alt, lattice_points_alt)
            inner_angles_alt = np.zeros(12)
            
            for rot_deg in np.linspace(0, 60, 31):
                if time.time() > deadline - 20:
                    break
                try_packing(centers_alt, inner_angles_alt, rot_deg)
    
    # Fallback
    if best_inner is None:
        e1_std = np.array([2.0, 0.0])
        e2_std = np.array([1.0, sqrt3])
        centers_std = _lattice_centers(e1_std, e2_std, lattice_points)
        inner_angles_std = np.zeros(12)
        s, center = _min_outer_side_analytic(centers_std, inner_angles_std, 0)
        best_s = s
        best_inner = np.column_stack([centers_std, inner_angles_std])
        best_outer = np.array([center[0], center[1], 0])
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
