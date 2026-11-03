import numpy as np
import time
from scipy.optimize import minimize

N = 11

def _hex_vertices(cx, cy, angle_deg=0.0):
    """Vertices of a unit regular hexagon centered at (cx, cy) rotated by angle_deg."""
    angle_rad = np.deg2rad(angle_deg)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angles = base_angles + angle_rad
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def _outer_normals(outer_angle_deg=0.0):
    """
    Return the 6 outward unit normals of a regular hexagon with given rotation.
    For a regular hexagon with 'pointy-top' orientation (vertex at angle 0),
    the outward normals are at angles 30, 90, 150, 210, 270, 330 degrees.
    Rotating the hexagon by outer_angle_deg shifts these.
    """
    theta_rad = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        alpha = theta_rad + np.deg2rad(30 + k * 60)
        normals.append((np.cos(alpha), np.sin(alpha)))
    return np.array(normals)

def _min_outer_side_from_centers(centers, inner_angles, outer_angle_deg=0.0):
    """
    Compute minimum outer hexagon side length S such that all inner hexagons fit.
    S = 2 * max_proj / sqrt(3) where max_proj is the max dot product of any inner
    vertex with any outer normal.
    """
    normals = _outer_normals(outer_angle_deg)
    max_proj = 0.0
    for (cx, cy), ang in zip(centers, inner_angles):
        verts = _hex_vertices(cx, cy, ang)
        # Project all vertices onto all normals
        projs = verts @ normals.T  # shape (6, 6)
        m = np.max(projs)
        if m > max_proj:
            max_proj = m
    S = 2.0 * max_proj / np.sqrt(3.0)
    return S

def _check_no_overlap(centers, inner_angles, eps=1e-9):
    """
    Check that no two inner hexagons overlap.
    Two unit regular hexagons (same orientation) are disjoint iff distance between centers >= sqrt(3).
    For different orientations, we do a more careful check using vertex-edge separation.
    Here we use a conservative check: if all orientations are the same, use distance check.
    Otherwise, use a polygon separation test.
    """
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            # If angles are the same (mod 60), use simple distance check
            ang_diff = (inner_angles[i] - inner_angles[j]) % 60.0
            if abs(ang_diff) < 1e-6 or abs(ang_diff - 60.0) < 1e-6:
                d = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                if d < np.sqrt(3.0) - eps:
                    return False
            else:
                # General case: check if any vertex of one is inside the other, or edges cross
                # Conservative: check min distance between vertices
                v1 = _hex_vertices(*centers[i], inner_angles[i])
                v2 = _hex_vertices(*centers[j], inner_angles[j])
                # If any vertex of v1 is inside v2 or vice versa, they overlap
                # Use a simple check: min distance between any pair of vertices < some threshold
                # This is not perfect but good enough for local search
                min_dist = np.inf
                for p in v1:
                    d = np.min(np.linalg.norm(v2 - p, axis=1))
                    if d < min_dist:
                        min_dist = d
                # If min vertex-vertex distance is very small, likely overlapping
                # A safer check: use scipy's convex hull or just check center distance
                d_centers = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                if d_centers < 1.9:  # Conservative threshold for different orientations
                    # Do a more careful check
                    # Check if any edge of v1 intersects any edge of v2
                    overlap = False
                    for a1, a2 in [(v1[k], v1[(k+1)%6]) for k in range(6)]:
                        for b1, b2 in [(v2[k], v2[(k+1)%6]) for k in range(6)]:
                            # Check segment intersection
                            # Using cross product method
                            d1 = np.cross(b2 - b1, a1 - b1)
                            d2 = np.cross(b2 - b1, a2 - b1)
                            d3 = np.cross(a2 - a1, b1 - a1)
                            d4 = np.cross(a2 - a1, b2 - a1)
                            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                                overlap = True
                                break
                        if overlap:
                            break
                    if overlap:
                        return False
                    # Also check if a vertex is inside the other hexagon
                    for p in v1:
                        if _point_in_hexagon(p[0], p[1], centers[j][0], centers[j][1], inner_angles[j]):
                            return False
                    for p in v2:
                        if _point_in_hexagon(p[0], p[1], centers[i][0], centers[i][1], inner_angles[i]):
                            return False
    return True

def _point_in_hexagon(px, py, cx, cy, angle_deg=0.0):
    """Check if point (px, py) is inside a unit hexagon centered at (cx, cy)."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # Unit hexagon with vertices at angles 0,60,120,180,240,300
    # The hexagon is the intersection of 6 half-planes
    # For each normal direction, check dot product <= 1 (inradius = sqrt(3)/2)
    inradius = np.sqrt(3.0) / 2.0
    for k in range(6):
        alpha = np.deg2rad(30 + k * 60)
        nx = np.cos(alpha)
        ny = np.sin(alpha)
        if lx * nx + ly * ny > inradius + 1e-9:
            return False
    return True

def _objective(params):
    """
    Objective function for optimization.
    params: flat array of [x1, y1, x2, y2, ..., xN, yN]
    Returns the outer side length S.
    Assumes all inner hexagons have angle 0.0 (aligned).
    """
    centers = [(params[2*i], params[2*i+1]) for i in range(N)]
    inner_angles = [0.0] * N
    
    # Check for overlaps - if overlapping, return a large penalty
    if not _check_no_overlap(centers, inner_angles, eps=1e-7):
        return 1e6
    
    S = _min_outer_side_from_centers(centers, inner_angles, outer_angle_deg=0.0)
    return S

def _initial_triangular_packing():
    """
    Generate an initial triangular lattice packing of 11 hexagons.
    Center + 6 neighbors + 4 from next shell.
    All aligned (angle 0).
    """
    s3 = np.sqrt(3.0)
    centers = []
    # Center
    centers.append((0.0, 0.0))
    # 6 neighbors
    centers.append((s3, 0.0))
    centers.append((s3/2, 1.5))
    centers.append((-s3/2, 1.5))
    centers.append((-s3, 0.0))
    centers.append((-s3/2, -1.5))
    centers.append((s3/2, -1.5))
    # 4 more from next shell (choose 4 out of 6 to break symmetry optimally)
    # Next shell candidates (distance 3 from center):
    ring_A = [
        (1.5*s3, 1.5),
        (0.0, 3.0),
        (-1.5*s3, 1.5),
        (-1.5*s3, -1.5),
        (0.0, -3.0),
        (1.5*s3, -1.5)
    ]
    # Choose 4 that are mutually non-overlapping (distance >= sqrt(3))
    # Try all combinations of 4 from 6
    from itertools import combinations
    best_combo = None
    best_S = float('inf')
    for combo in combinations(range(6), 4):
        extra = [ring_A[idx] for idx in combo]
        all_centers = centers + extra
        # Check mutual distances in extra
        valid = True
        for i in range(4):
            for j in range(i+1, 4):
                d = np.linalg.norm(np.array(extra[i]) - np.array(extra[j]))
                if d < s3 - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        S = _min_outer_side_from_centers(all_centers, [0.0]*N, outer_angle_deg=0.0)
        if S < best_S:
            best_S = S
            best_combo = combo
    if best_combo is not None:
        for idx in best_combo:
            centers.append(ring_A[idx])
    else:
        # Fallback: add 4 from ring A
        for idx in [0, 1, 2, 3]:
            centers.append(ring_A[idx])
    return centers

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin (150s limit)
    
    # Step 1: Get initial triangular packing
    centers = _initial_triangular_packing()
    inner_angles = [0.0] * N
    
    # Compute initial S
    best_S = _min_outer_side_from_centers(centers, inner_angles, outer_angle_deg=0.0)
    best_centers = list(centers)
    
    # Step 2: Local optimization using Nelder-Mead
    # Flatten centers
    x0 = []
    for (cx, cy) in centers:
        x0.append(cx)
        x0.append(cy)
    x0 = np.array(x0)
    
    # Run Nelder-Mead
    try:
        result = minimize(
            _objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if time.time() < deadline - 5:
            if result.fun < best_S and result.fun < 1e5:
                # Verify no overlaps
                opt_centers = [(result.x[2*i], result.x[2*i+1]) for i in range(N)]
                if _check_no_overlap(opt_centers, inner_angles, eps=1e-7):
                    opt_S = _min_outer_side_from_centers(opt_centers, inner_angles, outer_angle_deg=0.0)
                    if opt_S < best_S:
                        best_S = opt_S
                        best_centers = opt_centers
    except Exception:
        pass
    
    # Step 3: Try different outer rotations
    # For each candidate outer rotation, recompute S
    # The inner hexagons stay aligned at 0 degrees, but the outer hexagon rotates
    # Actually, we can also rotate the entire configuration
    best_overall_S = best_S
    best_overall_centers = best_centers
    best_overall_outer_angle = 0.0
    
    for outer_ang in range(0, 180, 15):
        if time.time() > deadline - 5:
            break
        S = _min_outer_side_from_centers(best_centers, inner_angles, outer_angle_deg=float(outer_ang))
        if S < best_overall_S:
            best_overall_S = S
            best_overall_outer_angle = float(outer_ang)
    
    # Step 4: Try rotating the entire configuration
    for rot_deg in range(0, 180, 15):
        if time.time() > deadline - 5:
            break
        rot_rad = np.deg2rad(rot_deg)
        cos_r = np.cos(rot_rad)
        sin_r = np.sin(rot_rad)
        rotated_centers = []
        for (x, y) in best_centers:
            rx = x * cos_r - y * sin_r
            ry = x * sin_r + y * cos_r
            rotated_centers.append((rx, ry))
        S = _min_outer_side_from_centers(rotated_centers, inner_angles, outer_angle_deg=0.0)
        if S < best_overall_S:
            best_overall_S = S
            best_overall_centers = rotated_centers
            best_overall_outer_angle = 0.0
    
    # Build return value
    inner = [[c[0], c[1], a] for c, a in zip(best_overall_centers, inner_angles)]
    return inner, (0.0, 0.0), best_overall_S, best_overall_outer_angle

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
