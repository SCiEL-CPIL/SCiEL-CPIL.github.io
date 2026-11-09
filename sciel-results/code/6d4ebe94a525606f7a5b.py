import numpy as np
import time
from itertools import combinations
from scipy.optimize import linprog, minimize

N = 11
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts

def _hex_edge_normals(angle_deg):
    """Return 6 outward normal vectors of a regular hexagon."""
    angle = np.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi / 6.0 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals

def _min_enclosing_side_lp(centers, inner_angles, outer_angle_deg):
    """
    Compute minimum side length of a regular hexagon (fixed orientation)
    enclosing all inner hexagons using Linear Programming.
    Returns (side, center).
    """
    sqrt3 = np.sqrt(3)
    normals = _hex_edge_normals(outer_angle_deg)
    
    # Collect all vertices
    all_points = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i]) 
                            for i in range(len(centers))])
    
    # A_i = max projection in direction n_i
    projections = all_points @ normals.T
    A = projections.max(axis=0)
    
    # Variables: x = [cx, cy, r]
    # Minimize r
    c_obj = np.array([0.0, 0.0, 1.0])
    
    # Constraints: n_i . c - r <= -A_i
    A_mat = np.zeros((6, 3))
    b_vec = np.zeros(6)
    
    for i in range(6):
        A_mat[i, 0] = normals[i, 0]
        A_mat[i, 1] = normals[i, 1]
        A_mat[i, 2] = -1.0
        b_vec[i] = -A[i]
        
    res = linprog(c_obj, A_ub=A_mat, b_ub=b_vec, method='highs')
    
    if res.status != 0:
        return float('inf'), np.array([0.0, 0.0])
        
    cx, cy, r = res.x
    side = 2 * r / sqrt3
    return side, np.array([cx, cy])

def _hex_overlap(c1, a1, c2, a2, tol=1e-9):
    """Check if two unit hexagons overlap using SAT."""
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            nx, ny = -ey, ex
            norm = np.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                axes.append((nx / norm, ny / norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    for ax, ay in axes:
        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True

def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    """Check containment and non-overlap."""
    sqrt3 = np.sqrt(3)
    outer_normals = _hex_edge_normals(outer_angle_deg)
    inradius = outer_side * sqrt3 / 2
    
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    
    return True

def _optimize_outer(centers, inner_angles):
    """Find optimal outer hexagon for fixed inner positions."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Sweep angle finely
    for outer_angle in np.linspace(0, 60, 121):
        side, center = _min_enclosing_side_lp(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    return best_side, best_center, best_angle

def _try_arrangement(centers, inner_angles=None):
    """Try an arrangement and return (side, result) or (inf, None)."""
    if inner_angles is None:
        inner_angles = [0.0] * N
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return float('inf'), None
    
    # Find optimal outer hexagon
    side, center, angle = _optimize_outer(centers, inner_angles)
    
    # Verify validity
    if _check_validity(centers, inner_angles, center, side, angle):
        return side, (centers, inner_angles, center, side, angle)
    
    return float('inf'), None

def _get_valid_fallback():
    """Return a known valid, non-overlapping configuration."""
    centers = []
    angles = [0.0] * N
    x = 0
    y = 0
    for i in range(N):
        centers.append((x, y))
        x += 4.0
        if i % 2 == 1:
            y += 4.0
            x = -4.0 + (i // 2) * 8.0
    
    side, center, angle = _optimize_outer(centers, angles)
    return centers, angles, center, side, angle

def _force_directed_relax(centers, angles, outer_side, outer_center, outer_angle, max_iter=500, tol=1e-8):
    """
    Force-directed relaxation: push overlapping hexagons apart,
    pull them toward center if outside boundary.
    """
    centers = [list(c) for c in centers]
    angles = list(angles)
    sqrt3 = np.sqrt(3)
    outer_normals = _hex_edge_normals(outer_angle)
    inradius = outer_side * sqrt3 / 2
    
    for it in range(max_iter):
        displacements = [[0.0, 0.0] for _ in range(N)]
        
        # Repulsion between overlapping hexagons
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(centers[i], angles[i], centers[j], angles[j]):
                    # Push apart along center-to-center vector
                    dx = centers[j][0] - centers[i][0]
                    dy = centers[j][1] - centers[i][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 1e-12:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    # Estimate penetration depth
                    v1 = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
                    v2 = _hex_vertices(centers[j][0], centers[j][1], 1.0, angles[j])
                    
                    # Use SAT to find minimum penetration
                    def get_axes(verts):
                        axes = []
                        n = len(verts)
                        for k in range(n):
                            x1, y1 = verts[k]
                            x2, y2 = verts[(k + 1) % n]
                            ex = x2 - x1
                            ey = y2 - y1
                            nx, ny = -ey, ex
                            norm = np.sqrt(nx**2 + ny**2)
                            if norm > 1e-12:
                                axes.append((nx / norm, ny / norm))
                        return axes
                    
                    axes = get_axes(v1) + get_axes(v2)
                    min_overlap = float('inf')
                    min_axis = (dx, dy)
                    
                    for ax, ay in axes:
                        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
                        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
                        overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
                        if overlap < 0:
                            min_overlap = float('inf')
                            break
                        if overlap < min_overlap:
                            min_overlap = overlap
                            min_axis = (ax, ay)
                    
                    if min_overlap < float('inf') and min_overlap > 0:
                        # Push each hexagon half the penetration depth apart
                        push = min_overlap * 0.5
                        displacements[i][0] -= min_axis[0] * push
                        displacements[i][1] -= min_axis[1] * push
                        displacements[j][0] += min_axis[0] * push
                        displacements[j][1] += min_axis[1] * push
        
        # Containment: push hexagons inside if they extend outside
        for i in range(N):
            verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            for vx, vy in verts:
                for nx, ny in outer_normals:
                    dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                    if dist > inradius:
                        # Push center inward
                        displacements[i][0] -= nx * (dist - inradius)
                        displacements[i][1] -= ny * (dist - inradius)
        
        # Apply displacements with damping
        max_disp = 0.0
        for i in range(N):
            centers[i][0] += displacements[i][0] * 0.5
            centers[i][1] += displacements[i][1] * 0.5
            max_disp = max(max_disp, np.sqrt(displacements[i][0]**2 + displacements[i][1]**2))
        
        if max_disp < tol:
            break
    
    return centers, angles

def _generate_initial_configurations():
    """Generate diverse initial configurations for multi-start."""
    configs = []
    
    # Config 1: Hexagonal lattice (center + 6 ring + 4 from second ring)
    # Use a spacing that guarantees no overlap (distance > 2 * inradius = sqrt(3) ~ 1.732)
    # For hexagons of side 1, inradius is sqrt(3)/2 ~ 0.866.
    # Distance between centers in hex lattice: 2 * inradius * 2 = 2*sqrt(3) ~ 3.464 for touching?
    # Actually, for regular hexagons side 1, touching distance is 2 * inradius * 2? No.
    # Distance between centers of touching hexagons is 2 * inradius = sqrt(3) ~ 1.732?
    # Let's check: Inradius r = s * sqrt(3) / 2. Distance between parallel sides is 2r = s*sqrt(3).
    # If they touch side-to-side, center distance is s*sqrt(3).
    # So d1 = sqrt(3) is touching. We should use slightly more to be safe, e.g. 1.8.
    # But for optimization, we want them close. Let's use exact touching distance for lattice.
    d1 = np.sqrt(3)
    d2 = 2 * np.sqrt(3)
    
    # Center at (0,0)
    centers = [(0.0, 0.0)]
    # First ring (6 hexagons)
    for i in range(6):
        a = np.pi/3*i
        centers.append((d1 * np.cos(a), d1 * np.sin(a)))
        
    # Second ring (12 hexagons, we need 4 more to reach 11)
    # Positions in second ring are at distance 2*d1 or d1*sqrt(3)?
    # In hex lattice, neighbors of neighbors.
    # Let's just generate a larger lattice and pick 11 closest to origin.
    lattice_points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Axial to pixel
            x = d1 * (q + r/2.0)
            y = d1 * (np.sqrt(3)/2.0 * r)
            lattice_points.append((x, y))
            
    # Sort by distance to origin
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    # Take first 11
    centers = lattice_points[:11]
    configs.append((centers, [0.0] * N))
    
    # Config 2: Linear arrangement with slight zigzag (relaxed)
    for spacing in [1.8, 1.9]:
        centers = []
        for i in range(N):
            x = (i - N//2) * spacing
            y = 0.5 * spacing * (1 if i % 2 == 0 else -1)
            centers.append((x, y))
        configs.append((centers, [0.0] * N))
        
    return configs

def _local_search(centers, angles, max_time=30.0):
    """
    Use Nelder-Mead to minimize the outer side length.
    The objective function returns the side length if valid, else a penalty.
    """
    start_time = time.time()
    x0 = np.array([c[0] for c in centers] + [c[1] for c in centers])
    
    def objective(x):
        if time.time() - start_time > max_time:
            return float('inf')
            
        new_centers = []
        for i in range(N):
            new_centers.append((x[2*i], x[2*i+1]))
            
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(new_centers[i], angles[i], new_centers[j], angles[j]):
                    return 100.0 # Penalty
        
        side, center, angle = _optimize_outer(new_centers, angles)
        
        if not np.isfinite(side):
            return 100.0
            
        # Verify containment (LP should handle it, but double check)
        if _check_validity(new_centers, angles, center, side, angle):
            return side
        else:
            return 100.0

    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    if res.success and res.fun < 100.0:
        new_centers = []
        for i in range(N):
            new_centers.append((res.x[2*i], res.x[2*i+1]))
        side, center, angle = _optimize_outer(new_centers, angles)
        if _check_validity(new_centers, angles, center, side, angle):
            return new_centers, angles, side, center, angle
            
    return centers, angles, float('inf'), None, None

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_result = None
    
    # Generate diverse initial configurations
    initial_configs = _generate_initial_configurations()
    
    best_init_centers = None
    best_init_angles = None
    best_init_side = float('inf')
    
    # Evaluate each initial configuration
    for centers, angles in initial_configs:
        if time.time() - start_time > 20:
            break
        side, result = _try_arrangement(centers, angles)
        if side < best_init_side:
            best_init_side = side
            best_init_centers = [list(c) for c in centers]
            best_init_angles = list(angles)
            best_result = result

    if best_result is None:
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        best_result = (fb_centers, fb_angles, fb_center, fb_side, fb_angle)
        best_init_centers = fb_centers
        best_init_angles = fb_angles
        best_init_side = fb_side

    # Force-directed relaxation from best initial
    if best_init_centers and time.time() - start_time < DEADLINE - 40:
        est_side, est_center, est_angle = _optimize_outer(best_init_centers, best_init_angles)
        
        relaxed_centers, relaxed_angles = _force_directed_relax(
            best_init_centers, best_init_angles, est_side, est_center, est_angle, max_iter=100
        )
        
        if time.time() - start_time < DEADLINE - 30:
            r_side, r_result = _try_arrangement(relaxed_centers, relaxed_angles)
            if r_side < best_init_side:
                best_init_side = r_side
                best_init_centers = [list(c) for c in relaxed_centers]
                best_init_angles = list(relaxed_angles)
                best_result = r_result

    # Run Local Search (Nelder-Mead)
    if best_init_centers and time.time() - start_time < DEADLINE - 10:
        ls_centers, ls_angles, ls_side, ls_center, ls_angle = _local_search(
            best_init_centers, best_init_angles, max_time=DEADLINE - time.time() - 10
        )
        
        if ls_side < best_side:
            best_side = ls_side
            best_result = (ls_centers, ls_angles, ls_center, ls_side, ls_angle)

    # Final verification and formatting
    if best_result is None:
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        best_result = (fb_centers, fb_angles, fb_center, fb_side, fb_angle)
        
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    if not _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle):
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        centers, inner_angles, outer_center, outer_side, outer_angle = fb_centers, fb_angles, fb_center, fb_side, fb_angle

    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
