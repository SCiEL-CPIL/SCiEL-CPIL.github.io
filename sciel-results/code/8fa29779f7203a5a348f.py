import numpy as np
import time
from itertools import combinations
from scipy.optimize import linprog

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
    d1 = 2 * np.sqrt(3)
    d2 = 4 * np.sqrt(3)
    first_ring = [(d1 * np.cos(np.pi/3*i), d1 * np.sin(np.pi/3*i)) for i in range(6)]
    second_ring = []
    for i in range(6):
        a = np.pi/3*i
        second_ring.append((d2 * np.cos(a), d2 * np.sin(a)))
        a2 = np.pi/3*i + np.pi/6
        second_ring.append((d2 * np.cos(a2), d2 * np.sin(a2)))
    
    # Try different subsets of second ring
    subsets = [
        [0, 2, 4, 6],
        [0, 3, 6, 9],
        [1, 3, 5, 7],
        [0, 1, 2, 3],
        [0, 2, 5, 7],
        [0, 4, 8, 10],
    ]
    for subset in subsets:
        centers = [(0.0, 0.0)] + first_ring + [second_ring[i] for i in subset]
        configs.append((centers, [0.0] * N))
    
    # Config 2: Linear arrangement with slight zigzag
    for spacing in [3.5, 3.8, 4.0]:
        centers = []
        for i in range(N):
            x = (i - N//2) * spacing
            y = 0.5 * spacing * (1 if i % 2 == 0 else -1)
            centers.append((x, y))
        configs.append((centers, [0.0] * N))
    
    # Config 3: Circular arrangement
    for radius in [2.5, 3.0, 3.5]:
        centers = [(0.0, 0.0)]
        for i in range(N - 1):
            angle = 2 * np.pi * i / (N - 1)
            centers.append((radius * np.cos(angle), radius * np.sin(angle)))
        configs.append((centers, [0.0] * N))
    
    # Config 4: Random with seed
    np.random.seed(123)
    for _ in range(3):
        centers = [(0.0, 0.0)]
        for i in range(N - 1):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(1.5, 3.5)
            centers.append((r * np.cos(angle), r * np.sin(angle)))
        configs.append((centers, [0.0] * N))
    
    return configs

def _simulate_annealing(initial_centers, initial_angles, max_time=120.0):
    """
    Simulated Annealing to optimize positions and angles.
    """
    start_time = time.time()
    centers = [list(c) for c in initial_centers]
    angles = list(initial_angles)
    
    # Initial cost
    current_side, current_center, current_angle = _optimize_outer(centers, angles)
    if not np.isfinite(current_side):
        return initial_centers, initial_angles, float('inf'), None, None, None
        
    best_side = current_side
    best_centers = [list(c) for c in centers]
    best_angles = list(angles)
    best_center = current_center
    best_angle = current_angle
    
    T = 0.5
    T_min = 0.01
    cooling_rate = 0.999
    
    np.random.seed(42)
    
    while time.time() - start_time < max_time:
        if T < T_min:
            break
            
        # Generate new state
        new_centers = [list(c) for c in centers]
        new_angles = list(angles)
        
        # Random perturbation
        idx = np.random.randint(N)
        delta_x = np.random.normal(0, 0.1)
        delta_y = np.random.normal(0, 0.1)
        new_centers[idx][0] += delta_x
        new_centers[idx][1] += delta_y
        
        if np.random.rand() < 0.5:
            rot_idx = np.random.randint(N)
            delta_ang = np.random.normal(0, 5.0)
            new_angles[rot_idx] += delta_ang
            
        if np.random.rand() < 0.1:
            dx = np.random.normal(0, 0.05)
            dy = np.random.normal(0, 0.05)
            for i in range(N):
                new_centers[i][0] += dx
                new_centers[i][1] += dy
                
        if np.random.rand() < 0.1:
            drot = np.random.normal(0, 2.0)
            for i in range(N):
                new_angles[i] += drot

        # Check validity (overlaps)
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(new_centers[i], new_angles[i], new_centers[j], new_angles[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            T *= cooling_rate
            continue
            
        # Calculate new cost
        new_side, new_center, new_angle = _optimize_outer(new_centers, new_angles)
        
        if not np.isfinite(new_side):
            T *= cooling_rate
            continue
            
        delta = new_side - current_side
        
        if delta < 0:
            centers = new_centers
            angles = new_angles
            current_side = new_side
            current_center = new_center
            current_angle = new_angle
            
            if current_side < best_side:
                best_side = current_side
                best_centers = [list(c) for c in centers]
                best_angles = list(angles)
                best_center = current_center
                best_angle = current_angle
        else:
            prob = np.exp(-delta / T)
            if np.random.rand() < prob:
                centers = new_centers
                angles = new_angles
                current_side = new_side
                current_center = new_center
                current_angle = new_angle
                
        T *= cooling_rate
        
    return best_centers, best_angles, best_side, best_center, best_angle

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
    if best_init_centers and time.time() - start_time < DEADLINE - 30:
        est_side, est_center, est_angle = _optimize_outer(best_init_centers, best_init_angles)
        
        relaxed_centers, relaxed_angles = _force_directed_relax(
            best_init_centers, best_init_angles, est_side, est_center, est_angle, max_iter=100
        )
        
        if time.time() - start_time < DEADLINE - 25:
            r_side, r_result = _try_arrangement(relaxed_centers, relaxed_angles)
            if r_side < best_init_side:
                best_init_side = r_side
                best_init_centers = [list(c) for c in relaxed_centers]
                best_init_angles = list(relaxed_angles)
                best_result = r_result

    # Run Simulated Annealing
    if best_init_centers and time.time() - start_time < DEADLINE - 10:
        sa_centers, sa_angles, sa_side, sa_center, sa_angle = _simulate_annealing(
            best_init_centers, best_init_angles, max_time=DEADLINE - time.time() - 10
        )
        
        if _check_validity(sa_centers, sa_angles, sa_center, sa_side, sa_angle):
            if sa_side < best_side:
                best_side = sa_side
                best_result = (sa_centers, sa_angles, sa_center, sa_side, sa_angle)

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
