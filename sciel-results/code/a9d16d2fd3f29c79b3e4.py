"""Optimized hexagon packing for N=11 unit hexagons in a regular hexagon."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _hex_vertices_batch(positions, angles, side=1.0):
    """Vectorized: return (N*6, 2) array of all vertices for N hexagons."""
    n = len(positions)
    pos = np.asarray(positions, dtype=np.float64)
    ang = np.radians(np.asarray(angles, dtype=np.float64))
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    offsets_x = side * np.cos(total_angles)
    offsets_y = side * np.sin(total_angles)
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    return np.column_stack([verts_x.ravel(), verts_y.ravel()])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    s = side
    sqrt3 = np.sqrt(3)
    if lx > s + 1e-10 or lx < -s - 1e-10:
        return False
    if sqrt3 * lx + ly > sqrt3 * s + 1e-10:
        return False
    if -sqrt3 * lx + ly > sqrt3 * s + 1e-10:
        return False
    if -sqrt3 * lx - ly > sqrt3 * s + 1e-10:
        return False
    if sqrt3 * lx - ly > sqrt3 * s + 1e-10:
        return False
    return True

def _min_side_vectorized(all_verts, cx, cy, outer_angle_deg):
    """
    Compute minimum outer side length S such that all vertices in all_verts are inside
    the outer hexagon centered at (cx,cy) with rotation outer_angle_deg.
    """
    verts = np.asarray(all_verts, dtype=np.float64)
    n_pts = len(verts)
    if n_pts == 0:
        return 0.0
    
    dx = verts[:, 0] - cx
    dy = verts[:, 1] - cy
    
    a = np.radians(outer_angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    c1 = lx
    c2 = -lx
    c3 = lx + ly / sqrt3
    c4 = -lx + ly / sqrt3
    c5 = -lx - ly / sqrt3
    c6 = lx - ly / sqrt3
    
    required = np.maximum(
        np.maximum(np.maximum(c1, c2), np.maximum(c3, c4)),
        np.maximum(c5, c6)
    )
    return float(np.max(required))

def _hexagon_overlap(verts1, verts2, tol=1e-9):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem)."""
    axes = []
    for verts in [verts1, verts2]:
        for i in range(len(verts)):
            v1 = verts[i]
            v2 = verts[(i + 1) % len(verts)]
            edge = v2 - v1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append([nx/norm, ny/norm])
    
    for axis in axes:
        proj1 = np.dot(verts1, axis)
        proj2 = np.dot(verts2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True

def _check_no_overlap_fast(positions, angles, tol=1e-7):
    """Check that no two inner hexagons overlap. Returns True if valid."""
    n = len(positions)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i]))
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagon_overlap(verts_list[i], verts_list[j], tol):
                return False
    return True

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle, tol=1e-6):
    """Find minimum outer side length such that all inner hexagons fit inside the outer hexagon."""
    cx, cy = outer_center
    pos = np.asarray(inner_positions, dtype=np.float64)
    ang = np.asarray(inner_angles, dtype=np.float64)
    all_verts = _hex_vertices_batch(pos, ang, side=1.0)
    return _min_side_vectorized(all_verts, cx, cy, outer_angle)

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if packing is valid: no overlaps, all inside outer hexagon."""
    cx, cy = outer_center
    verts_list = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        verts_list.append(verts)
    
    for i in range(len(verts_list)):
        for j in range(i + 1, len(verts_list)):
            if _hexagon_overlap(verts_list[i], verts_list[j]):
                return False
    
    outer_verts = _hex_vertices(cx, cy, outer_side, outer_angle)
    for verts in verts_list:
        for v in verts:
            if not _point_in_hexagon(v[0], v[1], cx, cy, outer_side, outer_angle):
                return False
    return True

def _initial_packing():
    """Create initial hexagonal grid packing for N hexagons."""
    positions = []
    angles = []
    sqrt3 = np.sqrt(3)
    
    # Start with the 7-hexagon cluster (center + 6 neighbors)
    positions.append([0.0, 0.0])
    angles.append(0.0)
    for i in range(6):
        a = i * np.pi / 3
        x = 2.0 * np.cos(a)
        y = 2.0 * np.sin(a)
        positions.append([x, y])
        angles.append(0.0)
    
    # Add 4 more hexagons at the next ring positions
    extra_positions = [
        [4.0, 0.0],
        [-4.0, 0.0],
        [2.0, 2.0 * sqrt3],
        [-2.0, 2.0 * sqrt3],
    ]
    for ep in extra_positions:
        positions.append(list(ep))
        angles.append(0.0)
    
    positions = positions[:N]
    angles = angles[:N]
    return np.array(positions, dtype=np.float64), np.array(angles, dtype=np.float64)

def _overlap_penalty_smooth(positions, angles, beta=20.0):
    """Compute smooth overlap penalty for all pairs."""
    n = len(positions)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            # For hexagons of side 1, minimum distance between centers for non-overlap is 2*sqrt(3)/2 * 2 = 2*sqrt(3) ≈ 3.464
            # But with rotation, the minimum distance can be as low as 2 (when aligned)
            # Use a smooth penalty that activates when dist < 3.0
            if dist < 3.0:
                # Smooth overlap measure
                overlap_depth = 3.0 - dist
                penalty += np.exp(beta * overlap_depth)
    return penalty

def _objective(params, outer_center, outer_angle, beta=20.0):
    """
    Objective function: minimize outer side length + overlap penalty.
    params: flat array [x_0, y_0, ..., x_{n-1}, y_{n-1}, ang_0, ..., ang_{n-1}]
    """
    n = N
    pos = params[:2*n].reshape(n, 2)
    ang = params[2*n:]
    
    all_verts = _hex_vertices_batch(pos, ang, side=1.0)
    
    cx, cy = outer_center
    dx = all_verts[:, 0] - cx
    dy = all_verts[:, 1] - cy
    
    a = np.radians(outer_angle)
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    c1 = lx
    c2 = -lx
    c3 = lx + ly / sqrt3
    c4 = -lx + ly / sqrt3
    c5 = -lx - ly / sqrt3
    c6 = lx - ly / sqrt3
    
    all_c = np.vstack([c1, c2, c3, c4, c5, c6])
    max_c = np.max(all_c)
    
    # Add overlap penalty
    penalty = _overlap_penalty_smooth(pos, ang, beta)
    
    return float(max_c) + 5.0 * penalty

def _local_search(pos, ang, outer_center, outer_angle, step, min_step, time_limit, max_iterations=5000):
    """Perform local search with decreasing step sizes."""
    n = N
    best_side = _min_outer_side(pos, ang, outer_center, outer_angle)
    best_pos = pos.copy()
    best_ang = ang.copy()
    iterations = 0
    
    while step >= min_step and time.time() < time_limit and iterations < max_iterations:
        iterations += 1
        improved = False
        
        # Try random perturbations
        for _ in range(3):
            if time.time() > time_limit:
                break
            
            new_pos = best_pos.copy()
            new_ang = best_ang.copy()
            
            # Perturb a random hexagon
            i = np.random.randint(n)
            new_pos[i, 0] += np.random.uniform(-step, step)
            new_pos[i, 1] += np.random.uniform(-step, step)
            new_ang[i] += np.random.uniform(-step, step)
            
            # Check overlaps
            if not _check_no_overlap_fast(new_pos, new_ang):
                continue
            
            new_side = _min_outer_side(new_pos, new_ang, outer_center, outer_angle)
            if new_side < best_side - 1e-10:
                best_pos = new_pos
                best_ang = new_ang
                best_side = new_side
                improved = True
        
        if not improved:
            step *= 0.8
    
    return best_pos, best_ang, best_side

def _simulated_annealing(pos, ang, outer_center, outer_angle, time_limit, n_iter=20000):
    """Simulated annealing to find good packing."""
    n = N
    current_pos = pos.copy()
    current_ang = ang.copy()
    current_side = _min_outer_side(current_pos, current_ang, outer_center, outer_angle)
    
    best_pos = current_pos.copy()
    best_ang = current_ang.copy()
    best_side = current_side
    
    T = 0.5  # Initial temperature
    cooling = 0.9995
    
    for i in range(n_iter):
        if time.time() > time_limit:
            break
        
        new_pos = current_pos.copy()
        new_ang = current_ang.copy()
        
        # Perturb a random hexagon
        idx = np.random.randint(n)
        new_pos[idx, 0] += np.random.normal(0, T)
        new_pos[idx, 1] += np.random.normal(0, T)
        new_ang[idx] += np.random.normal(0, T * 0.5)
        
        # Check overlaps
        if not _check_no_overlap_fast(new_pos, new_ang):
            T *= cooling
            continue
        
        new_side = _min_outer_side(new_pos, new_ang, outer_center, outer_angle)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_side - current_side
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            current_pos = new_pos
            current_ang = new_ang
            current_side = new_side
            
            if new_side < best_side:
                best_pos = new_pos.copy()
                best_ang = new_ang.copy()
                best_side = new_side
        
        T *= cooling
    
    return best_pos, best_ang, best_side

def construct():
    global START_TIME, DEADLINE
    
    # Outer hexagon center and angle (fixed at origin, 0 degrees for simplicity)
    outer_cx, outer_cy = 0.0, 0.0
    outer_angle = 0.0
    
    # Phase 1: Get initial packing
    pos, ang = _initial_packing()
    best_pos, best_ang = pos.copy(), ang.copy()
    best_side = _min_outer_side(pos, ang, (outer_cx, outer_cy), outer_angle)
    
    # Phase 2: Local search
    if time.time() < DEADLINE - 30.0:
        pos, ang, side = _local_search(
            pos, ang, (outer_cx, outer_cy), outer_angle,
            step=0.3, min_step=0.001, time_limit=time.time() + 20.0, max_iterations=3000
        )
        if side < best_side:
            best_pos, best_ang, best_side = pos.copy(), ang.copy(), side
    
    # Phase 3: Simulated annealing
    if time.time() < DEADLINE - 15.0:
        pos, ang, side = _simulated_annealing(
            pos, ang, (outer_cx, outer_cy), outer_angle,
            time_limit=time.time() + 10.0, n_iter=15000
        )
        if side < best_side:
            best_pos, best_ang, best_side = pos.copy(), ang.copy(), side
    
    # Phase 4: Fine local search
    if time.time() < DEADLINE - 5.0:
        pos, ang, side = _local_search(
            pos, ang, (outer_cx, outer_cy), outer_angle,
            step=0.05, min_step=0.0001, time_limit=time.time() + 5.0, max_iterations=2000
        )
        if side < best_side:
            best_pos, best_ang, best_side = pos.copy(), ang.copy(), side
    
    # Final verification
    inner_list = [[float(best_pos[i][0]), float(best_pos[i][1]), float(best_ang[i])] for i in range(N)]
    
    # Recompute the exact side length
    best_side = _min_outer_side(best_pos, best_ang, (outer_cx, outer_cy), outer_angle)
    
    if not _check_validity(inner_list, (outer_cx, outer_cy), best_side, outer_angle):
        # Fallback: use initial packing
        pos_init, ang_init = _initial_packing()
        inner_list = [[float(pos_init[i][0]), float(pos_init[i][1]), float(ang_init[i])] for i in range(N)]
        best_side = _min_outer_side(pos_init, ang_init, (outer_cx, outer_cy), outer_angle)
        
        if not _check_validity(inner_list, (outer_cx, outer_cy), best_side, outer_angle):
            # Last resort: spread them out
            inner_list = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner_list, (0.0, 0.0), 40.0, 0.0
    
    return inner_list, (float(outer_cx), float(outer_cy)), float(best_side), float(outer_angle)
