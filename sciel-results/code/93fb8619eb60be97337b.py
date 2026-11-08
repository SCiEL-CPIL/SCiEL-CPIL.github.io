import numpy as np
import time
from itertools import combinations
from scipy.optimize import linprog

N = 11
DEADLINE = 145.0  # safety margin before 150s kill

def _get_valid_fallback():
    """Return a known valid, non-overlapping configuration."""
    # Place hexagons in a grid with spacing 4.0 (safe for side 1)
    centers = []
    angles = [0.0] * N
    x = 0
    y = 0
    for i in range(N):
        centers.append((x, y))
        x += 4.0
        if i % 2 == 1:
            y += 4.0
            x = -4.0 + (i // 2) * 8.0 # Just a rough spiral/grid
    
    # Calculate enclosing hexagon
    side, center, angle = _optimize_outer(centers, angles)
    return centers, angles, center, side, angle

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
    # A_matrix shape (6, 3)
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
    
    # Verify validity (containment is guaranteed by construction of enclosing side, 
    # but check just in case of numerical issues)
    if _check_validity(centers, inner_angles, center, side, angle):
        return side, (centers, inner_angles, center, side, angle)
    
    return float('inf'), None

def _is_valid_configuration(centers, inner_angles, outer_center, outer_side, outer_angle):
    """Strictly check if the configuration is valid (non-overlapping and contained)."""
    return _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle)

def _force_directed_relax(centers, angles, max_iter=5000, tol=1e-6):
    """Force-directed relaxation to reduce overlaps and compact the arrangement."""
    centers = [list(c) for c in centers]
    angles = list(angles)
    
    sqrt3 = np.sqrt(3)
    
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
    
    for iteration in range(max_iter):
        forces = [[0.0, 0.0] for _ in range(N)]
        torques = [0.0] * N
        max_penetration = 0.0
        
        for i in range(N):
            for j in range(i + 1, N):
                v1 = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
                v2 = _hex_vertices(centers[j][0], centers[j][1], 1.0, angles[j])
                
                axes = get_axes(v1) + get_axes(v2)
                min_overlap = float('inf')
                best_axis = None
                
                for ax, ay in axes:
                    proj1 = v1[:, 0] * ax + v1[:, 1] * ay
                    proj2 = v2[:, 0] * ax + v2[:, 1] * ay
                    overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
                    if overlap > max_penetration:
                        max_penetration = overlap
                    if overlap > 0 and overlap < min_overlap:
                        min_overlap = overlap
                        best_axis = (ax, ay)
                
                if min_overlap > 0 and best_axis is not None:
                    ax, ay = best_axis
                    # Push apart along the axis of minimum overlap
                    forces[i][0] -= ax * min_overlap
                    forces[i][1] -= ay * min_overlap
                    forces[j][0] += ax * min_overlap
                    forces[j][1] += ay * min_overlap
    
        if max_penetration < tol:
            break
    
        # Apply forces with damping
        step_size = 0.5
        for i in range(N):
            centers[i][0] += forces[i][0] * step_size
            centers[i][1] += forces[i][1] * step_size
    
    return centers, angles

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
        # 1. Move a random hexagon
        idx = np.random.randint(N)
        delta_x = np.random.normal(0, 0.1)
        delta_y = np.random.normal(0, 0.1)
        new_centers[idx][0] += delta_x
        new_centers[idx][1] += delta_y
        
        # 2. Rotate a random hexagon
        if np.random.rand() < 0.5:
            rot_idx = np.random.randint(N)
            delta_ang = np.random.normal(0, 5.0) # degrees
            new_angles[rot_idx] += delta_ang
            
        # 3. Global translation
        if np.random.rand() < 0.1:
            dx = np.random.normal(0, 0.05)
            dy = np.random.normal(0, 0.05)
            for i in range(N):
                new_centers[i][0] += dx
                new_centers[i][1] += dy
                
        # 4. Global rotation
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
            # Reject invalid moves
            T *= cooling_rate
            continue
            
        # Calculate new cost
        new_side, new_center, new_angle = _optimize_outer(new_centers, new_angles)
        
        if not np.isfinite(new_side):
            T *= cooling_rate
            continue
            
        delta = new_side - current_side
        
        if delta < 0:
            # Accept
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
            # Accept with probability
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
    
    # Generate initial hexagonal lattice arrangement
    d1 = np.sqrt(3)
    d2 = 2 * np.sqrt(3)
    
    # First ring
    first_ring = []
    for i in range(6):
        angle = np.pi / 3 * i
        first_ring.append((d1 * np.cos(angle), d1 * np.sin(angle)))
    
    # Second ring positions (12 positions)
    second_ring_positions = []
    for i in range(6):
        angle = np.pi / 3 * i
        second_ring_positions.append((d2 * np.cos(angle), d2 * np.sin(angle)))
        angle2 = np.pi / 3 * i + np.pi / 6
        second_ring_positions.append((d2 * np.cos(angle2), d2 * np.sin(angle2)))
        
    # We need 1 + 6 + 4 = 11 hexagons.
    # Choose 4 from the 12 second-ring positions.
    # Due to symmetry, we can just pick one representative set, e.g., indices 0, 2, 4, 6 (axial)
    # Or 0, 1, 2, 3?
    # Let's try a few symmetric subsets.
    
    # Subset 1: Axial (0, 2, 4, 6) - wait, indices in second_ring_positions list.
    # The list is built in a specific order.
    # Let's just pick the first 4? No, symmetry matters.
    # Let's try the subset that forms a symmetric shape.
    # Indices 0, 2, 4, 6 correspond to the "corners" of the hexagon?
    # Actually, let's just try all combinations of 4 from 12 is C(12,4) = 495.
    # This is fast enough for initialization.
    
    best_init_centers = None
    best_init_angles = None
    best_init_side = float('inf')
    
    # The previous initialization used distances that were too small (sqrt(3) instead of 2*sqrt(3) for touching edges in a lattice).
    # Let's use a valid lattice spacing.
    # For aligned hexagons, center-to-center distance for touching is 2*sqrt(3).
    d1 = 2 * np.sqrt(3)
    d2 = 4 * np.sqrt(3) # Next ring is further out
    
    # Recalculate rings with correct distances
    first_ring = []
    for i in range(6):
        angle = np.pi / 3 * i
        first_ring.append((d1 * np.cos(angle), d1 * np.sin(angle)))
        
    second_ring_positions = []
    for i in range(6):
        angle = np.pi / 3 * i
        second_ring_positions.append((d2 * np.cos(angle), d2 * np.sin(angle)))
        angle2 = np.pi / 3 * i + np.pi / 6
        second_ring_positions.append((d2 * np.cos(angle2), d2 * np.sin(angle2)))

    for subset in combinations(range(12), 4):
        if time.time() - start_time > 10:
            break
        centers = [(0.0, 0.0)] + first_ring + [second_ring_positions[i] for i in subset]
        inner_angles = [0.0] * N
        side, result = _try_arrangement(centers, inner_angles)
        if side < best_init_side:
            best_init_side = side
            best_init_centers = [list(c) for c in centers]
            best_init_angles = list(inner_angles)
            best_result = result

    if best_result is None:
        # Fallback: Use a very sparse grid to ensure validity
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        best_result = (fb_centers, fb_angles, fb_center, fb_side, fb_angle)
        best_init_centers = fb_centers
        best_init_angles = fb_angles
        best_init_side = fb_side

    # Run Force-Directed Relaxation first
    if best_init_centers and time.time() - start_time < DEADLINE - 10:
        relaxed_centers, relaxed_angles = _force_directed_relax(best_init_centers, best_init_angles)
        
        # Check if relaxed configuration is valid and better
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(relaxed_centers[i], relaxed_angles[i], relaxed_centers[j], relaxed_angles[j]):
                    break
        else:
            side, center, angle = _optimize_outer(relaxed_centers, relaxed_angles)
            if np.isfinite(side) and _check_validity(relaxed_centers, relaxed_angles, center, side, angle):
                if side < best_side:
                    best_side = side
                    best_result = (relaxed_centers, relaxed_angles, center, side, angle)
                    best_init_centers = relaxed_centers
                    best_init_angles = relaxed_angles
    
    # Then run Simulated Annealing from the best configuration found
    if best_init_centers and time.time() - start_time < DEADLINE - 10:
        sa_centers, sa_angles, sa_side, sa_center, sa_angle = _simulate_annealing(
            best_init_centers, best_init_angles, max_time=DEADLINE - time.time() - 10
        )
        
        # Verify SA result is valid before accepting
        if _is_valid_configuration(sa_centers, sa_angles, sa_center, sa_side, sa_angle):
            if sa_side < best_side:
                best_side = sa_side
                best_result = (sa_centers, sa_angles, sa_center, sa_side, sa_angle)
        else:
            # SA failed to find valid config or returned invalid one, keep previous best
            pass

    # Final verification and formatting
    if best_result is None:
        # Should not happen, but use fallback
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        best_result = (fb_centers, fb_angles, fb_center, fb_side, fb_angle)
        
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    # Final strict check
    if not _is_valid_configuration(centers, inner_angles, outer_center, outer_side, outer_angle):
        # If somehow invalid, use fallback
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        centers, inner_angles, outer_center, outer_side, outer_angle = fb_centers, fb_angles, fb_center, fb_side, fb_angle

    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
