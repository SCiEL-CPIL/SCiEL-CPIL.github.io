"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin before 150s kill
SQRT3 = np.sqrt(3.0)

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

def _all_inner_vertices(pos, ang):
    """Return (6*N, 2) array of all inner hexagon vertices."""
    n = len(pos)
    verts = np.empty((n * 6, 2))
    for i in range(n):
        v = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
        verts[i*6:(i+1)*6] = v
    return verts

def _min_side_analytic(all_verts, cx, cy, outer_angle_deg):
    """Compute minimum outer hex side length containing all_verts, centered at (cx,cy) with rotation."""
    # For a regular hexagon with side s, center (cx,cy), rotation a,
    # a point (px,py) is inside iff for all 6 face normals n_k:
    #   n_k · ((px,cy) - (cx,cy)) <= s
    # where n_k are unit normals to the 6 faces pointing outward.
    # So min s = max over all points and all 6 normals of n_k · (p - center)
    a = np.radians(outer_angle_deg)
    # Face normals for a hexagon with vertices at angles a, a+60, ..., a+300:
    # The face between vertex at angle a+(k-1)*60 and a+k*60 has outward normal at angle a+(k-0.5)*60
    # Actually, for vertices at angles 0,60,...,300 (flat-top), the face normals are at 30,90,150,210,270,330
    # More generally, face normal k is at angle a + (k + 0.5) * 60 degrees
    normals = np.array([
        [np.cos(a + (k + 0.5) * np.pi / 3), np.sin(a + (k + 0.5) * np.pi / 3)]
        for k in range(6)
    ])  # (6, 2)
    
    # Relative positions
    rel = all_verts - np.array([cx, cy])  # (M, 2)
    # Project onto each normal
    projections = rel @ normals.T  # (M, 6)
    # The required side is the max projection over all points and all normals
    s = projections.max()
    return max(s, 1.0)  # side must be at least 1

def _hexagon_overlap_fast(verts1, verts2, tol=1e-9):
    """Check if two convex hexagons overlap using SAT."""
    axes = []
    for verts in [verts1, verts2]:
        for i in range(6):
            v1 = verts[i]
            v2 = verts[(i + 1) % 6]
            edge = v2 - v1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append([nx/norm, ny/norm])
    
    for axis in axes:
        proj1 = verts1 @ axis
        proj2 = verts2 @ axis
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True

def _check_all_valid(pos, ang, outer_side, ocx, ocy, oang):
    """Full validity check: no overlaps, all inside outer hex."""
    n = len(pos)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i]))
    
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagon_overlap_fast(verts_list[i], verts_list[j]):
                return False
    
    # Check containment
    all_verts = np.vstack(verts_list)
    s_needed = _min_side_analytic(all_verts, ocx, ocy, oang)
    return s_needed <= outer_side + 1e-6

def _initial_packing():
    """Create a good initial hexagonal grid packing of N unit hexagons."""
    positions = []
    angles = []
    # Generate hex grid positions, then pick N closest to center
    sqrt3 = np.sqrt(3)
    # Hex grid: neighbors at distance 2 in 6 directions
    # Generate a larger grid and pick N closest to origin
    candidates = []
    for row in range(-4, 5):
        for col in range(-4, 5):
            x = col * 2.0 + (row % 2) * 1.0
            y = row * sqrt3
            candidates.append([x, y])
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)[:N]
    positions = candidates[idx]
    angles = np.zeros(N)
    return positions, angles

def construct():
    global DEADLINE
    
    np.random.seed(123)
    
    # Initial packing
    pos, ang = _initial_packing()
    n = len(pos)
    
    # Outer hexagon: center at centroid, angle 0
    centroid = pos.mean(axis=0)
    ocx, ocy = float(centroid[0]), float(centroid[1])
    oang = 0.0
    
    # Compute initial side
    all_verts = _all_inner_vertices(pos, ang)
    current_side = _min_side_analytic(all_verts, ocx, ocy, oang)
    
    best_side = current_side
    best_pos = pos.copy()
    best_ang = ang.copy()
    best_ocx, best_ocy, best_oang = ocx, ocy, oang
    
    # Simulated annealing
    T = 0.5  # initial temperature
    T_min = 0.001
    cooling = 0.995
    max_iter = 500000
    
    iter_count = 0
    stall = 0
    
    while time.time() < DEADLINE and iter_count < max_iter:
        iter_count += 1
        
        # Pick random hexagon to move
        i = np.random.randint(n)
        
        # Random perturbation
        scale = T * 2.0
        dx = np.random.normal(0, scale)
        dy = np.random.normal(0, scale)
        da = np.random.normal(0, scale * 5.0)  # angle perturbation in degrees
        
        new_pos = pos.copy()
        new_pos[i, 0] += dx
        new_pos[i, 1] += dy
        new_ang = ang.copy()
        new_ang[i] += da
        # Keep angles in [0, 360)
        new_ang[i] = new_ang[i] % 360.0
        
        # Quick check: does moved hexagon overlap with any other?
        new_verts_i = _hex_vertices(new_pos[i, 0], new_pos[i, 1], 1.0, new_ang[i])
        overlap = False
        for j in range(n):
            if j == i:
                continue
            other_verts = _hex_vertices(pos[j, 0], pos[j, 1], 1.0, ang[j])
            if _hexagon_overlap_fast(new_verts_i, other_verts):
                overlap = True
                break
        if overlap:
            T *= cooling
            continue
        
        # Compute new side
        new_all_verts = _all_inner_vertices(new_pos, new_ang)
        new_side = _min_side_analytic(new_all_verts, ocx, ocy, oang)
        
        # Acceptance criterion
        delta = new_side - current_side
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            pos = new_pos
            ang = new_ang
            current_side = new_side
            stall = 0
            if current_side < best_side:
                best_side = current_side
                best_pos = pos.copy()
                best_ang = ang.copy()
                best_ocx, best_ocy, best_oang = ocx, ocy, oang
        else:
            stall += 1
        
        T *= cooling
        
        # Periodically re-center outer hexagon
        if iter_count % 500 == 0:
            centroid = pos.mean(axis=0)
            ocx, ocy = float(centroid[0]), float(centroid[1])
            current_side = _min_side_analytic(_all_inner_vertices(pos, ang), ocx, ocy, oang)
            if current_side < best_side:
                best_side = current_side
                best_pos = pos.copy()
                best_ang = ang.copy()
                best_ocx, best_ocy, best_oang = ocx, ocy, oang
        
        if T < T_min:
            break
    
    # Final verification
    if not _check_all_valid(best_pos, best_ang, best_side, best_ocx, best_ocy, best_oang):
        # Fallback: use a safe linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2 + 1)][:N]
        return inner, (0.0, 0.0), 50.0, 0.0
    
    inner = [[float(best_pos[i, 0]), float(best_pos[i, 1]), float(best_ang[i])] for i in range(N)]
    return inner, (float(best_ocx), float(best_ocy)), float(best_side), float(best_oang)

# EVOLVE-BLOCK-END
