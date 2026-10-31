"""
Force-directed granular packing + discrete polyhex initialization + continuous refinement.

Strategy:
1. Generate diverse initial configurations (polyhex clusters + random perturbations).
2. Run force-directed simulation with repulsive potentials and soft boundary constraint
   to dynamically compress clusters into dense equilibrium states.
3. Refine with Nelder-Mead continuous optimization over rotation, translation, and
   individual hexagon positions/angles.
4. Return the best valid configuration found within the time budget.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 11

# ---- Geometry helpers ------------------------------------------------------

def _hex_vertices(cx, cy, s, ang_deg):
    """Return 6 vertices of a regular hexagon (flat-top at ang=0)."""
    a = math.radians(ang_deg)
    angs = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])

def _all_vertices(positions, angles, s=1.0):
    """Return all 6*n vertices of n hexagons."""
    n = len(positions)
    v = np.empty((n * 6, 2))
    for i in range(n):
        v[i*6:(i+1)*6] = _hex_vertices(positions[i][0], positions[i][1], s, angles[i])
    return v

def _min_enclosing_hex_side(positions, angles, phi_rad, center=(0.0, 0.0)):
    """
    Compute the exact side length of the smallest regular hexagon with rotation phi_rad
    and center `center` that encloses all given unit hexagons.
    """
    verts = _all_vertices(positions, angles, 1.0) - np.array(center)
    thetas = phi_rad + np.array([(2*k + 1) * np.pi / 6 for k in range(6)])
    max_proj = 0.0
    for t in thetas:
        proj = verts[:, 0] * math.cos(t) + verts[:, 1] * math.sin(t)
        mp = np.max(proj)
        if mp > max_proj:
            max_proj = mp
    return max_proj / (math.sqrt(3) / 2)

def _smooth_side(positions, angles, x, beta=30.0):
    """
    Smooth surrogate for outer side length.
    x = [phi, tx, ty]: outer rotation, and translation of the cluster.
    """
    phi, tx, ty = x[0], x[1], x[2]
    verts = _all_vertices(positions, angles, 1.0)
    verts = verts - np.array([tx, ty])
    thetas = phi + np.array([(2*k + 1) * np.pi / 6 for k in range(6)])
    
    projs = np.zeros((6, len(verts)))
    for k, t in enumerate(thetas):
        projs[k] = verts[:, 0] * math.cos(t) + verts[:, 1] * math.sin(t)
    
    m = np.max(projs, axis=1)
    lse_per_dir = m + (1.0 / beta) * np.log(np.sum(np.exp(projs - m[:, None]), axis=1))
    
    m2 = np.max(lse_per_dir)
    lse_final = m2 + (1.0 / beta) * math.log(np.sum(np.exp((lse_per_dir - m2) * beta)))
    return lse_final / (math.sqrt(3) / 2)

# ---- SAT overlap check -----------------------------------------------------

def _sat_gap(p1, a1, p2, a2, s=1.0):
    """Compute minimum gap between two hexagons using SAT. Negative means overlap."""
    v1 = _hex_vertices(p1[0], p1[1], s, a1)
    v2 = _hex_vertices(p2[0], p2[1], s, a2)
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    min_gap = float('inf')
    for e in list(edges1) + list(edges2):
        nv = np.array([-e[1], e[0]])
        n = np.linalg.norm(nv)
        if n < 1e-12:
            continue
        nv /= n
        proj1 = v1 @ nv
        proj2 = v2 @ nv
        gap = min(np.max(proj1) - np.min(proj2), np.max(proj2) - np.min(proj1))
        if gap < min_gap:
            min_gap = gap
    return min_gap

def _check_disjoint(positions, angles, tol=1e-9):
    """Check that all hexagons are pairwise disjoint (non-overlapping)."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_gap(positions[i], angles[i], positions[j], angles[j]) < -tol:
                return False
    return True

# ---- Polyhex generation via BFS on axial grid ------------------------------

def _axial_to_xy(q, r):
    """Convert axial coordinates to Cartesian for flat-top hexagons, side 1."""
    s3 = math.sqrt(3)
    return (1.5 * q, s3 * (r + q / 2.0))

def _generate_polyhexes(n_cells=11):
    """
    Generate all free polyhexes with n_cells cells using BFS.
    Returns a list of sets of axial coordinates.
    """
    directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    def normalize(cells):
        cells = list(cells)
        min_q = min(c[0] for c in cells)
        min_r = min(c[1] for c in cells)
        translated = [(c[0] - min_q, c[1] - min_r) for c in cells]
        return frozenset(translated)
    
    def canonical(cells):
        cells_list = list(cells)
        forms = set()
        for rot in range(6):
            c = list(cells_list)
            for _ in range(rot):
                c = [(-r, q + r) for q, r in c]
            forms.add(normalize(c))
            c_ref = [(-q - r, r) for q, r in c]
            forms.add(normalize(c_ref))
        return min(forms)
    
    seen_canonical = set()
    polyhexes = []
    initial = frozenset([(0, 0)])
    
    def bfs_expand(current_set, depth, target):
        if depth == target:
            canon = canonical(current_set)
            if canon not in seen_canonical:
                seen_canonical.add(canon)
                polyhexes.append(set(canon))
            return
        
        boundary = set()
        for q, r in current_set:
            for dq, dr in directions:
                pos = (q + dq, r + dr)
                if pos not in current_set:
                    boundary.add(pos)
        
        for pos in boundary:
            new_set = frozenset(current_set | {pos})
            bfs_expand(new_set, depth + 1, target)
    
    bfs_expand(initial, 1, n_cells)
    return polyhexes

def _polyhex_to_positions(cells):
    """Convert a set of axial coordinates to Cartesian positions (flat-top, side 1)."""
    return [_axial_to_xy(q, r) for q, r in cells]

# ---- Force-directed granular packing simulation ----------------------------

def _force_directed_pack(positions, angles, target_side, max_steps=2000, dt=0.01, deadline=None):
    """
    Run force-directed simulation to compress hexagons while maintaining non-overlap.
    
    Uses:
    - Pairwise repulsive potential when hexagons overlap
    - Soft boundary constraint to keep hexagons inside the target outer hexagon
    - Gentle attraction toward center to encourage compaction
    
    Returns refined positions and angles.
    """
    pos = np.array(positions, dtype=float)  # (N, 2)
    ang = np.array(angles, dtype=float)     # (N,) in degrees
    s3 = math.sqrt(3)
    
    # Precompute hexagon vertices for each hexagon (updated each step)
    def get_verts(p, a_deg):
        a_rad = math.radians(a_deg)
        angs = a_rad + np.arange(6) * np.pi / 3
        return p[None, :] + np.column_stack([math.cos(angs), math.sin(angs)]) * 1.0
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        forces = np.zeros_like(pos)
        torques = np.zeros(N)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                v1 = get_verts(pos[i], ang[i])
                v2 = get_verts(pos[j], ang[j])
                
                # SAT check for overlap and compute penetration
                edges1 = np.roll(v1, -1, axis=0) - v1
                edges2 = np.roll(v2, -1, axis=0) - v2
                min_pen = 0.0
                min_normal = None
                
                for e in list(edges1) + list(edges2):
                    nv = np.array([-e[1], e[0]])
                    n = np.linalg.norm(nv)
                    if n < 1e-12:
                        continue
                    nv = nv / n
                    proj1 = v1 @ nv
                    proj2 = v2 @ nv
                    overlap = min(np.max(proj1) - np.min(proj2), np.max(proj2) - np.min(proj1))
                    if overlap < min_pen:
                        min_pen = overlap
                        min_normal = -nv if np.dot(pos[i] - pos[j], nv) > 0 else nv
                
                if min_pen < 0:  # Overlap
                    pen = -min_pen
                    # Repulsive force proportional to penetration
                    f_mag = 10.0 * pen
                    forces[i] += f_mag * min_normal
                    forces[j] -= f_mag * min_normal
                    # Small torque to help separate
                    torques[i] += 0.1 * pen * (np.random.random() - 0.5)
                    torques[j] -= 0.1 * pen * (np.random.random() - 0.5)
        
        # Boundary constraint: keep all vertices inside outer hexagon
        # Outer hexagon centered at origin with side length target_side, rotation 0
        outer_apothem = target_side * s3 / 2.0
        outer_normals = np.array([[(2*k + 1) * math.pi / 6 for k in range(6)]])
        outer_normals = np.array([np.array([math.cos(t), math.sin(t)]) for t in outer_normals[0]])
        
        for i in range(N):
            v = get_verts(pos[i], ang[i])
            for nk in range(6):
                proj = v @ outer_normals[nk]
                max_p = np.max(proj)
                if max_p > outer_apothem:
                    # Push back
                    pen = max_p - outer_apothem
                    f_mag = 5.0 * pen
                    forces[i] -= f_mag * outer_normals[nk]
        
        # Gentle centering force to encourage compaction
        for i in range(N):
            forces[i] += -0.5 * pos[i]
        
        # Update positions and angles
        for i in range(N):
            pos[i] += dt * forces[i]
            ang[i] += dt * torques[i]
            # Keep angles bounded
            ang[i] = ang[i] % 360.0
        
        # Damping: scale down forces over time to settle
        if step % 200 == 199:
            dt *= 0.8
    
    return [tuple(p) for p in pos], list(ang)

def _compress_with_force(positions, angles, initial_side, deadline):
    """
    Iteratively reduce target side length using force-directed packing.
    Returns the best (positions, angles, side) found.
    """
    pos = list(positions)
    ang = list(angles)
    best_side = initial_side
    best_pos = list(positions)
    best_ang = list(angles)
    
    # Try reducing side length in steps
    side = initial_side
    while side > 1.0 and time.time() < deadline:
        # Attempt to pack at this side length
        new_pos, new_ang = _force_directed_pack(
            pos, ang, side, 
            max_steps=500, dt=0.02, 
            deadline=deadline
        )
        
        # Check if result is valid
        if _check_disjoint(new_pos, new_ang):
            actual_side = _min_enclosing_hex_side(new_pos, new_ang, 0.0, (0.0, 0.0))
            if actual_side <= side * 1.001:
                pos = new_pos
                ang = new_ang
                best_pos = list(pos)
                best_ang = list(ang)
                best_side = actual_side
                side *= 0.98  # Reduce target
            else:
                # Didn't fit, back off slightly
                side *= 1.02
                if side > initial_side:
                    break
        else:
            # Invalid, back off
            side *= 1.05
            if side > initial_side:
                break
    
    return best_pos, best_ang, best_side

# ---- Continuous refinement -------------------------------------------------

def _refine_continuous(positions, angles, phi_init, center_init, deadline):
    """
    Refine configuration using Nelder-Mead over [phi, tx, ty, ...individual adjustments].
    """
    N_hex = len(positions)
    
    def obj_full(x):
        phi = x[0]
        tx = x[1]
        ty = x[2]
        # Individual angle adjustments (small)
        adj_angles = list(angles)
        for i in range(N_hex):
            adj_angles[i] = angles[i] + x[3 + i] * 0.1  # Small perturbation
        
        adj_pos = [(positions[i][0] + tx, positions[i][1] + ty) for i in range(N_hex)]
        
        # Smooth side
        side = _smooth_side(adj_pos, adj_angles, np.array([phi, 0, 0]), beta=25.0)
        
        # Overlap penalty
        overlap_pen = 0.0
        for i in range(N_hex):
            for j in range(i + 1, N_hex):
                gap = _sat_gap(adj_pos[i], adj_angles[i], adj_pos[j], adj_angles[j])
                if gap < 0:
                    overlap_pen += 100.0 * gap * gap
        
        return side + overlap_pen
    
    x0 = np.array([phi_init, center_init[0], center_init[1]] + [0.0] * N_hex)
    
    try:
        res = minimize(obj_full, x0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True})
        if res.fun < float('inf'):
            phi_sol = res.x[0]
            tx_sol = res.x[1]
            ty_sol = res.x[2]
            new_angles = [angles[i] + res.x[3 + i] * 0.1 for i in range(N_hex)]
            new_pos = [(positions[i][0] + tx_sol, positions[i][1] + ty_sol) for i in range(N_hex)]
            
            if _check_disjoint(new_pos, new_angles):
                side = _min_enclosing_hex_side(new_pos, new_angles, phi_sol, (0.0, 0.0))
                return new_pos, new_angles, phi_sol, side
    except Exception:
        pass
    
    return None

# ---- Initial configuration generators --------------------------------------

def _generate_initial_configs(deadline):
    """
    Generate diverse initial configurations for force-directed packing.
    Returns list of (positions, angles, initial_side).
    """
    configs = []
    
    # 1. Polyhex-based configurations
    try:
        t_start = time.time()
        polyhexes = _generate_polyhexes(N)
        elapsed = time.time() - t_start
        print(f"Generated {len(polyhexes)} polyhexes in {elapsed:.1f}s")
        
        # Sort by bounding box area
        def bbox_area(cells):
            xs = [_axial_to_xy(q, r)[0] for q, r in cells]
            ys = [_axial_to_xy(q, r)[1] for q, r in cells]
            return (max(xs) - min(xs)) * (max(ys) - min(ys))
        
        polyhexes.sort(key=bbox_area)
        
        # Take top 30 for initial configs
        for cells in polyhexes[:30]:
            if time.time() > deadline - 60:
                break
            positions = _polyhex_to_positions(cells)
            angles = [0.0] * N
            side = _min_enclosing_hex_side(positions, angles, 0.0, (0.0, 0.0))
            configs.append((positions, angles, side))
    except Exception as e:
        print(f"Polyhex generation failed: {e}")
    
    # 2. Hand-crafted compact clusters
    # 3-4-3 arrangement
    c1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
          (2,0), (1,1), (0,2), (-1,2)]
    if time.time() < deadline - 60:
        positions = _polyhex_to_positions(c1)
        angles = [0.0] * N
        side = _min_enclosing_hex_side(positions, angles, 0.0, (0.0, 0.0))
        configs.append((positions, angles, side))
    
    # 2-3-3-2 arrangement
    c2 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
          (2,0), (0,2), (-2,2), (0,-2)]
    if time.time() < deadline - 60:
        positions = _polyhex_to_positions(c2)
        angles = [0.0] * N
        side = _min_enclosing_hex_side(positions, angles, 0.0, (0.0, 0.0))
        configs.append((positions, angles, side))
    
    # 3. Random compact clusters
    np.random.seed(42)
    for trial in range(10):
        if time.time() > deadline - 60:
            break
        
        # Start with hexagonal lattice and perturb
        positions = []
        angles = [0.0] * N
        # Place in a compact hexagonal pattern
        # Row 0: 3 hexagons
        # Row 1: 3 hexagons  
        # Row 2: 3 hexagons
        # Row 3: 2 hexagons
        s3 = math.sqrt(3)
        for i in range(N):
            row = i // 3
            col = i % 3
            x = 1.5 * col + (0.75 if row % 2 == 1 else 0)
            y = s3 / 2 * row
            # Add small random perturbation
            x += np.random.uniform(-0.2, 0.2)
            y += np.random.uniform(-0.2, 0.2)
            positions.append((x, y))
        
        # Center
        cx = np.mean([p[0] for p in positions])
        cy = np.mean([p[1] for p in positions])
        positions = [(p[0] - cx, p[1] - cy) for p in positions]
        
        # Check if disjoint
        if _check_disjoint(positions, angles):
            side = _min_enclosing_hex_side(positions, angles, 0.0, (0.0, 0.0))
            configs.append((positions, angles, side))
    
    # 4. Circular arrangement with random angles
    for trial in range(5):
        if time.time() > deadline - 60:
            break
        
        positions = []
        angles = []
        for i in range(N):
            theta = 2 * math.pi * i / N
            r = 2.5
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            positions.append((x, y))
            angles.append(np.random.uniform(0, 60))
        
        if _check_disjoint(positions, angles):
            side = _min_enclosing_hex_side(positions, angles, 0.0, (0.0, 0.0))
            configs.append((positions, angles, side))
    
    # Sort by initial side length
    configs.sort(key=lambda x: x[2])
    
    return configs

# ---- Main construct --------------------------------------------------------

def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # Safety margin before 150s kill
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_center = (0.0, 0.0)
    
    # Generate initial configurations
    print("Generating initial configurations...")
    configs = _generate_initial_configs(deadline)
    print(f"Have {len(configs)} initial configurations")
    
    if not configs:
        # Fallback: linear arrangement
        best_positions = [(1.5 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        best_angles = [0.0] * N
        best_phi = 0.0
        best_center = (0.0, 0.0)
        best_side = _min_enclosing_hex_side(best_positions, best_angles, best_phi, best_center)
        print(f"Fallback side: {best_side}")
        return [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)], best_center, best_side, math.degrees(best_phi)
    
    # Process each configuration
    for idx, (positions, angles, init_side) in enumerate(configs):
        if time.time() > deadline - 30:
            break
        
        print(f"Processing config {idx+1}/{len(configs)}: init_side={init_side:.4f}")
        
        # Step 1: Force-directed compression
        if time.time() < deadline - 20:
            pos_fd, ang_fd, side_fd = _compress_with_force(
                positions, angles, init_side, deadline - 10
            )
            
            if side_fd < best_side and _check_disjoint(pos_fd, ang_fd):
                best_positions = pos_fd
                best_angles = ang_fd
                best_side = side_fd
                best_phi = 0.0
                best_center = (0.0, 0.0)
                print(f"  After FD: side={best_side:.4f}")
        
        # Step 2: Continuous refinement
        if time.time() < deadline - 10:
            if best_positions is not None:
                result = _refine_continuous(
                    best_positions, best_angles, best_phi, best_center, deadline
                )
                if result is not None:
                    new_pos, new_ang, new_phi, new_side = result
                    if new_side < best_side:
                        best_positions = new_pos
                        best_angles = new_ang
                        best_phi = new_phi
                        best_side = new_side
                        print(f"  After refinement: side={best_side:.4f}")
        
        # Step 3: Fine-tune rotation and center
        if time.time() < deadline - 5 and best_positions is not None:
            # Coarse rotation search
            for phi_deg in np.arange(0, 60, 3.0):
                if time.time() > deadline - 5:
                    break
                phi = math.radians(phi_deg)
                side = _min_enclosing_hex_side(best_positions, best_angles, phi, best_center)
                if side < best_side:
                    best_side = side
                    best_phi = phi
            
            # Fine rotation search around best
            if best_phi > 0:
                for phi_deg in np.arange(best_phi * 180 / math.pi - 2, 
                                         best_phi * 180 / math.pi + 2.01, 0.1):
                    if time.time() > deadline - 3:
                        break
                    phi = math.radians(phi_deg)
                    side = _min_enclosing_hex_side(best_positions, best_angles, phi, best_center)
                    if side < best_side:
                        best_side = side
                        best_phi = phi
            
            # Center optimization
            for dx in np.arange(-0.3, 0.301, 0.05):
                for dy in np.arange(-0.3, 0.301, 0.05):
                    if time.time() > deadline - 2:
                        break
                    center_try = (dx, dy)
                    side = _min_enclosing_hex_side(best_positions, best_angles, best_phi, center_try)
                    if side < best_side:
                        best_side = side
                        best_center = center_try
    
    # Final verification
    if best_positions is None or not _check_disjoint(best_positions, best_angles):
        best_positions = [(1.5 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        best_angles = [0.0] * N
        best_phi = 0.0
        best_center = (0.0, 0.0)
        best_side = _min_enclosing_hex_side(best_positions, best_angles, best_phi, best_center)
    
    # Add tiny epsilon for safety
    best_side *= (1.0 + 1e-9)
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    print(f"Final: side={best_side:.6f}, angle={math.degrees(best_phi):.3f}")
    return inner, best_center, best_side, math.degrees(best_phi)

if __name__ == "__main__":
    result = construct()
    print(f"Result: side={result[2]}, angle={result[3]}")
