import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3


def _hex_edge_normals(angle_deg):
    """Return 6 edge normal directions for a hexagon with given rotation."""
    a = math.radians(angle_deg)
    return a + math.pi/6.0 + np.arange(6) * math.pi/3.0


def _hex_support(angle_deg, theta):
    """Support function of unit hexagon (centered at origin, rotated by angle_deg) in direction theta."""
    a = math.radians(angle_deg)
    rel = (theta - a) % (2 * math.pi)
    k = round(rel / (math.pi / 3.0))
    delta = abs(rel - k * math.pi / 3.0)
    return math.cos(delta)


def _min_separation_same_angle(d, angle_deg):
    """Minimum separation between two unit hexagons with same rotation.
    Positive = separated, negative = overlapping.
    Returns the minimum clearance over all 6 edge-normal directions."""
    a = math.radians(angle_deg)
    dx, dy = d
    min_sep = float('inf')
    for k in range(6):
        theta = a + math.pi/6.0 + k * math.pi/3.0
        nx, ny = math.cos(theta), math.sin(theta)
        proj = abs(dx * nx + dy * ny)
        s = _hex_support(angle_deg, theta)
        sep = proj - 2.0 * s
        if sep < min_sep:
            min_sep = sep
    return min_sep


def _separation_direction(d, angle_deg):
    """Return the direction of minimum separation (unit vector from i to j) and the min sep value."""
    a = math.radians(angle_deg)
    dx, dy = d
    min_sep = float('inf')
    best_dir = None
    for k in range(6):
        theta = a + math.pi/6.0 + k * math.pi/3.0
        nx, ny = math.cos(theta), math.sin(theta)
        proj = dx * nx + dy * ny
        s = _hex_support(angle_deg, theta)
        sep = proj - 2.0 * s
        if sep < min_sep:
            min_sep = sep
            best_dir = (nx, ny)
    if best_dir is None:
        return (1.0, 0.0), 0.0
    return best_dir, min_sep


def _hex_vertices_batch(positions, angle_deg):
    """Vertices for multiple hexagons at given positions, all with same rotation."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    return positions[:, np.newaxis, :] + base[np.newaxis, :, :]


def _enclosing_hex_side_lp(verts, outer_angle_deg):
    """Find minimum side length of regular hexagon with given orientation containing all vertices via LP."""
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    dots = verts @ normals.T  # (M, 6)
    
    A = np.zeros((2 * M * 6, 3))
    b = np.zeros(2 * M * 6)
    
    idx = 0
    for k in range(6):
        nk = normals[k]
        A[idx:idx+M, 0] = nk[0]
        A[idx:idx+M, 1] = nk[1]
        A[idx:idx+M, 2] = -1.0
        b[idx:idx+M] = dots[:, k]
        idx += M
        A[idx:idx+M, 0] = -nk[0]
        A[idx:idx+M, 1] = -nk[1]
        A[idx:idx+M, 2] = -1.0
        b[idx:idx+M] = -dots[:, k]
        idx += M
    
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            return res.x[2] * 2.0 / _SQRT3, (res.x[0], res.x[1])
    except Exception:
        pass
    
    # Fallback
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    return r * 2.0 / _SQRT3, (cx, cy)


def _enclosing_hex_side_fast(verts, outer_angle_deg):
    """Fast approximation: use centroid as center."""
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    return r * 2.0 / _SQRT3


def _spring_mass_simulate(contact_edges, n_nodes, inner_angle, max_iters=10000, lr=0.02):
    """Spring-mass simulation with rest lengths = edge-to-edge distance.
    
    contact_edges: list of (i, j) pairs representing contact graph
    Returns optimized positions.
    """
    # Rest length: for two touching unit hexagons with same orientation,
    # the center-to-center distance is 2 * apothem = sqrt(3)
    rest_length = _SQRT3 * 1.001  # small gap to ensure strict disjointness
    
    # Initialize positions on honeycomb lattice
    u = (_SQRT3, 0.0)
    v = (_SQRT3 / 2, 1.5)
    
    # Generate initial positions from honeycomb coordinates
    # For a compact cluster of 11: center + 6 ring + 4 more
    init_coords = [(0, 0),
                   (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                   (2, 0), (1, 1), (-1, 2), (-2, 0)]
    
    pos = np.array([(ti * u[0] + tj * v[0], ti * u[1] + tj * v[1]) 
                    for ti, tj in init_coords[:n_nodes]], dtype=float)
    
    # Precompute edge normals for inner hexagons
    inner_normals = _hex_edge_normals(inner_angle)
    inner_cos = np.cos(inner_normals)
    inner_sin = np.sin(inner_normals)
    inner_support = np.array([_hex_support(inner_angle, theta) for theta in inner_normals])
    
    best_pos = pos.copy()
    best_energy = float('inf')
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 30.0:
            break
        
        forces = np.zeros((n_nodes, 2))
        
        # Spring forces along contact edges
        for i, j in contact_edges:
            d = pos[j] - pos[i]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.array([1e-5, 0.0])
                dist = 1e-5
            direction = d / dist
            # Hooke's law: force proportional to displacement from rest length
            f = 2.0 * (dist - rest_length)
            forces[i] += f * direction
            forces[j] -= f * direction
        
        # Repulsive forces for non-contact pairs that are too close
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                # Skip if already in contact graph
                if (i, j) in contact_edges or (j, i) in contact_edges:
                    continue
                d = pos[j] - pos[i]
                dist = np.linalg.norm(d)
                if dist < rest_length * 1.1:
                    # Compute actual separation
                    min_sep = float('inf')
                    for k in range(6):
                        nx = inner_cos[k]
                        ny = inner_sin[k]
                        proj = abs(d[0] * nx + d[1] * ny)
                        s = inner_support[k]
                        sep = proj - 2.0 * s
                        if sep < min_sep:
                            min_sep = sep
                    
                    if min_sep < 0:
                        # Overlapping: strong repulsion
                        f = 50.0 * (-min_sep)
                        if dist > 1e-10:
                            direction = d / dist
                        else:
                            direction = np.array([1.0, 0.0])
                        forces[i] -= f * direction
                        forces[j] += f * direction
        
        # Gentle attraction toward center for compactness
        for i in range(n_nodes):
            forces[i] -= 0.05 * pos[i]
        
        # Update positions
        pos += lr * forces
        
        # Compute energy: sum of spring energies + overlap penalty
        energy = 0.0
        for i, j in contact_edges:
            d = pos[j] - pos[i]
            dist = np.linalg.norm(d)
            energy += 0.5 * (dist - rest_length) ** 2
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                d = pos[j] - pos[i]
                min_sep = float('inf')
                for k in range(6):
                    nx = inner_cos[k]
                    ny = inner_sin[k]
                    proj = abs(d[0] * nx + d[1] * ny)
                    s = inner_support[k]
                    sep = proj - 2.0 * s
                    if sep < min_sep:
                        min_sep = sep
                if min_sep < 0:
                    energy += 100.0 * min_sep ** 2
        
        # Compute enclosing hex side for tracking
        all_vertices = _hex_vertices_batch(pos, inner_angle).reshape(-1, 2)
        side = _enclosing_hex_side_fast(all_vertices, 0.0)
        total_score = side + energy
        
        if total_score < best_energy:
            best_energy = total_score
            best_pos = pos.copy()
        
        # Check convergence
        if it > 100 and np.abs(forces).max() < 1e-8:
            break
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            lr *= 0.95
    
    return best_pos


def _optimize_positions(positions, inner_angle, outer_angle, max_iters=500):
    """Local optimization of positions using Nelder-Mead."""
    pos_arr = np.array(positions)
    n = len(pos_arr)
    
    x0 = pos_arr.flatten()
    
    def _objective(params):
        positions = params.reshape(-1, 2)
        all_vertices = _hex_vertices_batch(positions, inner_angle).reshape(-1, 2)
        side = _enclosing_hex_side_fast(all_vertices, outer_angle)
        
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                di = positions[j] - positions[i]
                dist = _min_separation_same_angle(di, inner_angle)
                if dist < 0:
                    penalty += dist * dist * 100.0
        
        return side + penalty
    
    try:
        res = minimize(_objective, x0, method='Nelder-Mead',
                      options={'maxiter': max_iters, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        return res.x.reshape(-1, 2)
    except Exception:
        return pos_arr


def _optimize_outer_angle(positions, inner_angle, coarse_angles, fine_angles=None):
    """Optimize outer hexagon orientation."""
    pos_arr = np.array(positions)
    all_vertices = _hex_vertices_batch(pos_arr, inner_angle).reshape(-1, 2)
    
    best_side = float('inf')
    best_angle = coarse_angles[0]
    
    # Coarse search
    for angle in coarse_angles:
        side = _enclosing_hex_side_fast(all_vertices, angle)
        if side < best_side:
            best_side = side
            best_angle = angle
    
    # Fine search around best
    if fine_angles is None:
        fine_angles = list(np.arange(best_angle - 1.0, best_angle + 1.0, 0.1))
    
    for angle in fine_angles:
        side = _enclosing_hex_side_fast(all_vertices, angle)
        if side < best_side:
            best_side = side
            best_angle = angle
    
    return best_side, best_angle


def _high_precision_polish(positions, inner_angle, outer_angle):
    """High-precision polish using LP-based objective."""
    pos_arr = np.array(positions)
    n = len(pos_arr)
    
    x0 = pos_arr.flatten()
    
    def _objective(params):
        positions = params.reshape(-1, 2)
        all_vertices = _hex_vertices_batch(positions, inner_angle).reshape(-1, 2)
        side, _ = _enclosing_hex_side_lp(all_vertices, outer_angle)
        
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                di = positions[j] - positions[i]
                dist = _min_separation_same_angle(di, inner_angle)
                if dist < 0:
                    penalty += dist * dist * 1000.0
        
        return side + penalty
    
    try:
        res = minimize(_objective, x0, method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        return res.x.reshape(-1, 2)
    except Exception:
        return pos_arr


def _build_contact_graph(layout_idx):
    """Build contact graph for different cluster topologies."""
    # Layout 0: Center + 6 ring + 4 from second ring
    # Nodes: 0=center, 1-6=ring1, 7-10=ring2
    if layout_idx == 0:
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
            (1, 7), (2, 8), (3, 9), (4, 10)
        ]
        return edges
    elif layout_idx == 1:
        # Row-based 3-5-3
        # Row 1 (y=-1.5): nodes 0,1,2
        # Row 2 (y=0): nodes 3,4,5,6,7
        # Row 3 (y=1.5): nodes 8,9,10
        edges = [
            (0, 1), (1, 2),
            (3, 4), (4, 5), (5, 6), (6, 7),
            (8, 9), (9, 10),
            (0, 3), (1, 3), (1, 4), (2, 4), (2, 5),
            (3, 8), (4, 8), (4, 9), (5, 9), (5, 10), (6, 10)
        ]
        return edges
    elif layout_idx == 2:
        # 4-3-4 pattern
        # Row 1: 0,1,2,3
        # Row 2: 4,5,6
        # Row 3: 7,8,9,10
        edges = [
            (0, 1), (1, 2), (2, 3),
            (4, 5), (5, 6),
            (7, 8), (8, 9), (9, 10),
            (0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6),
            (4, 7), (4, 8), (5, 8), (5, 9), (6, 9), (6, 10)
        ]
        return edges
    elif layout_idx == 3:
        # Diamond: 1-3-5-3-1
        # Row 0: 0
        # Row 1: 1,2,3
        # Row 2: 4,5,6,7,8
        # Row 3: 9,10,11 -> wait, we only have 11 nodes
        # Let's do: 0, then 1,2,3, then 4,5,6,7, then 8,9,10
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (2, 3),
            (1, 4), (2, 4), (2, 5), (3, 5), (3, 6),
            (4, 5), (5, 6), (6, 7),
            (4, 8), (5, 8), (5, 9), (6, 9), (6, 10), (7, 10),
            (8, 9), (9, 10)
        ]
        return edges
    else:
        # Fallback: same as layout 0
        return _build_contact_graph(0)


def _initial_positions_for_graph(contact_edges, n_nodes, layout_idx):
    """Generate initial positions based on layout type."""
    u = (_SQRT3 * 1.001, 0.0)
    v = (_SQRT3 / 2 * 1.001, 1.5 * 1.001)
    
    if layout_idx == 0:
        # Center + 6 ring + 4 more
        coords = [(0, 0),
                  (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                  (2, 0), (1, 1), (-1, 2), (-2, 0)]
    elif layout_idx == 1:
        # 3-5-3 rows
        coords = []
        for k in range(-1, 2):
            coords.append((k, -1))
        for k in range(-2, 3):
            coords.append((k, 0))
        for k in range(-1, 2):
            coords.append((k, 1))
    elif layout_idx == 2:
        # 4-3-4 rows
        coords = []
        for k in range(-2, 2):
            coords.append((k, -1))
        for k in range(-1, 2):
            coords.append((k, 0))
        for k in range(-2, 2):
            coords.append((k, 1))
    elif layout_idx == 3:
        # Diamond
        coords = [(0, 0),
                  (-1, -1), (0, -1), (1, -1),
                  (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
                  (-1, 1), (0, 1), (1, 1)]
        # Remove duplicate (0,0)
        seen = set()
        unique_coords = []
        for c in coords:
            if c not in seen:
                seen.add(c)
                unique_coords.append(c)
        coords = unique_coords[:n_nodes]
    else:
        coords = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        coords.extend(ring1)
        ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1)]
        coords.extend(ring2[:4])
    
    positions = []
    for ti, tj in coords[:n_nodes]:
        x = ti * u[0] + tj * v[0]
        y = ti * u[1] + tj * v[1]
        positions.append((x, y))
    
    return positions


def _validate_and_repair(positions, inner_angle, outer_angle, center, side, max_iterations=200):
    """Validate that all hexagons are non-overlapping and inside the outer hexagon.
    If overlaps exist, push them apart. If any hexagon is outside, pull it in.
    Returns (valid, repaired_positions, repaired_center, repaired_side)."""
    pos = np.array(positions, dtype=float)
    n = len(pos)
    GAP = 1e-6  # required minimum clearance
    
    for it in range(max_iterations):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Check all pairs for overlap
        max_violation = 0.0
        any_overlap = False
        forces = np.zeros((n, 2))
        
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[j] - pos[i]
                sep = _min_separation_same_angle(d, inner_angle)
                if sep < GAP:
                    any_overlap = True
                    violation = GAP - sep
                    if violation > max_violation:
                        max_violation = violation
                    # Push apart along the direction of minimum separation
                    dir_vec, _ = _separation_direction(d, inner_angle)
                    # If d is near zero, use a default direction
                    dist = np.linalg.norm(d)
                    if dist < 1e-8:
                        dir_vec = (1.0, 0.0)
                    else:
                        # Project d onto dir_vec to get sign
                        proj = d[0] * dir_vec[0] + d[1] * dir_vec[1]
                        if proj < 0:
                            dir_vec = (-dir_vec[0], -dir_vec[1])
                    push = 0.5 * violation * 1.1
                    forces[i] -= push * np.array(dir_vec)
                    forces[j] += push * np.array(dir_vec)
        
        if not any_overlap:
            break
        
        # Check if all hexagons are inside outer hexagon
        # (This is handled by the enclosing hex computation, so we trust it for now)
        
        # Apply forces with damping
        pos += 0.3 * forces
        
        # If still overlapping after many iterations, give up on this layout
        if it > max_iterations:
            break
    
    # Final validation
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[j] - pos[i]
            sep = _min_separation_same_angle(d, inner_angle)
            if sep < GAP:
                return False, pos, center, side
    
    return True, pos, center, side


def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    inner_angle = 30.0
    num_layouts = 4
    
    for layout_idx in range(num_layouts):
        if time.time() > _DEADLINE - 20.0:
            break
        
        # Step 1: Build contact graph and get initial positions
        contact_edges = _build_contact_graph(layout_idx)
        initial_positions = _initial_positions_for_graph(contact_edges, N, layout_idx)
        
        # Step 2: Spring-mass simulation
        if time.time() < _DEADLINE - 40.0:
            sim_positions = _spring_mass_simulate(contact_edges, N, inner_angle, 
                                                   max_iters=5000, lr=0.02)
        else:
            sim_positions = np.array(initial_positions)
        
        # Step 3: Optimize outer angle
        coarse_angles = list(np.arange(0, 30.0, 2.0))
        side_coarse, angle_coarse = _optimize_outer_angle(sim_positions, inner_angle, coarse_angles)
        
        # Step 4: Local position optimization
        if time.time() < _DEADLINE - 30.0:
            opt_positions = _optimize_positions(sim_positions, inner_angle, angle_coarse, 
                                                 max_iters=300)
        else:
            opt_positions = sim_positions
        
        # Step 5: Refine outer angle
        fine_angles = list(np.arange(angle_coarse - 2.0, angle_coarse + 2.0, 0.25))
        side_fine, angle_fine = _optimize_outer_angle(opt_positions, inner_angle, fine_angles)
        
        # Step 6: High-precision polish
        if time.time() < _DEADLINE - 15.0:
            polish_positions = _high_precision_polish(opt_positions, inner_angle, angle_fine)
        else:
            polish_positions = opt_positions
        
        # Step 7: Final LP computation
        pos_arr = np.array(polish_positions)
        all_vertices = _hex_vertices_batch(pos_arr, inner_angle).reshape(-1, 2)
        exact_side, center = _enclosing_hex_side_lp(all_vertices, angle_fine)
        
        # Step 8: Validate and repair
        if time.time() < _DEADLINE - 10.0:
            valid, repaired_pos, repaired_center, repaired_side = _validate_and_repair(
                polish_positions, inner_angle, angle_fine, center, exact_side)
            if valid:
                # Recompute enclosing hex with repaired positions
                all_vertices = _hex_vertices_batch(np.array(repaired_pos), inner_angle).reshape(-1, 2)
                exact_side, center = _enclosing_hex_side_lp(all_vertices, angle_fine)
                if exact_side < best_side:
                    best_side = exact_side
                    best_inner = [[pos[0], pos[1], inner_angle] for pos in repaired_pos]
                    best_center = center
                    best_outer_angle = angle_fine
            else:
                # Skip this layout if validation failed
                continue
        else:
            # No time for validation, skip
            continue
    
    if best_inner is None:
        # Fallback: place hexagons far apart to guarantee no overlap
        inner = [[8.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 100.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
