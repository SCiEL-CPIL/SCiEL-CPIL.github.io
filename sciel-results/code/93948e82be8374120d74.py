import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 115.0  # Safety margin before 120s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    sqrt3 = math.sqrt(3)
    
    # Precompute flat arrays for speed
    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]
    
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5:
            return True
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False

    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        ox = offsets[:, 0]
        oy = offsets[:, 1]
        
        # Fast evaluation: only 6 critical directions per phi, vectorized
        # The outer hexagon side for direction phi is determined by max projection
        # We scan phi in 1-degree steps then refine
        
        best_side = float('inf')
        best_phi = 0.0
        
        # Precompute the 6 normal angle offsets
        normal_offsets = np.pi/6 + np.arange(6) * np.pi/3
        
        # Vectorized scan over phi values
        phi_vals = np.arange(0, 60, 1.0)
        phi_radians = np.radians(phi_vals)
        
        # For each phi, compute all 6 normal directions
        # normal_angles[phi_idx, normal_idx]
        all_normals = phi_radians[:, None] + normal_offsets[None, :]
        nx_all = np.cos(all_normals)  # shape (60, 6)
        ny_all = np.sin(all_normals)  # shape (60, 6)
        
        # proj[phi_idx, vert_idx, normal_idx] = ox[vert_idx]*nx + oy[vert_idx]*ny
        # Use broadcasting: ox[None, :, None] * nx_all[:, None, :] + oy[None, :, None] * ny_all[:, None, :]
        proj = ox[None, :, None] * nx_all[:, None, :] + oy[None, :, None] * ny_all[:, None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=1)  # shape (60, 6)
        side_vals = (2.0 / sqrt3) * np.max(max_projs, axis=1)  # shape (60,)
        
        best_idx = np.argmin(side_vals)
        best_side = side_vals[best_idx]
        best_phi = phi_vals[best_idx]
        
        # Fine tune with minimize_scalar in a small window
        try:
            lo = max(0.0, best_phi - 1.5)
            hi = min(60.0, best_phi + 1.5)
            
            def eval_phi_fast(phi_deg):
                phi = np.radians(phi_deg)
                normal_angles = phi + normal_offsets
                nx = np.cos(normal_angles)
                ny = np.sin(normal_angles)
                proj = ox[:, None] * nx[None, :] + oy[:, None] * ny[None, :]
                abs_proj = np.abs(proj)
                max_projs = np.max(abs_proj, axis=0)
                side_vals = (2.0 / sqrt3) * max_projs
                return np.max(side_vals)
            
            res = minimize_scalar(eval_phi_fast, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8, 'maxiter': 30})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # Honeycomb initial configuration - use a more compact hexagonal arrangement
    # For 12 hexagons, use a hexagonal number pattern: center + 1 ring (6) + 5 from second ring
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    # Use a 4-3-4-1 or similar compact arrangement
    # Try: center, 6 neighbors, then 5 more placed optimally
    lattice_coords = [
        (0, 0),       # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # first ring
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1),  # 5 from second ring
    ]
    base_positions = np.zeros((N, 2))
    base_angles = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
        base_angles[idx] = 0.0

    best_side = float('inf')
    best_positions = base_positions.copy()
    best_angles = base_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Strategy: Use spectral graph partitioning to get a good initial layout,
    # then refine with simulated annealing.
    
    # Spectral Graph Partitioning Approach:
    # 1. Build a graph where nodes are hexagons
    # 2. Edge weights are inversely proportional to the Minkowski sum distance
    # 3. Use Fiedler vector to get a 2D embedding
    # 4. This gives a continuous layout that minimizes max support function violation
    
    def spectral_initial_layout():
        """
        Use spectral graph partitioning with Minkowski-sum-aware edge weights.
        Edge weights encode the minimum translation distance between pairs,
        so the Fiedler embedding places hexagons close where they can pack tightly.
        """
        n = N
        positions = base_positions.copy()
        angles = base_angles.copy()
        
        # Compute pairwise Minkowski-sum distances (min translation to separate)
        # For two unit hexagons at relative position d and relative angle delta,
        # the min separation distance is the support function of the Minkowski sum
        # minus the projection of d. We approximate using the honeycomb structure.
        
        # In a honeycomb, neighbors are at distance sqrt(3) apart with 0 relative angle.
        # The Minkowski sum of two unit hexagons is a hexagon with side 2, 
        # so the support function in direction u is 2*|cos| type function.
        
        # Build weight matrix based on honeycomb adjacency
        # Hexagons that are honeycomb neighbors get high weight
        # Use the lattice structure to determine adjacency
        
        # Adjacency in honeycomb: two hexagons are neighbors if their lattice
        # coords differ by one of the 6 neighbor vectors
        neighbor_vectors = [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]
        
        # Map lattice coords to index
        lattice_to_idx = {}
        for idx, (i, j) in enumerate(lattice_coords):
            lattice_to_idx[(i,j)] = idx
        
        W = np.zeros((n, n))
        for i in range(n):
            li, lj = lattice_coords[i]
            for (di, dj) in neighbor_vectors:
                key = (li+di, lj+dj)
                if key in lattice_to_idx:
                    j = lattice_to_idx[key]
                    W[i, j] = 1.0
                    W[j, i] = 1.0
        
        # Add second-neighbor connections with lower weight
        second_neighbor_vectors = [(2,0),(-2,0),(0,2),(0,-2),(2,-2),(-2,2),(1,1),(-1,-1)]
        for i in range(n):
            li, lj = lattice_coords[i]
            for (di, dj) in second_neighbor_vectors:
                key = (li+di, lj+dj)
                if key in lattice_to_idx:
                    j = lattice_to_idx[key]
                    W[i, j] += 0.3
                    W[j, i] += 0.3
        
        # Compute degree matrix
        D = np.diag(W.sum(axis=1))
        
        # Compute Laplacian L = D - W
        L = D - W
        
        # Compute eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            idx_sorted = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx_sorted]
            eigenvectors = eigenvectors[:, idx_sorted]
            
            # Use the first two non-trivial eigenvectors (Fiedler and next)
            embed_1 = eigenvectors[:, 1]
            embed_2 = eigenvectors[:, 2]
        except Exception:
            # Fallback: use honeycomb positions directly
            return base_positions.copy(), base_angles.copy(), \
                   *compute_min_outer_side(base_positions, base_angles)
        
        # Scale to match honeycomb extent
        extent_1 = np.max(embed_1) - np.min(embed_1)
        extent_2 = np.max(embed_2) - np.min(embed_2)
        honeycomb_extent = np.max(base_positions[:, 0]) - np.min(base_positions[:, 0])
        scale = honeycomb_extent / max(extent_1, extent_2, 0.1)
        
        new_positions = np.zeros((n, 2))
        new_positions[:, 0] = embed_1 * scale
        new_positions[:, 1] = embed_2 * scale
        
        # Center
        new_positions[:, 0] -= np.mean(new_positions[:, 0])
        new_positions[:, 1] -= np.mean(new_positions[:, 1])
        
        # Check validity
        angles_rad = np.zeros(n)
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                h1 = (new_positions[i, 0], new_positions[i, 1], angles_rad[i])
                h2 = (new_positions[j, 0], new_positions[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            side, phi, cx, cy = compute_min_outer_side(new_positions, np.zeros(n))
            return new_positions, np.zeros(n), side, phi, cx, cy
        
        # Fallback to honeycomb
        side, phi, cx, cy = compute_min_outer_side(base_positions, base_angles)
        return base_positions.copy(), base_angles.copy(), side, phi, cx, cy
    
    # Try spectral initialization
    spec_positions, spec_angles, spec_side, spec_phi, spec_cx, spec_cy = spectral_initial_layout()
    
    if spec_side < best_side:
        best_side = spec_side
        best_positions = spec_positions.copy()
        best_angles = spec_angles.copy()
        best_phi = spec_phi
        best_cx = spec_cx
        best_cy = spec_cy
    else:
        spec_positions = best_positions.copy()
        spec_angles = best_angles.copy()
    
    # Run multi-start SA with adaptive parameters
    sa_time_budget = DEADLINE * 0.65
    
    # Multi-start: try several different initial configurations
    starts = [
        ('honeycomb', base_positions.copy(), base_angles.copy()),
        ('spec', spec_positions.copy(), spec_angles.copy()),
    ]
    
    # Add a few randomized starts
    for _ in range(3):
        rand_pos = base_positions.copy()
        rand_ang = np.zeros(N)
        for i in range(N):
            rand_pos[i, 0] += random.gauss(0, 0.3)
            rand_pos[i, 1] += random.gauss(0, 0.3)
            rand_ang[i] = random.uniform(-15, 15)
        # Quick validity check
        angles_rad = np.radians(rand_ang)
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                h1 = (rand_pos[i, 0], rand_pos[i, 1], angles_rad[i])
                h2 = (rand_pos[j, 0], rand_pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            starts.append(('rand', rand_pos, rand_ang))
    
    # Allocate time to each start
    num_starts = len(starts)
    time_per_start = sa_time_budget / num_starts
    
    for start_name, init_pos, init_ang in starts:
        if time.time() - t0 > sa_time_budget - 1.0:
            break
        
        current_positions = init_pos.copy()
        current_angles = init_ang.copy()
        current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
        
        if current_side < best_side:
            best_side = current_side
            best_positions = current_positions.copy()
            best_angles = current_angles.copy()
            best_phi = current_phi
            best_cx = current_cx
            best_cy = current_cy
        
        # SA parameters - tune based on starting quality
        T = 0.12
        T_min = 0.00003
        cooling = 0.99992
        step_size = 0.12
        step_decay = 0.99998
        
        step_count = 0
        max_steps = 300000
        accept_count = 0
        total_count = 0
        eval_count = 0
        no_improve_count = 0
        improve_threshold = 250
        start_time = time.time()
        
        while time.time() - start_time < time_per_start and step_count < max_steps:
            step_count += 1
            total_count += 1
            
            # Occasionally move two hexagons together for better exploration
            if random.random() < 0.15:
                idx1 = random.randint(0, N - 1)
                idx2 = random.randint(0, N - 1)
                if idx1 == idx2:
                    idx2 = (idx1 + 1) % N
                new_positions = current_positions.copy()
                new_angles = current_angles.copy()
                
                dx1 = random.gauss(0, step_size)
                dy1 = random.gauss(0, step_size)
                da1 = random.gauss(0, 1.5)
                dx2 = random.gauss(0, step_size)
                dy2 = random.gauss(0, step_size)
                da2 = random.gauss(0, 1.5)
                
                new_positions[idx1, 0] += dx1
                new_positions[idx1, 1] += dy1
                new_angles[idx1] += da1
                new_positions[idx2, 0] += dx2
                new_positions[idx2, 1] += dy2
                new_angles[idx2] += da2
                
                angles_rad_new = np.radians(new_angles)
                valid = True
                for idx in [idx1, idx2]:
                    h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
                    for j in range(N):
                        if j == idx:
                            continue
                        h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
                        if not hexagons_disjoint_fast(h_moved, h_other):
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    T *= cooling
                    if T < T_min: T = T_min
                    step_size *= step_decay
                    if step_size < 0.002: step_size = 0.002
                    continue
                
                eval_count += 1
                new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)
            else:
                idx = random.randint(0, N - 1)
                new_positions = current_positions.copy()
                new_angles = current_angles.copy()
                
                dx = random.gauss(0, step_size)
                dy = random.gauss(0, step_size)
                da = random.gauss(0, 2.0)
                
                new_positions[idx, 0] += dx
                new_positions[idx, 1] += dy
                new_angles[idx] += da
                
                angles_rad_new = np.radians(new_angles)
                valid = True
                h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
                for j in range(N):
                    if j == idx:
                        continue
                    h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
                    if not hexagons_disjoint_fast(h_moved, h_other):
                        valid = False
                        break
                
                if not valid:
                    T *= cooling
                    if T < T_min: T = T_min
                    step_size *= step_decay
                    if step_size < 0.002: step_size = 0.002
                    continue
                
                eval_count += 1
                new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)
            
            if new_side < current_side:
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                accept_count += 1
                no_improve_count = 0
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
            else:
                delta = new_side - current_side
                if delta < 0: delta = 0
                if random.random() < math.exp(-delta / (T * 1.3)):
                    current_positions = new_positions
                    current_angles = new_angles
                    current_side = new_side
                    current_phi = new_phi
                    current_cx = new_cx
                    current_cy = new_cy
                    accept_count += 1
                no_improve_count += 1
            
            # Adaptive step size
            if total_count % 40 == 0:
                acc_rate = accept_count / max(total_count, 1)
                if acc_rate < 0.04:
                    step_size *= 0.85
                    if step_size < 0.002: step_size = 0.002
                elif acc_rate > 0.40:
                    step_size *= 1.12
                    if step_size > 0.30: step_size = 0.30
                accept_count = 0
                total_count = 0
            
            # Periodic restart from best if stuck
            if no_improve_count > improve_threshold:
                current_positions = best_positions.copy()
                current_angles = best_angles.copy()
                for i in range(N):
                    current_positions[i, 0] += random.gauss(0, 0.04)
                    current_positions[i, 1] += random.gauss(0, 0.04)
                    current_angles[i] += random.gauss(0, 0.8)
                current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
                no_improve_count = 0
                T = max(T, 0.06)
                step_size = max(step_size, 0.10)
            
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.002: step_size = 0.002

    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 2.0:
        # 1. Improved Coordinate Descent with line search
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side_cd, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_cd = current_side_cd
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.06
            min_step = 0.00005
            no_improve_cd = 0
            max_no_improve = 30
            
            while time.time() - t_start < time_limit and step > min_step:
                improved = False
                for idx in range(N):
                    if time.time() - t_start > time_limit:
                        return best_side_cd, best_pos_cd, best_ang_cd
                    
                    # Try moving in x, y, and angle dimensions with finer line search
                    for dim in range(3):
                        for direction in [1, -1]:
                            best_local = best_side_cd
                            best_local_pos = best_pos_cd.copy()
                            best_local_ang = best_ang_cd.copy()
                            
                            # Finer line search with more fractions
                            for frac in [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.15]:
                                trial_step = step * frac
                                new_pos = best_pos_cd.copy()
                                new_ang = best_ang_cd.copy()
                                
                                if dim == 0:
                                    new_pos[idx, 0] += direction * trial_step
                                elif dim == 1:
                                    new_pos[idx, 1] += direction * trial_step
                                else:
                                    new_ang[idx] += direction * trial_step
                                
                                angles_rad_new = np.radians(new_ang)
                                valid = True
                                h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                                for j in range(N):
                                    if j == idx: continue
                                    h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                    if not hexagons_disjoint_fast(h_moved, h_other):
                                        valid = False
                                        break
                                
                                if not valid:
                                    continue
                                
                                new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)
                                
                                if new_side < best_local:
                                    best_local = new_side
                                    best_local_pos = new_pos
                                    best_local_ang = new_ang
                            
                            if best_local < best_side_cd - 1e-13:
                                best_pos_cd = best_local_pos
                                best_ang_cd = best_local_ang
                                best_side_cd = best_local
                                improved = True
                                no_improve_cd = 0
                                break
                        
                        if improved:
                            break
                        if time.time() - t_start > time_limit:
                            return best_side_cd, best_pos_cd, best_ang_cd
                    else:
                        continue
                    break
                
                if not improved:
                    no_improve_cd += 1
                    step *= 0.72
                    if no_improve_cd > max_no_improve:
                        break
                else:
                    step *= 0.88
                    no_improve_cd = 0
            
            return best_side_cd, best_pos_cd, best_ang_cd

        # Run CD multiple times with different step initializations
        cd_time_total = min(remaining_time * 0.55, 30.0)
        cd_runs = 3
        cd_time_each = cd_time_total / cd_runs
        
        for run in range(cd_runs):
            if time.time() - t0 > DEADLINE - 5.0:
                break
            cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time_each)
            
            if cd_side < best_side:
                best_side = cd_side
                best_positions = cd_pos
                best_angles = cd_ang
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # 2. Nelder-Mead Polish
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 10.0:
            def polish_objective(params):
                pos = np.zeros((N, 2))
                ang = np.zeros(N)
                for i in range(N):
                    pos[i, 0] = params[3*i]
                    pos[i, 1] = params[3*i+1]
                    ang[i] = params[3*i+2]
                    
                angles_rad = np.radians(ang)
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                        h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            return 1e6
                
                side, phi, cx, cy = compute_min_outer_side(pos, ang)
                return side

            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]
                
            try:
                res = minimize(polish_objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 30000, 'xatol': 1e-10, 'fatol': 1e-12,
                                        'adaptive': True})
                
                if res.fun < best_side:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3*i]
                        new_pos[i, 1] = res.x[3*i+1]
                        new_ang[i] = res.x[3*i+2]
                    
                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
            except Exception:
                pass
        
        # 3. Final fine-tuning with very small steps
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 3.0:
            def final_finetune(pos, ang, time_limit):
                t_start = time.time()
                side_ft, _, _, _ = compute_min_outer_side(pos, ang)
                best_side_ft = side_ft
                best_pos_ft = pos.copy()
                best_ang_ft = ang.copy()
                
                step = 0.01
                while time.time() - t_start < time_limit and step > 0.0001:
                    improved = False
                    for idx in range(N):
                        if time.time() - t_start > time_limit:
                            break
                        for dim in range(3):
                            for direction in [1, -1]:
                                for frac in [1.0, 0.5, 0.25]:
                                    trial_step = step * frac
                                    new_pos = best_pos_ft.copy()
                                    new_ang = best_ang_ft.copy()
                                    
                                    if dim == 0:
                                        new_pos[idx, 0] += direction * trial_step
                                    elif dim == 1:
                                        new_pos[idx, 1] += direction * trial_step
                                    else:
                                        new_ang[idx] += direction * trial_step
                                    
                                    angles_rad_new = np.radians(new_ang)
                                    valid = True
                                    h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                                    for j in range(N):
                                        if j == idx: continue
                                        h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                        if not hexagons_disjoint_fast(h_moved, h_other):
                                            valid = False
                                            break
                                    
                                    if not valid:
                                        continue
                                    
                                    new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)
                                    
                                    if new_side < best_side_ft - 1e-14:
                                        best_pos_ft = new_pos
                                        best_ang_ft = new_ang
                                        best_side_ft = new_side
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if not improved:
                        step *= 0.6
                    else:
                        step *= 0.85
                
                return best_side_ft, best_pos_ft, best_ang_ft
            
            ft_time = min(remaining_time - 1.0, 5.0)
            if ft_time > 1.0:
                ft_side, ft_pos, ft_ang = final_finetune(best_positions, best_angles, ft_time)
                if ft_side < best_side:
                    best_side = ft_side
                    best_positions = ft_pos
                    best_angles = ft_ang
                    best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification and cleanup
    if not is_valid_configuration(best_positions, best_angles):
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    # One last recomputation to ensure consistency
    best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
