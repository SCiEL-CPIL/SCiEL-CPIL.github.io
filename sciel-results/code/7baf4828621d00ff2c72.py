import numpy as np
import time
import itertools

N = 11
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # Unit hexagon vertices centered at origin, angle 0
    unit_hex_verts = np.array([
        [1.0, 0.0],
        [0.5, SQRT3 / 2],
        [-0.5, SQRT3 / 2],
        [-1.0, 0.0],
        [-0.5, -SQRT3 / 2],
        [0.5, -SQRT3 / 2]
    ])

    # Normals of outer hexagon (centered at 0,0, rotation 0)
    normals = np.array([
        [1.0, 0.0],
        [0.5, SQRT3 / 2],
        [-0.5, SQRT3 / 2],
        [-1.0, 0.0],
        [-0.5, -SQRT3 / 2],
        [0.5, -SQRT3 / 2]
    ])

    def get_hex_verts_batch(centers, angles):
        """Get vertices of all hexagons. centers: (N,2), angles: (N,) in degrees."""
        angle_rad = np.deg2rad(angles)
        c = np.cos(angle_rad)[:, None, None]
        s = np.sin(angle_rad)[:, None, None]
        # Rotate each hexagon: v @ R^T where R = [[c,-s],[s,c]]
        # verts[i] = unit_hex_verts @ [[c_i, s_i], [-s_i, c_i]]
        # = [vx*c - vy*s, vx*s + vy*c] for each vertex
        verts = np.zeros((N, 6, 2))
        for k in range(6):
            vx = unit_hex_verts[k, 0]
            vy = unit_hex_verts[k, 1]
            verts[:, k, 0] = vx * c[:, 0, 0] - vy * s[:, 0, 0]
            verts[:, k, 1] = vx * s[:, 0, 0] + vy * c[:, 0, 0]
        # Translate
        verts += centers[:, None, :]
        return verts

    def calc_side_fast(centers, angles):
        """Calculate required outer hexagon side length using vectorized operations."""
        verts = get_hex_verts_batch(centers, angles)  # (N, 6, 2)
        # Project all vertices onto all normals: (N, 6, 6)
        # For each hexagon i, vertex j, normal k: dot(verts[i,j,:], normals[k,:])
        # verts shape: (N, 6, 2), normals shape: (6, 2)
        # We want: for each i, max over j,k of dot(verts[i,j,:], normals[k,:])
        # = for each i, max over j of (verts[i,j,:] @ normals.T) max over k
        # verts[i,j,:] @ normals.T gives (6,) vector of projections
        projections = np.einsum('ijc,kc->ijk', verts, normals)  # (N, 6, 6)
        max_apothem = np.max(projections)
        # Side S = 2 * apothem / sqrt(3)
        return 2.0 * max_apothem / SQRT3

    def hexagon_overlap_test_fast(centers, angles, eps=1e-9):
        """
        Check if any two hexagons overlap. Returns True if no overlap.
        Uses vertex-in-polygon test.
        """
        verts_all = get_hex_verts_batch(centers, angles)  # (N, 6, 2)
        
        for i in range(N):
            for j in range(i + 1, N):
                # Check if any vertex of i is inside j
                verts_i = verts_all[i]  # (6, 2)
                verts_j = verts_all[j]  # (6, 2)
                cx_j, cy_j = centers[j]
                angle_j = angles[j]
                
                # Transform vertices of i to local frame of j
                dx = verts_i[:, 0] - cx_j
                dy = verts_i[:, 1] - cy_j
                angle_rad = np.deg2rad(-angle_j)
                c = np.cos(angle_rad)
                s = np.sin(angle_rad)
                lx = c * dx - s * dy
                ly = s * dx + c * dy
                
                # Check each point against hexagon j
                for k in range(6):
                    v1 = unit_hex_verts[k]
                    v2 = unit_hex_verts[(k + 1) % 6]
                    ex = v2[0] - v1[0]
                    ey = v2[1] - v1[1]
                    cross = ex * (ly - v1[1]) - ey * (lx - v1[0])
                    if np.any(cross > eps):
                        # Also check vertices of j inside i
                        verts_j_local = verts_j
                        cx_i, cy_i = centers[i]
                        angle_i = angles[i]
                        dx2 = verts_j_local[:, 0] - cx_i
                        dy2 = verts_j_local[:, 1] - cy_i
                        angle_rad2 = np.deg2rad(-angle_i)
                        c2 = np.cos(angle_rad2)
                        s2 = np.sin(angle_rad2)
                        lx2 = c2 * dx2 - s2 * dy2
                        ly2 = s2 * dx2 + c2 * dy2
                        
                        overlap_found = False
                        for m in range(6):
                            v1m = unit_hex_verts[m]
                            v2m = unit_hex_verts[(m + 1) % 6]
                            exm = v2m[0] - v1m[0]
                            eym = v2m[1] - v1m[1]
                            cross2 = exm * (ly2 - v1m[1]) - eym * (lx2 - v1m[0])
                            if np.any(cross2 > eps):
                                overlap_found = True
                                break
                        
                        if overlap_found:
                            return False
                        else:
                            return False  # vertex of i is inside j
                # If we get here, no vertex of i is inside j, check vertices of j in i
                # (Already handled above in the else branch)
        return True

    def calc_side(centers, angles):
        """Calculate required outer hexagon side length."""
        return calc_side_fast(centers, angles)

    def hexagon_overlap_test(centers, angles, eps=1e-9):
        """Check if any two hexagons overlap. Returns True if no overlap."""
        return hexagon_overlap_test_fast(centers, angles, eps)

    # Hexagonal lattice spacing
    d = SQRT3  # Distance between centers of adjacent hexagons

    best_side = float('inf')
    best_centers = None
    best_angles = None

    def objective(centers, angles):
        """Objective function: side length + penalty for overlaps."""
        side = calc_side_fast(centers, angles)
        
        # Penalty for overlaps using distance heuristic
        penalty = 0.0
        min_dist = SQRT3 * 0.999  # Slightly less to allow touching
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 5000
        
        return side + penalty

    def local_search(centers, angles, max_iters=20000, step_init=0.1):
        """Local search to minimize side length while avoiding overlaps."""
        current_centers = centers.copy()
        current_angles = angles.copy()
        current_obj = objective(current_centers, current_angles)
        best_centers = current_centers.copy()
        best_angles = current_angles.copy()
        best_obj = current_obj
        step = step_init
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            idx = np.random.randint(N)
            dx = np.random.normal(0, step)
            dy = np.random.normal(0, step)
            dangle = np.random.normal(0, step * 3)
            
            new_centers = current_centers.copy()
            new_angles = current_angles.copy()
            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            new_angles[idx] += dangle
            
            new_obj = objective(new_centers, new_angles)
            
            if new_obj < current_obj:
                current_centers = new_centers
                current_angles = new_angles
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_centers = current_centers.copy()
                    best_angles = current_angles.copy()
            
            # Decay step size
            step *= 0.9998
        
        return best_centers, best_angles, best_obj

    # Generate multiple initial configurations
    np.random.seed(42)
    
    initial_configs = []
    
    # Config 1: Hexagonal cluster centered at origin
    centers1 = []
    centers1.append((0, 0))
    for k in range(6):
        angle = k * np.pi / 3
        centers1.append((d * np.cos(angle), d * np.sin(angle)))
    for k in range(4):
        angle = k * np.pi / 3 + np.pi / 6
        x = 2 * d * np.cos(angle)
        y = 2 * d * np.sin(angle)
        if len(centers1) < N:
            centers1.append((x, y))
    
    while len(centers1) < N:
        angle = np.random.uniform(0, 2 * np.pi)
        r = 2.5 * d
        centers1.append((r * np.cos(angle), r * np.sin(angle)))
    
    centers1 = np.array(centers1[:N])
    angles1 = np.zeros(N)
    initial_configs.append((centers1, angles1))
    
    # Config 2: Linear arrangement
    centers2 = np.array([(i * d - (N - 1) * d / 2, 0) for i in range(N)])
    angles2 = np.zeros(N)
    initial_configs.append((centers2, angles2))
    
    # Config 3: Two rows
    centers3 = []
    for i in range(N):
        if i < 6:
            centers3.append((i * d - 2.5 * d, 0))
        else:
            centers3.append(((i - 6) * d - 1.5 * d + d / 2, 1.5))
    centers3 = np.array(centers3)
    angles3 = np.zeros(N)
    initial_configs.append((centers3, angles3))
    
    # Config 4: Random scatter
    centers4 = np.random.uniform(-3 * d, 3 * d, (N, 2))
    angles4 = np.random.uniform(-10, 10, N)
    initial_configs.append((centers4, angles4))
    
    # Config 5: Tighter hexagonal pattern
    centers5 = []
    centers5.append((-d / 2, 0))
    centers5.append((d / 2, 0))
    centers5.append((-d, 1.5))
    centers5.append((0, 1.5))
    centers5.append((d, 1.5))
    centers5.append((-d / 2, 3.0))
    centers5.append((d / 2, 3.0))
    centers5.append((-d, -1.5))
    centers5.append((0, -1.5))
    centers5.append((d, -1.5))
    centers5.append((0, -3.0))
    
    centers5 = np.array(centers5[:N])
    angles5 = np.zeros(N)
    initial_configs.append((centers5, angles5))
    
    # Config 6: Another hexagonal arrangement
    centers6 = []
    centers6.append((0, 0))
    for k in range(6):
        angle = k * np.pi / 3
        centers6.append((d * np.cos(angle), d * np.sin(angle)))
    for k in [0, 2, 4, 6]:
        angle = k * np.pi / 3
        x = 2 * d * np.cos(angle)
        y = 2 * d * np.sin(angle)
        centers6.append((x, y))
    
    centers6 = np.array(centers6[:N])
    angles6 = np.zeros(N)
    initial_configs.append((centers6, angles6))
    
    # Config 7: 3-4-3-1 arrangement
    centers7 = []
    # Row 1: 3 hexagons
    for i in range(3):
        centers7.append((i * d - d, 0))
    # Row 2: 4 hexagons
    for i in range(4):
        centers7.append((i * d - 1.5 * d + d / 2, 1.5))
    # Row 3: 3 hexagons
    for i in range(3):
        centers7.append((i * d - d, 3.0))
    # Row 4: 1 hexagon
    centers7.append((0, 4.5))
    
    centers7 = np.array(centers7[:N])
    angles7 = np.zeros(N)
    initial_configs.append((centers7, angles7))
    
    # Config 8: 2-3-4-2 arrangement
    centers8 = []
    for i in range(2):
        centers8.append((i * d - d / 2, 0))
    for i in range(3):
        centers8.append((i * d - d + d / 2, 1.5))
    for i in range(4):
        centers8.append((i * d - 1.5 * d + d / 2, 3.0))
    for i in range(2):
        centers8.append((i * d - d / 2, 4.5))
    
    centers8 = np.array(centers8[:N])
    angles8 = np.zeros(N)
    initial_configs.append((centers8, angles8))
    
    # Run local search from each initial configuration
    for config_idx, (init_centers, init_angles) in enumerate(initial_configs):
        if time.time() > deadline - 8:
            break
        
        try:
            result_centers, result_angles, result_obj = local_search(
                init_centers, init_angles, max_iters=25000, step_init=0.12
            )
            
            # Check if the result is valid (no overlaps) using strict test
            if hexagon_overlap_test(result_centers, result_angles):
                side = calc_side_fast(result_centers, result_angles)
                if side < best_side:
                    best_side = side
                    best_centers = result_centers.copy()
                    best_angles = result_angles.copy()
        except Exception:
            pass
    
    # Final refinement: local search on the best solution
    if best_centers is not None and time.time() < deadline - 5:
        current_centers = best_centers.copy()
        current_angles = best_angles.copy()
        current_obj = objective(current_centers, current_angles)
        
        step_size = 0.03
        for it in range(10000):
            if time.time() > deadline - 2:
                break
            
            idx = np.random.randint(N)
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            dangle = np.random.normal(0, step_size * 8)
            
            new_centers = current_centers.copy()
            new_angles = current_angles.copy()
            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            new_angles[idx] += dangle
            
            new_obj = objective(new_centers, new_angles)
            
            if new_obj < current_obj:
                current_centers = new_centers
                current_angles = new_angles
                current_obj = new_obj
            
            step_size *= 0.9995
        
        if current_obj < objective(best_centers, best_angles):
            best_centers = current_centers
            best_angles = current_angles
            best_side = calc_side_fast(best_centers, best_angles)
    
    # Construct output
    if best_centers is None:
        # Fallback: widely spaced configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = []
    for i in range(N):
        x, y = best_centers[i, 0], best_centers[i, 1]
        ang = best_angles[i]
        inner.append([float(x), float(y), float(ang)])
    
    return inner, (0.0, 0.0), float(best_side), 0.0
