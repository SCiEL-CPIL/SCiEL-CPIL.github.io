import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill

    # Unit hexagon vertices centered at origin
    unit_hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Normals of outer hexagon faces
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    def get_hex_verts(x, y, angle_deg):
        """Get vertices of hexagon at (x, y) with given rotation."""
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        verts = unit_hex_verts @ R.T
        verts += np.array([x, y])
        return verts

    def calc_side(centers, angles, outer_center=(0.0, 0.0), outer_rotation=0.0):
        """Calculate required outer hexagon side length for given configuration."""
        max_apothem = 0.0
        # Rotate normals by outer rotation
        outer_rot_rad = np.deg2rad(outer_rotation)
        c, s = np.cos(outer_rot_rad), np.sin(outer_rot_rad)
        rot_matrix = np.array([[c, -s], [s, c]])
        rotated_normals = normals @ rot_matrix.T

        for k in range(N):
            verts = get_hex_verts(centers[k, 0], centers[k, 1], angles[k])
            # Translate relative to outer center
            verts_centered = verts - np.array(outer_center)
            for v in verts_centered:
                for n in rotated_normals:
                    proj = np.dot(v, n)
                    if proj > max_apothem:
                        max_apothem = proj
        # Side S = 2 * apothem / sqrt(3)
        return 2.0 * max_apothem / np.sqrt(3)

    def check_non_overlap(centers, angles, tol=1e-9):
        """Check if hexagons are non-overlapping using vertex containment."""
        for i in range(N):
            for j in range(i + 1, N):
                verts_i = get_hex_verts(centers[i, 0], centers[i, 1], angles[i])
                verts_j = get_hex_verts(centers[j, 0], centers[j, 1], angles[j])

                # Check if any vertex of i is inside j
                for v in verts_i:
                    inside = True
                    for k in range(6):
                        v1 = verts_j[k]
                        v2 = verts_j[(k + 1) % 6]
                        edge = v2 - v1
                        normal = np.array([-edge[1], edge[0]])
                        if np.dot(normal, v - v1) < -tol:
                            inside = False
                            break
                    if inside:
                        return False

                # Check if any vertex of j is inside i
                for v in verts_j:
                    inside = True
                    for k in range(6):
                        v1 = verts_i[k]
                        v2 = verts_i[(k + 1) % 6]
                        edge = v2 - v1
                        normal = np.array([-edge[1], edge[0]])
                        if np.dot(normal, v - v1) < -tol:
                            inside = False
                            break
                    if inside:
                        return False
        return True

    def smooth_overlap_penalty(centers, angles, beta=50.0):
        """Calculate smooth penalty for overlaps using vertex-to-edge distances."""
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                verts_i = get_hex_verts(centers[i, 0], centers[i, 1], angles[i])
                verts_j = get_hex_verts(centers[j, 0], centers[j, 1], angles[j])

                # Check vertices of i against edges of j
                for v in verts_i:
                    for k in range(6):
                        v1 = verts_j[k]
                        v2 = verts_j[(k + 1) % 6]
                        edge = v2 - v1
                        edge_len = np.linalg.norm(edge)
                        if edge_len < 1e-12:
                            continue
                        normal = np.array([-edge[1], edge[0]]) / edge_len
                        dist = np.dot(normal, v - v1)
                        if dist < 0:
                            # Vertex is on the "inside" side of the edge
                            penalty += np.exp(-beta * (-dist))

                # Check vertices of j against edges of i
                for v in verts_j:
                    for k in range(6):
                        v1 = verts_i[k]
                        v2 = verts_i[(k + 1) % 6]
                        edge = v2 - v1
                        edge_len = np.linalg.norm(edge)
                        if edge_len < 1e-12:
                            continue
                        normal = np.array([-edge[1], edge[0]]) / edge_len
                        dist = np.dot(normal, v - v1)
                        if dist < 0:
                            penalty += np.exp(-beta * (-dist))

        return penalty

    def objective(centers, angles, outer_center=(0.0, 0.0), outer_rotation=0.0, beta=50.0):
        """Objective: side length + smooth overlap penalty."""
        side = calc_side(centers, angles, outer_center, outer_rotation)
        penalty = smooth_overlap_penalty(centers, angles, beta)
        return side + penalty

    def simulated_annealing(initial_centers, initial_angles, outer_center, outer_rotation,
                            max_iters=20000, start_temp=0.5, end_temp=0.001):
        """Run simulated annealing to optimize the packing."""
        current_centers = initial_centers.copy()
        current_angles = initial_angles.copy()
        current_obj = objective(current_centers, current_angles, outer_center, outer_rotation)

        best_centers = current_centers.copy()
        best_angles = current_angles.copy()
        best_obj = current_obj

        best_valid_side = float('inf')
        best_valid_centers = None
        best_valid_angles = None

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            temp = start_temp * (end_temp / start_temp) ** (it / max_iters)

            # Generate perturbation
            new_centers = current_centers.copy()
            new_angles = current_angles.copy()

            if np.random.random() < 0.8:
                # Move one hexagon
                idx = np.random.randint(N)
                step_scale = 0.2 * (1.0 + 0.5 * np.sin(it / 1000))
                new_centers[idx, 0] += np.random.normal(0, step_scale)
                new_centers[idx, 1] += np.random.normal(0, step_scale)
                new_angles[idx] += np.random.normal(0, step_scale * 30)
            else:
                # Move two hexagons
                idx1 = np.random.randint(N)
                idx2 = np.random.randint(N)
                while idx2 == idx1:
                    idx2 = np.random.randint(N)
                step_scale = 0.15
                new_centers[idx1, 0] += np.random.normal(0, step_scale)
                new_centers[idx1, 1] += np.random.normal(0, step_scale)
                new_angles[idx1] += np.random.normal(0, step_scale * 20)
                new_centers[idx2, 0] += np.random.normal(0, step_scale)
                new_centers[idx2, 1] += np.random.normal(0, step_scale)
                new_angles[idx2] += np.random.normal(0, step_scale * 20)

            new_obj = objective(new_centers, new_angles, outer_center, outer_rotation)

            delta = new_obj - current_obj
            if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                current_centers = new_centers
                current_angles = new_angles
                current_obj = new_obj

                if current_obj < best_obj:
                    best_obj = current_obj
                    best_centers = current_centers.copy()
                    best_angles = current_angles.copy()

                # Check validity periodically (every 50 iters for speed)
                if it % 50 == 0 and check_non_overlap(current_centers, current_angles):
                    side = calc_side(current_centers, current_angles, outer_center, outer_rotation)
                    if side < best_valid_side:
                        best_valid_side = side
                        best_valid_centers = current_centers.copy()
                        best_valid_angles = current_angles.copy()

        # Final check
        if best_valid_centers is None and check_non_overlap(best_centers, best_angles):
            side = calc_side(best_centers, best_angles, outer_center, outer_rotation)
            best_valid_centers = best_centers.copy()
            best_valid_angles = best_angles.copy()
            best_valid_side = side

        if best_valid_centers is not None:
            return best_valid_centers, best_valid_angles, best_valid_side
        else:
            return best_centers, best_angles, best_obj

    # Generate initial configurations
    np.random.seed(42)

    d = np.sqrt(3)  # Distance between centers of adjacent hexagons

    initial_configs = []

    # Config 1: Hexagonal cluster (flower pattern extended)
    centers1 = []
    centers1.append((0, 0))  # Center
    for k in range(6):
        angle = k * np.pi / 3
        centers1.append((d * np.cos(angle), d * np.sin(angle)))
    for k in range(6):
        angle = k * np.pi / 3 + np.pi / 6
        x = 2 * d * np.cos(angle)
        y = 2 * d * np.sin(angle)
        if len(centers1) < N:
            centers1.append((x, y))
    while len(centers1) < N:
        angle = np.random.uniform(0, 2 * np.pi)
        r = 2.5 * d
        centers1.append((r * np.cos(angle), r * np.sin(angle)))
    initial_configs.append((np.array(centers1[:N]), np.zeros(N)))

    # Config 2: Linear arrangement
    centers2 = np.array([(i * d - (N - 1) * d / 2, 0) for i in range(N)])
    initial_configs.append((centers2, np.zeros(N)))

    # Config 3: Two staggered rows
    centers3 = []
    for i in range(6):
        centers3.append((i * d - 2.5 * d, 0))
    for i in range(5):
        centers3.append((i * d - 2.0 * d + d / 2, 1.5))
    initial_configs.append((np.array(centers3[:N]), np.zeros(N)))

    # Config 4: Random scatter
    centers4 = np.random.uniform(-2.5 * d, 2.5 * d, (N, 2))
    angles4 = np.random.uniform(-10, 10, N)
    initial_configs.append((centers4, angles4))

    # Config 5: Compact 3-4-3-1 arrangement
    centers5 = []
    # Row 0: 3 hexagons
    for i in range(3):
        centers5.append((i * d - d, 0))
    # Row 1: 4 hexagons
    for i in range(4):
        centers5.append((i * d - 1.5 * d + d / 2, 1.5))
    # Row 2: 3 hexagons
    for i in range(3):
        centers5.append((i * d - d, 3.0))
    # Row 3: 1 hexagon
    centers5.append((0, -1.5))
    initial_configs.append((np.array(centers5[:N]), np.zeros(N)))

    # Config 6: Another compact arrangement
    centers6 = []
    # Center and surrounding
    centers6.append((0, 0))
    for k in range(6):
        angle = k * np.pi / 3
        centers6.append((d * np.cos(angle), d * np.sin(angle)))
    # Add 4 more in a ring
    for k in range(4):
        angle = k * np.pi / 2 + np.pi / 4
        r = 2.2 * d
        centers6.append((r * np.cos(angle), r * np.sin(angle)))
    initial_configs.append((np.array(centers6[:N]), np.zeros(N)))

    # Run simulated annealing from each initial configuration
    best_side = float('inf')
    best_centers = None
    best_angles = None

    for config_idx, (init_centers, init_angles) in enumerate(initial_configs):
        if time.time() > deadline - 15:
            break

        try:
            result_centers, result_angles, result_obj = simulated_annealing(
                init_centers, init_angles, (0.0, 0.0), 0.0,
                max_iters=25000, start_temp=0.4, end_temp=0.001
            )

            if check_non_overlap(result_centers, result_angles):
                side = calc_side(result_centers, result_angles, (0.0, 0.0), 0.0)
                if side < best_side:
                    best_side = side
                    best_centers = result_centers.copy()
                    best_angles = result_angles.copy()
        except Exception:
            pass

    # Try with different outer rotations
    if best_centers is not None and time.time() < deadline - 20:
        for outer_rot in [30, 60, 90]:
            if time.time() > deadline - 10:
                break
            try:
                result_centers, result_angles, result_obj = simulated_annealing(
                    best_centers.copy(), best_angles.copy(), (0.0, 0.0), outer_rot,
                    max_iters=10000, start_temp=0.3, end_temp=0.001
                )
                if check_non_overlap(result_centers, result_angles):
                    side = calc_side(result_centers, result_angles, (0.0, 0.0), outer_rot)
                    if side < best_side:
                        best_side = side
                        best_centers = result_centers.copy()
                        best_angles = result_angles.copy()
            except Exception:
                pass

    # Final refinement: local search with exact overlap checking
    if best_centers is not None and time.time() < deadline - 10:
        current_centers = best_centers.copy()
        current_angles = best_angles.copy()
        current_side = calc_side(current_centers, current_angles, (0.0, 0.0), 0.0)

        # Verify current solution is valid
        if not check_non_overlap(current_centers, current_angles):
            # Try to fix overlaps
            for it in range(5000):
                if time.time() > deadline - 5:
                    break
                idx = np.random.randint(N)
                new_centers = current_centers.copy()
                new_angles = current_angles.copy()
                new_centers[idx, 0] += np.random.normal(0, 0.1)
                new_centers[idx, 1] += np.random.normal(0, 0.1)
                new_angles[idx] += np.random.normal(0, 2.0)

                if check_non_overlap(new_centers, new_angles):
                    new_side = calc_side(new_centers, new_angles, (0.0, 0.0), 0.0)
                    if new_side < current_side:
                        current_centers = new_centers
                        current_angles = new_angles
                        current_side = new_side

        # Fine-tuning to reduce side length
        step_size = 0.05
        for it in range(10000):
            if time.time() > deadline - 3:
                break

            idx = np.random.randint(N)
            new_centers = current_centers.copy()
            new_angles = current_angles.copy()
            new_centers[idx, 0] += np.random.normal(0, step_size)
            new_centers[idx, 1] += np.random.normal(0, step_size)
            new_angles[idx] += np.random.normal(0, step_size * 10)

            if check_non_overlap(new_centers, new_angles):
                new_side = calc_side(new_centers, new_angles, (0.0, 0.0), 0.0)
                if new_side < current_side:
                    current_centers = new_centers
                    current_angles = new_angles
                    current_side = new_side

            step_size *= 0.9998

        if current_side < best_side:
            best_centers = current_centers
            best_angles = current_angles
            best_side = current_side

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
