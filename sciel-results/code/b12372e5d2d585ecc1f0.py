import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600 - 15  # Safety margin

    # Precompute base angles and normals for unit hexagon (flat-top)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    # Normals (perpendicular to edges) are at 30, 90, 150, 210, 270, 330 degrees
    base_angles = np.deg2rad(np.arange(30, 360, 60))
    base_normals = np.column_stack((np.cos(base_angles), np.sin(base_angles)))  # (6, 2)

    # Outer hexagon is flat-top, centered at origin.
    # Its normals are the same as base_normals.
    outer_normals = base_normals

    # Precompute base vertex offsets (unit hexagon at angle 0)
    base_vertex_angles = np.deg2rad(np.arange(0, 360, 60))
    base_vertices = np.column_stack((np.cos(base_vertex_angles), np.sin(base_vertex_angles)))  # (6, 2)

    def get_rotations(angles_deg):
        """Get rotation matrices for each hexagon. Returns (11, 2, 2)."""
        a_rad = np.deg2rad(angles_deg)
        c = np.cos(a_rad)  # (11,)
        s = np.sin(a_rad)  # (11,)
        R = np.zeros((len(angles_deg), 2, 2))
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
        return R

    def get_all_vertices(positions, angles):
        """Get all vertices for all 11 hexagons. Returns (11, 6, 2)."""
        Rs = get_rotations(angles)  # (11, 2, 2)
        # base_vertices: (6, 2), Rs: (11, 2, 2) -> (11, 6, 2)
        rotated = np.einsum('ij,akj->aik', base_vertices, Rs)  # (11, 6, 2)
        return rotated + positions[:, np.newaxis, :]  # (11, 6, 2)

    def get_all_normals(angles):
        """Get all edge normals for all 11 hexagons. Returns (11, 6, 2)."""
        Rs = get_rotations(angles)  # (11, 2, 2)
        # base_normals: (6, 2), Rs: (11, 2, 2) -> (11, 6, 2)
        rotated = np.einsum('ij,akj->aik', base_normals, Rs)  # (11, 6, 2)
        return rotated

    def calc_overlap_penalty(v1, n1, v2, n2):
        """
        Calculate the minimum overlap depth if two hexagons intersect.
        Returns 0 if separated.
        v1, v2: (6, 2) vertex arrays
        n1, n2: (6, 2) normal arrays
        """
        min_overlap = np.inf

        # Check all 12 normals
        for i in range(6):
            n = n1[i]
            p1 = v1 @ n
            p2 = v2 @ n
            min1, max1 = p1.min(), p1.max()
            min2, max2 = p2.min(), p2.max()

            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                return 0.0  # Separated

            overlap = min(max1, max2) - max(min1, min2)
            if overlap < 0:
                overlap = 0
            if overlap < min_overlap:
                min_overlap = overlap

        for i in range(6):
            n = n2[i]
            p1 = v1 @ n
            p2 = v2 @ n
            min1, max1 = p1.min(), p1.max()
            min2, max2 = p2.min(), p2.max()

            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                return 0.0  # Separated

            overlap = min(max1, max2) - max(min1, min2)
            if overlap < 0:
                overlap = 0
            if overlap < min_overlap:
                min_overlap = overlap

        return min_overlap

    def get_min_side(positions, angles):
        """
        Compute the minimum side length of the outer hexagon.
        Vectorized: project all vertices onto outer normals.
        """
        all_verts = get_all_vertices(positions, angles)  # (11, 6, 2)
        # Project onto each of the 6 outer normals
        # all_verts: (11, 6, 2), outer_normals: (6, 2)
        # projs: (11, 6, 6) where projs[i, v, n] = vertex v of hex i dotted with normal n
        projs = np.einsum('ijk,jk->ij', all_verts, outer_normals)  # (11, 6, 6)
        max_proj = projs.max()
        # Apothem = S * sqrt(3) / 2
        # S >= 2 * max_proj / sqrt(3)
        S = 2 * max_proj / np.sqrt(3)
        return S

    def containment_penalty(positions, angles, S):
        """
        Compute penalty for inner hexagons not contained in outer hexagon.
        Outer hexagon has center (0,0), side S, flat-top.
        A point p is inside if for each edge normal n_k: p . n_k <= S * sqrt(3)/2
        Penalty = sum of violations squared.
        """
        all_verts = get_all_vertices(positions, angles)  # (11, 6, 2)
        projs = np.einsum('ijk,jk->ij', all_verts, outer_normals)  # (11, 6, 6)
        apothem = S * np.sqrt(3) / 2
        violations = np.maximum(0, projs - apothem)  # (11, 6, 6)
        return np.sum(violations ** 2)

    def objective(x):
        if time.time() > deadline:
            return 1e10

        positions = x[:22].reshape(11, 2)
        angles = x[22:33]

        # Calculate min side
        S = get_min_side(positions, angles)

        # Check overlaps and compute penalty
        penalty = 0.0
        all_verts = get_all_vertices(positions, angles)  # (11, 6, 2)
        all_normals = get_all_normals(angles)  # (11, 6, 2)

        for i in range(11):
            v1 = all_verts[i]
            n1 = all_normals[i]
            for j in range(i + 1, 11):
                v2 = all_verts[j]
                n2 = all_normals[j]
                overlap = calc_overlap_penalty(v1, n1, v2, n2)
                if overlap > 0:
                    penalty += overlap ** 2

        # Containment penalty: ensure all inner hexagons fit in outer
        cont_pen = containment_penalty(positions, angles, S)

        return S + 1000 * penalty + 100 * cont_pen

    def is_separated(v1, n1, v2, n2):
        """Check if two hexagons are separated (no overlap)."""
        # Check all 12 normals
        for i in range(6):
            n = n1[i]
            p1 = v1 @ n
            p2 = v2 @ n
            min1, max1 = p1.min(), p1.max()
            min2, max2 = p2.min(), p2.max()

            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                return True  # Separated

        for i in range(6):
            n = n2[i]
            p1 = v1 @ n
            p2 = v2 @ n
            min1, max1 = p1.min(), p1.max()
            min2, max2 = p2.min(), p2.max()

            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                return True  # Separated

        return False

    def is_contained(positions, angles, S):
        """Check if all inner hexagons are contained in outer hexagon."""
        all_verts = get_all_vertices(positions, angles)  # (11, 6, 2)
        projs = np.einsum('ijk,jk->ij', all_verts, outer_normals)  # (11, 6, 6)
        apothem = S * np.sqrt(3) / 2
        return np.all(projs <= apothem + 1e-6)

    def validate(positions, angles, S):
        """Full validation: separation, containment, edge lengths."""
        all_verts = get_all_vertices(positions, angles)  # (11, 6, 2)
        all_normals = get_all_normals(angles)  # (11, 6, 2)

        # Check separation
        for i in range(11):
            for j in range(i + 1, 11):
                if not is_separated(all_verts[i], all_normals[i], all_verts[j], all_normals[j]):
                    return False

        # Check containment
        if not is_contained(positions, angles, S):
            return False

        # Check edge lengths within 1e-8 of 1
        for i in range(11):
            v = all_verts[i]  # (6, 2)
            for k in range(6):
                edge = v[(k + 1) % 6] - v[k]
                if abs(np.linalg.norm(edge) - 1.0) > 1e-8:
                    return False

        return True

    # Initial guesses
    sqrt3 = np.sqrt(3)

    # Guess 1: 7 flower + 4 corners
    init_pos1 = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
        [3, sqrt3],
        [-3, sqrt3],
        [-3, -sqrt3],
        [3, -sqrt3]
    ])
    init_angles1 = np.zeros(11)

    # Guess 2: more compact arrangement
    init_pos2 = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
        [2.5, 1.5 * sqrt3],
        [-2.5, 1.5 * sqrt3],
        [-2.5, -1.5 * sqrt3],
        [2.5, -1.5 * sqrt3]
    ])
    init_angles2 = np.zeros(11)

    # Guess 3: slightly rotated
    init_pos3 = init_pos1.copy()
    init_angles3 = np.deg2rad(np.deg2rad(0)) * np.ones(11)  # still 0 but let's try small angles
    init_angles3 = np.zeros(11)

    initial_guesses = [
        np.concatenate([init_pos1.flatten(), init_angles1]),
        np.concatenate([init_pos2.flatten(), init_angles2]),
        np.concatenate([init_pos3.flatten(), init_angles3]),
    ]

    best_x = None
    best_val = np.inf

    for x0 in initial_guesses:
        val = objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()

    # Multi-start optimization
    n_starts = 5
    rng = np.random.RandomState(42)

    for start_idx in range(n_starts):
        if time.time() > deadline - 30:
            break

        if start_idx == 0:
            x0 = best_x.copy()
        else:
            # Perturb best solution
            x0 = best_x + rng.randn(33) * 0.1

        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 30000, 'xatol': 1e-9, 'fatol': 1e-11,
                                       'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass

    # Fine-tuning pass
    if time.time() < deadline - 30:
        try:
            result = minimize(objective, best_x, method='Nelder-Mead',
                              options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-12,
                                       'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass

    # Extract best result
    positions = best_x[:22].reshape(11, 2)
    angles = best_x[22:33]

    # Final validation
    S = get_min_side(positions, angles)
    S = max(S, np.sqrt(11))

    # Check if valid
    valid = validate(positions, angles, S)

    if not valid:
        # Try to fix by scaling positions slightly inward
        scale = 0.999
        positions_fixed = positions * scale
        S_fixed = get_min_side(positions_fixed, angles)
        S_fixed = max(S_fixed, np.sqrt(11))
        valid_fixed = validate(positions_fixed, angles, S_fixed)
        if valid_fixed:
            positions = positions_fixed
            S = S_fixed
        else:
            # Fallback to seed
            inner_hex_data = np.array(
                [
                    [0, 0, 0],
                    [-2.5, 0, 0],
                    [2.5, 0, 0],
                    [-1.25, 2.17, 0],
                    [1.25, 2.17, 0],
                    [-1.25, -2.17, 0],
                    [1.25, -2.17, 0],
                    [-3.75, 2.17, 0],
                    [3.75, 2.17, 0],
                    [-3.75, -2.17, 0],
                    [3.75, -2.17, 0],
                ]
            )
            outer_hex_data = np.array([0, 0, 0])
            outer_hex_side_length = 8
            return inner_hex_data, outer_hex_data, outer_hex_side_length

    inner_hex_data = np.column_stack((positions, angles))
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = S

    return inner_hex_data, outer_hex_data, outer_hex_side_length
