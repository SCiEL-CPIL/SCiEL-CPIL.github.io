import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
SQRT3 = math.sqrt(3)

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Unit hexagon vertices (flat-top orientation, side=1)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, SQRT3/2],
        [-0.5, SQRT3/2],
        [-1.0, 0.0],
        [-0.5, -SQRT3/2],
        [0.5, -SQRT3/2]
    ])

    def rotate_points(pts, angle):
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return pts @ R.T

    def get_hexagon_vertices(cx, cy, angle):
        v = rotate_points(hex_verts, angle)
        return v + np.array([cx, cy])

    def compute_S(positions, angles, outer_center, outer_angle):
        """
        Compute the minimum outer hexagon side length S such that all inner
        hexagons (at given positions and angles) are contained within the
        outer hexagon centered at outer_center with rotation outer_angle.
        Returns (S, valid) where valid is True if no overlaps.
        """
        # Outer hexagon normals (pointing inward from each face)
        # For a regular hexagon with flat-top, the face normals are at angles
        # 90, 150, 210, 270, 330, 30 degrees (perpendicular to edges)
        # Actually for a hexagon with vertices at angles 0,60,..., the face
        # normals are at 30, 90, 150, 210, 270, 330
        outer_normals = np.array([
            [math.cos(math.pi/6 + k * math.pi/3), math.sin(math.pi/6 + k * math.pi/3)]
            for k in range(6)
        ])

        # For each inner hexagon, project all vertices onto each outer normal
        # The maximum projection determines the required apothem
        max_proj = 0.0
        for i in range(N):
            verts = get_hexagon_vertices(positions[i,0], positions[i,1], angles[i])
            centered = verts - outer_center
            projs = centered @ outer_normals.T  # (6, 6)
            # For each direction, we need max and min projection
            for k in range(6):
                pmax = np.max(projs[:, k])
                pmin = np.min(projs[:, k])
                max_proj = max(max_proj, pmax, -pmin)

        S = (max_proj + APOTHEM) / APOTHEM

        # Check for overlaps between inner hexagons
        # Two hexagons overlap if their distance is less than 2*apothem in some direction
        # More precisely: check if any vertex of one is inside the other, or edges intersect
        # Simple check: minimum distance between centers must be >= 2*APOTHEM for same orientation
        # But with rotation, we need a more careful check
        # Use: for each pair, check if the minimum of |dot(d, n)| over 6 directions of one
        # is >= APOTHEM + APOTHEM = 2*APOTHEM... this isn't quite right for rotated hexagons.
        # Better: check if any vertex of hex i is inside hex j
        for i in range(N):
            verts_i = get_hexagon_vertices(positions[i,0], positions[i,1], angles[i])
            for j in range(i+1, N):
                verts_j = get_hexagon_vertices(positions[j,0], positions[j,1], angles[j])
                # Check if any vertex of i is inside j
                # A point is inside a convex polygon if it's on the same side of all edges
                # For a regular hexagon, we can use the 6 face normals
                # Get face normals of hex j (inward pointing)
                edges_j = np.roll(verts_j, -1, axis=0) - verts_j  # edge vectors
                # Inward normals: rotate edge vectors by -90 degrees
                inward_normals_j = np.array([[-e[1], e[0]] for e in edges_j])
                # Normalize
                norms = np.linalg.norm(inward_normals_j, axis=1, keepdims=True)
                inward_normals_j = inward_normals_j / norms
                # For each vertex of i, check if inside j
                for v in verts_i:
                    rel = v - verts_j[0]
                    # Check against all 6 faces
                    inside = True
                    for k in range(6):
                        if np.dot(rel, inward_normals_j[k]) < -1e-9:
                            inside = False
                            break
                    if inside:
                        return S, False
                # Also check vertices of j inside i
                verts_i_edges = np.roll(verts_i, -1, axis=0) - verts_i
                inward_normals_i = np.array([[-e[1], e[0]] for e in verts_i_edges])
                norms_i = np.linalg.norm(inward_normals_i, axis=1, keepdims=True)
                inward_normals_i = inward_normals_i / norms_i
                for v in verts_j:
                    rel = v - verts_i[0]
                    inside = True
                    for k in range(6):
                        if np.dot(rel, inward_normals_i[k]) < -1e-9:
                            inside = False
                            break
                    if inside:
                        return S, False
        return S, True

    def smooth_objective(params):
        """
        params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, cx, cy, outer_angle]
        Total: 3*N + 3 = 36 parameters
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10

        positions = params[:3*N].reshape(N, 3)[:, :2]
        angles = params[:3*N].reshape(N, 3)[:, 2]
        outer_center = params[3*N:3*N+2]
        outer_angle = params[3*N+2]

        # Compute S using smooth approximation
        outer_normals = np.array([
            [math.cos(math.pi/6 + outer_angle + k * math.pi/3),
             math.sin(math.pi/6 + outer_angle + k * math.pi/3)]
            for k in range(6)
        ])

        # For smoothness, use log-sum-exp on the projections
        all_projs = []
        for i in range(N):
            verts = get_hexagon_vertices(positions[i,0], positions[i,1], angles[i])
            centered = verts - outer_center
            projs = centered @ outer_normals.T  # (6, 6)
            # For each of 6 directions, take max and -min
            for k in range(6):
                all_projs.append(np.max(projs[:, k]))
                all_projs.append(-np.min(projs[:, k]))

        all_projs = np.array(all_projs)
        # Log-sum-exp for smooth max
        max_val = np.max(all_projs)
        if max_val > 100:
            return 1e10
        temperature = 0.01
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM

        # Overlap penalty
        penalty = 0.0
        for i in range(N):
            verts_i = get_hexagon_vertices(positions[i,0], positions[i,1], angles[i])
            for j in range(i+1, N):
                verts_j = get_hexagon_vertices(positions[j,0], positions[j,1], angles[j])
                # Quick distance check
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 2.5:  # Only check if potentially overlapping
                    # Check vertex containment
                    for v in verts_i:
                        rel = v - verts_j[0]
                        edges_j = np.roll(verts_j, -1, axis=0) - verts_j
                        inward_n = np.array([[-e[1], e[0]] for e in edges_j])
                        norms = np.linalg.norm(inward_n, axis=1, keepdims=True)
                        inward_n = inward_n / norms
                        for k in range(6):
                            d = np.dot(rel, inward_n[k])
                            if d < 0:
                                penalty += (-d) ** 2 * 100.0
                    for v in verts_j:
                        rel = v - verts_i[0]
                        edges_i = np.roll(verts_i, -1, axis=0) - verts_i
                        inward_n_i = np.array([[-e[1], e[0]] for e in edges_i])
                        norms_i = np.linalg.norm(inward_n_i, axis=1, keepdims=True)
                        inward_n_i = inward_n_i / norms_i
                        for k in range(6):
                            d = np.dot(rel, inward_n_i[k])
                            if d < 0:
                                penalty += (-d) ** 2 * 100.0

        return S + penalty

    def evaluate_exact(positions, angles, outer_center, outer_angle):
        """Exact evaluation for final result."""
        S, valid = compute_S(positions, angles, outer_center, outer_angle)
        if not valid:
            return float('inf')
        return S

    best_S = float('inf')
    best_params = None

    rng = np.random.default_rng(42)

    # Multi-start local search
    n_starts = 30
    max_iter_per_start = 500

    for start_idx in range(n_starts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

        # Random initial configuration
        # Place hexagons in a rough grid pattern with random perturbations
        if start_idx == 0:
            # Start with a known good lattice: hexagonal packing
            # 3-3-3-2 arrangement
            base_positions = []
            # Row 0: 3 hexagons
            for i in range(3):
                base_positions.append([i * SQRT3, 0.0])
            # Row 1: 3 hexagons
            for i in range(3):
                base_positions.append([(i + 0.5) * SQRT3, 1.5])
            # Row 2: 3 hexagons
            for i in range(3):
                base_positions.append([i * SQRT3, 3.0])
            # Row 3: 2 hexagons
            for i in range(2):
                base_positions.append([(i + 0.5) * SQRT3, 4.5])
            positions = np.array(base_positions)
            # Center the configuration
            center_offset = positions.mean(axis=0)
            positions = positions - center_offset
            angles = np.zeros(N)
            outer_center = np.zeros(2)
            outer_angle = 0.0
        else:
            # Random perturbation around a lattice
            base_positions = []
            for i in range(3):
                base_positions.append([i * SQRT3, 0.0])
            for i in range(3):
                base_positions.append([(i + 0.5) * SQRT3, 1.5])
            for i in range(3):
                base_positions.append([i * SQRT3, 3.0])
            for i in range(2):
                base_positions.append([(i + 0.5) * SQRT3, 4.5])
            positions = np.array(base_positions)
            center_offset = positions.mean(axis=0)
            positions = positions - center_offset
            # Add random perturbations
            perturb_scale = 0.1 + 0.3 * (start_idx / n_starts)
            positions = positions + rng.normal(0, perturb_scale, size=(N, 2))
            angles = rng.uniform(-math.pi/6, math.pi/6, size=N)
            outer_center = rng.normal(0, 0.1, size=2)
            outer_angle = rng.uniform(0, math.pi/6)

        params_init = np.concatenate([
            positions.ravel(), angles, outer_center, [outer_angle]
        ])

        try:
            result = minimize(
                smooth_objective, params_init,
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_start, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result.fun < best_S + 0.1:
                # Extract and verify
                pos = result.x[:3*N].reshape(N, 3)[:, :2]
                ang = result.x[:3*N].reshape(N, 3)[:, 2]
                oc = result.x[3*N:3*N+2]
                oa = result.x[3*N+2]
                S_exact = evaluate_exact(pos, ang, oc, oa)
                if S_exact < best_S:
                    best_S = S_exact
                    best_params = (pos.copy(), ang.copy(), oc.copy(), oa)
        except Exception:
            pass

    # If no valid solution found, use fallback
    if best_params is None:
        # Fallback: simple 3-3-3-2 grid
        base_positions = []
        for i in range(3):
            base_positions.append([i * SQRT3, 0.0])
        for i in range(3):
            base_positions.append([(i + 0.5) * SQRT3, 1.5])
        for i in range(3):
            base_positions.append([i * SQRT3, 3.0])
        for i in range(2):
            base_positions.append([(i + 0.5) * SQRT3, 4.5])
        positions = np.array(base_positions)
        center_offset = positions.mean(axis=0)
        positions = positions - center_offset
        angles = np.zeros(N)
        outer_center = np.zeros(2)
        outer_angle = 0.0
        best_S = evaluate_exact(positions, angles, outer_center, outer_angle)
        if best_S == float('inf'):
            # Very large fallback
            best_S = 10.0
        best_params = (positions, angles, outer_center, outer_angle)

    positions, angles, outer_center, outer_angle = best_params

    # Final polish: try to tighten further with a very low temperature
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 5:
        def final_smooth(params):
            positions_f = params[:3*N].reshape(N, 3)[:, :2]
            angles_f = params[:3*N].reshape(N, 3)[:, 2]
            outer_center_f = params[3*N:3*N+2]
            outer_angle_f = params[3*N+2]

            outer_normals = np.array([
                [math.cos(math.pi/6 + outer_angle_f + k * math.pi/3),
                 math.sin(math.pi/6 + outer_angle_f + k * math.pi/3)]
                for k in range(6)
            ])

            all_projs = []
            for i in range(N):
                verts = get_hexagon_vertices(positions_f[i,0], positions_f[i,1], angles_f[i])
                centered = verts - outer_center_f
                projs = centered @ outer_normals.T
                for k in range(6):
                    all_projs.append(np.max(projs[:, k]))
                    all_projs.append(-np.min(projs[:, k]))

            all_projs = np.array(all_projs)
            max_val = np.max(all_projs)
            if max_val > 100:
                return 1e10
            temperature = 0.001
            shifted = (all_projs - max_val) / temperature
            shifted = np.clip(shifted, -50, 50)
            log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
            D = temperature * log_sum
            S = (D + APOTHEM) / APOTHEM

            penalty = 0.0
            for i in range(N):
                verts_i = get_hexagon_vertices(positions_f[i,0], positions_f[i,1], angles_f[i])
                for j in range(i+1, N):
                    verts_j = get_hexagon_vertices(positions_f[j,0], positions_f[j,1], angles_f[j])
                    dist = np.linalg.norm(positions_f[i] - positions_f[j])
                    if dist < 2.5:
                        for v in verts_i:
                            rel = v - verts_j[0]
                            edges_j = np.roll(verts_j, -1, axis=0) - verts_j
                            inward_n = np.array([[-e[1], e[0]] for e in edges_j])
                            norms = np.linalg.norm(inward_n, axis=1, keepdims=True)
                            inward_n = inward_n / norms
                            for k in range(6):
                                d = np.dot(rel, inward_n[k])
                                if d < 0:
                                    penalty += (-d) ** 2 * 1000.0
                        for v in verts_j:
                            rel = v - verts_i[0]
                            edges_i = np.roll(verts_i, -1, axis=0) - verts_i
                            inward_n_i = np.array([[-e[1], e[0]] for e in edges_i])
                            norms_i = np.linalg.norm(inward_n_i, axis=1, keepdims=True)
                            inward_n_i = inward_n_i / norms_i
                            for k in range(6):
                                d = np.dot(rel, inward_n_i[k])
                                if d < 0:
                                    penalty += (-d) ** 2 * 1000.0

            return S + penalty

        init_final = np.concatenate([
            positions.ravel(), angles, outer_center, [outer_angle]
        ])
        try:
            res_final = minimize(
                final_smooth, init_final,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-13}
            )
            if res_final.fun < best_S + 0.01:
                pos_f = res_final.x[:3*N].reshape(N, 3)[:, :2]
                ang_f = res_final.x[:3*N].reshape(N, 3)[:, 2]
                oc_f = res_final.x[3*N:3*N+2]
                oa_f = res_final.x[3*N+2]
                S_f = evaluate_exact(pos_f, ang_f, oc_f, oa_f)
                if S_f < best_S:
                    best_S = S_f
                    positions = pos_f
                    angles = ang_f
                    outer_center = oc_f
                    outer_angle = oa_f
        except Exception:
            pass

    # Prepare output
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(math.degrees(angles[i]))])

    center = [float(outer_center[0]), float(outer_center[1])]
    side = float(best_S)
    rotation = float(math.degrees(outer_angle))

    return inner, center, side, rotation
