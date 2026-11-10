# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, minimizing outer side length.
    Returns (inner, outer, s) where inner is (12,3) array of (x,y,angle_deg),
    outer is (3,) array (x,y,angle_deg), s is outer side length.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit

    def hex_vertices(cx, cy, d, angle_deg):
        """Compute 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
        a = np.radians(angle_deg)
        angles = a + np.radians(60 * np.arange(6))
        return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

    def get_all_inner_vertices(inner_hex_data):
        """Get all 72 vertices of all 12 inner hexagons."""
        verts = []
        for i in range(12):
            cx, cy, ang = inner_hex_data[i]
            verts.append(hex_vertices(cx, cy, 1.0, ang))
        return np.vstack(verts)

    def hexagon_containment_check(verts, outer_center, outer_side, outer_angle_deg):
        """
        Check if all vertices are inside the outer hexagon.
        For each outer edge (p1, p2), cross(p2-p1, v-p1) must be >= -1e-6.
        """
        outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle_deg)
        for k in range(len(verts)):
            v = verts[k]
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False
        return True

    def hexagons_separated(hex1, hex2):
        """
        Check if two unit hexagons are separated (non-overlapping) using SAT.
        Returns True if separated (no overlap), False if overlapping.
        """
        v1 = hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
        v2 = hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])

        # Get all edge normals from both hexagons
        normals = []
        for v in [v1, v2]:
            for i in range(6):
                p1 = v[i]
                p2 = v[(i + 1) % 6]
                edge = p2 - p1
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-15:
                    n = n / norm
                    normals.append(n)

        # Project both sets of vertices onto each normal
        for n in normals:
            proj1 = v1 @ n
            proj2 = v2 @ n
            # If there's a gap, they're separated
            if proj1.max() < proj2.min() - 1e-8 or proj2.max() < proj1.min() - 1e-8:
                return True

        # No separating axis found -> overlapping
        return False

    def all_separated(inner_hex_data):
        """Check all pairs of inner hexagons are separated."""
        for i in range(12):
            for j in range(i + 1, 12):
                if not hexagons_separated(inner_hex_data[i], inner_hex_data[j]):
                    return False
        return True

    def min_containing_hex_side(verts, angle_deg):
        """
        Find minimum side length of a regular hexagon with given orientation
        that contains all vertices. Returns (s, center).
        """
        a = np.radians(angle_deg)
        normals = []
        for i in range(6):
            edge_angle = a + np.radians(60 * i)
            normal_angle = edge_angle + np.radians(90)
            normals.append(np.array([np.cos(normal_angle), np.sin(normal_angle)]))
        normals = np.array(normals)

        projections = verts @ normals.T  # (N, 6)

        s_min = 0
        for i in range(6):
            span = projections[:, i].max() - projections[:, i].min()
            s_min = max(s_min, span / np.sqrt(3))

        # Find center that satisfies all constraints
        mids = (projections.max(axis=0) + projections.min(axis=0)) / 2
        center = np.linalg.lstsq(normals, mids, rcond=None)[0]

        return s_min, center

    def compute_valid_outer(inner_hex_data, outer_angle_deg):
        """Compute a valid outer hexagon that contains all inner hexagons."""
        verts = get_all_inner_vertices(inner_hex_data)
        s, center = min_containing_hex_side(verts, outer_angle_deg)
        # Add small margin for numerical safety
        s *= 1.0005
        return center, s

    def objective(params):
        """
        params: 36 (inner: 12 x [cx, cy, angle]) + 1 (outer angle) = 37
        Returns outer side length s.
        """
        if time.time() > deadline:
            return 1e6

        inner = params[:36].reshape(12, 3)
        outer_angle = params[36]

        # Check separation
        for i in range(12):
            for j in range(i + 1, 12):
                if not hexagons_separated(inner[i], inner[j]):
                    return 1e6

        # Compute minimum containing outer hexagon
        verts = get_all_inner_vertices(inner)
        s, center = min_containing_hex_side(verts, outer_angle)

        # Verify containment with the computed outer
        outer_center = center
        if not hexagon_containment_check(verts, outer_center, s, outer_angle):
            # Expand slightly
            s *= 1.001

        return s

    # Initial arrangement: hexagonal grid with correct spacing
    # For unit hexagons, center-to-center distance for touching = sqrt(3)
    sqrt3 = np.sqrt(3)

    # Arrangement: 4-3-4-1 in hex grid
    # Using flat-top hexagons
    # Row spacing: 1.5 (for flat-top, vertical distance between rows = 1.5 * side = 1.5)
    # Horizontal spacing: sqrt(3)
    # Odd rows offset by sqrt(3)/2

    positions = []
    # Row 0: 4 hexagons
    for col in range(4):
        positions.append([col * sqrt3, 0.0, 0.0])
    # Row 1: 3 hexagons (offset)
    for col in range(3):
        positions.append([(col + 0.5) * sqrt3, 1.5, 0.0])
    # Row 2: 4 hexagons
    for col in range(4):
        positions.append([col * sqrt3, 3.0, 0.0])
    # Row 3: 1 hexagon
    positions.append([sqrt3, 4.5, 0.0])

    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_mean = (max(xs) + min(xs)) / 2
    cy_mean = (max(ys) + min(ys)) / 2
    for p in positions:
        p[0] -= cx_mean
        p[1] -= cy_mean

    inner_init = np.array(positions)

    # Verify initial arrangement is valid
    assert all_separated(inner_init), "Initial arrangement has overlaps!"

    # Compute initial outer
    center_init, s_init = compute_valid_outer(inner_init, 0.0)

    # Initial params for optimization
    x0 = np.concatenate([inner_init.flatten(), [0.0]])

    best_params = x0.copy()
    best_s = s_init

    # Optimization
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-9,
                                  'adaptive': True, 'initial_simplex': None})

        if result.fun < best_s and result.fun < 1e5:
            best_params = result.x
            best_s = result.fun

    except Exception:
        pass

    # Try some random restarts if time permits
    np.random.seed(123)
    for restart in range(30):
        if time.time() > deadline - 100:
            break

        # Small random perturbation
        perturbation = np.random.randn(37) * 0.3
        x0_pert = x0.copy()
        x0_pert[:36] += perturbation[:36]

        try:
            result = minimize(objective, x0_pert, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9,
                                      'adaptive': True})

            if result.fun < best_s and result.fun < 1e5:
                best_params = result.x
                best_s = result.fun

        except Exception:
            pass

    # Extract final result
    inner_final = best_params[:36].reshape(12, 3)
    outer_angle_final = best_params[36]

    # Compute final outer hexagon
    center_final, s_final = compute_valid_outer(inner_final, outer_angle_final)

    # Final validation
    verts_final = get_all_inner_vertices(inner_final)

    # Ensure separation
    for i in range(12):
        for j in range(i + 1, 12):
            if not hexagons_separated(inner_final[i], inner_final[j]):
                # Nudge apart
                inner_final[j, 0] += 0.01

    # Ensure containment
    center_final, s_final = compute_valid_outer(inner_final, outer_angle_final)
    verts_final = get_all_inner_vertices(inner_final)

    if not hexagon_containment_check(verts_final, center_final, s_final, outer_angle_final):
        s_final *= 1.01

    outer_final = np.array([center_final[0], center_final[1], outer_angle_final])

    return inner_final, outer_final, s_final


# EVOLVE-BLOCK-END
