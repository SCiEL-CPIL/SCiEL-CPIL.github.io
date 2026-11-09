# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon with center (cx,cy), side d, rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _check_validity(inner, outer, s, tol=1e-8):
    """Validate the packing. Returns True if valid."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if s < np.sqrt(12.0) - 1e-12:
        return False

    # Compute outer vertices
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    if len(np.unique(np.round(outer_verts, 10), axis=0)) < 6:
        return False

    # Check unit edge lengths of inner hexagons
    for i in range(12):
        verts = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(6):
            v1 = verts[j]
            v2 = verts[(j + 1) % 6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > tol:
                return False

    # Separation check: SAT for each pair of inner hexagons
    for i in range(12):
        vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(i + 1, 12):
            vj = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            # Collect edge normals from both hexagons
            normals = []
            for verts in (vi, vj):
                for k in range(6):
                    e = verts[(k + 1) % 6] - verts[k]
                    n = np.array([-e[1], e[0]])
                    normals.append(n)
            found_separating = False
            for n in normals:
                proj_i = vi @ n
                proj_j = vj @ n
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                gap = max(min_i, min_j) - min(max_i, max_j)
                if gap > 1e-6:
                    found_separating = True
                    break
            if not found_separating:
                return False

    # Containment check: each inner vertex must be inside outer hexagon
    for i in range(12):
        vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for v in vi:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False

    return True


def hexagon_packing_12():
    """
    Packs 12 unit regular hexagons in a regular hexagon using a hexagonal lattice arrangement
    with small gaps to satisfy separation constraints.
    Returns (inner, outer, s) where inner is (12,3), outer is (3,), s is side length.
    """
    deadline = time.time() + 620

    sqrt3 = np.sqrt(3.0)

    # Use a small expansion factor to create gaps between hexagons
    # Base spacing for touching hexagons is sqrt(3), we use sqrt(3) * (1 + eps)
    eps = 0.001  # 0.1% expansion for clear separation
    spacing = sqrt3 * (1.0 + eps)

    # Hexagonal lattice positions for 12 hexagons
    base_positions = []
    base_positions.append([0.0, 0.0])
    for k in range(6):
        angle = np.radians(60.0 * k)
        base_positions.append([spacing * np.cos(angle), spacing * np.sin(angle)])
    for k in [0, 1, 2, 4, 5]:
        angle = np.radians(60.0 * k)
        base_positions.append([2 * spacing * np.cos(angle), 2 * spacing * np.sin(angle)])

    base_positions = np.array(base_positions)

    # Compute the required outer hexagon side length for the base configuration
    # Outermost vertex distance = 2*spacing + sqrt(3)/2
    # For outer hexagon with side s, inradius = s*sqrt(3)/2
    # We need inradius >= 2*spacing + sqrt(3)/2 (approximately, depends on orientation)

    best_inner = None
    best_outer = None
    best_s = float('inf')

    # Try various global rotations and outer hexagon orientations
    for rot_deg in range(0, 60, 3):
        if time.time() > deadline:
            break

        theta = np.radians(rot_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        rotated_positions = base_positions @ R.T

        for outer_rot in range(0, 60, 3):
            if time.time() > deadline:
                break

            outer_angle = np.radians(outer_rot)
            normal_angles = outer_angle + np.radians(60.0 * np.arange(6))
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

            # Compute all inner vertices
            all_verts = []
            for pos in rotated_positions:
                verts = _hex_vertices(pos[0], pos[1], 1.0, 0.0)
                all_verts.append(verts)
            all_verts = np.vstack(all_verts)

            supports = all_verts @ normals.T
            max_supports = supports.max(axis=0)
            required_s = np.max(max_supports) / (sqrt3 / 2.0)

            if required_s < best_s:
                best_s = required_s
                inner = np.zeros((12, 3))
                for i, pos in enumerate(rotated_positions):
                    inner[i, 0] = pos[0]
                    inner[i, 1] = pos[1]
                    inner[i, 2] = 0.0
                outer = np.array([0.0, 0.0, float(outer_rot)])
                best_inner = inner
                best_outer = outer

    if best_inner is None:
        inner = np.zeros((12, 3))
        for i, pos in enumerate(base_positions):
            inner[i, 0] = pos[0]
            inner[i, 1] = pos[1]
            inner[i, 2] = 0.0
        outer = np.array([0.0, 0.0, 0.0])
        s = 5.0 * sqrt3 / 2.0
        return inner, outer, s

    # Validate
    if not _check_validity(best_inner, best_outer, best_s):
        # Increase eps and retry
        eps = 0.01
        spacing = sqrt3 * (1.0 + eps)
        base_positions = []
        base_positions.append([0.0, 0.0])
        for k in range(6):
            angle = np.radians(60.0 * k)
            base_positions.append([spacing * np.cos(angle), spacing * np.sin(angle)])
        for k in [0, 1, 2, 4, 5]:
            angle = np.radians(60.0 * k)
            base_positions.append([2 * spacing * np.cos(angle), 2 * spacing * np.sin(angle)])
        base_positions = np.array(base_positions)

        inner = np.zeros((12, 3))
        for i, pos in enumerate(base_positions):
            inner[i, 0] = pos[0]
            inner[i, 1] = pos[1]
            inner[i, 2] = 0.0
        outer = np.array([0.0, 0.0, 0.0])
        # Compute s
        all_verts = []
        for pos in base_positions:
            verts = _hex_vertices(pos[0], pos[1], 1.0, 0.0)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        normals = np.column_stack([np.cos(np.radians(60 * np.arange(6))), np.sin(np.radians(60 * np.arange(6)))])
        supports = all_verts @ normals.T
        max_supports = supports.max(axis=0)
        s = np.max(max_supports) / (sqrt3 / 2.0)
        return inner, outer, s

    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
