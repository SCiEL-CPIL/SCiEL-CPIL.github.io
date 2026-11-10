# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 560.0  # safety margin before 630s kill


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.stack([cx + d * np.cos(th), cy + d * np.sin(th)], axis=1)


def _lattice_hexagon(cx, cy, d, a_deg):
    """A regular hexagon oriented to match a hexagonal lattice (flat-top)."""
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.stack([cx + d * np.cos(th), cy + d * np.sin(th)], axis=1)


def _build_lattice_clusters():
    """Return list of (12,2) coordinate arrays on triangular lattice (side 1)."""
    dx, dy = 1.5, np.sqrt(3) / 2
    clusters = []

    # Cluster A: center + full ring1 (6) + 5 of ring2
    A = [(0, 0)]
    for i in range(6):
        A.append((dx * np.cos(np.deg2rad(60 * i)), dx * np.sin(np.deg2rad(60 * i))))
    for i in range(5):
        A.append((2 * dx * np.cos(np.deg2rad(60 * i)), 2 * dx * np.sin(np.deg2rad(60 * i))))
    clusters.append(np.array(A))

    # Cluster B: center + full ring1 (6) + 5 of ring2 (rotated selection)
    B = [(0, 0)]
    for i in range(6):
        B.append((dx * np.cos(np.deg2rad(60 * i)), dx * np.sin(np.deg2rad(60 * i))))
    for i in range(5):
        B.append((2 * dx * np.cos(np.deg2rad(60 * i + 30)),
                  2 * dx * np.sin(np.deg2rad(60 * i + 30))))
    clusters.append(np.array(B))

    # Cluster C: 3x4 parallelogram block (12 hexes)
    C = []
    for r in range(3):
        for c in range(4):
            C.append((dx * (c - 1.5), dy * (r - 1)))
    clusters.append(np.array(C))

    return clusters


def _min_enclosing_hexagon_size(points):
    """
    Given a set of points, find the minimum side length s of a regular hexagon
    (centered at origin, orientation free) that contains all points.
    Returns best (s, a_deg).
    """
    best_s = np.inf
    best_a = 0.0
    # Try orientations 0..179 deg in 0.5 deg steps, then refine
    for a_deg in np.arange(0, 180, 0.5):
        a = np.deg2rad(a_deg)
        th = a + np.deg2rad(60.0) * np.arange(6)
        # Hexagon: |dot(p, n_i)| <= s/2 for all 6 edge normals n_i = (cos(th_i), sin(th_i))
        # where n_i is the outward normal of edge i, i.e. direction to vertex i.
        # Actually for a regular hexagon with vertex direction th_i, the edge normal
        # (inward) is at th_i, and the constraint is p . n_i <= s/2 where n_i = (cos th_i, sin th_i).
        # Wait: for a regular hexagon centered at origin with vertex at angle th_i and side s,
        # the apothem is s * sqrt(3)/2. The half-width in direction n_i (outward normal) is apothem = s*sqrt(3)/2.
        # The constraint: for all 6 directions n_i (edge normals), |p . n_i| <= apothem.
        # Edge normals are perpendicular to edges. For a regular hexagon with vertices at th_i,
        # the edge between vertex i and i+1 has normal direction th_i + 30deg.
        # Let me use: apothem = s * sqrt(3)/2. Edge normal directions = th_i + 30deg.
        normals = th + np.deg2rad(30.0)
        # For each point, max over 6 normals of |p . n|
        # = max over normals of (p . n) since we take absolute via both +n and -n
        # Actually |p.n| = max(p.n, -p.n), so max over all 12 directions
        all_normals = np.concatenate([normals, normals + np.pi])
        proj = points @ all_normals.T  # (N, 12)
        max_proj = proj.max(axis=1)  # (N,)
        needed_apothem = max_proj.max()
        s = needed_apothem * 2.0 / np.sqrt(3)
        if s < best_s:
            best_s = s
            best_a = a_deg
    # Refine around best_a
    for a_deg in np.arange(best_a - 1.0, best_a + 1.0, 0.05):
        a = np.deg2rad(a_deg)
        th = a + np.deg2rad(60.0) * np.arange(6)
        normals = th + np.deg2rad(30.0)
        all_normals = np.concatenate([normals, normals + np.pi])
        proj = points @ all_normals.T
        max_proj = proj.max(axis=1)
        needed_apothem = max_proj.max()
        s = needed_apothem * 2.0 / np.sqrt(3)
        if s < best_s:
            best_s = s
            best_a = a_deg
    return best_s, best_a


def _validate(inner, outer, s, verbose=False):
    """Check validity. Returns (is_valid, max_violation)."""
    # Check finite
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer)) and np.isfinite(s)):
        return False, np.inf
    if s < np.sqrt(12) - 1e-9:
        return False, np.inf

    # Check inner hexagons are unit side
    for i in range(12):
        cx, cy, a = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, a)
        for j in range(6):
            edge_len = np.linalg.norm(verts[(j + 1) % 6] - verts[j])
            if abs(edge_len - 1.0) > 1e-8:
                return False, abs(edge_len - 1.0)

    # Check outer hexagon vertices distinct
    ox, oy, oa = outer
    outer_verts = _hex_vertices(ox, oy, s, oa)
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False, 0.0

    # Shift all by outer center
    shift = np.array([ox, oy])
    inner_shifted = inner[:, :2] - shift
    outer_shifted_verts = outer_verts - shift

    # Separating axis test between pairs of inner hexagons
    for i in range(12):
        for j in range(i + 1, 12):
            vi = _hex_vertices(inner_shifted[i, 0], inner_shifted[i, 1], 1.0, inner[i, 2])
            vj = _hex_vertices(inner_shifted[j, 0], inner_shifted[j, 1], 1.0, inner[j, 2])
            # Edge normals of both hexagons
            all_ok = True
            for verts in [vi, vj]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    normal = normal / np.linalg.norm(normal)
                    proj_i = vi @ normal
                    proj_j = vj @ normal
                    # Intervals [min_i, max_i] and [min_j, max_j]
                    if proj_i.max() < proj_j.min() - 1e-6 or proj_j.max() < proj_i.min() - 1e-6:
                        # Separated: OK
                        continue
                    else:
                        all_ok = False
                        break
                if not all_ok:
                    break
            if not all_ok:
                return False, 0.0  # overlap

    # Check inner vertices inside outer hexagon
    for i in range(12):
        vi = _hex_vertices(inner_shifted[i, 0], inner_shifted[i, 1], 1.0, inner[i, 2])
        for v in vi:
            for k in range(6):
                p1 = outer_shifted_verts[k]
                p2 = outer_shifted_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False, -cross

    return True, 0.0


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns (inner, outer, s).
    """
    t_start = time.time()
    clusters = _build_lattice_clusters()

    best_result = None
    best_s = np.inf

    for cluster in clusters:
        if time.time() > _DEADLINE:
            break
        # Compute min enclosing hexagon
        s, a_deg = _min_enclosing_hexagon_size(cluster)

        # Build inner hex data: all at angle 0 (lattice orientation)
        inner = np.zeros((12, 3))
        inner[:, 0] = cluster[:, 0]
        inner[:, 1] = cluster[:, 1]
        inner[:, 2] = 0.0  # all aligned with lattice

        # Outer hexagon centered at origin with side s and angle a_deg
        outer = np.array([0.0, 0.0, a_deg])

        # Validate
        valid, _ = _validate(inner, outer, s)
        if valid:
            if s < best_s:
                best_s = s
                best_result = (inner.copy(), outer.copy(), s)

    if best_result is None:
        # Fallback: large outer hexagon
        inner = np.zeros((12, 3))
        outer = np.array([0.0, 0.0, 0.0])
        s = 10.0
        return inner, outer, s

    return best_result


# EVOLVE-BLOCK-END
