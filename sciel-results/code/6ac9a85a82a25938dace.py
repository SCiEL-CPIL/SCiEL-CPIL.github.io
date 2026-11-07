"""Portfolio-search program for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 290.0  # safety margin below 300s


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    if positions is None:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _dedupe(pts, tol=0.05):
    unique = [pts[0]]
    for p in pts[1:]:
        if all(np.linalg.norm(p - q) > tol for q in unique):
            unique.append(p)
    return np.array(unique)


def _scale_to_spacing(pts, spacing=0.72):
    if len(pts) < 2:
        return pts
    rmax = np.max(np.linalg.norm(pts, axis=1))
    if rmax > 0:
        target_r = spacing * len(pts) ** (1.0 / 3.0) * 0.55
        pts = pts * (target_r / rmax)
    return pts


def _make_fcc(n):
    """FCC-like cluster: nearest n atoms from FCC lattice."""
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 3
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    pts.append(np.array([i + dx, j + dy, k + dz]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    return _scale_to_spacing(pts)


def _make_bcc(n):
    """BCC-like cluster."""
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 3
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                pts.append(np.array([i, j, k]))
                pts.append(np.array([i + 0.5, j + 0.5, k + 0.5]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    return _scale_to_spacing(pts)


def _make_mackay(n):
    """Mackay icosahedral cluster with concentric shells.
    Shell radii: 0, 1, sqrt(2), sqrt(3), 2, sqrt(5), ...
    Counts per shell: 1, 12, 42, 92, 156, ...
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 0, phi], [-1, 0, phi], [1, 0, -phi], [-1, 0, -phi],
        [0, phi, 1], [0, -phi, 1], [0, phi, -1], [0, -phi, -1],
        [phi, 1, 0], [-phi, 1, 0], [phi, -1, 0], [-phi, -1, 0]], dtype=float)
    verts /= np.linalg.norm(verts[0])

    # Generate points on icosahedral shells
    # Use a systematic approach: generate all points in a cubical grid
    # and keep those with icosahedral symmetry, or use shell-by-shell construction.

    # Simpler: generate points at each shell radius using icosahedral directions
    # Shell 1: 12 vertices at r=1
    # Shell 2: 42 points (30 edge midpoints + 12 face centers... actually 42 = 12+30)
    # Let's use a known construction: for Mackay clusters, the points are
    # all integer lattice points (i,j,k) with |i|+|j|+|k| <= m and icosahedral constraints.

    # Alternative: use the fact that Mackay clusters are nested icosahedra.
    # For n=1: center
    # For n=13: 1 + 12
    # For n=55: 1 + 12 + 42
    # For n=147: 1 + 12 + 42 + 92
    # For n=309: 1 + 12 + 42 + 92 + 156

    # Generate icosahedral directions for each shell
    # Shell 1 (12 points): the 12 vertices
    # Shell 2 (42 points): 30 edge midpoints + 12 face centers
    # Shell 3 (92 points): more complex

    # Let's use a simpler approach: generate a large set of icosahedral points
    # and take the n closest to the center.

    # Generate points on icosahedral shells
    pts = [np.zeros(3)]

    # Shell 1: 12 vertices at r=1
    for v in verts:
        pts.append(1.0 * v)

    # Shell 2: 42 points at r=sqrt(2)
    # 30 edge midpoints
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.05:
                edges.append((i, j))
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        mid /= np.linalg.norm(mid)
        pts.append(np.sqrt(2) * mid)

    # 12 face centers
    face_centers = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                d_ki = np.linalg.norm(verts[k] - verts[i])
                if d_ij < 1.05 and d_jk < 1.05 and d_ki < 1.05:
                    center = (verts[i] + verts[j] + verts[k]) / 3
                    center /= np.linalg.norm(center)
                    face_centers.append(center)
    unique_fc = []
    for fc in face_centers:
        if all(np.linalg.norm(fc - q) > 0.1 for q in unique_fc):
            unique_fc.append(fc)
    for v in unique_fc:
        pts.append(np.sqrt(2) * v)

    # Shell 3: 92 points at r=sqrt(3)
    # Generate more icosahedral directions
    # Use combinations of vertices and edge midpoints
    shell3_dirs = []
    # Add more directions from icosahedral symmetry
    # For now, add scaled versions of existing directions
    for v in verts:
        shell3_dirs.append(v)
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        mid /= np.linalg.norm(mid)
        shell3_dirs.append(mid)
    for v in unique_fc:
        shell3_dirs.append(v)

    # Add more unique directions
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if np.linalg.norm(verts[i] - verts[j]) < 1.05 and \
                   np.linalg.norm(verts[j] - verts[k]) < 1.05:
                    # Find a direction perpendicular to the edge
                    edge_dir = verts[k] - verts[j]
                    edge_dir /= np.linalg.norm(edge_dir)
                    # Perpendicular in the plane
                    perp = np.cross(verts[i] - verts[j], edge_dir)
                    if np.linalg.norm(perp) > 0.01:
                        perp /= np.linalg.norm(perp)
                        shell3_dirs.append(perp)

    # Deduplicate shell3_dirs
    unique_s3 = []
    for d in shell3_dirs:
        d = d / np.linalg.norm(d)
        if all(np.linalg.norm(d - q) > 0.1 for q in unique_s3):
            unique_s3.append(d)

    for v in unique_s3:
        pts.append(np.sqrt(3) * v)

    # Shell 4: 156 points at r=2
    shell4_dirs = []
    for v in verts:
        shell4_dirs.append(v)
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        mid /= np.linalg.norm(mid)
        shell4_dirs.append(mid)
    for v in unique_fc:
        shell4_dirs.append(v)
    for d in unique_s3:
        shell4_dirs.append(d)
    # Add more
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if np.linalg.norm(verts[i] - verts[j]) < 1.05 and \
                   np.linalg.norm(verts[j] - verts[k]) < 1.05 and \
                   np.linalg.norm(verts[k] - verts[i]) < 1.05:
                    center = (verts[i] + verts[j] + verts[k]) / 3
                    center /= np.linalg.norm(center)
                    shell4_dirs.append(center)

    unique_s4 = []
    for d in shell4_dirs:
        d = d / np.linalg.norm(d)
        if all(np.linalg.norm(d - q) > 0.08 for q in unique_s4):
            unique_s4.append(d)

    for v in unique_s4:
        pts.append(2.0 * v)

    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    pts = _dedupe(pts)

    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    return _scale_to_spacing(pts)


def _make_random_compact(n, rng, spacing=0.72):
    """Generate random compact cluster with minimum distance constraint."""
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    attempts = 0
    min_dist = 0.55 * spacing
    while len(points) < n and attempts < 30000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
    if len(points) == n:
        return np.array(points)
    return None


def _perturb(positions, rng, scale):
    return positions + rng.normal(0, scale, positions.shape)


def _basin_hop(start, potential, kind, rng, max_kicks=15, kick_scale=0.05, time_budget=None):
    """Simple basin-hopping: relax, kick, accept if lower energy."""
    best_energy, best_pos = np.inf, start.copy()
    current_energy, current_pos = np.inf, start.copy()

    for kick in range(max_kicks):
        if time_budget is not None and _time_left() < time_budget:
            break
        try:
            energy, positions = relax(current_pos, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                current_energy = energy
                current_pos = positions
                if energy < best_energy:
                    best_energy = energy
                    best_pos = positions.copy()
        except Exception:
            pass

        # Kick
        kick_scale_current = kick_scale * (1 + 0.1 * kick)
        kicked = _perturb(current_pos, rng, kick_scale_current)
        if not _is_valid(kicked):
            kicked = current_pos + rng.normal(0, kick_scale_current * 0.5, current_pos.shape)
        if not _is_valid(kicked):
            break
        current_pos = kicked

    return best_energy, best_pos


def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(n * 12345 + 7)
    best_energy = np.inf
    best_positions = None

    # Generate structured templates
    templates = []
    try:
        fcc = _make_fcc(n)
        if _is_valid(fcc):
            templates.append(("fcc", fcc))
    except Exception:
        pass

    try:
        bcc = _make_bcc(n)
        if _is_valid(bcc):
            templates.append(("bcc", bcc))
    except Exception:
        pass

    try:
        mackay = _make_mackay(n)
        if _is_valid(mackay):
            templates.append(("mackay", mackay))
    except Exception:
        pass

    # Add random compact starts
    n_random = min(6, max(3, n // 15))
    for _ in range(n_random):
        rc = _make_random_compact(n, rng)
        if rc is not None and _is_valid(rc):
            templates.append(("random", rc))

    # Phase 1: Relax each template directly
    for name, template in templates:
        if _time_left() < 10.0:
            break
        try:
            energy, positions = relax(template, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # Phase 2: Basin-hopping from each template
    for name, template in templates:
        if _time_left() < 15.0:
            break
        try:
            energy, positions = _basin_hop(
                template, potential, kind, rng,
                max_kicks=10, kick_scale=0.04, time_budget=10.0
            )
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # Phase 3: Perturbed templates + relax
    perturbation_scales = [0.03, 0.06, 0.1, 0.15]
    for name, template in templates[:4]:  # Limit to first 4 templates
        if _time_left() < 10.0:
            break
        for scale in perturbation_scales:
            if _time_left() < 5.0:
                break
            pert = _perturb(template, rng, scale)
            if not _is_valid(pert):
                continue
            try:
                energy, positions = relax(pert, potential, kind)
                if np.isfinite(energy) and _is_valid(positions):
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
            except Exception:
                pass

    # Phase 4: More random starts if time remains
    if _time_left() > 20.0:
        for _ in range(5):
            if _time_left() < 10.0:
                break
            rc = _make_random_compact(n, rng)
            if rc is not None and _is_valid(rc):
                try:
                    energy, positions = _basin_hop(
                        rc, potential, kind, rng,
                        max_kicks=8, kick_scale=0.05, time_budget=8.0
                    )
                    if np.isfinite(energy) and _is_valid(positions):
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = positions
                except Exception:
                    pass

    # Fallback
    if best_positions is None:
        try:
            best_positions = _make_fcc(n)
        except Exception:
            best_positions = rng.uniform(-1, 1, (n, 3))

    return best_positions
# EVOLVE-BLOCK-END
