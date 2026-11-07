"""Portfolio-search program for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 300.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _make_fcc(n):
    """Generate an FCC-like cluster by taking atoms from a cubic grid with FCC offsets."""
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 3
    # FCC basis vectors
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                # FCC lattice: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) offsets
                for off in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + off[0], j + off[1], k + off[2]], dtype=float)
                    pts.append(p)
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
        if len(pts) >= n:
            break
    pts = np.array(pts[:n])
    # Center the cluster
    pts -= pts.mean(axis=0)
    return pts


def _make_bcc(n):
    """Generate a BCC-like cluster."""
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 3
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                # BCC: (0,0,0) and (0.5,0.5,0.5)
                pts.append(np.array([i, j, k], dtype=float))
                pts.append(np.array([i + 0.5, j + 0.5, k + 0.5], dtype=float))
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
        if len(pts) >= n:
            break
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    return pts


def _make_icosahedron_shell(n):
    """Generate an icosahedral-like cluster with shells."""
    # Icosahedron vertices (golden ratio based)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])  # normalize
    
    pts = [np.zeros(3)]  # center atom
    # First shell: 12 vertices
    for v in verts:
        pts.append(v * 1.0)
    
    if len(pts) >= n:
        return np.array(pts[:n])
    
    # Second shell: 30 atoms at edge midpoints
    edges = [(0,1),(1,3),(3,2),(2,0),
             (0,4),(1,5),(2,6),(3,7),
             (0,8),(1,9),(2,10),(3,11),
             (4,5),(5,7),(7,6),(6,4),
             (8,9),(9,11),(11,10),(10,8),
             (4,8),(5,9),(6,10),(7,11),
             (4,10),(5,11),(6,9),(7,8)]
    shell2 = []
    for a, b in edges:
        mid = (verts[a] + verts[b]) / 2
        mid = mid / np.linalg.norm(mid) * np.sqrt(2)  # scale to shell 2 radius
        shell2.append(mid)
    for v in shell2:
        pts.append(v)
        if len(pts) >= n:
            break
    
    if len(pts) >= n:
        return np.array(pts[:n])
    
    # Third shell: fill with more points
    rng = np.random.default_rng(42)
    while len(pts) < n:
        v = verts[rng.integers(0, 12)]
        p = v * (np.sqrt(2) + rng.uniform(0.1, 0.5))
        pts.append(p)
    
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    return pts


def _make_random_compact(n, spacing=0.72, seed=None):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    for _ in range(2000):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
                points.append(p)
    if len(points) < n:
        # Fallback: just place remaining atoms randomly
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points[:n])


def construct(instance):
    """Best of multiple structural starting clusters, each relaxed locally."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    if _time_left() < 5:
        # Emergency: return a simple valid cluster
        pts = _make_random_compact(size, seed=0)
        energy, positions = relax(pts, potential, kind)
        return positions
    
    # Generate diverse starting structures
    candidates = []
    
    # FCC-based
    try:
        fcc_pts = _make_fcc(size)
        if _is_valid(fcc_pts):
            candidates.append(fcc_pts)
    except Exception:
        pass
    
    # BCC-based
    try:
        bcc_pts = _make_bcc(size)
        if _is_valid(bcc_pts):
            candidates.append(bcc_pts)
    except Exception:
        pass
    
    # Icosahedral
    try:
        ico_pts = _make_icosahedron_shell(size)
        if _is_valid(ico_pts):
            candidates.append(ico_pts)
    except Exception:
        pass
    
    # Random compact (multiple seeds)
    for seed in [0, 1, 2, 3]:
        try:
            rand_pts = _make_random_compact(size, seed=seed)
            if _is_valid(rand_pts):
                candidates.append(rand_pts)
        except Exception:
            pass
    
    best_energy = np.inf
    best_positions = None
    
    for idx, start_pts in enumerate(candidates):
        if _time_left() < 10:
            break
        try:
            energy, positions = relax(start_pts, potential, kind)
            if _is_valid(positions) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback
        pts = _make_random_compact(size, seed=99)
        energy, positions = relax(pts, potential, kind)
        best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
