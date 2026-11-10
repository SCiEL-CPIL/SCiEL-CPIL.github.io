"""Sutton-Chen cluster global minima: multi-start from physically-motivated seeds."""
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = 330.0
_MARGIN = 15.0  # seconds to keep before the kill


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _T0)


_T0 = time.time()


def _icosahedron(n):
    """Generate n points on/near an icosahedral shell (FCC-like dense packing)."""
    # 12 vertices of a regular icosahedron
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    points = []
    # Layer 1: center
    points.append(np.array([0.0, 0.0, 0.0]))
    # Layer 2: icosahedron vertices scaled
    r2 = 1.44
    for v in verts:
        points.append(v * r2)
    # If we need more, add points on a second shell (20 face-centers)
    if n > 13:
        # Face centers of icosahedron
        edges = [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(6,7),(7,8),(8,6),
                 (9,10),(10,11),(11,9),(0,4),(4,1),(1,5),(5,0),
                 (2,3),(3,7),(7,2),(2,6),(6,8),(8,2),
                 (9,6),(6,10),(10,11),(11,9)]
        # Actually let's just add more shells
        r3 = 2.8
        for v in verts:
            points.append(v * r3)
            if len(points) >= n:
                break
        # Add rotated copies
        if len(points) < n:
            rot = np.array([[0,1,0],[-1,0,0],[0,0,1]])
            for v in verts:
                p = rot @ v
                p = p / np.linalg.norm(p)
                points.append(p * r3 * 0.9)
                if len(points) >= n:
                    break
    return np.array(points[:n])


def _fcc_fragment(n):
    """Generate n points from an FCC lattice fragment."""
    points = []
    a = 2.8  # lattice constant for SC (equilibrium ~2.8-3.0)
    # FCC basis
    basis = np.array([[0,0,0],[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]])
    # Grow a cubic region
    max_coord = int(np.ceil((n / 4.0) ** (1.0/3.0))) + 2
    candidates = []
    for ix in range(-max_coord, max_coord+1):
        for iy in range(-max_coord, max_coord+1):
            for iz in range(-max_coord, max_coord+1):
                for b in basis:
                    p = np.array([ix, iy, iz]) * a + b * a
                    candidates.append(p)
    candidates = np.array(candidates)
    # Center
    candidates -= candidates.mean(axis=0)
    # Sort by distance from center, take nearest n
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    points = candidates[order[:n]]
    return points


def _hcp_fragment(n):
    """Generate n points from an HCP-like arrangement."""
    points = []
    a = 2.85
    c = a * np.sqrt(8.0/3.0)
    basis_ab = np.array([
        [0,0,0], [1,0,0], [0.5, np.sqrt(3)/2, 0],
        [0.5, 0, 0], [1.5, 0, 0], [1, np.sqrt(3)/2, 0],
        [0.5, np.sqrt(3)/2, 0], [1.5, np.sqrt(3)/2, 0]
    ])
    # Two-layer AB stacking
    for layer in range(6):
        z = layer * c / 2.0
        offset = (layer % 2) * 0.5
        for ix in range(-3, 4):
            for iy in range(-3, 4):
                if layer % 2 == 0:
                    p = np.array([ix*a + offset*a, iy*a*np.sqrt(3)/2, z])
                else:
                    p = np.array([ix*a + 0.75*a, iy*a*np.sqrt(3)/2 + 0.25*a, z])
                points.append(p)
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    return points[order[:n]]


def _random_compact(n, seed, spacing=2.8):
    """Random compact cluster with minimum separation."""
    rng = np.random.default_rng(seed)
    radius = spacing * (n / 4.0) ** (1.0/3.0) * 1.2
    points = [np.zeros(3)]
    min_sep = 0.4 * spacing
    attempts = 0
    while len(points) < n and attempts < n * 200:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_sep for q in points):
            points.append(p)
    # If not enough, add random
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n])


def _sanitize(positions, n):
    """Ensure positions are finite and no two atoms closer than 0.35."""
    if not np.all(np.isfinite(positions)):
        return None
    # Check pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(positions)
    if np.min(d) < 0.35:
        # Nudge apart
        positions = positions.copy()
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                d_ij = np.linalg.norm(positions[i] - positions[j])
                if d_ij < 0.35:
                    if d_ij < 1e-8:
                        positions[j] += np.array([0.36, 0, 0])
                    else:
                        diff = (positions[j] - positions[i]) / d_ij
                        positions[j] = positions[i] + diff * 0.36
        d2 = pdist(positions)
        if np.min(d2) < 0.35:
            return None
    return positions


def construct(instance):
    global _T0
    _T0 = time.time()
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]

    best_energy = np.inf
    best_positions = None

    def try_relax(positions, label=""):
        nonlocal best_energy, best_positions
        if _time_left() < 5.0:
            return
        positions = np.array(positions, dtype=float)
        if positions.shape != (n, 3):
            return
        try:
            energy, new_pos = relax(positions, potential, kind)
            new_pos = _sanitize(new_pos, n)
            if new_pos is not None and np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = new_pos.copy()
        except Exception:
            pass

    # Seed 1: Icosahedral
    try:
        pos = _icosahedron(n)
        try_relax(pos, "ico")
    except Exception:
        pass

    # Seed 2: FCC fragment
    try:
        pos = _fcc_fragment(n)
        try_relax(pos, "fcc")
    except Exception:
        pass

    # Seed 3: HCP fragment
    try:
        pos = _hcp_fragment(n)
        try_relax(pos, "hcp")
    except Exception:
        pass

    # Seeds 4+: Random compact with different seeds
    for s in range(20):
        if _time_left() < 10.0:
            break
        try:
            pos = _random_compact(n, seed=42 + s, spacing=2.8)
            try_relax(pos, f"rand{s}")
        except Exception:
            pass

    # More random if time remains
    s = 20
    while _time_left() > 15.0:
        try:
            pos = _random_compact(n, seed=1000 + s, spacing=2.8)
            try_relax(pos, f"rand2_{s}")
        except Exception:
            pass
        s += 1

    # Final fallback
    if best_positions is None:
        best_positions = _random_compact(n, seed=0)

    return best_positions
