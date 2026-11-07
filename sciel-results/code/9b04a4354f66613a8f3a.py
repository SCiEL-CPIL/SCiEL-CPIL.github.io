"""Basin Hopping search for Sutton-Chen TM cluster global minima."""
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
    """Generate a compact FCC-like cluster of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 2
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                pts.append([i, j, k])
    radius = 0.6 * n ** (1.0 / 3.0)
    pts = [p for p in pts if np.linalg.norm(p) <= radius]
    if len(pts) < n:
        mc2 = mc + 2
        for i in range(-mc2, mc2 + 1):
            for j in range(-mc2, mc2 + 1):
                for k in range(-mc2, mc2 + 1):
                    p = [i, j, k]
                    if p not in pts:
                        pts.append(p)
        pts = [p for p in pts if np.linalg.norm(p) <= radius + 1.5]
    pts.sort(key=lambda p: np.linalg.norm(p))
    pts = pts[:n]
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            pts.append(p.tolist())
    return np.array(pts, dtype=float)


def _make_bcc(n):
    """Generate a compact BCC-like cluster of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 2
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                pts.append([i, j, k])
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    radius = 0.6 * n ** (1.0 / 3.0)
    pts = [p for p in pts if np.linalg.norm(p) <= radius]
    if len(pts) < n:
        mc2 = mc + 2
        for i in range(-mc2, mc2 + 1):
            for j in range(-mc2, mc2 + 1):
                for k in range(-mc2, mc2 + 1):
                    for p in [[i, j, k], [i + 0.5, j + 0.5, k + 0.5]]:
                        if p not in pts:
                            pts.append(p)
        pts = [p for p in pts if np.linalg.norm(p) <= radius + 1.5]
    pts.sort(key=lambda p: np.linalg.norm(p))
    pts = pts[:n]
    if len(pts) < n:
        rng = np.random.default_rng(43)
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            pts.append(p.tolist())
    return np.array(pts, dtype=float)


def _make_icosahedron(n):
    """Generate a Mackay icosahedral cluster of n atoms.
    
    Mackay icosahedra are known near-minima for Sutton-Chen clusters.
    Shells: M(1), M(12), M(55), M(120), ...
    We build concentric icosahedral shells and take the first n atoms.
    """
    pts = []
    
    # Icosahedral vertices (golden ratio based)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])  # normalize to unit sphere
    
    # Shell 1: 1 atom at center
    pts.append([0.0, 0.0, 0.0])
    
    # Shell 2: 12 atoms at icosahedral vertices, radius ~1.5 (first neighbor)
    r1 = 1.5
    for v in ico_verts:
        pts.append(list(r1 * v))
    
    # Shell 3: 30 atoms (edges of icosahedron, at midpoints scaled)
    # The 30 edge-midpoints of an icosahedron form a dodecahedron
    # Compute edge midpoints
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if d < 1.0:  # adjacent vertices (edge length ~1.05 for unit ico)
                edges.append((i, j))
    
    r2 = 2.5
    for i, j in edges:
        mid = (ico_verts[i] + ico_verts[j]) / 2
        mid = mid / np.linalg.norm(mid)
        pts.append(list(r2 * mid))
    
    # Shell 4: 20 atoms at face centers (pentagonal faces)
    # Find faces of the icosahedron
    faces = []
    # For each triple of vertices, check if they form a face
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                d3 = np.linalg.norm(ico_verts[k] - ico_verts[i])
                if d1 < 1.0 and d2 < 1.0 and d3 < 1.0:
                    faces.append((i, j, k))
    
    r3 = 3.3
    for i, j, k in faces:
        centroid = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3
        centroid = centroid / np.linalg.norm(centroid)
        pts.append(list(r3 * centroid))
    
    # If we need more atoms, add additional shells
    if n > len(pts):
        # Add atoms on a cubic grid to fill up
        mc = int(np.ceil(n ** (1/3))) + 2
        for i in range(-mc, mc + 1):
            for j in range(-mc, mc + 1):
                for k in range(-mc, mc + 1):
                    p = [i * 0.8, j * 0.8, k * 0.8]
                    if p not in pts:
                        pts.append(p)
        # Sort by distance and take first n
        pts.sort(key=lambda p: np.linalg.norm(p))
        pts = pts[:n]
    
    pts = pts[:n]
    
    if len(pts) < n:
        rng = np.random.default_rng(44)
        radius = 0.7 * n ** (1.0 / 3.0)
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            pts.append(p.tolist())
    
    return np.array(pts, dtype=float)


def _make_random_cluster(n, seed=0):
    """Generate a random compact cluster of n atoms."""
    rng = np.random.default_rng(seed)
    radius = 0.7 * n ** (1.0 / 3.0)
    pts = []
    for _ in range(n):
        while True:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                break
        pts.append(p)
    return np.array(pts, dtype=float)


def _basin_hopping_step(positions, kick_strength, rng, potential, kind):
    """Perform one basin hopping step: kick + relax."""
    # Apply random perturbation
    kick = rng.normal(0, kick_strength, positions.shape)
    perturbed = positions + kick
    
    # Relax the perturbed structure
    try:
        energy, relaxed = relax(perturbed, potential, kind)
        if _is_valid(relaxed):
            return energy, relaxed
    except Exception:
        pass
    return np.inf, None


def construct(instance):
    """Basin Hopping search with multiple diverse starting structures."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting structures
    starts = [
        _make_icosahedron(size),  # Best: Mackay icosahedron
        _make_fcc(size),
        _make_bcc(size),
        _make_random_cluster(size, seed=size),
        _make_random_cluster(size, seed=size + 100),
    ]
    
    # First pass: relax each starting structure
    for start in starts:
        if _time_left() < 10.0:
            break
        try:
            energy, positions = relax(start, potential, kind)
            if _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            continue
    
    if best_positions is None:
        best_positions = _make_random_cluster(size, seed=size + 200)
        try:
            best_energy, best_positions = relax(best_positions, potential, kind)
        except Exception:
            best_energy = np.inf
    
    # Second pass: Basin Hopping from the best structure
    if _time_left() > 30.0:
        rng = np.random.default_rng(size)
        current_energy = best_energy
        current_positions = best_positions.copy()
        kick_strength = 0.3
        
        # Adaptive basin hopping
        for step in range(50):
            if _time_left() < 10.0:
                break
            
            new_energy, new_positions = _basin_hopping_step(
                current_positions, kick_strength, rng, potential, kind
            )
            
            if new_positions is not None:
                # Metropolis acceptance criterion
                if new_energy < current_energy:
                    current_energy = new_energy
                    current_positions = new_positions
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_positions = current_positions.copy()
                    # Decrease kick strength slightly on success
                    kick_strength *= 0.95
                else:
                    # Accept with probability exp(-dE/kT)
                    delta_e = new_energy - current_energy
                    if delta_e < 0.5:  # Only accept small uphill moves
                        accept_prob = np.exp(-delta_e / 0.1)
                        if rng.random() < accept_prob:
                            current_energy = new_energy
                            current_positions = new_positions
                    # Increase kick strength on rejection
                    kick_strength *= 1.05
                    kick_strength = min(kick_strength, 1.0)
        
        # Final relaxation of the best found structure
        if _time_left() > 5.0:
            try:
                final_energy, final_positions = relax(best_positions, potential, kind)
                if _is_valid(final_positions) and final_energy < best_energy:
                    best_energy = final_energy
                    best_positions = final_positions
            except Exception:
                pass
    
    return best_positions
# EVOLVE-BLOCK-END
