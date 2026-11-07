"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from itertools import product
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Time management
_START_TIME = time.time()
_TOTAL_TIME_LIMIT = 300.0  # seconds
_SAFETY_MARGIN = 10.0  # seconds

def _time_left():
    return _TOTAL_TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN

# --- Structural initialization helpers ---

def _bcc_positions(n, spacing=1.0, rng=None):
    """Generate n positions in a BCC-like arrangement."""
    if n == 0:
        return np.zeros((0, 3))
    if rng is None:
        rng = np.random.default_rng(0)
    
    a = spacing
    max_coord = int(np.ceil((n / 4) ** (1/3))) + 2
    
    candidates = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                candidates.append(np.array([ix * a, iy * a, iz * a]))
                candidates.append(np.array([(ix + 0.5) * a, (iy + 0.5) * a, (iz + 0.5) * a]))
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    candidates = candidates[order]
    
    # Greedily select with minimum distance
    selected = [candidates[0]]
    min_dist = 0.5 * a
    for p in candidates[1:]:
        if len(selected) >= n:
            break
        if all(np.linalg.norm(p - s) >= min_dist for s in selected):
            selected.append(p)
    
    if len(selected) < n:
        selected = [candidates[i] for i in range(n)]
    
    return np.array(selected[:n])


def _fcc_positions(n, spacing=1.0, rng=None):
    """Generate n positions in an FCC-like arrangement."""
    if n == 0:
        return np.zeros((0, 3))
    if rng is None:
        rng = np.random.default_rng(0)
    
    a = spacing
    max_coord = int(np.ceil((n / 2) ** (1/3))) + 2
    candidates = []
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                candidates.append(np.array([ix * a, iy * a, iz * a]))
                candidates.append(np.array([(ix + 0.5) * a, (iy + 0.5) * a, iz * a]))
                candidates.append(np.array([(ix + 0.5) * a, iy * a, (iz + 0.5) * a]))
                candidates.append(np.array([ix * a, (iy + 0.5) * a, (iz + 0.5) * a]))
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    candidates = candidates[order]
    
    selected = [candidates[0]]
    min_dist = 0.5 * a
    for p in candidates[1:]:
        if len(selected) >= n:
            break
        if all(np.linalg.norm(p - s) >= min_dist for s in selected):
            selected.append(p)
    
    if len(selected) < n:
        selected = [candidates[i] for i in range(n)]
    
    return np.array(selected[:n])


def _random_compact(n, spacing=0.72, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng()
    
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = n * 500
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.4 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    
    return np.array(points)


def _icosahedron(n, spacing=0.72):
    """Generate an icosahedral cluster."""
    if n <= 1:
        return np.zeros((n, 3))
    
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * spacing
    
    positions = [np.array([0.0, 0.0, 0.0])]
    if n <= 1:
        return np.array(positions)
    
    for v in verts:
        if len(positions) >= n:
            break
        positions.append(v)
    
    if len(positions) < n:
        octa = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ])
        octa = octa / np.linalg.norm(octa, axis=1, keepdims=True) * spacing * 1.6
        for v in octa:
            if len(positions) >= n:
                break
            positions.append(v)
    
    while len(positions) < n:
        p = np.random.uniform(-spacing * 2, spacing * 2, 3)
        if all(np.linalg.norm(p - q) >= 0.4 * spacing for q in positions):
            positions.append(p)
    
    return np.array(positions[:n])


def _octahedron(n, spacing=0.72):
    """Generate an octahedral cluster."""
    if n <= 1:
        return np.zeros((n, 3))
    
    octa = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ])
    octa = octa / np.linalg.norm(octa, axis=1, keepdims=True) * spacing
    
    positions = [np.array([0.0, 0.0, 0.0])]
    for v in octa:
        if len(positions) >= n:
            break
        positions.append(v)
    
    if len(positions) < n:
        cube = np.array(list(product([-1, 1], repeat=3)))
        cube = cube / np.linalg.norm(cube, axis=1, keepdims=True) * spacing * 1.4
        for v in cube:
            if len(positions) >= n:
                break
            positions.append(v)
    
    while len(positions) < n:
        p = np.random.uniform(-spacing * 2, spacing * 2, 3)
        if all(np.linalg.norm(p - q) >= 0.4 * spacing for q in positions):
            positions.append(p)
    
    return np.array(positions[:n])


def _cuboctahedron(n, spacing=0.72):
    """Generate a cuboctahedral cluster."""
    if n <= 1:
        return np.zeros((n, 3))
    
    verts = []
    for perm in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        for sign in [1, -1]:
            v = np.zeros(3)
            v[perm[0]] = sign * 1
            v[perm[1]] = sign * 1
            v[perm[2]] = 0
            verts.append(v)
    verts = np.array(verts)
    unique_verts = []
    for v in verts:
        if not any(np.allclose(v, u) for u in unique_verts):
            unique_verts.append(v)
    verts = np.array(unique_verts)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * spacing
    
    positions = [np.array([0.0, 0.0, 0.0])]
    for v in verts:
        if len(positions) >= n:
            break
        positions.append(v)
    
    while len(positions) < n:
        p = np.random.uniform(-spacing * 2, spacing * 2, 3)
        if all(np.linalg.norm(p - q) >= 0.4 * spacing for q in positions):
            positions.append(p)
    
    return np.array(positions[:n])


def _generate_starts(n, kind, rng):
    """Generate multiple diverse starting configurations."""
    spacing = 0.72
    starts = []
    
    try:
        bcc = _bcc_positions(n, spacing, rng)
        if len(bcc) == n and np.all(np.isfinite(bcc)):
            starts.append(bcc)
    except:
        pass
    
    try:
        fcc = _fcc_positions(n, spacing, rng)
        if len(fcc) == n and np.all(np.isfinite(fcc)):
            starts.append(fcc)
    except:
        pass
    
    try:
        icosa = _icosahedron(n, spacing)
        if len(icosa) == n and np.all(np.isfinite(icosa)):
            starts.append(icosa)
    except:
        pass
    
    try:
        octa = _octahedron(n, spacing)
        if len(octa) == n and np.all(np.isfinite(octa)):
            starts.append(octa)
    except:
        pass
    
    try:
        cubocta = _cuboctahedron(n, spacing)
        if len(cubocta) == n and np.all(np.isfinite(cubocta)):
            starts.append(cubocta)
    except:
        pass
    
    for _ in range(4):
        try:
            rand = _random_compact(n, spacing, rng)
            if len(rand) == n and np.all(np.isfinite(rand)):
                starts.append(rand)
        except:
            pass
    
    if len(starts) > 0:
        for _ in range(2):
            jittered = starts[0] + rng.normal(0, 0.05, (n, 3))
            if np.all(np.isfinite(jittered)):
                starts.append(jittered)
    
    return starts


def _validate(positions):
    """Check if positions are valid."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < 0.35:
                return False
    return True


def _relax_with_restarts(positions, kind, rng, max_restarts=5):
    """Relax positions with multiple restarts from jittered versions."""
    best_energy = np.inf
    best_pos = positions.copy()
    
    # First try the original
    try:
        energy, pos = relax(positions, 'sutton_chen', kind)
        if np.all(np.isfinite(pos)) and _validate(pos):
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
    except:
        pass
    
    # Try jittered versions
    for _ in range(max_restarts):
        jitter = rng.normal(0, 0.1, positions.shape)
        start = positions + jitter
        if not _validate(start):
            continue
        try:
            energy, pos = relax(start, 'sutton_chen', kind)
            if np.all(np.isfinite(pos)) and _validate(pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos.copy()
        except:
            pass
    
    return best_energy, best_pos


def _worker(args):
    """Worker function for parallel relaxation."""
    pos, kind, rng_seed = args
    rng = np.random.default_rng(rng_seed)
    try:
        energy, positions = _relax_with_restarts(pos, kind, rng, max_restarts=3)
        return energy, positions
    except Exception:
        return np.inf, None


def construct(instance):
    """Find near-global minimum for Sutton-Chen transition-metal clusters."""
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(n * 12345 + 67890)
    
    # Generate diverse starting configurations
    starts = _generate_starts(n, kind, rng)
    
    if not starts:
        starts = [_random_compact(n, 0.72, rng)]
    
    best_energy = np.inf
    best_positions = starts[0]
    
    # Use parallel workers for the first batch of starts
    n_workers = min(4, len(starts))
    worker_args = [(starts[i], kind, n * 12345 + 67890 + i * 1000) for i in range(n_workers)]
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_worker, args) for args in worker_args]
            for future in concurrent.futures.as_completed(futures):
                if _time_left() < 5.0:
                    break
                try:
                    energy, positions = future.result(timeout=10)
                    if positions is not None and energy < best_energy:
                        best_energy = energy
                        best_positions = positions
                except Exception:
                    continue
    except Exception:
        pass
    
    # Sequential processing for remaining starts
    for i in range(n_workers, len(starts)):
        if _time_left() < 5.0:
            break
        
        if not _validate(starts[i]):
            continue
        
        energy, positions = _relax_with_restarts(starts[i], kind, rng, max_restarts=3)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
    
    # If we have time, try more random starts
    attempts = 0
    max_attempts = 30
    while _time_left() > 10.0 and attempts < max_attempts:
        attempts += 1
        try:
            start = _random_compact(n, 0.72, rng)
            if not _validate(start):
                continue
            
            energy, positions = _relax_with_restarts(start, kind, rng, max_restarts=2)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            continue
    
    # Final validation
    if not _validate(best_positions):
        best_positions = _random_compact(n, 0.72, rng)
        for i in range(len(best_positions)):
            for j in range(i + 1, len(best_positions)):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.35:
                    direction = best_positions[i] - best_positions[j]
                    if np.linalg.norm(direction) > 1e-10:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = np.array([1.0, 0, 0])
                    best_positions[j] = best_positions[i] + direction * 0.36
    
    return best_positions
# EVOLVE-BLOCK-END
