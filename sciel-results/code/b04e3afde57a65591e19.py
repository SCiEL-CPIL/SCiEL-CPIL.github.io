"""Dzugutov cluster global minima search with multi-start structural heuristics."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax

_DEADLINE = 320.0  # safety margin before 330s kill
_start_time = None


def _dzugutov_energy(positions):
    """Compute Dzugutov energy for validation."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(positions)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-10:
                continue
            if r < a:
                E += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                E += B * np.exp(d / (r - b))
    return E


def _generate_icosahedral(n, rng):
    """Generate icosahedral-like cluster: 12-vertex icosahedron core + shells."""
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # Icosahedron vertices (12 vertices)
    scale = 1.5  # radius of first shell
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    ico = ico / np.linalg.norm(ico[0]) * scale

    points = [np.zeros(3)]  # center atom
    if n > 1:
        points.append(ico[0])
    if n > 2:
        points.append(ico[1])
    # Add remaining icosahedron vertices
    for v in ico[2:]:
        if len(points) >= n:
            break
        points.append(v)

    # Add second shell (20 vertices of dodecahedron, radius ~2.8)
    if len(points) < n:
        # Dodecahedron vertices
        r2 = 2.8
        # 20 vertices of a regular dodecahedron
        dodec = []
        # Using standard dodecahedron coordinates
        inv_phi = 1.0 / phi
        verts = [
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, inv_phi, phi], [0, inv_phi, -phi], [0, -inv_phi, phi], [0, -inv_phi, -phi],
            [inv_phi, phi, 0], [inv_phi, -phi, 0], [-inv_phi, phi, 0], [-inv_phi, -phi, 0],
            [phi, 0, inv_phi], [phi, 0, -inv_phi], [-phi, 0, inv_phi], [-phi, 0, -inv_phi],
        ]
        for v in verts:
            v = np.array(v, dtype=float)
            v = v / np.linalg.norm(v) * r2
            dodec.append(v)
        for v in dodec:
            if len(points) >= n:
                break
            # Check minimum distance
            if all(np.linalg.norm(v - p) >= 0.55 for p in points):
                points.append(v)

    # Fill remaining with random points on a shell
    if len(points) < n:
        r3 = 3.5
        while len(points) < n:
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * r3
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)

    return np.array(points[:n])


def _generate_fcc(n, rng):
    """Generate FCC lattice chunk."""
    a = 1.6  # lattice constant (equilibrium spacing)
    # Generate FCC points in a box
    points = []
    max_coord = 6
    # FCC basis
    basis = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for bx, by, bz in basis:
                    p = np.array([i * a + bx * a, j * a + by * a, k * a + bz * a])
                    if np.linalg.norm(p) < 4.5:
                        points.append(p)
    # Shuffle and take n
    if len(points) > n:
        idx = rng.choice(len(points), n, replace=False)
        points = [points[i] for i in idx]
    elif len(points) < n:
        # Add random points
        while len(points) < n:
            p = rng.uniform(-4.5, 4.5, 3)
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
    return np.array(points[:n])


def _generate_random_compact(n, rng):
    """Generate random compact cluster."""
    radius = 1.5 * n ** (1.0 / 3.0) * 0.7
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        attempts += 1
    # Fallback: fill with more spread out points
    while len(points) < n:
        p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
        points.append(p)
    return np.array(points[:n])


def _generate_body_centered(n, rng):
    """Generate BCC-like cluster (also has tetrahedral character)."""
    a = 1.5
    points = []
    max_coord = 5
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                p = np.array([i * a, j * a, k * a])
                if np.linalg.norm(p) < 4.0:
                    points.append(p)
                # BCC center
                p2 = np.array([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a])
                if np.linalg.norm(p2) < 4.0:
                    points.append(p2)
    if len(points) > n:
        idx = rng.choice(len(points), n, replace=False)
        points = [points[i] for i in idx]
    elif len(points) < n:
        while len(points) < n:
            p = rng.uniform(-4.0, 4.0, 3)
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
    return np.array(points[:n])


def _worker(args):
    """Worker: relax a single starting structure."""
    positions, potential, parameter = args
    try:
        energy, relaxed = relax(positions, potential, parameter)
        return (energy, relaxed)
    except Exception:
        return (np.inf, positions)


def construct(instance):
    """Find low-energy Dzugutov cluster via multi-start relaxation."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))

    rng = np.random.default_rng(size * 7919 + 13)

    # Generate diverse starting structures
    generators = [
        _generate_icosahedral,
        _generate_fcc,
        _generate_random_compact,
        _generate_body_centered,
    ]

    tasks = []
    # Multiple random restarts for each generator
    n_restarts = 3
    for gen in generators:
        for r in range(n_restarts):
            try:
                pos = gen(size, rng)
                if len(pos) == size:
                    tasks.append((pos, potential, parameter))
            except Exception:
                pass

    # Also add some perturbed icosahedral structures
    for r in range(2):
        try:
            pos = _generate_icosahedral(size, rng)
            # Add small random perturbations
            pos = pos + rng.normal(0, 0.1, pos.shape)
            tasks.append((pos, potential, parameter))
        except Exception:
            pass

    best_energy = np.inf
    best_positions = None

    # Run relaxations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for task in tasks:
            if time.time() - _start_time > _DEADLINE - 5:
                break
            futures.append(executor.submit(_worker, task))

        for future in concurrent.futures.as_completed(futures):
            if time.time() - _start_time > _DEADLINE - 3:
                break
            try:
                energy, positions = future.result(timeout=2)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass

    # If no valid result, try a simple sequential relaxation
    if best_positions is None:
        try:
            pos = _generate_icosahedral(size, rng)
            energy, best_positions = relax(pos, potential, parameter)
        except Exception:
            # Last resort: random
            pos = rng.uniform(-3, 3, (size, 3))
            best_positions = pos

    # Ensure minimum distance constraint
    # Check and fix if needed
    for i in range(len(best_positions)):
        for j in range(i + 1, len(best_positions)):
            d = np.linalg.norm(best_positions[i] - best_positions[j])
            if d < 0.5:
                # Push apart
                direction = best_positions[j] - best_positions[i]
                if np.linalg.norm(direction) < 1e-10:
                    direction = np.array([1.0, 0, 0])
                direction = direction / np.linalg.norm(direction)
                shift = (0.5 - d) / 2.0
                best_positions[i] -= shift * direction
                best_positions[j] += shift * direction

    return best_positions
# EVOLVE-BLOCK-END
