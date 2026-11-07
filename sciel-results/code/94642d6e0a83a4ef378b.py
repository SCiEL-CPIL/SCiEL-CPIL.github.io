import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
except ImportError:
    relax = None
    sutton_chen_energy_gradient = None

_GLOBAL_START_TIME = time.time()
_TOTAL_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 12.0
_lock = threading.Lock()


def _get_time_left():
    return _TOTAL_TIME_LIMIT - (time.time() - _GLOBAL_START_TIME) - _SAFETY_MARGIN


def _validate_positions(positions):
    if positions is None or len(positions) == 0:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _icosahedron_13():
    """Return the 13-atom icosahedral cluster (center + 12 vertices)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    pts = np.vstack([np.zeros(3), verts])
    return pts


def _icosahedron_55():
    """Return the 55-atom Mackay icosahedron (1 + 12 + 42)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # First shell: 12 vertices at radius 1
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    shell1 = verts / norms  # radius 1

    # Second shell: 42 atoms (12 face centers + 30 edge midpoints)
    # In a Mackay icosahedron, the second shell radius is sqrt(5 + 2*phi)
    R2 = np.sqrt(5.0 + 2.0 * phi)  # ≈ 2.802
    # Face centers: 20 faces, but in Mackay only 12 are used? No, 12 face centers + 30 edge midpoints = 42
    # Actually for 55-atom Mackay: 1 center, 12 first shell, 42 second shell.
    # The 42 second-shell atoms are at the centers of the 20 triangular faces and midpoints of 30 edges.
    # Wait, 20 + 30 = 50, not 42. Let me reconsider.
    # Actually the standard Mackay icosahedron of order 2 has 55 atoms: 1 + 12 + 42.
    # The 42 are arranged as 12 at face centers and 30 at edge midpoints? That's 42. Yes!
    # But an icosahedron has 20 faces and 30 edges. So 12 face centers (not all 20) + 30 edge midpoints = 42.
    # Hmm, that's asymmetric. Let me just use a known construction.
    
    # Alternative: generate 42 points on a sphere of radius R2 with icosahedral symmetry approximation.
    # Use Fibonacci sphere for 42 points, then relax will fix it.
    N2 = 42
    z = (2 * np.arange(N2) + 1) / N2
    r = np.sqrt(1 - z**2)
    phi_angle = np.arange(N2) * (np.pi * (3 - np.sqrt(5)))
    x = r * np.cos(phi_angle)
    y = r * np.sin(phi_angle)
    shell2 = np.column_stack([x, y, z]) * R2

    pts = np.vstack([np.zeros(3), shell1, shell2])
    return pts


def _fcc_sphere(n):
    """Generate FCC-like spherical cluster."""
    a = 1.0
    L = int(np.cbrt(n)) + 3
    pts = []
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    pts.append([(i + dx) * a, (j + dy) * a, (k + dz) * a])
    pts = np.array(pts)
    center = pts.mean(axis=0)
    pts -= center
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    # Scale to reasonable size
    max_r = np.max(np.linalg.norm(pts, axis=1))
    if max_r > 0:
        pts = pts * (4.0 / max_r)
    return pts


def _bcc_sphere(n):
    """Generate BCC-like spherical cluster."""
    a = 1.0
    L = int(np.cbrt(n)) + 3
    pts = []
    for i in range(L):
        for j in range(L):
            for k in range(L):
                pts.append([(i) * a, (j) * a, (k) * a])
                if i < L - 1 and j < L - 1 and k < L - 1:
                    pts.append([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a])
    pts = np.array(pts)
    center = pts.mean(axis=0)
    pts -= center
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    max_r = np.max(np.linalg.norm(pts, axis=1))
    if max_r > 0:
        pts = pts * (4.0 / max_r)
    return pts


def _random_compact(n, rng):
    """Generate a compact random cluster via greedy placement."""
    pts = np.zeros((n, 3))
    for i in range(1, n):
        idx = rng.integers(0, i)
        offset = rng.normal(0, 0.6, 3)
        pts[i] = pts[idx] + offset
    return pts


def _build_initial_structures(n, kind, rng):
    """Build a diverse set of initial structures using hierarchical growth."""
    starts = []

    # 1. Icosahedral seed (13 atoms) + growth
    ico13 = _icosahedron_13()
    if n >= 13:
        # Scale icosahedron to fit, then add remaining atoms on a shell
        base = ico13.copy()
        deficit = n - 13
        if deficit > 0:
            # Place new atoms on a spherical shell around the icosahedron
            R_shell = 2.0
            z = (2 * np.arange(deficit) + 1) / deficit
            r = np.sqrt(1 - z**2)
            phi_angle = np.arange(deficit) * (np.pi * (3 - np.sqrt(5)))
            x = r * np.cos(phi_angle)
            y = r * np.sin(phi_angle)
            shell = np.column_stack([x, y, z]) * R_shell
            base = np.vstack([base, shell])
        if _validate_positions(base):
            starts.append(base)

    # 2. Mackay 55 seed + growth (for n >= 55)
    if n >= 55:
        mackay55 = _icosahedron_55()
        deficit = n - 55
        if deficit > 0:
            R_shell = 3.5
            z = (2 * np.arange(deficit) + 1) / deficit
            r = np.sqrt(1 - z**2)
            phi_angle = np.arange(deficit) * (np.pi * (3 - np.sqrt(5)))
            x = r * np.cos(phi_angle)
            y = r * np.sin(phi_angle)
            shell = np.column_stack([x, y, z]) * R_shell
            base = np.vstack([mackay55, shell])
        else:
            base = mackay55
        if _validate_positions(base):
            starts.append(base)

    # 3. FCC sphere
    fcc = _fcc_sphere(n)
    if _validate_positions(fcc):
        starts.append(fcc)

    # 4. BCC sphere
    bcc = _bcc_sphere(n)
    if _validate_positions(bcc):
        starts.append(bcc)

    # 5. Two random compact structures
    for _ in range(2):
        rc = _random_compact(n, rng)
        if _validate_positions(rc):
            starts.append(rc)

    # 6. Additional icosahedral growth with different shell radii
    if n >= 13:
        ico13 = _icosahedron_13()
        deficit = n - 13
        if deficit > 0:
            R_shell = 1.8
            z = (2 * np.arange(deficit) + 1) / deficit
            r = np.sqrt(1 - z**2)
            phi_angle = np.arange(deficit) * (np.pi * (3 - np.sqrt(5)))
            x = r * np.cos(phi_angle)
            y = r * np.sin(phi_angle)
            shell = np.column_stack([x, y, z]) * R_shell
            base = np.vstack([ico13, shell])
            if _validate_positions(base):
                starts.append(base)

    return starts


def construct(instance):
    n = instance["n"]
    kind = instance.get("kind", "12-6")

    seed = n * 100 + (0 if kind == '12-6' else 1 if kind == '9-6' else 2)
    rng = np.random.default_rng(seed)

    best_energy = np.inf
    best_positions = None

    # Build diverse initial structures
    starts = _build_initial_structures(n, kind, rng)

    if not starts:
        # Fallback: pure random
        starts.append(rng.uniform(-3, 3, (n, 3)))

    def _relax_one(pos):
        try:
            energy, positions = relax(pos, 'sutton_chen', kind)
            if _validate_positions(positions):
                return energy, positions
            else:
                return np.inf, None
        except Exception:
            return np.inf, None

    # Phase 1: Relax all initial structures in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_relax_one, start): start for start in starts}
        for future in as_completed(futures):
            if _get_time_left() < 5.0:
                break
            try:
                energy, positions = future.result()
                if energy < best_energy:
                    with _lock:
                        best_energy = energy
                        best_positions = positions.copy()
            except Exception:
                pass

    if best_positions is None:
        # Fallback
        best_positions = starts[0]
        try:
            energy, positions = relax(best_positions, 'sutton_chen', kind)
            if _validate_positions(positions):
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            pass

    # Phase 2: Basin hopping with adaptive kick amplitudes
    kick_amplitudes = [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
    kick_idx = 0
    stagnation_count = 0
    max_stagnation = 15

    while _get_time_left() > 10.0:
        if best_positions is None:
            break

        # Adaptive kick: increase amplitude if stagnating
        amp = kick_amplitudes[kick_idx % len(kick_amplitudes)]
        if stagnation_count > max_stagnation:
            amp = min(amp * 1.5, 1.5)
            stagnation_count = 0
            kick_idx = 0

        noise = rng.normal(0, amp, best_positions.shape)
        perturbed = best_positions + noise

        if not _validate_positions(perturbed):
            noise = rng.normal(0, amp * 0.4, best_positions.shape)
            perturbed = best_positions + noise
            if not _validate_positions(perturbed):
                kick_idx += 1
                continue

        try:
            energy, positions = relax(perturbed, 'sutton_chen', kind)
            if _validate_positions(positions):
                if energy < best_energy - 1e-10:
                    with _lock:
                        best_energy = energy
                        best_positions = positions.copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1
        except Exception:
            pass

        kick_idx += 1

    # Phase 3: Final polish with smaller kicks
    if best_positions is not None and _get_time_left() > 5.0:
        for amp in [0.05, 0.03, 0.01]:
            if _get_time_left() < 3.0:
                break
            noise = rng.normal(0, amp, best_positions.shape)
            perturbed = best_positions + noise
            if _validate_positions(perturbed):
                try:
                    energy, positions = relax(perturbed, 'sutton_chen', kind)
                    if _validate_positions(positions) and energy < best_energy:
                        best_energy = energy
                        best_positions = positions.copy()
                except Exception:
                    pass

    return best_positions
