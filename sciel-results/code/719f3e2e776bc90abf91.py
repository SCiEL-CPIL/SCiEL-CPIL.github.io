"""
Sutton-Chen transition-metal cluster global minima search.
Combines: BCC-prioritized multi-start (inspiration 1) with proper icosahedron
generation and a perturb+re-relax refinement loop (current program structure).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

try:
    from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
except ImportError:
    relax = None
    sutton_chen_energy_gradient = None

_GLOBAL_START_TIME = time.time()
_TOTAL_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 12.0

def _get_time_left():
    return _TOTAL_TIME_LIMIT - (time.time() - _GLOBAL_START_TIME) - _SAFETY_MARGIN

def _validate_positions(positions):
    if positions is None:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n <= 1:
        return True
    # Efficient min-distance check
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 0.35:
                return False
    return True

def _icosahedron_vertices():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def _icosahedron_edges_faces():
    """Return edge list and face list for the 12-vertex icosahedron."""
    v = _icosahedron_vertices()
    # Icosahedron edges: pairs of vertices at the minimum distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            dists[i, j] = np.linalg.norm(v[i] - v[j])
    min_d = np.min(dists[dists > 0])
    tol = 1e-9
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_d) < tol:
                edges.append((i, j))
    # Faces: triangles of 3 mutually connected vertices
    adj = {i: set() for i in range(12)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    faces = []
    for i in range(12):
        for j in sorted(adj[i]):
            for k in sorted(adj[j]):
                if k > j and k in adj[i]:
                    faces.append((i, j, k))
    return edges, faces

def _generate_mackay_icosahedron(n, r1=1.0):
    """
    Generate a Mackay icosahedral cluster.
    Shell 1: 12 atoms (icosahedron vertices at radius r1)
    Shell 2: 42 atoms (30 edge midpoints + 12 face centers, at radius r2)
    Shell 3: 100 atoms
    For n=13: center + 12. For n=55: center + 12 + 42.
    For other n, truncate to the nearest shell total (with center).
    """
    if n <= 0:
        return None
    v = _icosahedron_vertices()
    edges, faces = _icosahedron_edges_faces()

    points = [np.zeros(3)]  # center

    # Shell 1: 12 vertices
    shell1 = v * r1
    points.extend([shell1[i] for i in range(12)])

    if n <= 13:
        return np.array(points[:n])

    # Shell 2: 42 atoms
    # 30 edge midpoints: push midpoint of each edge out to the sphere of radius r2
    # 12 face centers: normal of each face, at radius r2
    # In a Mackay icosahedron, r2 = r1 * sqrt(1 + phi) approximately? 
    # Actually the 2nd shell radius: for the standard Mackay icosahedron with
    # 1st shell at distance 1 from center, the 2nd shell is at distance sqrt(1+phi).
    # But the edge midpoints and face centers are at slightly different distances
    # in the true icosahedron; in Mackay they're placed on a sphere.
    # We place them on a sphere of radius r2.
    r2 = r1 * np.sqrt(1.0 + (1.0 + np.sqrt(5.0)) / 2.0)

    shell2 = []
    # 30 edge midpoints
    for i, j in edges:
        mid = (v[i] + v[j]) / 2.0
        mid = mid / np.linalg.norm(mid) * r2
        shell2.append(mid)
    # 12 face centers
    for (i, j, k) in faces:
        # Face normal
        n1 = np.cross(v[j] - v[i], v[k] - v[i])
        n1 = n1 / np.linalg.norm(n1)
        fc = n1 * r2
        shell2.append(fc)

    points.extend(shell2)

    if n <= 55:
        return np.array(points[:n])

    # Shell 3: 100 atoms (more complex; approximate with a larger sphere distribution)
    # For n up to 155, we add shell 3. Approximate by distributing 100 points
    # on a sphere of radius r3.
    r3 = r1 * np.sqrt(1.0 + (1.0 + np.sqrt(5.0)) / 2.0) * 1.5
    # Use a Fibonacci sphere for 100 points
    def fib_sphere(num, radius):
        pts = []
        golden = np.pi * (3.0 - np.sqrt(5.0))
        for k in range(num):
            y = 1.0 - (k / (num - 1)) * 2.0
            r = np.sqrt(1.0 - y*y)
            theta = golden * k
            pts.append([r * np.cos(theta), y, r * np.sin(theta)])
        return np.array(pts) * radius
    shell3 = fib_sphere(100, r3)
    points.extend([shell3[i] for i in range(100)])

    if n <= 155:
        return np.array(points[:n])

    return None

def _generate_fcc_chunk(n):
    """Generate an FCC-like cluster by stacking triangular layers."""
    if n <= 0:
        return np.zeros((0, 3))
    a = 0.72
    def get_layer(k, z=0.0):
        pts = []
        for i in range(k):
            for j in range(k - i):
                x = i * a + j * a * np.cos(np.pi/3)
                y = j * a * np.sin(np.pi/3)
                pts.append([x, y, z])
        return np.array(pts)
    points = []
    z = 0.0
    d = a * np.sqrt(2.0/3.0)
    current_total = 0
    layer_data = []
    k = 1
    while current_total < n:
        num_atoms = k * (k + 1) // 2
        layer_pts = get_layer(k, z)
        if len(layer_pts) > 0:
            center_xy = layer_pts[:, :2].mean(axis=0)
            layer_pts[:, :2] -= center_xy
        layer_data.append(layer_pts)
        current_total += num_atoms
        z += d
        k += 1
        if k > 20:
            break
    all_pts = np.vstack(layer_data) if layer_data else np.array([[0,0,0]])
    center = all_pts.mean(axis=0)
    all_pts -= center
    if len(all_pts) > n:
        dists = np.linalg.norm(all_pts, axis=1)
        order = np.argsort(dists)
        all_pts = all_pts[order[:n]]
    elif len(all_pts) < n:
        # Pad with random near the surface
        rng = np.random.default_rng(42)
        R = np.max(np.linalg.norm(all_pts, axis=1)) if len(all_pts) > 0 else 1.0
        while len(all_pts) < n:
            p = rng.uniform(-R, R, 3)
            all_pts = np.vstack([all_pts, [p]])
    return all_pts

def _generate_bcc_chunk(n):
    """Generate a BCC-like cluster. Measured to start closer to global min."""
    if n <= 0:
        return np.zeros((0, 3))
    a = 0.72
    points = []
    max_side = int(np.ceil(n ** (1.0/3.0))) + 3
    for x in range(max_side):
        for y in range(max_side):
            for z in range(max_side):
                points.append([x*a, y*a, z*a])
                if x < max_side-1 and y < max_side-1 and z < max_side-1:
                    points.append([(x+0.5)*a, (y+0.5)*a, (z+0.5)*a])
                if len(points) >= n + 5:
                    break
            if len(points) >= n + 5:
                break
        if len(points) >= n + 5:
            break
    points = np.array(points[:n])
    if len(points) < n:
        rng = np.random.default_rng(7)
        R = 1.0
        while len(points) < n:
            p = rng.uniform(-R, R, 3)
            points = np.vstack([points, [p]])
    center = points.mean(axis=0)
    points -= center
    return points

def _generate_random_close_pack(n, rng, radius=0.72):
    if n <= 0:
        return np.zeros((0, 3))
    R = radius * n ** (1.0/3.0) * 1.3
    points = [np.zeros(3)]
    min_dist = 0.38 * radius
    attempts = 0
    max_attempts = n * 800
    while len(points) < n and attempts < max_attempts:
        u = rng.uniform(0, 1)
        v = rng.uniform(0, 1)
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        r = R * np.cbrt(rng.uniform(0, 1))
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        p = np.array([x, y, z])
        ok = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                ok = False
                break
        if ok:
            points.append(p)
        attempts += 1
    while len(points) < n:
        points.append(rng.uniform(-R, R, 3))
    return np.array(points)

def _relax_and_record(start, kind, best_state):
    """Relax a start, return (energy, positions) if valid and better."""
    try:
        energy, positions = relax(start, 'sutton_chen', kind)
    except Exception:
        return None
    if not _validate_positions(positions):
        return None
    if energy < best_state[0]:
        best_state[0] = energy
        best_state[1] = positions.copy()
    return (energy, positions)

def construct(instance):
    n = instance["n"]
    kind = instance.get("kind", "12-6")

    seed = n * 100 + (0 if kind == '12-6' else 1 if kind == '9-6' else 2)
    rng = np.random.default_rng(seed)

    best_energy = np.inf
    best_positions = None

    starts = []

    # Priority 1: BCC (measured best)
    try:
        bcc = _generate_bcc_chunk(n)
        if len(bcc) == n and _validate_positions(bcc):
            starts.append(bcc)
    except Exception:
        pass

    # Priority 2: Mackay Icosahedron (magic numbers 13, 55, 147)
    try:
        ico = _generate_mackay_icosahedron(n)
        if ico is not None and len(ico) == n and _validate_positions(ico):
            starts.append(ico)
    except Exception:
        pass

    # Priority 3: FCC
    try:
        fcc = _generate_fcc_chunk(n)
        if len(fcc) == n and _validate_positions(fcc):
            starts.append(fcc)
    except Exception:
        pass

    # Priority 4: Random close packs
    num_rcp = 6
    for i in range(num_rcp):
        try:
            rcp = _generate_random_close_pack(n, rng)
            if len(rcp) == n and _validate_positions(rcp):
                jittered = rcp + rng.normal(0, 0.015, (n, 3))
                if _validate_positions(jittered):
                    starts.append(jittered)
                else:
                    starts.append(rcp)
        except Exception:
            pass

    if not starts:
        pts = np.zeros((n, 3))
        pts[:, 0] = np.arange(n) * 0.8
        starts.append(pts)

    # Phase 1: Relax all starts
    for start in starts:
        if _get_time_left() < 6.0:
            break
        _relax_and_record(start, kind, [best_energy, best_positions] if best_positions is not None else [best_energy, None])
        # Re-check best
        # (relax_and_record updates best_state in place; we use a mutable holder)

    # Use a mutable holder to track best across calls
    # Redo with a proper holder
    best_energy = np.inf
    best_positions = None
    holder = [np.inf, None]
    for start in starts:
        if _get_time_left() < 6.0:
            break
        res = _relax_and_record(start, kind, holder)
        if res is not None:
            holder[0] = res[0]
            holder[1] = res[1]
    best_energy = holder[0]
    best_positions = holder[1]

    # Phase 2: Perturbation + re-relaxation refinement from the best found
    # This is the refinement machinery: kick the best config slightly and re-relax
    # to escape local minima.
    if best_positions is not None and _get_time_left() > 8.0:
        perturb_scale = [0.05, 0.10, 0.20, 0.35, 0.50]
        for scale in perturb_scale:
            if _get_time_left() < 6.0:
                break
            # Try a few perturbations at this scale
            num_pert = 3
            for p in range(num_pert):
                if _get_time_left() < 4.0:
                    break
                try:
                    perturbed = best_positions + rng.normal(0, scale, (n, 3))
                    if not _validate_positions(perturbed):
                        continue
                    energy, positions = relax(perturbed, 'sutton_chen', kind)
                    if _validate_positions(positions) and energy < best_energy:
                        best_energy = energy
                        best_positions = positions.copy()
                except Exception:
                    continue

    # Phase 3: Use gradient to find a better direction from the best
    if best_positions is not None and _get_time_left() > 8.0 and sutton_chen_energy_gradient is not None:
        try:
            energy, grad = sutton_chen_energy_gradient(best_positions, kind)
            if np.all(np.isfinite(grad)):
                # Take a small step in the negative gradient direction, then relax
                gnorm = np.linalg.norm(grad)
                if gnorm > 1e-12:
                    step_size = 0.1
                    stepped = best_positions - step_size * (grad / gnorm)
                    if _validate_positions(stepped):
                        energy2, positions2 = relax(stepped, 'sutton_chen', kind)
                        if _validate_positions(positions2) and energy2 < best_energy:
                            best_energy = energy2
                            best_positions = positions2.copy()
        except Exception:
            pass

    # Final fallback
    if best_positions is None:
        best_positions = starts[0]

    return best_positions
# EVOLVE-BLOCK-END
