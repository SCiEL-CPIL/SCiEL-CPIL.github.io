"""Sutton-Chen cluster global minima search using parallel multi-start basin hopping."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _icosahedron_vertices(r=1.0):
    """Return 12 vertices of a regular icosahedron scaled to radius r."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    return verts * r


def _build_mackay_icosahedron(n, a=0.72):
    """Build a Mackay icosahedral cluster."""
    points = [np.zeros(3)]  # center
    
    # Shell 1: icosahedron vertices
    r1 = a
    v1 = _icosahedron_vertices(r1)
    for v in v1:
        if len(points) < n:
            points.append(v.copy())
    
    # Shell 2: 42 atoms
    if len(points) < n:
        r2 = a * 1.48
        v1_unit = _icosahedron_vertices(1.0)
        
        # Compute face centers
        phi = (1 + np.sqrt(5)) / 2
        verts = v1_unit
        n_verts = len(verts)
        dists = np.linalg.norm(verts[:, None] - verts[None, :], axis=2)
        min_d = dists.min()
        
        # Find faces
        faces = []
        for i in range(n_verts):
            for j in range(n_verts):
                if i == j:
                    continue
                for k in range(n_verts):
                    if i == k or j == k:
                        continue
                    if (abs(dists[i, j] - min_d) < 1e-9 and 
                        abs(dists[j, k] - min_d) < 1e-9 and 
                        abs(dists[k, i] - min_d) < 1e-9):
                        face = tuple(sorted([i, j, k]))
                        if face not in faces:
                            faces.append(face)
        
        # Face centers
        fc = []
        for f in faces:
            c = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
            fc.append(c)
        fc = np.array(fc)
        fc_norm = fc / np.linalg.norm(fc, axis=1, keepdims=True)
        fc_scaled = fc_norm * r2
        for p in fc_scaled:
            if len(points) < n:
                points.append(p)
        
        # Edge midpoints
        if len(points) < n:
            edges = []
            for i in range(n_verts):
                for j in range(i + 1, n_verts):
                    if abs(dists[i, j] - min_d) < 1e-9:
                        edges.append((i, j))
            
            em = []
            for i, j in edges:
                m = (verts[i] + verts[j]) / 2.0
                em.append(m)
            em = np.array(em)
            em_norm = em / np.linalg.norm(em, axis=1, keepdims=True)
            em_scaled = em_norm * r2
            for p in em_scaled:
                if len(points) < n:
                    points.append(p)
    
    # Shell 3 if needed
    if len(points) < n:
        r3 = a * 2.0
        theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        for t in theta:
            if len(points) < n:
                points.append([r3 * np.cos(t), r3 * np.sin(t), 0])
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _build_fcc_sphere(n, a=0.72):
    """Build an fcc spherical cluster."""
    basis = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
    ]
    side = int(np.ceil((n / 4) ** (1.0 / 3.0))) + 2
    candidates = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i * a + b[0] * a, j * a + b[1] * a, k * a + b[2] * a])
                    candidates.append(p)
    candidates = np.array(candidates)
    center = candidates.mean(axis=0)
    dists = np.linalg.norm(candidates - center, axis=1)
    order = np.argsort(dists)
    points = candidates[order[:n]]
    points -= points.mean(axis=0)
    return points


def _build_bcc_sphere(n, a=0.72):
    """Build a bcc spherical cluster."""
    r_est = (3.0 * n / (2.0 * 4.0 * np.pi)) ** (1.0 / 3.0) * 1.2
    max_idx = int(np.ceil(r_est / a)) + 2
    
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p1 = np.array([i, j, k]) * a
                if np.linalg.norm(p1) <= r_est:
                    points.append(p1)
                p2 = (np.array([i, j, k]) + 0.5) * a
                if np.linalg.norm(p2) <= r_est:
                    points.append(p2)
    
    if not points:
        return np.zeros((n, 3))
    
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    points -= points.mean(axis=0)
    return points


def _random_compact(n, spacing=0.72, seed=42):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    for _ in range(10000):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - np.array(q)) >= 0.5 * spacing for q in points):
                points.append(p)
    if len(points) < n:
        points = [list(rng.uniform(-radius, radius, 3)) for _ in range(n)]
    return np.array(points[:n])


def _generate_starts(n, kind, deadline):
    """Generate diverse starting configurations."""
    starts = []
    
    # 1. Mackay icosahedron
    try:
        starts.append(_build_mackay_icosahedron(n))
    except:
        pass
    
    # 2. FCC sphere
    try:
        starts.append(_build_fcc_sphere(n))
    except:
        pass
    
    # 3. BCC sphere
    try:
        starts.append(_build_bcc_sphere(n))
    except:
        pass
    
    # 4. Random compact with different seeds
    for seed in [42, 123, 456, 789]:
        try:
            starts.append(_random_compact(n, seed=seed))
        except:
            pass
    
    return starts


def _relax_and_basin_hop(positions, kind, n, deadline, bh_steps=15, step_size=0.5):
    """Relax and perform basin hopping."""
    try:
        best_energy, best_pos = relax(positions, 'sutton_chen', kind)
    except:
        return np.inf, None
    
    if not np.isfinite(best_energy):
        return np.inf, None
    
    current_energy, current_pos = best_energy, best_pos
    
    for step in range(bh_steps):
        if time.time() > deadline:
            break
        
        rng = np.random.default_rng(step * 1000 + int(n))
        noise = rng.normal(0, step_size, (n, 3))
        perturbed = current_pos + noise
        
        try:
            energy, pos = relax(perturbed, 'sutton_chen', kind)
        except:
            continue
        
        if np.isfinite(energy) and energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            current_pos = pos
            current_energy = energy
    
    # Validate
    if not np.all(np.isfinite(best_pos)):
        return np.inf, None
    
    if n > 1:
        d = np.sqrt(np.sum((best_pos[:, None, :] - best_pos[None, :, :]) ** 2, axis=-1))
        np.fill_diagonal(d, np.inf)
        if d.min() < 0.35:
            return np.inf, None
    
    return best_energy, best_pos


def _worker(args):
    """Worker function for parallel processing."""
    positions, kind, n, deadline, bh_steps, step_size = args
    return _relax_and_basin_hop(positions, kind, n, deadline, bh_steps, step_size)


def construct(instance):
    """Structure prediction via parallel multi-start basin hopping."""
    size = instance["n"]
    kind = instance["kind"]
    
    start_time = time.time()
    deadline = start_time + 285  # Safety margin before 330s kill
    
    # Generate diverse starting configurations
    starts = _generate_starts(size, kind, deadline)
    
    if not starts:
        rng = np.random.default_rng(0)
        return rng.uniform(-1, 1, (size, 3))
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Parallel relaxation of all starts
    # Use 4 threads max
    max_workers = 4
    
    # Prepare tasks
    tasks = []
    bh_steps = 10  # Moderate basin hopping steps per start
    step_size = 0.5
    
    for i, start in enumerate(starts):
        if time.time() > deadline:
            break
        tasks.append((start, kind, size, deadline, bh_steps, step_size))
    
    # Run in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, task): i for i, task in enumerate(tasks)}
        
        for future in as_completed(futures):
            if time.time() > deadline:
                break
            
            try:
                energy, positions = future.result()
                if np.isfinite(energy) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                continue
    
    if best_positions is None:
        # Fallback: just relax the first start
        try:
            best_energy, best_positions = relax(starts[0], 'sutton_chen', kind)
        except:
            rng = np.random.default_rng(0)
            best_positions = rng.uniform(-1, 1, (size, 3))
            return best_positions
    
    # Phase 2: If time remains, do more basin hopping from the best
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        # Estimate how many more BH steps we can afford
        max_extra_steps = min(20, int(remaining_time / 5))
        try:
            extra_energy, extra_pos = _relax_and_basin_hop(
                best_positions, kind, size, deadline,
                bh_steps=max_extra_steps, step_size=0.7
            )
            if np.isfinite(extra_energy) and extra_energy < best_energy:
                best_energy = extra_energy
                best_positions = extra_pos
        except:
            pass
    
    # Phase 3: If more time, try from second-best or perturbed
    if time.time() < deadline - 20:
        rng = np.random.default_rng(999)
        noise = rng.normal(0, 0.8, (size, 3))
        perturbed = best_positions + noise
        try:
            energy, positions = relax(perturbed, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
