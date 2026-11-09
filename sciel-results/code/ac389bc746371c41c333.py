"""Sutton-Chen transition-metal cluster global minima: adaptive basin-hopping + multi-start structured seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_cluster(n, spacing=0.72):
    """Generate an n-atom fcc-like cluster by filling fcc lattice sites nearest to origin."""
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 3
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                cubic = np.array([i, j, k], dtype=float)
                for b in basis:
                    p = (cubic + b) * spacing
                    sites.append(p)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    selected = sites[idx[:n]]
    selected = selected - selected.mean(axis=0)
    return selected


def _compressed_icosahedral(n, spacing=0.72):
    """Generate a compressed icosahedral cluster: inner core + outer shell with reduced radius."""
    if n <= 0:
        return np.empty((0, 3))
    phi = (1 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, -phi, 1/phi], [0, phi, -1/phi], [0, -phi, -1/phi],
        [phi, 1/phi, 0], [-phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, -1/phi, 0],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    for v in ico_verts[:12]:
        points.append(v * spacing * 0.95)
    
    if n <= 13:
        selected = np.array(points[:n])
        return selected - selected.mean(axis=0)
    
    # Second shell: 30 vertices of icosidodecahedron, compressed
    num_second = min(30, n - 13)
    # Generate icosidodecahedron vertices (midpoints of icosahedron edges)
    edge_mids = []
    ico_12 = ico_verts[:12]
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(ico_12[i] - ico_12[j])
            if 0.9 < d < 1.1:  # adjacent vertices
                mid = (ico_12[i] + ico_12[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                edge_mids.append(mid)
    # Remove duplicates
    unique_mids = []
    for m in edge_mids:
        if not any(np.linalg.norm(m - u) < 0.05 for u in unique_mids):
            unique_mids.append(m)
    
    for k in range(num_second):
        points.append(unique_mids[k % len(unique_mids)] * spacing * 1.7)
    
    remaining = n - len(points)
    if remaining > 0:
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for k in range(remaining):
            z = 1.0 - 2.0 * (k + 0.5) / remaining
            r = np.sqrt(1.0 - z * z)
            theta = golden_angle * k
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * spacing * 2.5)
    
    selected = np.array(points[:n])
    return selected - selected.mean(axis=0)


def _hcp_cluster(n, spacing=0.72):
    """Generate an n-atom hcp-like cluster by filling hcp lattice sites nearest to origin."""
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 3
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            x = a * (i + j / 2.0)
            y = a * np.sqrt(3.0) / 2.0 * j
            for layer in range(2):
                z = layer * c / 2.0
                if layer == 1:
                    x_offset = a / 2.0
                    y_offset = a * np.sqrt(3.0) / 6.0
                else:
                    x_offset = 0.0
                    y_offset = 0.0
                p = np.array([x + x_offset, y + y_offset, z], dtype=float)
                sites.append(p)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    selected = sites[idx[:n]]
    selected = selected - selected.mean(axis=0)
    return selected


def _icosahedral_cluster(n, spacing=0.72):
    """Generate an n-atom icosahedral shell cluster."""
    if n <= 0:
        return np.empty((0, 3))
    phi = (1 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, -phi, 1/phi], [0, phi, -1/phi], [0, -phi, -1/phi],
        [phi, 1/phi, 0], [-phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, -1/phi, 0],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    for v in ico_verts[:12]:
        points.append(v * spacing * 1.0)
    
    if n <= 13:
        selected = np.array(points[:n])
        return selected - selected.mean(axis=0)
    
    num_second = min(42, n - 13)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for k in range(num_second):
        z = 1.0 - 2.0 * (k + 0.5) / num_second
        r = np.sqrt(1.0 - z * z)
        theta = golden_angle * k
        p = np.array([r * np.cos(theta), r * np.sin(theta), z])
        points.append(p * spacing * 1.8)
    
    remaining = n - len(points)
    if remaining > 0:
        num_third = remaining
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for k in range(num_third):
            z = 1.0 - 2.0 * (k + 0.5) / num_third
            r = np.sqrt(1.0 - z * z)
            theta = golden_angle * (k + 0.3)
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * spacing * 2.6)
    
    selected = np.array(points[:n])
    return selected - selected.mean(axis=0)


def _random_compact(n, spacing=0.72, seed=42):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    min_dist = 0.6 * spacing
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    selected = np.array(points)
    return selected - selected.mean(axis=0)


def _jitter_cluster(base, n, spacing=0.72, seed=0, jitter_frac=0.15):
    """Add random jitter to a base cluster to create diverse starting points."""
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, jitter_frac * spacing, size=base.shape)
    return base + jitter


def _basin_hopping(positions, potential, kind, t0, deadline, rng, max_iter=80):
    """Adaptive basin hopping with Metropolis acceptance and dynamic perturbation scaling."""
    best_energy, best_pos = relax(positions, potential, kind)
    current_energy, current_pos = best_energy, best_pos.copy()
    
    if not np.isfinite(best_energy):
        return best_energy, best_pos
    
    # Adaptive temperature and perturbation scale
    temp = 0.5  # initial Metropolis temperature (in energy units)
    perturb_scale = 0.15  # initial perturbation std
    n_accepted = 0
    n_rejected = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate perturbed structure
        noise = rng.normal(0, perturb_scale, size=current_pos.shape)
        new_pos = current_pos + noise
        
        # Validate
        if not np.all(np.isfinite(new_pos)):
            continue
        n_atoms = new_pos.shape[0]
        valid = True
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if np.linalg.norm(new_pos[i] - new_pos[j]) < 0.35:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            n_rejected += 1
            continue
        
        # Relax
        try:
            new_energy, relaxed_pos = relax(new_pos, potential, kind)
        except Exception:
            n_rejected += 1
            continue
        
        if not np.isfinite(new_energy):
            n_rejected += 1
            continue
        
        # Metropolis acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-10)):
            current_energy, current_pos = new_energy, relaxed_pos
            n_accepted += 1
            if new_energy < best_energy:
                best_energy, best_pos = new_energy, relaxed_pos.copy()
        else:
            n_rejected += 1
        
        # Adaptive: adjust temperature and perturbation scale
        if (n_accepted + n_rejected) % 10 == 0 and (n_accepted + n_rejected) > 10:
            accept_rate = n_accepted / (n_accepted + n_rejected)
            # Target ~0.5 acceptance for Metropolis
            if accept_rate > 0.7:
                perturb_scale *= 1.3
                temp *= 1.2
            elif accept_rate < 0.3:
                perturb_scale *= 0.7
                temp *= 0.8
            perturb_scale = min(max(perturb_scale, 0.05), 0.5)
            temp = min(max(temp, 0.1), 5.0)
    
    return best_energy, best_pos


def _validate_positions(pos):
    """Check if positions are finite and no atoms are closer than 0.35."""
    if not np.all(np.isfinite(pos)):
        return False
    if pos.shape[0] < 2:
        return True
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            if np.linalg.norm(pos[i] - pos[j]) < 0.35:
                return False
    return True


def construct(instance):
    """Multi-start structured seed relaxation for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    t_start = time.time()
    deadline = t_start + 270.0
    
    spacing = 0.72
    geometries = []
    
    # 1. FCC-like
    try:
        fcc = _fcc_cluster(size, spacing)
        if fcc.shape[0] == size and _validate_positions(fcc):
            geometries.append(fcc)
    except Exception:
        pass
    
    # 2. HCP-like
    try:
        hcp = _hcp_cluster(size, spacing)
        if hcp.shape[0] == size and _validate_positions(hcp):
            geometries.append(hcp)
    except Exception:
        pass
    
    # 3. Icosahedral
    try:
        ico = _icosahedral_cluster(size, spacing)
        if ico.shape[0] == size and _validate_positions(ico):
            geometries.append(ico)
    except Exception:
        pass
    
    # 4. Compressed icosahedral
    try:
        comp_ico = _compressed_icosahedral(size, spacing)
        if comp_ico.shape[0] == size and _validate_positions(comp_ico):
            geometries.append(comp_ico)
    except Exception:
        pass
    
    # 5. Random compact (multiple seeds)
    for seed in [42, 123, 777, 2024]:
        try:
            rand = _random_compact(size, spacing, seed=seed)
            if rand.shape[0] == size and _validate_positions(rand):
                geometries.append(rand)
        except Exception:
            pass
    
    # 6. Jittered versions of structured geometries
    for i, base in enumerate(geometries[:4]):
        for jseed in [10, 20, 30]:
            try:
                jit = _jitter_cluster(base, size, spacing, seed=jseed + i * 100, jitter_frac=0.12)
                if jit.shape[0] == size and _validate_positions(jit):
                    geometries.append(jit)
            except Exception:
                pass

    if not geometries:
        positions = np.zeros((size, 3))
        positions[:, 0] = np.arange(size) * 0.5
        geometries.append(positions)
    
    best_energy = np.inf
    best_positions = None
    rng = np.random.default_rng(42)
    
    # Phase 1: Relax initial seeds (quick pass to find good basins)
    for geom in geometries:
        if time.time() > deadline - 60.0:
            break
        try:
            energy, positions = relax(geom, potential, kind)
            if np.isfinite(energy) and energy < best_energy and _validate_positions(positions):
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    if best_positions is None:
        best_positions = geometries[0].copy()
        try:
            best_energy, best_positions = relax(best_positions, potential, kind)
        except Exception:
            pass
    
    # Phase 2: Adaptive basin hopping on the best structure
    if time.time() < deadline - 30.0:
        bh_energy, bh_pos = _basin_hopping(
            best_positions, potential, kind, t_start, deadline - 20.0, rng, max_iter=100
        )
        if np.isfinite(bh_energy) and bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_pos
    
    # Phase 3: Run basin hopping from other promising seeds if time allows
    if time.time() < deadline - 30.0:
        # Sort geometries by their relaxation energy (re-relax top few)
        seed_results = []
        for geom in geometries[:6]:
            if time.time() > deadline - 30.0:
                break
            try:
                e, p = relax(geom, potential, kind)
                if np.isfinite(e):
                    seed_results.append((e, p))
            except Exception:
                continue
        
        seed_results.sort(key=lambda x: x[0])
        for e, p in seed_results:
            if time.time() > deadline - 30.0:
                break
            if e >= best_energy:
                break
            try:
                bh_e, bh_p = _basin_hopping(
                    p, potential, kind, t_start, deadline - 20.0, rng, max_iter=40
                )
                if np.isfinite(bh_e) and bh_e < best_energy:
                    best_energy = bh_e
                    best_positions = bh_p
            except Exception:
                continue
    
    # Final validation and safety
    if best_positions is None or not np.all(np.isfinite(best_positions)):
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * 0.5
    
    # Ensure minimum distance constraint
    n_atoms = best_positions.shape[0]
    for _ in range(50):
        fixed = False
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d = best_positions[i] - best_positions[j]
                r = np.linalg.norm(d)
                if r < 0.36:
                    if r < 1e-9:
                        best_positions[i] += np.array([0.01, 0.01, 0.01])
                    else:
                        push = (0.36 - r) / 2.0
                        best_positions[i] += push * d / r
                        best_positions[j] -= push * d / r
                    fixed = True
        if not fixed:
            break
    
    return best_positions
# EVOLVE-BLOCK-END
