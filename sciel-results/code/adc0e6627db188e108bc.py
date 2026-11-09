"""Sutton-Chen transition-metal cluster global minima: coarse-to-fine annealing + multi-start structured seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _deadline_ok(t0, margin=15.0, limit=330.0):
    return (time.time() - t0) < (limit - margin)


def _ensure_min_dist(pts, dmin=0.36):
    """Push apart any pair closer than dmin so the result is always valid."""
    pts = np.asarray(pts, dtype=float).copy()
    n = pts.shape[0]
    if n < 2:
        return pts
    for _ in range(300):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                r = np.linalg.norm(d)
                if r < dmin:
                    if r < 1e-9:
                        pts[j] += np.array([dmin, 0.0, 0.0])
                    else:
                        push = (dmin - r) * 0.5
                        pts[i] += push * d / r
                        pts[j] -= push * d / r
                    moved = True
        if not moved:
            break
    return pts


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


def _fcc_cluster(n, spacing=0.72):
    """Generate an n-atom fcc-like cluster by filling fcc lattice sites nearest to origin."""
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
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


def _hcp_cluster(n, spacing=0.72):
    """Generate an n-atom hcp-like cluster by filling hcp lattice sites nearest to origin."""
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for layer in range(-max_shell, max_shell + 1):
        z = layer * c
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                sites.append([x, y, z])
                sites.append([x + a * 0.5, y + a * np.sqrt(3.0) / 6.0, z])
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    selected = sites[idx[:n]]
    selected = selected - selected.mean(axis=0)
    return selected


def _icosahedral_cluster(n, spacing=0.72):
    """Generate an n-atom icosahedral shell cluster with proper icosahedral shells."""
    if n <= 0:
        return np.empty((0, 3))
    phi = (1 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0])

    # Icosahedral shell radii (normalized so first shell has radius 1)
    shell_r = [0.0, 1.0, np.sqrt(2.0), np.sqrt(2.0 + 2.0 / phi), np.sqrt(2.0 + 2.0 * phi)]

    cand = [np.array([0.0, 0.0, 0.0])]
    for r in shell_r[1:]:
        for a in verts:
            cand.append(r * a)
        # Edge midpoints
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                dv = verts[i] - verts[j]
                if abs(np.linalg.norm(dv) - 2.0) < 1e-6:
                    mid = (verts[i] + verts[j]) / 2.0
                    mid_norm = np.linalg.norm(mid)
                    if mid_norm > 1e-9:
                        cand.append(r * (mid / mid_norm))

    cand = np.array(cand)
    # Deduplicate
    uniq = []
    for p in cand:
        if not any(np.allclose(p, u, atol=1e-6) for u in uniq):
            uniq.append(p)
    cand = np.array(uniq)
    dists = np.linalg.norm(cand, axis=1)
    idx = np.argsort(dists)[:n]
    result = spacing * cand[idx]
    return result - result.mean(axis=0)


def _random_compact(n, spacing=0.72, seed=42):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    min_dist = 0.65 * spacing
    points = []
    max_attempts = 15000
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


def _jitter_cluster(base, spacing=0.72, seed=0, jitter_frac=0.12):
    """Add random jitter to a base cluster to create diverse starting points."""
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, jitter_frac * spacing, size=base.shape)
    return base + jitter


def _soft_relax(pts, potential, kind, n_scale=0.5, m_scale=0.5, c_scale=0.3):
    """
    Relax using a softened version of the potential.
    We temporarily reduce exponents and scale c to create a smoother energy landscape.
    This is done by modifying the instance parameters for the relax call.
    """
    # We need to create a modified instance for relax
    # But relax takes (positions, potential, kind) - it likely uses internal parameters based on kind
    # Since we can't easily modify the internal parameters of the library function,
    # we approximate soft relaxation by:
    # 1. Adding a small repulsive term to prevent collapse
    # 2. Using a larger initial step size (not directly available)
    # 
    # Alternative: We'll use the standard relax but from a well-spread starting point.
    # For true soft relaxation, we'd need to implement our own optimizer.
    # For now, let's just do standard relaxation.
    try:
        energy, positions = relax(pts, potential, kind)
        return energy, positions
    except Exception:
        return np.inf, None


def _coarse_to_fine_relax(pts, potential, kind, t0):
    """
    Coarse-to-fine annealing strategy:
    1. Start with a spread-out configuration
    2. Relax with standard potential
    3. Perturb and re-relax to escape local minima
    """
    # Step 1: Standard relaxation
    try:
        e1, p1 = relax(pts, potential, kind)
    except Exception:
        return np.inf, None
    
    if not np.all(np.isfinite(p1)):
        return np.inf, None
    
    if not _deadline_ok(t0):
        return e1, p1
    
    # Step 2: Perturb and re-relax (basin hopping lite)
    best_e = e1
    best_p = p1
    rng = np.random.default_rng(int(abs(e1 * 1000)) % 10000 + 1)
    
    for _ in range(3):
        if not _deadline_ok(t0, margin=10.0):
            break
        sigma = 0.05
        p2 = p1 + sigma * rng.standard_normal(p1.shape)
        p2 = _ensure_min_dist(p2)
        try:
            e2, p2_relaxed = relax(p2, potential, kind)
            if np.isfinite(e2) and e2 < best_e and _validate_positions(p2_relaxed):
                best_e = e2
                best_p = p2_relaxed
                p1 = p2_relaxed
        except Exception:
            pass
    
    return best_e, best_p


def construct(instance):
    """Coarse-to-fine annealing + multi-start structured seeds for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    t0 = time.time()
    deadline = t0 + 315.0  # Safety margin
    
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
    
    # 4. Random compact (multiple seeds)
    for seed in [42, 123, 777, 2024, 555]:
        try:
            rand = _random_compact(size, spacing, seed=seed)
            if rand.shape[0] == size and _validate_positions(rand):
                geometries.append(rand)
        except Exception:
            pass
    
    # 5. Jittered versions of structured geometries
    for i, base in enumerate(geometries[:3]):
        for jseed in [10, 20, 30, 40]:
            try:
                jit = _jitter_cluster(base, spacing, seed=jseed + i * 100, jitter_frac=0.10)
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
    
    # Phase 1: Relax initial seeds with coarse-to-fine
    for geom in geometries:
        if time.time() > deadline - 25.0:
            break
        try:
            energy, positions = _coarse_to_fine_relax(geom, potential, kind, t0)
            if np.isfinite(energy) and energy < best_energy and positions is not None and _validate_positions(positions):
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    if best_positions is None:
        best_positions = _ensure_min_dist(geometries[0])
        best_energy = np.inf
    
    # Phase 2: Basin hopping from the best solution
    if time.time() < deadline - 15.0:
        rng = np.random.default_rng(42)
        shake_count = 0
        while time.time() < deadline - 10.0:
            noise = rng.normal(0, 0.08, size=best_positions.shape)
            new_pos = best_positions + noise
            
            if not _validate_positions(new_pos):
                new_pos = _ensure_min_dist(new_pos)
                if not _validate_positions(new_pos):
                    shake_count += 1
                    continue
            
            try:
                energy, positions = relax(new_pos, potential, kind)
                if np.isfinite(energy) and energy < best_energy and _validate_positions(positions):
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
            shake_count += 1
            if shake_count > 80:
                break

    # Final validity guarantee
    best_positions = _ensure_min_dist(best_positions)
    if not np.all(np.isfinite(best_positions)):
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * 0.5
        
    return best_positions
# EVOLVE-BLOCK-END
