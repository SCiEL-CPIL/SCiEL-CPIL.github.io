"""Combined: structured high-symmetry seeds + multi-start local relax + basin-hopping refinement."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = 325.0
_T0 = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _icosahedron_verts():
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / n

def _cuboctahedron_verts():
    v = []
    for x in (-1, 1):
        for y in (-1, 1):
            v.append([x, y, 0.0])
            v.append([x, 0.0, y])
            v.append([0.0, x, y])
    v = np.array(v, float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def _rhombicubocta_verts():
    t = 1.0 + 2.0 ** 0.5
    v = []
    for a in (-1, 1):
        for b in (-1, 1):
            for c in (-1, 1):
                v.append([a, b, c * t])
                v.append([a, b * t, c])
                v.append([a * t, b, c])
    v = np.array(v, float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def _fcc_lattice(n):
    R = 2.5
    pts = []
    s = int(R) + 1
    for x in range(-s, s + 1):
        for y in range(-s, s + 1):
            for z in range(-s, s + 1):
                for shift in [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]:
                    p = np.array([x + shift[0], y + shift[1], z + shift[2]])
                    if np.linalg.norm(p) <= R:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) == 0:
        return pts
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    pts = pts[order]
    if len(pts) > n:
        pts = pts[:n]
    if len(pts) >= 2:
        nn = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        nn = nn[nn > 1e-6]
        if len(nn):
            scale = 0.72 / np.median(nn)
            pts = pts * scale
    return pts

def _bcc_lattice(n):
    R = 2.5
    pts = []
    s = int(R) + 1
    for x in range(-s, s + 1):
        for y in range(-s, s + 1):
            for z in range(-s, s + 1):
                for shift in [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]:
                    p = np.array([x + shift[0], y + shift[1], z + shift[2]])
                    if np.linalg.norm(p) <= R:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) == 0:
        return pts
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    pts = pts[order]
    if len(pts) > n:
        pts = pts[:n]
    if len(pts) >= 2:
        nn = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        nn = nn[nn > 1e-6]
        if len(nn):
            scale = 0.72 / np.median(nn)
            pts = pts * scale
    return pts

def _random_cluster(size, rng, spacing=0.72, min_dist=0.85):
    radius = spacing * size ** (1.0 / 3.0) * 0.9
    pts = []
    attempts = 0
    while len(pts) < size and attempts < size * size * 40:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
        attempts += 1
    while len(pts) < size:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts[:size])

def _incremental_energy(pos, kind):
    """Compute the full Sutton-Chen energy for a cluster."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(r, np.inf)
    r2 = r ** 2
    r6 = r2 ** 3
    r12 = r6 ** 2
    r9 = r2 ** 4 * r2 ** 0.5
    r8 = r2 ** 4
    if kind == '12-6':
        n_exp, m_exp, c = 12, 6, 144.41
        pair = 1.0 / r12 - c * np.sqrt(1.0 / r6)
    elif kind == '9-6':
        n_exp, m_exp, c = 9, 6, 39.432
        pair = 1.0 / r9 - c * np.sqrt(1.0 / r6)
    elif kind == '10-8':
        n_exp, m_exp, c = 10, 8, 34.408
        pair = 1.0 / r8 - c * np.sqrt(1.0 / r8)
    else:
        return 0.0
    mask = r < 50.0
    return 0.5 * pair[mask].sum()

def _greedy_insertion(size, kind, rng, n_starts=4, n_candidates=60, spacing=0.72):
    """Coarse-to-fine: build cluster by greedily inserting atoms at positions
    that minimize incremental energy. Returns a list of candidate structures."""
    results = []
    for start in range(n_starts):
        if _time_left() < 15.0:
            break
        # Initial nucleus: 2-3 atoms
        nucleus = []
        if start == 0:
            nucleus = [np.array([0.0, 0.0, 0.0]), np.array([spacing, 0.0, 0.0])]
        elif start == 1:
            nucleus = [np.array([0.0, 0.0, 0.0]), np.array([spacing * 0.9, spacing * 0.5, 0.0])]
        elif start == 2:
            nucleus = [np.array([0.0, 0.0, 0.0]), np.array([spacing, 0.0, 0.0]),
                       np.array([spacing * 0.5, spacing * 0.866, 0.0])]
        else:
            nucleus = [rng.uniform(-spacing, spacing, 3) for _ in range(min(3, size))]
        
        if len(nucleus) >= size:
            pos = np.array(nucleus[:size])
            pos = _sanitize(pos, size)
            if pos is not None:
                results.append(pos)
            continue
        
        pos = np.array(nucleus)
        radius = spacing * size ** (1.0 / 3.0) * 1.2
        
        for atom_idx in range(len(nucleus), size):
            if _time_left() < 10.0:
                break
            # Generate candidate positions for the new atom
            cands = []
            # Shell candidates: on a sphere of increasing radius
            for shell_r in [spacing * (1.0 + 0.15 * k) for k in range(4)]:
                for theta in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    for phi in np.linspace(0, np.pi, 8, endpoint=False):
                        c = np.array([shell_r * np.sin(phi) * np.cos(theta),
                                      shell_r * np.sin(phi) * np.sin(theta),
                                      shell_r * np.cos(phi)])
                        cands.append(c)
            # Random candidates
            for _ in range(30):
                c = rng.uniform(-radius, radius, 3)
                cands.append(c)
            
            cands = np.array(cands)
            # Filter: minimum distance from existing atoms
            valid = []
            for c in cands:
                if len(pos) > 0:
                    min_d = np.min(np.linalg.norm(pos - c, axis=1))
                else:
                    min_d = np.inf
                if min_d > 0.40:
                    valid.append(c)
            
            if not valid:
                # Fallback: place at random position
                valid = [rng.uniform(-radius, radius, 3)]
            
            # Evaluate incremental energy for each valid candidate
            best_e = np.inf
            best_c = valid[0]
            for c in valid[:n_candidates]:
                test_pos = np.vstack([pos, c.reshape(1, 3)])
                e = _incremental_energy(test_pos, kind)
                if e < best_e:
                    best_e = e
                    best_c = c
            
            pos = np.vstack([pos, best_c.reshape(1, 3)])
        
        # Center the cluster
        pos = pos - pos.mean(axis=0)
        pos = _sanitize(pos, size)
        if pos is not None:
            results.append(pos)
    
    return results

def _sanitize(pos, size):
    """Ensure finite coords and min distance > 0.35."""
    if not np.all(np.isfinite(pos)):
        return None
    if size < 2:
        return pos
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    if d.min() < 0.35:
        scale = 0.35 / d.min() + 0.05
        pos = pos * scale
    return pos

def _basin_hopping(pos, potential, kind, rng, n_iter=12, step_scale=0.25):
    """Simple basin-hopping loop around a relaxed structure."""
    best_energy, best_pos = None, None
    try:
        e0, p0 = relax(pos, potential, kind)
        if np.all(np.isfinite(p0)):
            best_energy = float(e0)
            best_pos = np.asarray(p0, float)
    except Exception:
        return pos

    for it in range(n_iter):
        if _time_left() < 10.0:
            break
        noise = step_scale * (1.0 - 0.5 * it / n_iter)
        perturbed = best_pos + rng.normal(0, noise, size=(best_pos.shape[0], 3))
        perturbed = _sanitize(perturbed, best_pos.shape[0])
        if perturbed is None:
            continue
        try:
            energy, positions = relax(perturbed, potential, kind)
        except Exception:
            continue
        positions = np.asarray(positions, float)
        if not np.all(np.isfinite(positions)):
            continue
        if best_energy is None or energy < best_energy:
            best_energy = float(energy)
            best_pos = positions
    return best_pos

def _polish(pos, potential, kind, n_rounds=3):
    """High-precision polish: multiple tight relaxations to converge to a deeper local minimum."""
    best_pos = np.asarray(pos, float)
    best_energy = None
    for _ in range(n_rounds):
        if _time_left() < 5.0:
            break
        try:
            energy, positions = relax(best_pos, potential, kind)
            positions = np.asarray(positions, float)
            if not np.all(np.isfinite(positions)):
                break
            if best_energy is None or energy < best_energy:
                best_energy = float(energy)
                best_pos = positions
            else:
                break
        except Exception:
            break
    return best_pos

def construct(instance):
    global _T0
    _T0 = time.time()
    size = int(instance["n"])
    potential = instance["potential"]
    kind = instance.get("kind")
    spacing = {"morse": 1.0, "sutton_chen": 0.72, "dzugutov": 1.1}.get(potential, 0.72)
    rng = np.random.default_rng(hash((potential, kind, size)) & 0xFFFFFFFF)

    # Build candidate seeds: energy-guided greedy insertion + geometric templates + random
    seeds = []
    
    # Primary: coarse-to-fine energy-guided insertion (structural change)
    greedy_seeds = _greedy_insertion(size, kind, rng, n_starts=5, n_candidates=50, spacing=spacing)
    for gs in greedy_seeds:
        seeds.append(gs)
    
    # Secondary: geometric templates (fcc, bcc)
    for c in [_fcc_lattice(size), _bcc_lattice(size)]:
        if c is not None and len(c) == size:
            c = _sanitize(c, size)
            if c is not None:
                seeds.append(c)
    
    # Tertiary: random seeds
    for _ in range(6):
        seeds.append(_random_cluster(size, rng, spacing, 0.85))

    best = None
    for seed in seeds:
        if _time_left() < 15.0:
            break
        try:
            energy, positions = relax(np.asarray(seed, float), potential, kind)
        except Exception:
            continue
        positions = np.asarray(positions, float)
        if not np.all(np.isfinite(positions)):
            continue
        if best is None or energy < best[0]:
            best = (float(energy), positions)
    
    # Additional: second round of greedy insertion with different parameters
    if _time_left() > 40.0:
        greedy2 = _greedy_insertion(size, kind, rng, n_starts=3, n_candidates=40, spacing=spacing * 0.9)
        for gs in greedy2:
            try:
                energy, positions = relax(gs, potential, kind)
                positions = np.asarray(positions, float)
                if not np.all(np.isfinite(positions)):
                    continue
                if best is None or energy < best[0]:
                    best = (float(energy), positions)
            except Exception:
                continue
    
    # Diverse random seeds with varying spacing
    for spacing_try in [0.65, 0.72, 0.80]:
        if _time_left() < 10.0:
            break
        for _ in range(3):
            if _time_left() < 10.0:
                break
            seed = _random_cluster(size, rng, spacing_try, 0.85)
            try:
                energy, positions = relax(seed, potential, kind)
                positions = np.asarray(positions, float)
                if not np.all(np.isfinite(positions)):
                    continue
                if best is None or energy < best[0]:
                    best = (float(energy), positions)
            except Exception:
                continue

    if best is None:
        for _ in range(10):
            if _time_left() < 6.0:
                break
            seed = _random_cluster(size, rng, spacing, 0.9)
            try:
                energy, positions = relax(seed, potential, kind)
                if np.all(np.isfinite(positions)):
                    if best is None or energy < best[0]:
                        best = (float(energy), np.asarray(positions, float))
            except Exception:
                continue

    if best is None:
        pts = _random_cluster(size, rng, spacing, 1.0)
        return np.asarray(pts, float)

    # Basin hopping refinement: stochastic global search to escape local minima
    best_energy, best_pos = best
    bh_result = _basin_hopping(best_pos, potential, kind, rng, n_iter=20, step_scale=0.30)
    if bh_result is not None and np.all(np.isfinite(bh_result)):
        try:
            e_bh, _ = sutton_chen_energy_gradient(bh_result, kind)
            if np.isfinite(e_bh) and e_bh < best_energy:
                best_energy = float(e_bh)
                best_pos = bh_result
        except Exception:
            pass
    
    # Second basin-hopping from a different starting point (second-best seed)
    if _time_left() > 30.0 and best is not None:
        # Try basin hopping from the second-best relaxed structure if available
        pass

    # Multi-scale perturbation refinement: try different noise scales
    for noise_scale in [0.35, 0.25, 0.15, 0.08, 0.04]:
        if _time_left() < 12.0:
            break
        for trial in range(4):
            if _time_left() < 12.0:
                break
            perturbed = best_pos + rng.normal(0, noise_scale, size=(size, 3))
            perturbed = _sanitize(perturbed, size)
            if perturbed is None:
                continue
            try:
                energy, positions = relax(perturbed, potential, kind)
            except Exception:
                continue
            positions = np.asarray(positions, float)
            if not np.all(np.isfinite(positions)):
                continue
            if energy < best_energy:
                best_energy = float(energy)
                best_pos = positions

    # Final high-precision polish pass
    polished = _polish(best_pos, potential, kind, n_rounds=3)
    if polished is not None and np.all(np.isfinite(polished)):
        try:
            e_polish, _ = sutton_chen_energy_gradient(polished, kind)
            if np.isfinite(e_polish) and e_polish < best_energy:
                best_energy = float(e_polish)
                best_pos = polished
        except Exception:
            pass

    return best_pos
# EVOLVE-BLOCK-END
