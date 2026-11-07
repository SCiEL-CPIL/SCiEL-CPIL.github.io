"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    dists = np.sqrt(np.sum((positions[:, np.newaxis, :] - positions[np.newaxis, :, :]) ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    min_d = np.min(dists[mask])
    if min_d < min_dist:
        return False
    return True


def _generate_fcc(n):
    """Generate n atoms in FCC-like arrangement (closest to origin)."""
    max_r = int(np.ceil(n ** (1 / 3))) + 5
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k], dtype=float) / np.sqrt(2.0)
                    candidates.append((np.dot(p, p), p))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_bcc(n):
    """Generate n atoms in BCC-like arrangement."""
    max_r = int(np.ceil(n ** (1 / 3))) + 5
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k], dtype=float)
                candidates.append((np.dot(p1, p1), p1))
                p2 = p1 + 0.5
                candidates.append((np.dot(p2, p2), p2))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_mackay(n):
    """Generate Mackay icosahedral cluster."""
    phi = (1 + np.sqrt(5)) / 2
    v = [
        np.array([0, 1, phi]), np.array([0, 1, -phi]),
        np.array([0, -1, phi]), np.array([0, -1, -phi]),
        np.array([1, phi, 0]), np.array([1, -phi, 0]),
        np.array([-1, phi, 0]), np.array([-1, -phi, 0]),
        np.array([phi, 0, 1]), np.array([phi, 0, -1]),
        np.array([-phi, 0, 1]), np.array([-phi, 0, -1])
    ]
    v = [p / np.linalg.norm(p) for p in v]
    
    pts = [np.zeros(3)]
    if n <= 13:
        pts.extend(v[:min(12, n - 1)])
        return np.array(pts[:n])
    
    # Shell 1: 12 atoms at radius 1
    pts.extend(v)
    
    if n <= 55:
        return np.array(pts[:n])
    
    # Shell 2: 42 atoms at radius 2
    # 30 edge midpoints
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(v[i] - v[j])
            if 1.5 < d < 2.5:
                edges.append((i, j))
    
    shell2_pts = []
    for i, j in edges[:30]:
        mid = (v[i] + v[j]) / 2
        mid = mid * 2.0 / np.linalg.norm(mid)
        shell2_pts.append(mid)
    
    # 12 more atoms: use vertices of a dodecahedron or just random on sphere
    rng = np.random.default_rng(42)
    for _ in range(200):
        if len(shell2_pts) >= 42:
            break
        p = rng.normal(size=3)
        p = p * 2.0 / np.linalg.norm(p)
        if all(np.linalg.norm(p - q) > 0.4 for q in shell2_pts):
            shell2_pts.append(p)
    
    pts.extend(shell2_pts[:42])
    return np.array(pts[:n])


def _generate_random(n, seed):
    """Generate random positions in a compact sphere."""
    rng = np.random.default_rng(seed)
    r = 0.4 * np.sqrt(n)
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    pts = pts / norms * r * rng.random((n, 1)) ** (1 / 3)
    return pts


def _growth_cluster(n, kind, seed=0):
    """Build cluster by growing from a small relaxed cluster."""
    rng = np.random.default_rng(seed)
    
    # Start with a small cluster
    start_n = min(13, n)
    if start_n <= 13:
        pos = _generate_mackay(start_n)
    else:
        pos = _generate_fcc(start_n)
    
    # Relax the starting cluster
    if _time_left() < 10:
        return pos
    try:
        e, pos = relax(pos, 'sutton_chen', kind)
        if not _is_valid(pos):
            pos = _generate_fcc(start_n)
    except:
        pass
    
    # Grow one atom at a time
    for target_n in range(start_n + 1, n + 1):
        if _time_left() < 5:
            break
        
        # Find best position to add a new atom
        best_e = np.inf
        best_pos = None
        
        # Try several candidate positions
        for attempt in range(50):
            # Place new atom near the surface
            # Pick a random existing atom and offset it
            i = rng.integers(0, len(pos))
            offset = rng.normal(0, 0.8, 3)
            new_atom = pos[i] + offset
            
            # Check validity
            candidate = np.vstack([pos, new_atom.reshape(1, 3)])
            if not _is_valid(candidate):
                continue
            
            # Quick energy estimate (not full relax, just check it's not terrible)
            try:
                e, _ = sutton_chen_energy_gradient(candidate, kind)
                if e < best_e:
                    best_e = e
                    best_pos = candidate.copy()
            except:
                continue
        
        if best_pos is not None:
            pos = best_pos
        else:
            # Fallback: just add a random atom
            new_atom = pos[rng.integers(0, len(pos))] + rng.normal(0, 1.0, 3)
            pos = np.vstack([pos, new_atom.reshape(1, 3)])
        
        # Relax periodically
        if target_n % 5 == 0 or target_n == n:
            if _time_left() < 5:
                break
            try:
                e, pos = relax(pos, 'sutton_chen', kind)
                if not _is_valid(pos):
                    pos = _generate_fcc(target_n)
            except:
                pass
    
    return pos


def _basin_hop_chain(positions, kind, n_steps, rng, sigma0=0.3, sigma_decay=0.95):
    """Run a basin hopping chain with adaptive step size."""
    e, current = relax(positions, 'sutton_chen', kind)
    if not (np.isfinite(e) and _is_valid(current)):
        return positions, np.inf
    
    best_e = e
    best_pos = current.copy()
    sigma = sigma0
    
    for i in range(n_steps):
        if _time_left() < 10.0:
            break
        
        # Perturb
        pos = current + rng.normal(0, sigma, (len(current), 3))
        
        if not _is_valid(pos):
            sigma *= 0.9
            continue
        
        try:
            e_new, relaxed = relax(pos, 'sutton_chen', kind)
            if np.isfinite(e_new) and _is_valid(relaxed):
                # Accept if better, or with Metropolis probability
                if e_new < best_e:
                    best_e = e_new
                    best_pos = relaxed.copy()
                    sigma = min(sigma * 1.05, 0.5)
                
                # Metropolis acceptance
                if e_new < e or rng.random() < np.exp(-(e_new - e) / max(0.1, e * 0.01)):
                    current = relaxed
                    e = e_new
                else:
                    sigma *= 0.95
        except:
            continue
            
    return best_pos, best_e


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    # Phase 1: Generate diverse initial structures
    generators = [
        lambda: _generate_fcc(n),
        lambda: _generate_bcc(n),
        lambda: _generate_mackay(n),
        lambda: _generate_random(n, 1),
        lambda: _generate_random(n, 2),
        lambda: _generate_random(n, 3),
        lambda: _growth_cluster(n, kind, seed=0),
        lambda: _growth_cluster(n, kind, seed=1),
    ]
    
    initial_candidates = []
    for gen in generators:
        if _time_left() < 15.0:
            break
        try:
            base = gen()
            if _is_valid(base):
                energy, relaxed = relax(base, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    initial_candidates.append((energy, relaxed))
        except Exception:
            continue
    
    if not initial_candidates:
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            best_pos = _generate_random(n, 0)
        return best_pos
    
    # Sort by energy
    initial_candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Basin Hopping from the best candidates
    n_chains = 4
    rngs = [np.random.default_rng(100 + i) for i in range(n_chains)]
    
    for i in range(min(n_chains, len(initial_candidates))):
        if _time_left() < 25.0:
            break
        
        e_init, pos_init = initial_candidates[i]
        if e_init < best_energy:
            best_energy = e_init
            best_pos = pos_init.copy()
        
        # Run BH chain
        bh_pos, bh_energy = _basin_hop_chain(pos_init, kind, n_steps=80, rng=rngs[i])
        
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos
    
    # Phase 3: If time remains, run more BH from the global best
    if best_pos is not None and _time_left() > 20.0:
        rng_final = np.random.default_rng(999)
        bh_pos, bh_energy = _basin_hop_chain(best_pos, kind, n_steps=40, rng=rng_final)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos

    return best_pos

# EVOLVE-BLOCK-END
