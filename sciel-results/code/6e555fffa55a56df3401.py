"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill


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


def _generate_hcp(n):
    """Generate n atoms in HCP-like arrangement."""
    a = 1.0
    c = 1.633 * a
    # HCP lattice vectors
    a1 = np.array([a, 0, 0])
    a2 = np.array([a/2, a*np.sqrt(3)/2, 0])
    a3 = np.array([0, 0, c])
    
    # Basis atoms in unit cell
    basis = [np.zeros(3), a1/3 + a2/3, a1/3 + a2/3 + a3/2]
    
    max_r = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                for b in basis:
                    p = i * a1 + j * a2 + k * a3 + b
                    candidates.append((np.dot(p, p), p))
    
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
    if n > 1:
        pts.extend(v[:min(12, n - 1)])
    
    if n > 13:
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
        
        faces = []
        for i in range(12):
            for j in range(i + 1, 12):
                for k in range(j + 1, 12):
                    d1 = np.linalg.norm(v[i] - v[j])
                    d2 = np.linalg.norm(v[j] - v[k])
                    d3 = np.linalg.norm(v[i] - v[k])
                    if 1.5 < d1 < 2.5 and 1.5 < d2 < 2.5 and 1.5 < d3 < 2.5:
                        center = (v[i] + v[j] + v[k]) / 3
                        center = center * 2.0 / np.linalg.norm(center)
                        faces.append(center)
        
        shell2_pts.extend(faces[:12])
        
        unique_pts = []
        for p in shell2_pts:
            if not any(np.linalg.norm(p - q) < 0.1 for q in unique_pts):
                unique_pts.append(p)
        
        remaining = min(42, n - len(pts))
        pts.extend(unique_pts[:remaining])
    
    if n > 55:
        rng = np.random.default_rng(123)
        remaining = min(92, n - len(pts))
        for _ in range(remaining * 5):
            if len(pts) >= n:
                break
            p = rng.normal(size=3)
            p = p * 3.0 / np.linalg.norm(p)
            if all(np.linalg.norm(p - q) > 0.5 for q in pts):
                pts.append(p)
    
    return np.array(pts[:n])


def _generate_random(n, seed):
    """Generate random positions in a sphere."""
    rng = np.random.default_rng(seed)
    r = 0.4 * np.sqrt(n)
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    pts = pts / norms * r * rng.random((n, 1)) ** (1 / 3)
    return pts


def _differential_evolution(positions_list, kind, n_gen=15, pop_size=5, rng=None):
    """
    Run differential evolution over a population of relaxed structures.
    Each individual is a relaxed (n,3) array. Mutation: best/1/bin.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n = len(positions_list[0])
    dim = n * 3
    
    # Flatten positions and ensure valid
    pop = []
    energies = []
    for p in positions_list:
        pos = p.reshape(n, 3)
        if not _is_valid(pos):
            pos = _generate_random(n, rng.integers(10000))
        try:
            e, relaxed = relax(pos, 'sutton_chen', kind)
            if np.isfinite(e) and _is_valid(relaxed):
                pop.append(relaxed.flatten())
                energies.append(e)
            else:
                pop.append(pos.flatten())
                energies.append(np.inf)
        except Exception:
            pop.append(pos.flatten())
            energies.append(np.inf)
    
    # Pad population if needed
    while len(pop) < pop_size:
        idx = rng.integers(len(pop))
        new_p = pop[idx].copy()
        new_p += rng.normal(0, 0.1, dim)
        pos = new_p.reshape(n, 3)
        if _is_valid(pos):
            try:
                e, relaxed = relax(pos, 'sutton_chen', kind)
                if np.isfinite(e) and _is_valid(relaxed):
                    pop.append(relaxed.flatten())
                    energies.append(e)
                else:
                    pop.append(new_p)
                    energies.append(np.inf)
            except Exception:
                pop.append(new_p)
                energies.append(np.inf)
        else:
            pos = _generate_random(n, rng.integers(10000))
            pop.append(pos.flatten())
            energies.append(np.inf)
    
    best_e = min(energies)
    best_pos = pop[np.argmin(energies)].copy()
    
    for gen in range(n_gen):
        if _time_left() < 10.0:
            break
        
        # Sort by energy
        order = np.argsort(energies)
        pop = [pop[i] for i in order]
        energies = [energies[i] for i in order]
        
        for i in range(pop_size):
            if _time_left() < 10.0:
                break
            
            # Select three distinct random individuals (excluding i)
            candidates = [j for j in range(pop_size) if j != i]
            if len(candidates) < 3:
                continue
            a, b, c = rng.choice(candidates, size=3, replace=False)
            
            # Mutation: best/1/bin
            F = 0.8
            CR = 0.9
            v = pop[0] + F * (pop[a] - pop[b])
            
            # Crossover
            trial = np.where(rng.random(dim) < CR, v, pop[i])
            
            # Evaluate trial
            pos = trial.reshape(n, 3)
            if not _is_valid(pos):
                continue
            
            try:
                e_trial, relaxed = relax(pos, 'sutton_chen', kind)
                if np.isfinite(e_trial) and _is_valid(relaxed):
                    if e_trial < energies[i]:
                        pop[i] = relaxed.flatten()
                        energies[i] = e_trial
                        if e_trial < best_e:
                            best_e = e_trial
                            best_pos = relaxed.flatten().copy()
            except Exception:
                continue
    
    return best_pos.reshape(n, 3), best_e


def _basin_hop_chain(positions, kind, n_steps, rng):
    """Run a basin hopping chain with adaptive perturbation."""
    e, current = relax(positions, 'sutton_chen', kind)
    if not (np.isfinite(e) and _is_valid(current)):
        return positions, np.inf
    
    best_e = e
    best_pos = current.copy()
    
    for i in range(n_steps):
        if _time_left() < 8.0:
            break
        
        sigma = 0.06 + 0.1 * (1 + np.sin(i * 0.2)) / 2
        pos = current + rng.normal(0, sigma, (len(current), 3))
        
        if not _is_valid(pos):
            continue
        
        try:
            e_new, relaxed = relax(pos, 'sutton_chen', kind)
            if np.isfinite(e_new) and _is_valid(relaxed):
                if e_new < best_e:
                    best_e = e_new
                    best_pos = relaxed.copy()
                # Metropolis acceptance
                if e_new <= e or rng.random() < np.exp(-(e_new - e) / 0.1):
                    current = relaxed
                    e = e_new
        except Exception:
            continue
            
    return best_pos, best_e


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    # Phase 1: Generate diverse initial structures via multiple methods
    initial_candidates = []
    
    # FCC initial
    if _time_left() > 25.0:
        try:
            base = _generate_fcc(n)
            if _is_valid(base):
                energy, relaxed = relax(base, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    initial_candidates.append((energy, relaxed))
        except Exception:
            pass
    
    # HCP initial
    if _time_left() > 25.0:
        try:
            base = _generate_hcp(n)
            if _is_valid(base):
                energy, relaxed = relax(base, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    initial_candidates.append((energy, relaxed))
        except Exception:
            pass
    
    # BCC initial
    if _time_left() > 20.0:
        try:
            base = _generate_bcc(n)
            if _is_valid(base):
                energy, relaxed = relax(base, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    initial_candidates.append((energy, relaxed))
        except Exception:
            pass
    
    # Mackay initial (especially good for n=55)
    if n in [13, 55, 147] or _time_left() > 20.0:
        try:
            base = _generate_mackay(n)
            if _is_valid(base):
                energy, relaxed = relax(base, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    initial_candidates.append((energy, relaxed))
        except Exception:
            pass
    
    # Random initial structures
    for seed in [10, 20, 30, 40]:
        if _time_left() < 15.0:
            break
        try:
            base = _generate_random(n, seed)
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
    
    # Update best from initial candidates
    for e, pos in initial_candidates:
        if e < best_energy:
            best_energy = e
            best_pos = pos.copy()
    
    # Phase 2: Differential Evolution over the best candidates
    if _time_left() > 30.0:
        try:
            # Take top 5 candidates for DE
            de_candidates = [pos for _, pos in initial_candidates[:min(5, len(initial_candidates))]]
            de_pos, de_energy = _differential_evolution(de_candidates, kind, n_gen=12, pop_size=5, 
                                                         rng=np.random.default_rng(123))
            if de_energy < best_energy:
                best_energy = de_energy
                best_pos = de_pos.copy()
        except Exception:
            pass
    
    # Phase 3: Basin Hopping from the best candidates
    n_chains = min(3, len(initial_candidates))
    rngs = [np.random.default_rng(300 + i) for i in range(n_chains)]
    
    for i in range(n_chains):
        if _time_left() < 20.0:
            break
        
        e_init, pos_init = initial_candidates[i]
        
        # Run BH chain with adaptive steps based on remaining time
        n_steps = max(8, int(min(60, _time_left() / max(1.5, 0.5))))
        
        bh_pos, bh_energy = _basin_hop_chain(pos_init, kind, n_steps=n_steps, rng=rngs[i])
        
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos.copy()
    
    # Phase 4: If time remains, continue BH from the global best
    if best_pos is not None and _time_left() > 15.0:
        rng_final = np.random.default_rng(999)
        n_steps_final = max(5, int(_time_left() / 2.5))
        bh_pos, bh_energy = _basin_hop_chain(best_pos, kind, n_steps=n_steps_final, rng=rng_final)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos.copy()
    
    # Final validation
    if best_pos is not None and not _is_valid(best_pos):
        # Try to repair
        try:
            e, repaired = relax(best_pos, 'sutton_chen', kind)
            if np.isfinite(e) and _is_valid(repaired):
                best_pos = repaired
        except Exception:
            best_pos = _generate_fcc(n)
    
    return best_pos

# EVOLVE-BLOCK-END
