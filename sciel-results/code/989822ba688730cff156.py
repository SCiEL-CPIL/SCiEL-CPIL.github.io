"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
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
    for i in range(n - 1):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _generate_fcc(n):
    """Generate n atoms in FCC-like arrangement (closest to origin)."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
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
    max_r = int(np.ceil(n ** (1 / 3))) + 3
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
    pts = []
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    a = 1.0
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            x = i + 0.5 * j
            y = (np.sqrt(3) / 2) * j
            for layer in range(4):
                z = layer * 0.816
                if layer % 2 == 1:
                    x += 1.0 / 3
                    y += 1.0 / np.sqrt(3)
                p = np.array([x, y, z], dtype=float)
                candidates.append((np.dot(p, p), p))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_icosahedral(n):
    """Generate icosahedral cluster with shells."""
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
        rng = np.random.default_rng(123 + n)
        remaining = n - len(pts)
        dirs = rng.normal(size=(remaining, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        radius = 1.9
        for d in dirs:
            pts.append(d * radius)
    
    return np.array(pts[:n])


def _generate_random(n, seed):
    """Generate random positions in a sphere."""
    rng = np.random.default_rng(seed)
    r = 0.3 * np.sqrt(n)
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    pts = pts / norms * r * rng.random((n, 1)) ** (1 / 3)
    return pts


def _differential_evolution(positions, kind, n_restarts=10, population_size=20):
    """
    Use differential evolution-style global search:
    - Maintain a population of candidate structures
    - Generate new candidates via mutation (differential + random)
    - Evaluate with local relaxation
    - Keep best candidates
    """
    n = len(positions)
    rng = np.random.default_rng(42 + n)
    
    # Initial population from diverse templates
    population = []
    energies = []
    
    # Add base templates
    for gen in [_generate_fcc, _generate_bcc, _generate_hcp, _generate_icosahedral]:
        try:
            base = gen(n)
            if _is_valid(base):
                population.append(base)
                e, _ = sutton_chen_energy_gradient(base, kind)
                energies.append(e)
        except Exception:
            pass
    
    # Add random initializations
    for i in range(population_size - len(population)):
        pos = _generate_random(n, 1000 + i * 31 + n)
        if _is_valid(pos):
            population.append(pos)
            e, _ = sutton_chen_energy_gradient(pos, kind)
            energies.append(e)
    
    if len(population) < 4:
        return None, np.inf
    
    best_energy = min(energies)
    best_idx = np.argmin(energies)
    best_pos = population[best_idx].copy()
    
    # DE iterations
    for gen_iter in range(n_restarts):
        if _time_left() < 15.0:
            break
        
        # Generate new candidates via mutation
        new_candidates = []
        new_energies = []
        
        n_mutations = max(3, population_size // 2)
        for m in range(n_mutations):
            if _time_left() < 15.0:
                break
            
            # Select three distinct random individuals
            idxs = rng.choice(len(population), size=3, replace=False)
            a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
            
            # Differential mutation: x' = a + F * (b - c)
            F = 0.8
            mutant = a + F * (b - c)
            
            # Add small random noise
            mutant += rng.normal(0, 0.05, mutant.shape)
            
            if not _is_valid(mutant):
                continue
            
            try:
                energy, relaxed = relax(mutant, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    new_candidates.append(relaxed)
                    new_energies.append(energy)
            except Exception:
                continue
        
        # Combine with existing population
        all_pos = population + new_candidates
        all_energies = energies + new_energies
        
        if len(all_energies) == 0:
            break
        
        # Select top candidates
        order = np.argsort(all_energies)
        population = [all_pos[i] for i in order[:population_size]]
        energies = [all_energies[i] for i in order[:population_size]]
        
        if energies[0] < best_energy:
            best_energy = energies[0]
            best_pos = population[0].copy()
    
    return best_pos, best_energy


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    # Time budget for this instance
    instance_budget = max(10.0, _time_left() - 5.0)
    
    # Phase 1: Try base templates with local relaxation
    generators = [_generate_fcc, _generate_bcc, _generate_hcp, _generate_icosahedral]
    
    for gen in generators:
        if _time_left() < 15.0:
            break
        try:
            base = gen(n)
            if not _is_valid(base):
                base = base * 1.2
            if _is_valid(base):
                energy, relaxed = relax(base, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed.copy()
        except Exception:
            continue
    
    # Phase 2: Differential evolution global search
    if _time_left() > 30.0:
        de_pos, de_energy = _differential_evolution(
            best_pos if best_pos is not None else _generate_fcc(n),
            kind,
            n_restarts=8,
            population_size=15
        )
        if de_pos is not None and de_energy < best_energy:
            best_energy = de_energy
            best_pos = de_pos
    
    # Phase 3: Basin-hopping refinement on best found
    if best_pos is not None and _time_left() > 20.0:
        rng = np.random.default_rng(42 + n + hash(kind) % 1000)
        for i in range(15):
            if _time_left() < 15.0:
                break
            sigma = 0.05 + 0.03 * (i / 15)
            pos = best_pos + rng.normal(0, sigma, best_pos.shape)
            if not _is_valid(pos):
                continue
            try:
                energy, relaxed = relax(pos, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed.copy()
            except Exception:
                continue
    
    # Fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            best_pos = _generate_bcc(n)
        if not _is_valid(best_pos):
            rng2 = np.random.default_rng(999)
            best_pos = rng2.uniform(-2, 2, (n, 3))
            for attempt in range(100):
                if _is_valid(best_pos):
                    break
                best_pos = rng2.uniform(-2, 2, (n, 3))
    
    return best_pos

# EVOLVE-BLOCK-END
