"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
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
    """Generate a cluster of n atoms in an FCC-like structure."""
    max_r = int(np.ceil(n ** (1 / 3))) + 4
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_bcc(n):
    """Generate a cluster of n atoms in a BCC-like structure."""
    max_r = int(np.ceil(n ** (1 / 3))) + 4
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                candidates.append((np.linalg.norm(p1), p1))
                candidates.append((np.linalg.norm(p2), p2))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_hcp(n):
    """Generate a cluster of n atoms in an HCP-like structure."""
    a = 1.0
    c = np.sqrt(8.0 / 3.0) * a
    candidates = []
    max_r = int(np.ceil(n ** (1 / 3))) + 4
    
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                # Layer A
                x = a * (i + 0.5 * j)
                y = a * np.sqrt(3) / 2 * j
                z = c * k
                p1 = np.array([x, y, z])
                candidates.append((np.linalg.norm(p1), p1))
                
                # Layer B (offset)
                x2 = a * (i + 0.5 * j + 1.0 / np.sqrt(3.0))
                y2 = a * np.sqrt(3) / 2 * (j + 1.0 / 3.0)
                z2 = c * (k + 1.0 / 3.0)
                p2 = np.array([x2, y2, z2])
                candidates.append((np.linalg.norm(p2), p2))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_icosahedron(n):
    """Generate a cluster of n atoms with icosahedral symmetry."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    pts = [np.zeros(3)]
    pts.extend(verts.tolist())
    
    # Add second shell (30 atoms at edges midpoints, pushed outward)
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 0.6:
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * 1.06
                pts.append(mid)
    
    # Add third shell if needed
    while len(pts) < n:
        # Face centroids
        added = False
        for i in range(12):
            for j in range(i + 1, 12):
                for k in range(j + 1, 12):
                    if len(pts) >= n:
                        break
                    d1 = np.linalg.norm(verts[i] - verts[j])
                    d2 = np.linalg.norm(verts[j] - verts[k])
                    d3 = np.linalg.norm(verts[k] - verts[i])
                    if d1 < 0.6 and d2 < 0.6 and d3 < 0.6:
                        centroid = (verts[i] + verts[j] + verts[k]) / 3
                        centroid = centroid / np.linalg.norm(centroid) * 1.18
                        pts.append(centroid)
                        added = True
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
        if len(pts) < n and not added:
            # Outer shell
            for v in verts:
                if len(pts) >= n:
                    break
                outer = v * 1.35
                pts.append(outer)
    
    return np.array(pts[:n])


def _generate_random_compact(n, rng, radius=2.5):
    """Generate n random points within a sphere."""
    pts = []
    while len(pts) < n:
        p = rng.uniform(-1, 1, size=(max(n - len(pts), 10), 3))
        norms = np.linalg.norm(p, axis=1)
        valid = p[norms <= 1.0]
        for v in valid:
            if len(pts) < n:
                pts.append(v * radius)
    return np.array(pts[:n])


def _differential_evolution_search(n, kind, rng, time_budget=25.0):
    """
    Perform differential evolution on the cluster configuration.
    Uses a population of candidate structures, mutating and crossing over.
    """
    pop_size = min(30, max(10, n // 2))
    # Initialize population with diverse starting points
    templates = [
        _generate_fcc(n),
        _generate_bcc(n),
        _generate_hcp(n),
        _generate_icosahedron(n),
        _generate_random_compact(n, rng, radius=2.0),
        _generate_random_compact(n, rng, radius=3.0),
    ]
    
    population = []
    for i in range(pop_size):
        template = templates[i % len(templates)]
        noise_scale = 0.1 + 0.2 * (i % 5)
        pos = template + rng.normal(0, noise_scale, (n, 3))
        population.append(pos)
    
    # Evaluate initial population
    energies = []
    for pos in population:
        try:
            energy, relaxed = relax(pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                energies.append(energy)
                population[population.index(pos)] = relaxed
            else:
                energies.append(np.inf)
        except Exception:
            energies.append(np.inf)
    
    best_idx = np.argmin(energies)
    best_energy = energies[best_idx]
    best_pos = population[best_idx].copy()
    
    # DE parameters
    F = 0.8  # Mutation factor
    CR = 0.9  # Crossover rate
    
    start_t = time.time()
    gen = 0
    while time.time() - start_t < time_budget and _time_left() > 10.0:
        gen += 1
        for i in range(pop_size):
            # Select three distinct individuals
            idxs = [i]
            while len(idxs) < 4:
                j = rng.integers(0, pop_size)
                if j not in idxs:
                    idxs.append(j)
            
            # Mutation: x_best + F * (x_r1 - x_r2)
            mutant = population[best_idx] + F * (population[idxs[1]] - population[idxs[2]])
            
            # Crossover
            trial = population[i].copy()
            for d in range(n * 3):
                if rng.random() < CR or d == rng.integers(0, n * 3):
                    trial.flat[d] = mutant.flat[d]
            
            # Evaluate trial
            try:
                energy, relaxed = relax(trial, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < energies[i]:
                        population[i] = relaxed
                        energies[i] = energy
                        if energy < best_energy:
                            best_energy = energy
                            best_pos = relaxed.copy()
                            best_idx = i
                else:
                    energies[i] = np.inf
            except Exception:
                energies[i] = np.inf
        
        # Update best
        cur_best_idx = np.argmin(energies)
        if energies[cur_best_idx] < best_energy:
            best_energy = energies[cur_best_idx]
            best_pos = population[cur_best_idx].copy()
            best_idx = cur_best_idx
    
    return best_pos, best_energy


def _basin_hopping(positions, kind, rng, n_trials=50):
    """
    Basin hopping: perturb, relax, accept/reject.
    """
    current_pos = positions.copy()
    current_energy, _ = sutton_chen_energy_gradient(current_pos, kind)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Simulated annealing schedule
    T0 = 1.0
    alpha = 0.95
    
    for i in range(n_trials):
        # Perturb
        step_size = 0.3 * (T0 ** (i / n_trials))
        new_pos = current_pos + rng.normal(0, step_size, (len(current_pos), 3))
        
        # Relax
        try:
            new_energy, relaxed = relax(new_pos, 'sutton_chen', kind)
            if not np.isfinite(new_energy) or not _is_valid(relaxed):
                continue
        except Exception:
            continue
        
        # Metropolis acceptance
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < np.exp(-delta / max(T0, 1e-10)):
            current_pos = relaxed
            current_energy = new_energy
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
        
        T0 *= alpha
    
    return best_pos, best_energy


def _polish(positions, kind):
    """High-precision polish pass."""
    try:
        energy, pos = relax(positions, 'sutton_chen', kind)
        if not np.isfinite(energy):
            return None, np.inf
        
        # Additional refinement
        for _ in range(3):
            e, g = sutton_chen_energy_gradient(pos, kind)
            if not np.isfinite(e):
                break
            # Small gradient steps
            lr = 0.005
            pos = pos - lr * g
            if not _is_valid(pos):
                break
        
        energy, pos = relax(pos, 'sutton_chen', kind)
        if np.isfinite(energy) and _is_valid(pos):
            return pos, energy
    except Exception:
        pass
    return None, np.inf


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    rng = np.random.default_rng(42 + n)
    
    # Generate base structures
    base_structures = [
        _generate_fcc(n),
        _generate_bcc(n),
        _generate_hcp(n),
        _generate_icosahedron(n),
        _generate_random_compact(n, rng, radius=2.0),
        _generate_random_compact(n, rng, radius=3.0),
    ]
    
    # Phase 1: Relax each base structure
    relaxed_bases = []
    for base in base_structures:
        if _time_left() < 40.0:
            break
        try:
            energy, relaxed_pos = relax(base, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed_pos):
                relaxed_bases.append(relaxed_pos)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos.copy()
        except Exception:
            continue
    
    # Phase 2: Differential evolution search
    if _time_left() > 60.0:
        de_pos, de_energy = _differential_evolution_search(n, kind, rng, time_budget=40.0)
        if de_energy < best_energy:
            best_energy = de_energy
            best_pos = de_pos.copy()
    
    # Phase 3: Basin hopping from best structure
    if best_pos is not None and _time_left() > 30.0:
        bh_pos, bh_energy = _basin_hopping(best_pos, kind, rng, n_trials=80)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos.copy()
    
    # Phase 4: Multi-start with perturbations from relaxed bases
    if _time_left() > 20.0:
        n_restarts = min(100, int(_time_left() / 0.5))
        for i in range(n_restarts):
            if _time_left() < 10.0:
                break
            
            if relaxed_bases:
                base = relaxed_bases[i % len(relaxed_bases)].copy()
            else:
                base = base_structures[i % len(base_structures)].copy()
            
            noise_scale = 0.05 + 0.2 * (i % 10) / 10.0
            pos = base + rng.normal(0, noise_scale, (n, 3))
            
            try:
                energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed_pos):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed_pos.copy()
            except Exception:
                continue
    
    # Phase 5: Final polish
    if best_pos is not None and _time_left() > 15.0:
        polished_pos, polished_energy = _polish(best_pos, kind)
        if polished_pos is not None and polished_energy < best_energy:
            best_energy = polished_energy
            best_pos = polished_pos
    
    # Fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
    
    return best_pos

# EVOLVE-BLOCK-END
