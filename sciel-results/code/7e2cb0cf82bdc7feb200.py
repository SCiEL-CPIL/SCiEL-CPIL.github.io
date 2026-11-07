"""Program for Sutton-Chen TM cluster global minima search.
Uses shrink-wrap annealing: start from a compact random sphere and gradually
relax with a soft-core repulsive potential to avoid high-energy barriers.
Combined with growth construction and multi-start basin hopping.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 280.0  # Safety margin before 300s kill


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
    max_r = int(np.ceil(n ** (1 / 3))) + 3
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
    max_r = int(np.ceil(n ** (1 / 3))) + 3
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
    a = 1.0 / np.sqrt(2.0)
    c = a * np.sqrt(8.0 / 3.0)
    
    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                x = a * (i + j / 2.0)
                y = a * np.sqrt(3.0) / 2.0 * j
                z = c * k
                candidates.append((np.sqrt(x*x + y*y + z*z), np.array([x, y, z])))
                x2 = a * (i + j / 2.0 + 0.5)
                y2 = a * np.sqrt(3.0) / 2.0 * (j + 1.0 / np.sqrt(3.0))
                z2 = c * (k + 1.0 / 3.0)
                candidates.append((np.sqrt(x2*x2 + y2*y2 + z2*z2), np.array([x2, y2, z2])))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_icosahedron(n):
    """Generate a Mackay icosahedral cluster of n atoms."""
    max_r = int(np.ceil(n ** (1 / 3))) + 2
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                candidates.append((np.linalg.norm(p), p))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_random_compact(n, rng, radius_scale=1.0):
    """Generate a random compact cluster."""
    radius = 1.2 + 0.4 * (n / 50.0) * radius_scale
    pos = rng.uniform(-radius, radius, (n, 3))
    return pos


def _shrink_wrap_anneal(n, kind, rng, n_steps=15):
    """
    Shrink-wrap annealing: start with a tightly packed sphere of random points,
    then gradually relax with decreasing soft-core repulsion to smoothly separate
    atoms and find low-energy configurations.
    """
    # Start with points in a small sphere (overlapping)
    radius_start = 0.5 + 0.2 * (n / 50.0)
    pos = rng.uniform(-radius_start, radius_start, (n, 3))
    
    # Annealing schedule: soft-core strength decreases from high to zero
    # Use a schedule that starts with strong repulsion and gradually relaxes
    alphas = np.linspace(1.0, 0.0, n_steps)
    
    best_energy = np.inf
    best_pos = pos.copy()
    
    for step, alpha in enumerate(alphas):
        if _time_left() < 20.0:
            break
        
        # Define the annealed energy function with soft-core repulsion
        def annealed_energy(x):
            p = x.reshape(-1, 3)
            # Compute Sutton-Chen energy
            sc_e, _ = sutton_chen_energy_gradient(p, kind)
            
            # Add soft-core repulsive term: alpha * sum_{i<j} 1/(r_ij^2 + eps)
            n_atoms = len(p)
            repulsive = 0.0
            for i in range(n_atoms - 1):
                d = p[i+1:] - p[i]
                r2 = np.sum(d * d, axis=1)
                repulsive += np.sum(1.0 / (r2 + 0.01))
            
            return sc_e + alpha * 10.0 * repulsive
        
        def annealed_gradient(x):
            p = x.reshape(-n_atoms, 3) if False else x.reshape(-1, 3)
            n_atoms = len(p)
            
            # Sutton-Chen gradient
            _, sc_grad = sutton_chen_energy_gradient(p, kind)
            grad = sc_grad.copy()
            
            # Soft-core gradient: -2*alpha*10 * sum_{j!=i} (p_i - p_j) / (r_ij^2 + eps)^2
            for i in range(n_atoms):
                for j in range(n_atoms):
                    if i == j:
                        continue
                    d = p[i] - p[j]
                    r2 = np.sum(d * d)
                    grad[i] += -2.0 * alpha * 10.0 * d / (r2 + 0.01) ** 2
            
            return grad.ravel()
        
        # Use L-BFGS-B to minimize the annealed energy
        try:
            result = minimize(
                annealed_energy,
                best_pos.ravel(),
                jac=annealed_gradient,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-8, 'maxls': 30}
            )
            if np.isfinite(result.fun):
                candidate = result.x.reshape(-1, 3)
                if _is_valid(candidate):
                    # Evaluate the true Sutton-Chen energy (without soft-core)
                    true_e, relaxed = relax(candidate, 'sutton_chen', kind)
                    if np.isfinite(true_e) and _is_valid(relaxed):
                        if true_e < best_energy:
                            best_energy = true_e
                            best_pos = relaxed
                    else:
                        # Keep the candidate as-is for next annealing step
                        best_pos = candidate
                else:
                    # If invalid, keep previous position
                    pass
        except Exception:
            continue
        
        # After each annealing step, try a direct relaxation
        try:
            true_e, relaxed = relax(best_pos, 'sutton_chen', kind)
            if np.isfinite(true_e) and _is_valid(relaxed):
                if true_e < best_energy:
                    best_energy = true_e
                    best_pos = relaxed
        except Exception:
            pass
    
    return best_energy, best_pos


def _polish(positions, kind, max_iter=500, tol=1e-10):
    """High-precision polish pass using gradient-based optimization."""
    def energy_fn(x):
        pos = x.reshape(-1, 3)
        e, _ = sutton_chen_energy_gradient(pos, kind)
        return e
    
    def gradient_fn(x):
        pos = x.reshape(-1, 3)
        _, g = sutton_chen_energy_gradient(pos, kind)
        return g.ravel()
    
    try:
        result = minimize(
            energy_fn,
            positions.ravel(),
            jac=gradient_fn,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': tol, 'ftol': tol, 'maxls': 50}
        )
        if result.success and np.isfinite(result.fun):
            polished = result.x.reshape(-1, 3)
            if _is_valid(polished):
                return result.fun, polished
    except Exception:
        pass
    
    return np.inf, positions


def _growth_construct(n, kind, rng):
    """
    Deterministic growth algorithm: build cluster by adding atoms one at a time.
    """
    if n <= 1:
        return np.zeros((n, 3))
    
    best_energy = np.inf
    best_pos = None
    
    for d0 in [0.9, 1.0, 1.1, 1.2]:
        pos = np.array([[0, 0, 0], [d0, 0, 0]])
        try:
            energy, relaxed = relax(pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed
        except Exception:
            continue
    
    if best_pos is None:
        best_pos = np.array([[0, 0, 0], [1.0, 0, 0]])
    
    current = best_pos.copy()
    
    for target_n in range(3, n + 1):
        if _time_left() < 40.0:
            break
        
        candidates = []
        
        for i in range(len(current)):
            for trial in range(6):
                direction = rng.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
                for dist in [0.8, 0.9, 1.0, 1.1, 1.2]:
                    new_pos = current[i] + direction * dist
                    candidates.append(new_pos)
        
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                midpoint = (current[i] + current[j]) / 2
                bond_vec = current[j] - current[i]
                bond_len = np.linalg.norm(bond_vec)
                if bond_len > 0.1:
                    perp = np.cross(bond_vec, [0, 0, 1])
                    if np.linalg.norm(perp) < 1e-6:
                        perp = np.cross(bond_vec, [1, 0, 0])
                    perp /= np.linalg.norm(perp)
                    for offset in [0.3, 0.5, 0.7]:
                        candidates.append(midpoint + perp * offset)
                        candidates.append(midpoint - perp * offset)
        
        best_e = np.inf
        best_p = None
        
        rng.shuffle(candidates)
        for candidate in candidates[:12]:
            if _time_left() < 35.0:
                break
            
            test_pos = np.vstack([current, candidate.reshape(1, 3)])
            try:
                energy, relaxed = relax(test_pos, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < best_e:
                        best_e = energy
                        best_p = relaxed
            except Exception:
                continue
        
        if best_p is not None:
            current = best_p
        else:
            new_atom = current[0] + rng.normal(0, 0.5, 3)
            current = np.vstack([current, new_atom.reshape(1, 3)])
    
    return current


def _basin_hopping(positions_init, kind, rng, n_trials=8, step_scale=0.3):
    """
    Basin hopping: perturb the initial structure, relax, and keep the best.
    """
    best_energy = np.inf
    best_pos = positions_init.copy()
    
    # First, relax the initial structure
    try:
        e, relaxed = relax(positions_init, 'sutton_chen', kind)
        if np.isfinite(e) and _is_valid(relaxed):
            best_energy = e
            best_pos = relaxed
    except Exception:
        pass
    
    for _ in range(n_trials):
        if _time_left() < 20.0:
            break
        
        # Generate a random perturbation
        noise = rng.normal(0, step_scale, positions_init.shape)
        perturbed = best_pos + noise
        
        if not _is_valid(perturbed):
            continue
        
        try:
            e, relaxed = relax(perturbed, 'sutton_chen', kind)
            if np.isfinite(e) and _is_valid(relaxed):
                if e < best_energy:
                    best_energy = e
                    best_pos = relaxed
        except Exception:
            continue
    
    return best_energy, best_pos


def _differential_evolution(positions_init, kind, pop_size=8, max_generations=20,
                             mutation_rate=0.7, crossover_rate=0.8, rng=None):
    """
    Custom differential evolution for cluster optimization.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_atoms = len(positions_init)
    
    population = []
    population.append(positions_init.copy())
    
    templates = [_generate_fcc, _generate_bcc, _generate_hcp, _generate_icosahedron]
    for i in range(1, pop_size):
        if _time_left() < 30.0:
            break
        
        if i % 2 == 1 and i // 2 < len(templates):
            base = templates[i // 2](n_atoms)
            noise = rng.normal(0, 0.1, base.shape)
            population.append(base + noise)
        else:
            radius = 1.2 + 0.4 * (n_atoms / 50.0)
            pos = rng.uniform(-radius, radius, (n_atoms, 3))
            population.append(pos)
    
    while len(population) < pop_size:
        base = _generate_fcc(n_atoms)
        noise = rng.normal(0, 0.15, base.shape)
        population.append(base + noise)
    
    population = np.array(population)
    
    fitness = np.full(pop_size, np.inf)
    for i in range(pop_size):
        if _time_left() < 25.0:
            break
        pos = population[i]
        if not _is_valid(pos):
            fitness[i] = np.inf
            continue
        try:
            energy, relaxed = relax(pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                fitness[i] = energy
                population[i] = relaxed
        except Exception:
            fitness[i] = np.inf
    
    best_idx = np.argmin(fitness)
    best_energy = fitness[best_idx]
    best_pos = population[best_idx].copy()
    
    for gen in range(max_generations):
        if _time_left() < 20.0:
            break
        
        for i in range(pop_size):
            if _time_left() < 15.0:
                break
            
            idxs = np.random.choice(pop_size, size=4, replace=False)
            a, b, c = idxs[0], idxs[1], idxs[2]
            
            if a == i or b == i or c == i:
                continue
            
            v = population[a] + mutation_rate * (population[b] - population[c])
            
            jrand = np.random.randint(0, n_atoms)
            u = population[i].copy()
            for j in range(n_atoms):
                if rng.random() < crossover_rate or j == jrand:
                    u[j] = v[j]
            
            if not _is_valid(u):
                continue
            
            try:
                energy, relaxed = relax(u, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < fitness[i]:
                        fitness[i] = energy
                        population[i] = relaxed
                        if energy < best_energy:
                            best_energy = energy
                            best_pos = relaxed.copy()
            except Exception:
                continue
    
    return best_energy, best_pos


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    rng = np.random.default_rng(42 + n)
    
    # Strategy 1: Shrink-wrap annealing (primary new strategy)
    try:
        sw_energy, sw_pos = _shrink_wrap_anneal(n, kind, rng, n_steps=12)
        if np.isfinite(sw_energy) and _is_valid(sw_pos):
            best_energy = sw_energy
            best_pos = sw_pos
    except Exception:
        pass
    
    # Strategy 2: Growth-based construction
    try:
        growth_pos = _growth_construct(n, kind, rng)
        if _is_valid(growth_pos):
            energy, relaxed = relax(growth_pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed
    except Exception:
        pass
    
    # Strategy 3: Template-based multi-start
    for gen_func in [_generate_fcc, _generate_bcc, _generate_hcp, _generate_icosahedron]:
        if _time_left() < 25.0:
            break
        
        base = gen_func(n)
        if not _is_valid(base):
            continue
        
        for noise_scale in [0.0, 0.05, 0.1, 0.2]:
            if _time_left() < 20.0:
                break
            
            noise = rng.normal(0, noise_scale, base.shape) if noise_scale > 0 else np.zeros_like(base)
            pos = base + noise
            
            try:
                energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
                
                if not np.isfinite(energy):
                    continue
                    
                if _is_valid(relaxed_pos):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed_pos
            except Exception:
                continue
    
    # Strategy 4: Basin hopping from best position found
    if best_pos is not None and _time_left() > 40.0:
        try:
            bh_energy, bh_pos = _basin_hopping(best_pos, kind, rng, n_trials=10, step_scale=0.3)
            if np.isfinite(bh_energy) and _is_valid(bh_pos):
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_pos = bh_pos
        except Exception:
            pass
    
    # Strategy 5: Differential Evolution if time remains
    if _time_left() > 50.0:
        try:
            if best_pos is not None:
                de_seed = best_pos.copy()
            else:
                de_seed = _generate_fcc(n)
            
            de_energy, de_pos = _differential_evolution(
                de_seed, kind, 
                pop_size=8, 
                max_generations=15,
                mutation_rate=0.7,
                crossover_rate=0.8,
                rng=rng
            )
            
            if np.isfinite(de_energy) and _is_valid(de_pos):
                if de_energy < best_energy:
                    best_energy = de_energy
                    best_pos = de_pos
        except Exception:
            pass
    
    # Strategy 6: Random restarts if time remains
    if _time_left() > 30.0:
        for i in range(15):
            if _time_left() < 20.0:
                break
            
            radius_scale = 1.0 + 0.2 * (i % 3)
            pos = _generate_random_compact(n, rng, radius_scale)
            
            try:
                energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
                
                if not np.isfinite(energy):
                    continue
                    
                if _is_valid(relaxed_pos):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed_pos
            except Exception:
                continue
    
    # Strategy 7: Final high-precision polish
    if best_pos is not None and _time_left() > 15.0:
        try:
            polished_energy, polished_pos = _polish(best_pos, kind, max_iter=200, tol=1e-12)
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_pos = polished_pos
        except Exception:
            pass
    
    # Final fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
    
    return best_pos

# EVOLVE-BLOCK-END
