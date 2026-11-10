# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient vectorized."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    iu = np.triu_indices(n, k=1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.sqrt(np.sum(diff * diff, axis=1))
    r_safe = np.where(r < 1e-10, 1e-10, r)
    
    x = rho * (1.0 - r_safe)
    ex = np.exp(x)
    
    # Energy: sum of ex*(ex-2)
    energy = float(np.sum(ex * (ex - 2.0)))
    
    # Gradient: dE_dr = -2*rho*ex*(ex-1)
    dE_dr = -2.0 * rho * ex * (ex - 1.0)
    factor = dE_dr / r_safe  # shape (n_pairs,)
    
    # Build gradient using scatter
    grad = np.zeros_like(positions)
    np.add.at(grad, iu[0], factor[:, None] * diff)
    np.add.at(grad, iu[1], -factor[:, None] * diff)
    
    return energy, grad

def _relax_lbfgs(positions, rho, max_iter=300, tol=1e-8):
    """Relax using L-BFGS-B with analytic gradient."""
    def fun(x):
        pos = x.reshape(-1, 3)
        e, _ = _morse_energy_and_grad(pos, rho)
        return e
    
    def jac(x):
        pos = x.reshape(-1, 3)
        _, g = _morse_energy_and_grad(pos, rho)
        return g.flatten()
    
    x0 = positions.flatten()
    try:
        result = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'gtol': tol, 'maxls': 50})
        final_pos = result.x.reshape(-1, 3)
        final_energy, _ = _morse_energy_and_grad(final_pos, rho)
        return final_energy, final_pos
    except Exception:
        return np.inf, positions

def _check_minimum_distance(positions, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    n = len(positions)
    if n < 2:
        return True
    iu = np.triu_indices(n, k=1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.sqrt(np.sum(diff * diff, axis=1))
    return bool(np.all(r >= min_dist))

def _fix_minimum_distance(positions, min_dist=0.5, max_iter=100):
    """Iteratively fix minimum distance violations."""
    n = len(positions)
    if n < 2:
        return positions
    
    for _ in range(max_iter):
        iu = np.triu_indices(n, k=1)
        diff = positions[iu[0]] - positions[iu[1]]
        r = np.sqrt(np.sum(diff * diff, axis=1))
        violations = r < min_dist
        if not np.any(violations):
            break
        
        for idx in np.where(violations)[0]:
            i, j = iu[0][idx], iu[1][idx]
            d = positions[i] - positions[j]
            norm = np.linalg.norm(d)
            if norm < 1e-10:
                d = np.array([1.0, 0.0, 0.0])
                norm = 1.0
            else:
                d = d / norm
            separation = (min_dist - norm) / 2.0 + 0.01
            positions[i] += separation * d
            positions[j] -= separation * d
    
    return positions

def _generate_growth(n, rho, seed=42, candidates_per_step=20, relax_steps=20):
    """
    Generate a cluster by incremental growth.
    Start with 2 atoms, add one by one, finding the best position for each new atom.
    """
    rng = np.random.default_rng(seed)
    
    # Start with 2 atoms at optimal distance (1.0)
    if n <= 1:
        return np.zeros((n, 3))
    
    positions = np.array([
        [-0.5, 0, 0],
        [0.5, 0, 0]
    ])
    
    for k in range(2, n):
        # Generate candidate positions for the new atom
        # Place them on a sphere around the center of mass or randomly in a box
        com = np.mean(positions, axis=0)
        radius = 1.5 * (k ** (1.0/3.0))
        
        best_energy = np.inf
        best_pos = None
        
        for _ in range(candidates_per_step):
            # Random direction on sphere
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            # Random distance
            r = rng.uniform(0.5, radius)
            candidate = com + r * u
            
            # Check minimum distance constraint roughly
            # If it's too close to any existing atom, skip or push out
            dists = np.linalg.norm(positions - candidate, axis=1)
            if np.min(dists) < 0.5:
                continue
            
            # Calculate energy of new atom with existing atoms
            # E_new = sum_{i in 0..k-1} V(r_i, r_new)
            # We only need the interaction of the new atom with existing ones
            # V(r) = exp(rho(1-r))(exp(rho(1-r))-2)
            r_vals = np.linalg.norm(positions - candidate, axis=1)
            r_vals = np.maximum(r_vals, 1e-10)
            x = rho * (1.0 - r_vals)
            ex = np.exp(x)
            e_new = np.sum(ex * (ex - 2.0))
            
            if e_new < best_energy:
                best_energy = e_new
                best_pos = candidate
        
        if best_pos is None:
            # Fallback: place randomly
            best_pos = com + rng.normal(0, 1.0, 3)
        
        # Add the new atom
        positions = np.vstack([positions, best_pos])
        
        # Optional: Relax the whole cluster slightly to allow adjustment
        # This is expensive, so we do it sparingly or with few iterations
        if k % 5 == 0 or k == n - 1:
            # Relax with limited iterations
            e, relaxed = _relax_lbfgs(positions, rho, max_iter=relax_steps)
            if _check_minimum_distance(relaxed, 0.5):
                positions = relaxed
            else:
                positions = _fix_minimum_distance(relaxed, 0.5, max_iter=20)
        
        # Enforce minimum distance
        if not _check_minimum_distance(positions, 0.5):
            positions = _fix_minimum_distance(positions, 0.5, max_iter=20)
            
    return positions

def _generate_random_compact(n, seed, min_dist=0.6):
    """Generate a random compact cluster with minimum spacing."""
    rng = np.random.default_rng(seed)
    radius = 1.2 * n ** (1.0 / 3.0)
    positions = []
    max_attempts = 10000
    attempts = 0
    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if len(positions) == 0:
            positions.append(p)
        else:
            dists = np.linalg.norm(np.array(positions) - p, axis=1)
            if np.min(dists) >= min_dist:
                positions.append(p)
    
    while len(positions) < n:
        p = rng.uniform(-radius, radius, 3)
        positions.append(p)
    
    return np.array(positions)

def _basin_hop(positions, rho, n_cycles=20, noise_scale=0.3, deadline=None):
    """Basin hopping: perturb, relax, keep if lower energy."""
    best_energy, best_pos = _relax_lbfgs(positions, rho, max_iter=300)
    if not _check_minimum_distance(best_pos, 0.5):
        best_pos = _fix_minimum_distance(best_pos, 0.5, max_iter=50)
        best_energy, _ = _morse_energy_and_grad(best_pos, rho)
    
    rng = np.random.default_rng(42)
    
    for cycle in range(n_cycles):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb
        noise = rng.normal(0, noise_scale, size=best_pos.shape)
        perturbed = best_pos + noise
        perturbed = _fix_minimum_distance(perturbed, 0.5, max_iter=30)
        
        # Relax
        e, relaxed = _relax_lbfgs(perturbed, rho, max_iter=200)
        
        if e < best_energy and _check_minimum_distance(relaxed, 0.5):
            best_energy = e
            best_pos = relaxed
        elif not _check_minimum_distance(relaxed, 0.5):
            relaxed = _fix_minimum_distance(relaxed, 0.5, max_iter=50)
            e, _ = _morse_energy_and_grad(relaxed, rho)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed
    
    return best_energy, best_pos

def construct(instance):
    """Find low-energy Morse cluster configurations."""
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin before 330s kill
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Incremental Growth (Primary Strategy)
    # Use multiple seeds for diversity in growth
    growth_seeds = [42, 123, 456, 789]
    for seed in growth_seeds:
        if time.time() > deadline - 60:
            break
        
        try:
            # Generate via growth
            positions = _generate_growth(n, rho, seed=seed, candidates_per_step=15, relax_steps=10)
            
            # Final relaxation
            energy, relaxed = _relax_lbfgs(positions, rho, max_iter=300)
            
            if _check_minimum_distance(relaxed, 0.5) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except Exception:
            continue
    
    # Phase 2: Random Restarts (Fallback/Diversity)
    if time.time() < deadline - 60:
        rng_seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
        for seed in rng_seeds:
            if time.time() > deadline - 60:
                break
            
            try:
                positions = _generate_random_compact(n, seed, min_dist=0.6)
                positions = _fix_minimum_distance(positions, 0.5, max_iter=50)
                energy, relaxed = _relax_lbfgs(positions, rho, max_iter=200)
                
                if _check_minimum_distance(relaxed, 0.5) and energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
            except Exception:
                continue
    
    # Fallback if no valid solution
    if best_positions is None:
        rng = np.random.default_rng(n)
        radius = 1.2 * n ** (1.0 / 3.0)
        positions = rng.uniform(-radius, radius, (n, 3))
        positions = _fix_minimum_distance(positions, 0.5, max_iter=100)
        energy, relaxed = _relax_lbfgs(positions, rho, max_iter=200)
        best_positions = relaxed
        best_energy = energy
        if not _check_minimum_distance(best_positions, 0.5):
            best_positions = _fix_minimum_distance(best_positions, 0.5, max_iter=100)
    
    # Phase 3: Basin Hopping from the best solution
    if time.time() < deadline - 10:
        try:
            e, pos = _basin_hop(best_positions, rho, n_cycles=20, noise_scale=0.3, deadline=deadline - 5)
            if e < best_energy and _check_minimum_distance(pos, 0.5):
                best_energy = e
                best_positions = pos
        except Exception:
            pass
    
    # Phase 4: Try library relax as additional check
    if time.time() < deadline - 5:
        try:
            lib_pos = relax(best_positions.copy(), 'morse', rho)
            if lib_pos is not None and len(lib_pos) == n:
                lib_pos = np.array(lib_pos)
                e, _ = _morse_energy_and_grad(lib_pos, rho)
                if _check_minimum_distance(lib_pos, 0.5) and e < best_energy:
                    best_energy = e
                    best_positions = lib_pos
        except Exception:
            pass
    
    # Final validation
    if best_positions is None:
        best_positions = _generate_random_compact(n, 42)
    
    if not _check_minimum_distance(best_positions, 0.5):
        best_positions = _fix_minimum_distance(best_positions, 0.5, max_iter=100)
    
    return best_positions
# EVOLVE-BLOCK-END
