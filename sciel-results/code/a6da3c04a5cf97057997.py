"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 150s kill

def _lj_energy(pos):
    """Compute total Lennard-Jones energy efficiently."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    iu = np.triu_indices(len(pos), k=1)
    return np.sum(pair_energy[iu])

def _lj_energy_grad(pos):
    """Compute gradient of LJ energy."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    # dE/dx_i = sum_{j!=i} 4 * (-12/r^14 + 6/r^8) * (x_i - x_j)
    # where r^2 = r2, so r^14 = r2^7, r^8 = r2^4
    coeff = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos
    return grad

def _make_mackay_icosahedron(r1=1.12, r2=2.05, r3=2.5):
    """Generate Mackay icosahedron with 55 atoms (1 + 12 + 30 + 12)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    r_circ = np.sqrt(1.0 + phi ** 2)
    
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    vertices = vertices / r_circ
    
    # Find edges by computing all pairwise distances
    d = np.linalg.norm(vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d)
    
    edge_mid_unit = np.zeros((30, 3))
    count = 0
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(d[i, j] - min_d) < 1e-10:
                mid = (vertices[i] + vertices[j]) / 2.0
                edge_mid_unit[count] = mid / np.linalg.norm(mid)
                count += 1
    
    positions = np.zeros((55, 3))
    positions[1:13] = r1 * vertices
    positions[13:43] = r2 * edge_mid_unit
    positions[43:55] = r3 * vertices
    
    return positions

def _relax_lbfgs(pos, ftol=1e-14, gtol=1e-12, maxiter=500):
    """Single L-BFGS relaxation pass."""
    def obj_and_grad(x):
        x = x.reshape(55, 3)
        e = _lj_energy(x)
        g = _lj_energy_grad(x)
        return e, g.flatten()
    
    res = minimize(obj_and_grad, pos.flatten(), jac=True, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
    return res.x.reshape(55, 3), res.fun

def _polish(pos, remaining_time):
    """Multi-stage polish with progressively tighter tolerances."""
    stages = [
        (1e-12, 1e-10, 200),
        (1e-14, 1e-12, 300),
        (1e-15, 1e-13, 500),
        (1e-16, 1e-14, 800),
    ]
    best = pos.copy()
    best_e = _lj_energy(pos)
    
    for ftol, gtol, maxiter in stages:
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        try:
            p, e = _relax_lbfgs(best, ftol=ftol, gtol=gtol, maxiter=maxiter)
            if e < best_e:
                best = p
                best_e = e
        except Exception:
            break
    
    return best, best_e

def _optimize_radial_params():
    """Optimize the 3 radial parameters of the Mackay icosahedron."""
    def symmetric_energy(params):
        r1, r2, r3 = params
        # Ensure radii are positive and reasonable
        if r1 < 0.5 or r2 < r1 + 0.5 or r3 < r2 + 0.5:
            return 1e10
        p = _make_mackay_icosahedron(r1, r2, r3)
        return _lj_energy(p)
    
    # Use Nelder-Mead for robust optimization of 3 parameters
    try:
        res = minimize(symmetric_energy, [1.04, 1.76, 1.98], method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        return res.x
    except Exception:
        return np.array([1.04, 1.76, 1.98])

def _differential_evolution_search(initial_pos, best_energy, num_generations=3, pop_size=6):
    """
    Differential evolution population-based search.
    Each individual is a 55-atom structure initialized by perturbing the Mackay icosahedron.
    Fitness is evaluated via a single L-BFGS relaxation.
    """
    rng = np.random.default_rng(123)
    elapsed = time.time() - START_TIME
    
    # Initialize population
    population = [initial_pos.copy()]
    for i in range(pop_size - 1):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        perturb_size = 0.05 + 0.02 * i
        perturbed = initial_pos + rng.normal(0, perturb_size, size=(55, 3))
        population.append(perturbed)
    
    # Evaluate initial population
    energies = [best_energy]
    for i in range(1, len(population)):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        try:
            p, e = _relax_lbfgs(population[i], ftol=1e-10, gtol=1e-8, maxiter=200)
            population[i] = p
            energies.append(e)
        except Exception:
            energies.append(1e10)
    
    best_energy_de = min(energies)
    best_pos_de = population[energies.index(best_energy_de)]
    
    # Evolution loop
    for gen in range(num_generations):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        
        # Sort population by energy
        sorted_indices = np.argsort(energies)
        sorted_pop = [population[i] for i in sorted_indices]
        sorted_energies = [energies[i] for i in sorted_indices]
        
        # Generate offspring via mutation and crossover
        for i in range(len(population)):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            # Select three distinct random individuals (excluding i)
            candidates = [j for j in range(len(population)) if j != i]
            a, b, c = rng.choice(candidates, size=3, replace=False)
            
            # Mutation: v = x_a + F * (x_b - x_c)
            F = 0.8
            mutant = sorted_pop[a] + F * (sorted_pop[b] - sorted_pop[c])
            
            # Crossover: binomial crossover with rate 0.5
            trial = np.zeros_like(mutant)
            cross_mask = rng.random((55, 3)) < 0.5
            trial[cross_mask] = mutant[cross_mask]
            trial[~cross_mask] = sorted_pop[i][~cross_mask]
            
            # Evaluate trial
            try:
                p, e = _relax_lbfgs(trial, ftol=1e-10, gtol=1e-8, maxiter=200)
                if e < sorted_energies[i]:
                    sorted_pop[i] = p
                    sorted_energies[i] = e
            except Exception:
                pass
        
        # Update population
        population = sorted_pop
        energies = sorted_energies
        
        # Track best
        current_best_e = min(energies)
        if current_best_e < best_energy_de:
            best_energy_de = current_best_e
            best_pos_de = population[energies.index(current_best_e)]
    
    return best_pos_de, best_energy_de

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Step 1: Optimize radial parameters
    optimal_params = _optimize_radial_params()
    pos = _make_mackay_icosahedron(*optimal_params)
    
    # Step 2: Initial polish
    best_pos, best_energy = _polish(pos, TIME_LIMIT - (time.time() - START_TIME))
    
    # Step 3: Differential evolution search if time remains
    elapsed = time.time() - START_TIME
    if TIME_LIMIT - elapsed > 30:
        de_pos, de_energy = _differential_evolution_search(
            best_pos, best_energy, 
            num_generations=2, 
            pop_size=5
        )
        if de_energy < best_energy:
            best_energy = de_energy
            best_pos = de_pos
    
    # Step 4: Final polish with tight tolerances
    if time.time() - START_TIME < TIME_LIMIT - 5:
        best_pos, best_energy = _polish(best_pos, TIME_LIMIT - (time.time() - START_TIME))
    
    # Final check: ensure minimum distance >= 0.5
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1e10)
    min_dist = np.min(dists)
    
    if min_dist < 0.5:
        scale = 0.5 / min_dist * 1.01
        best_pos = best_pos * scale
    
    return best_pos
# EVOLVE-BLOCK-END
