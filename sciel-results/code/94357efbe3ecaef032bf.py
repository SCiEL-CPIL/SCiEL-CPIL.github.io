"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

START_TIME = time.time()
TIME_LIMIT = 115.0 # Leave 5s buffer

def _lj_energy_and_grad(x):
    """Compute Lennard-Jones energy and gradient efficiently."""
    pos = x.reshape(-1, 3)
    n = pos.shape[0]
    
    # Compute pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1.0) # Avoid division by zero
    
    inv_r2 = 1.0 / dists_sq
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy: sum over i < j
    # The matrix inv_r12 - inv_r6 has 2x the energy because it sums i!=j
    energy = 2.0 * np.sum(inv_r12 - inv_r6)
    
    # Gradient
    # dE/dx_i = sum_{j != i} (x_i - x_j) * (24 * r_ij^-8 - 48 * r_ij^-14)
    inv_r8 = inv_r2 ** 4
    inv_r14 = inv_r2 ** 7
    
    coeff = 24.0 * inv_r8 - 48.0 * inv_r14
    np.fill_diagonal(coeff, 0.0)
    
    # grad[i] = sum_j coeff[i, j] * diff[i, j]
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad.flatten()

def _relax(pos, max_iter=500):
    """Relax a structure using L-BFGS-B."""
    if time.time() - START_TIME > TIME_LIMIT:
        return pos
    
    pos = pos.copy()
    try:
        result = minimize(
            _lj_energy_and_grad,
            pos.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
        )
        relaxed = result.x.reshape(-1, 3)
        
        # Check validity
        dists = pdist(relaxed, 'euclidean')
        if np.min(dists) < 0.5:
            return pos
        return relaxed
    except Exception:
        return pos

def _valid(pos):
    """Check if positions are valid (min distance >= 0.5)."""
    dists = pdist(pos, 'euclidean')
    return np.min(dists) >= 0.5

def _make_mackay_seed():
    """
    Construct a Mackay icosahedron for 55 atoms.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    
    # Scale Shell 1
    R1 = 1.122
    shell1 = verts * R1
    
    # Find edges (shortest distances between vertices)
    dists = pdist(verts)
    min_dist = np.min(dists)
    
    edges = []
    idx = 0
    n = len(verts)
    for i in range(n):
        for j in range(i+1, n):
            if abs(dists[idx] - min_dist) < 1e-6:
                edges.append((i, j))
            idx += 1
            
    # 30 edge midpoints
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        mid = mid / mid_norm
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    # 12 dodecahedron vertices (face centers of icosahedron)
    # We will construct the 20 face centers and pick 12
    face_centers = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_ik = np.linalg.norm(verts[i] - verts[k])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                if abs(d_ij - min_dist) < 1e-6 and abs(d_ik - min_dist) < 1e-6 and abs(d_jk - min_dist) < 1e-6:
                    center = verts[i] + verts[j] + verts[k]
                    center_norm = np.linalg.norm(center)
                    if center_norm > 1e-6:
                        center = center / center_norm
                        face_centers.append(center)
                        
    face_centers = np.array(face_centers)
    shell2_part2 = face_centers[:12]
    
    # Scale Shell 2
    R2 = 2.244
    
    shell2_part1 = midpoints * R2
    shell2_part2 = shell2_part2 * R2
    
    shell2 = np.vstack([shell2_part1, shell2_part2])
    
    center = np.array([[0.0, 0.0, 0.0]])
    all_points = np.vstack([center, shell1, shell2])
    
    return all_points

def _make_fcc_seed():
    """Generate an FCC fragment seed."""
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                pts.append([i, j, k])
                pts.append([i+0.5, j+0.5, k])
                pts.append([i+0.5, j, k+0.5])
                pts.append([i, j+0.5, k+0.5])
    fcc_pts = np.array(pts)
    
    # Scale to LJ distance ~1.12
    fcc_pts *= 1.58
    
    # Select 55 closest to center
    dists = np.linalg.norm(fcc_pts, axis=1)
    idxs = np.argsort(dists)[:55]
    return fcc_pts[idxs]

def _make_random_seed():
    """Generate a valid random seed."""
    while True:
        pts = np.random.randn(N, 3) * 2.0
        if _valid(pts):
            return pts

def _distort_mackay(noise_level=0.2):
    """Create a distorted Mackay seed."""
    base = _make_mackay_seed()
    distorted = base + np.random.randn(N, 3) * noise_level
    return distorted

def construct():
    """Genetic Algorithm for LJ cluster optimization."""
    np.random.seed(42)
    
    pop_size = 40
    population = []
    energies = []
    
    # Initial Seeds
    seeds = [
        _make_mackay_seed(),
        _distort_mackay(0.1),
        _distort_mackay(0.3),
        _make_fcc_seed(),
        _make_random_seed(),
        _make_random_seed()
    ]
    
    for s in seeds:
        population.append(s)
        energies.append(_lj_energy_and_grad(s.flatten())[0])
        
    # Fill rest with random
    while len(population) < pop_size:
        s = _make_random_seed()
        population.append(s)
        energies.append(_lj_energy_and_grad(s.flatten())[0])
        
    # Relax initial population
    for i in range(len(population)):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        population[i] = _relax(population[i])
        if _valid(population[i]):
            energies[i] = _lj_energy_and_grad(population[i].flatten())[0]
            
    best_idx = np.argmin(energies)
    best_energy = energies[best_idx]
    best_pos = population[best_idx].copy()
    
    max_generations = 200
    gen = 0
    stagnation_count = 0
    current_sigma = 0.3
    
    while gen < max_generations:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
            
        gen += 1
        
        # Sort population
        sorted_indices = np.argsort(energies)
        sorted_pop = [population[i] for i in sorted_indices]
        sorted_energies = [energies[i] for i in sorted_indices]
        
        # Elitism: Keep top 3
        elitism_count = 3
        new_population = [sorted_pop[i].copy() for i in range(elitism_count)]
        new_energies = [sorted_energies[i] for i in range(elitism_count)]
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
                
            # Tournament Selection
            def tournament_select():
                idxs = np.random.choice(len(population), 5, replace=False)
                best_idx = idxs[np.argmin([energies[i] for i in idxs])]
                return population[best_idx].copy()
                
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover
            if np.random.rand() < 0.7:
                # Uniform Crossover
                mask = np.random.rand(N) < 0.5
                child = np.where(mask[:, np.newaxis], parent1, parent2)
            else:
                # Keep better parent (approx)
                child = parent1.copy()
                
            # Adaptive Mutation
            # Decay sigma slowly, but increase if stagnating
            if stagnation_count > 5:
                current_sigma = min(1.0, current_sigma * 1.1)
            else:
                current_sigma = max(0.05, current_sigma * 0.98)
                
            child += np.random.randn(N, 3) * current_sigma
            
            # Relax
            relaxed_child = _relax(child)
            if _valid(relaxed_child):
                new_population.append(relaxed_child)
                new_energies.append(_lj_energy_and_grad(relaxed_child.flatten())[0])
            else:
                # If relaxation fails validity, try a small nudge and re-relax? 
                # Or just keep the unrelaxed one? 
                # Let's keep the unrelaxed one but it will likely be high energy.
                new_population.append(child)
                new_energies.append(_lj_energy_and_grad(child.flatten())[0])
                
        population = new_population
        energies = new_energies
        
        # Update best
        gen_best_idx = np.argmin(energies)
        if energies[gen_best_idx] < best_energy:
            best_energy = energies[gen_best_idx]
            best_pos = population[gen_best_idx].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1
            
        # Stagnation handling: Replace worst 10% with new random seeds
        if stagnation_count > 15:
            num_replace = max(5, pop_size // 10)
            for _ in range(num_replace):
                idx = np.random.randint(len(population))
                population[idx] = _make_random_seed()
                population[idx] = _relax(population[idx])
                if _valid(population[idx]):
                    energies[idx] = _lj_energy_and_grad(population[idx].flatten())[0]
            stagnation_count = 0
            
    # Final Polish: Take top 5 and run high-precision relaxation
    if time.time() - START_TIME < TIME_LIMIT - 5:
        sorted_indices = np.argsort(energies)
        top_candidates = [population[i].copy() for i in sorted_indices[:5]]
        
        for i, cand in enumerate(top_candidates):
            if time.time() - START_TIME > TIME_LIMIT - 2:
                break
            # Run with higher max_iter and tighter tolerances
            polished = _relax(cand, max_iter=500)
            if _valid(polished):
                e = _lj_energy_and_grad(polished.flatten())[0]
                if e < best_energy:
                    best_energy = e
                    best_pos = polished
        
    # Final final relaxation on the absolute best
    if time.time() - START_TIME < TIME_LIMIT:
        final_pos = _relax(best_pos, max_iter=1000)
        if _valid(final_pos):
            best_pos = final_pos
            
    return best_pos
# EVOLVE-BLOCK-END
