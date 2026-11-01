"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
import time

START_TIME = time.time()
TIME_LIMIT = 115.0

def _lj_energy_and_grad(x):
    """Compute Lennard-Jones energy and gradient efficiently."""
    pos = x.reshape(-1, 3)
    n = pos.shape[0]
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_r2 = 1.0 / dists_sq
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    energy = 2.0 * np.sum(inv_r12 - inv_r6)
    
    inv_r8 = inv_r2 ** 4
    inv_r14 = inv_r2 ** 7
    
    coeff = 24.0 * inv_r8 - 48.0 * inv_r14
    np.fill_diagonal(coeff, 0.0)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad.flatten()

def _relax(pos, max_iter=300, ftol=1e-10, gtol=1e-8):
    """Relax a structure using L-BFGS-B with time check."""
    if time.time() - START_TIME > TIME_LIMIT - 2:
        return pos
    
    pos = pos.copy()
    try:
        result = minimize(
            _lj_energy_and_grad,
            pos.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 15}
        )
        relaxed = result.x.reshape(-1, 3)
        
        dists = pdist(relaxed, 'euclidean')
        if np.min(dists) < 0.5:
            return pos
        return relaxed
    except Exception:
        return pos

def _valid(pos):
    """Check if positions are valid (min distance >= 0.5)."""
    if pos.shape != (N, 3):
        return False
    dists = pdist(pos, 'euclidean')
    return np.min(dists) >= 0.5

def _make_mackay_seed():
    """Construct a Mackay icosahedron for 55 atoms."""
    phi = (1 + np.sqrt(5)) / 2
    
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    
    R1 = 1.122
    shell1 = verts * R1
    
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
            
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        mid = mid / mid_norm
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
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
    fcc_pts *= 1.58
    dists = np.linalg.norm(fcc_pts, axis=1)
    idxs = np.argsort(dists)[:55]
    return fcc_pts[idxs]

def _make_bcc_seed():
    """Generate a BCC fragment seed."""
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                pts.append([i, j, k])
                pts.append([i+0.5, j+0.5, k+0.5])
    bcc_pts = np.array(pts)
    bcc_pts *= 1.44
    dists = np.linalg.norm(bcc_pts, axis=1)
    idxs = np.argsort(dists)[:55]
    return bcc_pts[idxs]

def _make_random_seed(scale=2.0):
    """Generate a valid random seed."""
    for _ in range(100):
        pts = np.random.randn(N, 3) * scale
        if _valid(pts):
            return pts
    return np.random.randn(N, 3) * 3.0

def _cluster_crossover(parent1, parent2, n_groups=5):
    """
    Cluster-based crossover: partition parent1 into spatial clusters,
    and replace each cluster with the corresponding region from parent2.
    This preserves local bonding patterns.
    """
    child = parent1.copy()
    
    # Use KDTree to find spatial clusters in parent1
    tree = cKDTree(parent1)
    
    # Assign atoms to clusters using k-means-like approach
    # Pick n_groups seed points spread out
    seed_indices = np.random.choice(N, n_groups, replace=False)
    seeds = parent1[seed_indices]
    
    # Assign each atom to nearest seed
    distances = np.zeros((N, n_groups))
    for g in range(n_groups):
        distances[:, g] = np.linalg.norm(parent1 - seeds[g], axis=1)
    
    assignments = np.argmin(distances, axis=1)
    
    # For each cluster, find the centroid in parent2 and replace
    for g in range(n_groups):
        mask = assignments == g
        n_atoms_in_cluster = np.sum(mask)
        if n_atoms_in_cluster == 0:
            continue
        
        # Centroid of this cluster in parent1
        centroid1 = np.mean(parent1[mask], axis=0)
        
        # Find atoms in parent2 closest to centroid1
        dists_to_centroid = np.linalg.norm(parent2 - centroid1, axis=1)
        closest_in_p2 = np.argsort(dists_to_centroid)[:n_atoms_in_cluster]
        
        # Replace: child atoms in this cluster get positions from parent2's closest atoms
        # with a small offset to avoid exact overlap
        child[mask] = parent2[closest_in_p2] + np.random.randn(n_atoms_in_cluster, 3) * 0.1
    
    return child

def _tournament_select(population, energies, k=3):
    """Tournament selection."""
    idxs = np.random.choice(len(population), k, replace=False)
    best_idx = idxs[np.argmin([energies[i] for i in idxs])]
    return population[best_idx].copy()

def construct():
    """Genetic Algorithm for LJ cluster optimization."""
    np.random.seed(42)
    
    pop_size = 30
    
    # Build diverse initial population
    population = []
    energies = []
    
    # Add multiple structural motifs
    initial_motifs = [
        _make_mackay_seed(),
        _make_mackay_seed() + np.random.randn(N, 3) * 0.15,
        _make_mackay_seed() + np.random.randn(N, 3) * 0.3,
        _make_mackay_seed() + np.random.randn(N, 3) * 0.5,
        _make_fcc_seed(),
        _make_bcc_seed(),
        _make_random_seed(2.0),
        _make_random_seed(3.0),
        _make_random_seed(1.5),
    ]
    
    for motif in initial_motifs:
        population.append(motif)
        energies.append(_lj_energy_and_grad(motif.flatten())[0])
    
    # Fill rest with random
    while len(population) < pop_size:
        s = _make_random_seed(2.5)
        population.append(s)
        energies.append(_lj_energy_and_grad(s.flatten())[0])
    
    # Relax initial population
    for i in range(len(population)):
        if time.time() - START_TIME > TIME_LIMIT - 30:
            break
        population[i] = _relax(population[i], max_iter=200)
        if _valid(population[i]):
            energies[i] = _lj_energy_and_grad(population[i].flatten())[0]
    
    best_idx = np.argmin(energies)
    best_energy = energies[best_idx]
    best_pos = population[best_idx].copy()
    
    max_generations = 150
    gen = 0
    stagnation_count = 0
    mutation_sigma = 0.25
    
    while gen < max_generations:
        if time.time() - START_TIME > TIME_LIMIT - 25:
            break
        
        gen += 1
        
        # Sort by energy
        sorted_indices = np.argsort(energies)
        sorted_pop = [population[i] for i in sorted_indices]
        sorted_energies = [energies[i] for i in sorted_indices]
        
        # Elitism: Keep top 4
        elitism_count = 4
        new_population = [sorted_pop[i].copy() for i in range(elitism_count)]
        new_energies = [sorted_energies[i] for i in range(elitism_count)]
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() - START_TIME > TIME_LIMIT - 15:
                break
            
            # Selection
            parent1 = _tournament_select(population, energies, k=3)
            parent2 = _tournament_select(population, energies, k=3)
            
            # Crossover: 60% cluster crossover, 40% keep better parent
            if np.random.rand() < 0.6:
                child = _cluster_crossover(parent1, parent2, n_groups=5)
            else:
                # Keep the better parent
                e1 = _lj_energy_and_grad(parent1.flatten())[0]
                e2 = _lj_energy_and_grad(parent2.flatten())[0]
                child = (parent1 if e1 < e2 else parent2).copy()
            
            # Adaptive mutation
            if stagnation_count > 8:
                mutation_sigma = min(0.8, mutation_sigma * 1.15)
            else:
                mutation_sigma = max(0.05, mutation_sigma * 0.97)
            
            child += np.random.randn(N, 3) * mutation_sigma
            
            # Relax
            relaxed_child = _relax(child, max_iter=250)
            if _valid(relaxed_child):
                new_population.append(relaxed_child)
                new_energies.append(_lj_energy_and_grad(relaxed_child.flatten())[0])
            else:
                new_population.append(child)
                new_energies.append(_lj_energy_and_grad(child.flatten())[0])
        
        population = new_population
        energies = new_energies
        
        # Update best
        gen_best_idx = np.argmin(energies)
        if energies[gen_best_idx] < best_energy - 1e-6:
            best_energy = energies[gen_best_idx]
            best_pos = population[gen_best_idx].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        # Stagnation handling: inject diverse random structures
        if stagnation_count > 12:
            num_replace = max(5, pop_size // 6)
            for _ in range(num_replace):
                idx = np.random.randint(len(population))
                # Mix of random and distorted Mackay
                if np.random.rand() < 0.5:
                    population[idx] = _make_random_seed(2.5)
                else:
                    population[idx] = _make_mackay_seed() + np.random.randn(N, 3) * 0.4
                population[idx] = _relax(population[idx], max_iter=200)
                if _valid(population[idx]):
                    energies[idx] = _lj_energy_and_grad(population[idx].flatten())[0]
            stagnation_count = 0
    
    # Final polish: top 5 with tighter tolerances
    if time.time() - START_TIME < TIME_LIMIT - 8:
        sorted_indices = np.argsort(energies)
        top_candidates = [population[i].copy() for i in sorted_indices[:5]]
        
        for cand in top_candidates:
            if time.time() - START_TIME > TIME_LIMIT - 3:
                break
            polished = _relax(cand, max_iter=400, ftol=1e-12, gtol=1e-10)
            if _valid(polished):
                e = _lj_energy_and_grad(polished.flatten())[0]
                if e < best_energy:
                    best_energy = e
                    best_pos = polished
    
    # Final relaxation on absolute best
    if time.time() - START_TIME < TIME_LIMIT - 1:
        final_pos = _relax(best_pos, max_iter=600, ftol=1e-12, gtol=1e-10)
        if _valid(final_pos):
            best_pos = final_pos
    
    return best_pos
# EVOLVE-BLOCK-END
