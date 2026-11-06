import numpy as np
import time
import math
import random

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    # Evaluator kills at 230s. We use 190s for safety.
    deadline = start_time + 190.0

    # 1. Fast Surrogate Solver Setup
    # Use 15x15 = 225 plane waves for speed (vs 289 in evaluator)
    half_pw = 7 
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # K-points covering the irreducible Brillouin zone
    # Triangle: (0,0) to (pi,0) to (pi/2, pi/2) to (0,0) ... wait, square lattice IBZ is triangle (0,0)-(pi,0)-(pi/2,pi/2)
    # Actually, for square lattice, IBZ is the triangle with vertices (0,0), (pi, 0), (pi/2, pi/2).
    # Let's sample a grid in the triangle.
    k_points = []
    n_k = 6
    for i in range(n_k + 1):
        for j in range(n_k + 1):
            kx = i / n_k * np.pi
            ky = j / n_k * np.pi
            # Check if (kx, ky) is in the IBZ triangle: kx >= 0, ky >= 0, kx + ky <= pi
            if kx >= 0 and ky >= 0 and kx + ky <= np.pi + 1e-9:
                k_points.append((kx, ky))
    # Add high symmetry points explicitly to ensure coverage
    k_points.append((np.pi, 0))
    k_points.append((np.pi/2, np.pi/2))
    k_points.append((0, 0))
    # Remove duplicates
    k_points = list(set(k_points))
    k_points = np.array(k_points)
    
    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score for a given cell (32x32 array of 0/1).
        Returns -1.0 if invalid or overlapping.
        """
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # FFT of epsilon (normalized)
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        
        # 1/eps_F for TM mode
        # Avoid division by zero (eps_F[0,0] is average eps, non-zero)
        inv_eps_F = 1.0 / eps_F
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
            kG2 = kG[:, 0]**2 + kG[:, 1]**2
            
            def matvec(v):
                w = kG2 * v
                # Reshape to 15x15 for 2D FFT
                u = np.fft.fft2(w.reshape(15, 15))
                z = inv_eps_F * u
                y = np.fft.ifft2(z)
                return y.flatten().real
            
            try:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import LinearOperator
                A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
                # We need bands 4 and 5 (indices 3 and 4)
                # eigsh finds smallest magnitude 'SA' or smallest 'SM'
                # TM eigenvalues are positive.
                vals, vecs = eigsh(A_op, k=6, which='SA', tol=1e-4)
                vals = np.sort(vals.real)
                
                if len(vals) < 5:
                    return -1.0
                
                omega4 = np.sqrt(max(0, vals[3]))
                omega5 = np.sqrt(max(0, vals[4]))
                
                if omega5 < min_omega5:
                    min_omega5 = omega5
                if omega4 > max_omega4:
                    max_omega4 = omega4
                    
            except Exception:
                return -1.0
        
        if min_omega5 == np.inf or max_omega4 == 0.0:
            return -1.0
            
        if min_omega5 <= max_omega4:
            return -1.0
            
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

    def enforce_full_symmetry(c):
        """
        Enforce C4v symmetry on 32x32 array.
        Fundamental domain: 0 <= i <= 15, 0 <= j <= 15, i <= j.
        """
        c_sym = np.zeros_like(c)
        for i in range(16):
            for j in range(i, 16):
                val = c[i, j]
                # 1. (i, j)
                c_sym[i, j] = val
                # 2. (j, i)
                c_sym[j, i] = val
                # 3. (31-i, 31-j)
                c_sym[31-i, 31-j] = val
                # 4. (31-j, 31-i)
                c_sym[31-j, 31-i] = val
                # 5. (i, 31-j)
                c_sym[i, 31-j] = val
                # 6. (31-j, i)
                c_sym[31-j, i] = val
                # 7. (31-i, j)
                c_sym[31-i, j] = val
                # 8. (j, 31-i)
                c_sym[j, 31-i] = val
        return c_sym

    def generate_random_cell():
        """Generate a random symmetric cell."""
        c = np.zeros((size, size), dtype=int)
        # Only randomize the fundamental domain
        for i in range(16):
            for j in range(i, 16):
                c[i, j] = random.randint(0, 1)
        return enforce_full_symmetry(c)

    def generate_motif_cell(motif_size, motif):
        """
        motif is a motif_size x motif_size array of 0/1.
        It is repeated to fill the 32x32 cell.
        Then symmetry is enforced.
        """
        cell = np.zeros((size, size), dtype=int)
        n_repeats = size // motif_size
        for i in range(n_repeats):
            for j in range(n_repeats):
                cell[i*motif_size:(i+1)*motif_size, j*motif_size:(j+1)*motif_size] = motif
        return enforce_full_symmetry(cell)

    # 2. Initialize Population
    # Start with a mix of structured motifs and random cells to ensure diversity
    population = []
    
    # Structured candidates
    # 4x4 checkerboard
    motif_4 = np.array([[1,0,1,0],
                        [0,1,0,1],
                        [1,0,1,0],
                        [0,1,0,1]])
    population.append(generate_motif_cell(4, motif_4))
    
    # 8x8 central square
    motif_8 = np.zeros((8,8), dtype=int)
    motif_8[2:6, 2:6] = 1
    population.append(generate_motif_cell(8, motif_8))
    
    # 16x16 central square
    motif_16 = np.zeros((16,16), dtype=int)
    motif_16[4:12, 4:12] = 1
    population.append(generate_motif_cell(16, motif_16))
    
    # 4 rods (classic)
    cell_rods = np.zeros((size, size), dtype=int)
    centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
    for cx, cy in centers:
        cell_rods[cx-3:cx+4, cy-3:cy+4] = 1
    population.append(enforce_full_symmetry(cell_rods))
    
    # Random candidates
    pop_size = 20
    for _ in range(pop_size - len(population)):
        population.append(generate_random_cell())
        
    # Evaluate initial population
    scores = []
    for c in population:
        scores.append(compute_score(c))
        
    best_idx = np.argmax(scores)
    best_cell = population[best_idx].copy()
    best_score = scores[best_idx]
    
    # 3. Genetic Algorithm Loop
    # Parameters
    pop_size = 20
    mutation_rate = 0.15
    crossover_rate = 0.7
    elite_count = 2
    
    generation = 0
    max_generations = 500
    
    while time.time() < deadline - 10.0 and generation < max_generations:
        generation += 1
        
        # Sort population by score
        # Note: scores might be -1.0
        indices = np.argsort(scores)
        indices = indices[::-1] # Descending
        
        # Create new population
        new_population = []
        
        # Elitism: keep the best ones
        for i in range(min(elite_count, pop_size)):
            idx = indices[i]
            new_population.append(population[idx].copy())
            # We will re-evaluate or keep score? Keep score to save time, but ensure consistency
            # Actually, we'll re-evaluate the whole new population at the end of gen to be safe,
            # or just track scores. Let's track scores.
            
        # Fill rest of population with offspring
        while len(new_population) < pop_size:
            # Select parents (Tournament selection)
            def select_parent():
                # Tournament size 3
                t_idx = np.random.choice(len(population), 3, replace=False)
                best_t_idx = t_idx[np.argmax([scores[i] for i in t_idx])]
                return population[best_t_idx]
            
            parent1 = select_parent()
            parent2 = select_parent()
            
            # Crossover
            if random.random() < crossover_rate:
                # Uniform crossover
                child = np.zeros_like(parent1)
                for i in range(size):
                    for j in range(size):
                        if random.random() < 0.5:
                            child[i, j] = parent1[i, j]
                        else:
                            child[i, j] = parent2[i, j]
            else:
                child = parent1.copy()
                
            # Enforce symmetry
            child = enforce_full_symmetry(child)
            
            # Mutation
            # Flip pixels in fundamental domain
            for i in range(16):
                for j in range(i, 16):
                    if random.random() < mutation_rate:
                        child[i, j] = 1 - child[i, j]
            
            # Re-enforce symmetry after mutation (though flipping in FD and enforcing is enough)
            child = enforce_full_symmetry(child)
            
            new_population.append(child)
            
        # Evaluate new population
        new_scores = []
        for c in new_population:
            new_scores.append(compute_score(c))
            
        # Update best
        gen_best_idx = np.argmax(new_scores)
        if new_scores[gen_best_idx] > best_score:
            best_score = new_scores[gen_best_idx]
            best_cell = new_population[gen_best_idx].copy()
            
        # Update population
        population = new_population
        scores = new_scores
        
        # Check time
        if time.time() > deadline - 20.0:
            break

    # 4. Final Polish: Local Search around best cell
    # Use remaining time for a quick local search to refine the best GA result
    if time.time() < deadline - 5.0:
        cell = best_cell.copy()
        T = 0.1
        T_min = 0.01
        cooling = 0.99
        
        steps = 0
        while time.time() < deadline - 2.0 and steps < 5000:
            steps += 1
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            
            new_cell = cell.copy()
            new_cell[i, j] = 1 - new_cell[i, j]
            new_cell = enforce_full_symmetry(new_cell)
            
            if np.array_equal(new_cell, cell):
                continue
                
            new_score = compute_score(new_cell)
            
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell.copy()
                cell = new_cell
            else:
                delta = new_score - best_score
                if delta < 0:
                    p = math.exp(delta / T)
                    if random.random() < p:
                        cell = new_cell
                        
            T = max(T_min, T * cooling)

    # Ensure the returned cell is a list of lists of ints
    return best_cell.tolist()
