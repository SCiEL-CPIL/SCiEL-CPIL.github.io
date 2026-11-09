import numpy as np
import time

def design(problem):
    """
    Genetic algorithm over C4v-symmetric binary pixel bitmasks to maximize
    the TM band gap between bands 4 and 5 in a 32x32 photonic crystal unit cell.
    """
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Define the fundamental domain for C4v symmetry
    # A pixel (i,j) in the fundamental domain maps to 8 symmetric positions
    # (unless it lies on a symmetry axis, in which case fewer)
    # Fundamental domain: i in [0, 15], j in [i, 15] (upper triangle of left half)
    # Actually, let's use: i in [0, size//2], j in [i, size//2]
    # For size=32, size//2 = 16, so i in [0,16], j in [i,16]
    
    # Generate all symmetric positions for each fundamental pixel
    def get_sym_positions(i, j, size):
        """Return all unique symmetric positions of (i,j) under C4v about center."""
        # The 8 elements of C4v applied to (i,j):
        # 1. (i, j)
        # 2. (size-1-i, j)      - reflect about horizontal midline
        # 3. (i, size-1-j)      - reflect about vertical midline
        # 4. (size-1-i, size-1-j) - 180 rotation
        # 5. (j, i)             - transpose (reflect about y=x)
        # 6. (size-1-j, i)      - reflect about y=x then horizontal
        # 7. (j, size-1-i)      - reflect about y=x then vertical
        # 8. (size-1-j, size-1-i) - 180 rotation then transpose
        positions = [
            (i, j),
            (size-1-i, j),
            (i, size-1-j),
            (size-1-i, size-1-j),
            (j, i),
            (size-1-j, i),
            (j, size-1-i),
            (size-1-j, size-1-i)
        ]
        # Deduplicate
        unique = list(set(positions))
        return unique
    
    # Build fundamental domain: pixels where i <= j and i <= size//2
    # More precisely, we want a set of representatives such that every pixel
    # is covered exactly once by the union of symmetric positions of fundamental pixels.
    # For C4v about center of a 32x32 grid:
    # Fundamental region: 0 <= i <= 15, i <= j <= 31-i  ... this is getting complex.
    # Simpler: iterate over all pixels, keep only the first one encountered in each orbit.
    
    orbit_reps = []
    assigned = set()
    for i in range(size):
        for j in range(size):
            if (i, j) not in assigned:
                orbit_reps.append((i, j))
                for (pi, pj) in get_sym_positions(i, j, size):
                    assigned.add((pi, pj))
    
    num_fundamental = len(orbit_reps)
    
    # Precompute symmetric positions for each fundamental pixel
    sym_positions = [get_sym_positions(i, j, size) for (i, j) in orbit_reps]
    
    # Plane wave setup for evaluation
    N_pw = 17  # 17x17 = 289 plane waves
    G_vals = np.arange(-N_pw//2, N_pw//2 + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # K points: 30 points in irreducible BZ
    k_points = []
    # Gamma-M: (t, 0) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, 0))
    # M-X: (0.5, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 10):
        k_points.append((0.5, t))
    # Gamma-X: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, t))
    
    # Remove duplicates
    seen = set()
    unique_k = []
    for k in k_points:
        key = (round(k[0], 10), round(k[1], 10))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
    
    # Precompute G-G' differences
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (num_pw, num_pw, 2)
    
    # Precompute indices for FFT lookup
    G_diff_indices = np.zeros((num_pw, num_pw, 2), dtype=int)
    G_diff_indices[:, :, 0] = G_diff[:, :, 0] % size
    G_diff_indices[:, :, 1] = G_diff[:, :, 1] % size
    
    def bitmask_to_grid(bits):
        """Convert a binary array of fundamental pixels to a full 32x32 grid."""
        grid = np.zeros((size, size), dtype=np.float64)
        for idx, bit in enumerate(bits):
            if bit:
                for (pi, pj) in sym_positions[idx]:
                    grid[pi, pj] = 1.0
        return grid
    
    def evaluate_grid(grid):
        """Compute the gap-midgap ratio using plane-wave method."""
        eps_grid = eps_low + (eps_high - eps_low) * grid
        inv_eps_grid = 1.0 / eps_grid
        
        # FFT of 1/epsilon
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)
        
        # Build coefficient matrix
        coeff_matrix = fft_inv_eps[G_diff_indices[:, :, 0], G_diff_indices[:, :, 1]]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)
            
            # M[i,j] = coeff_matrix[i,j] * omega_sq[i]
            M = coeff_matrix * omega_sq[:, np.newaxis]
            
            # Solve eigenvalue problem
            eigvals = np.linalg.eigvalsh(M)
            real_eigvals = np.sort(eigvals.real)
            
            if len(real_eigvals) >= 5:
                omega_4 = np.sqrt(max(0, real_eigvals[3]))
                omega_5 = np.sqrt(max(0, real_eigvals[4]))
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio
    
    def evaluate_bits(bits):
        """Evaluate a binary bitmask."""
        grid = bitmask_to_grid(bits)
        return evaluate_grid(grid)
    
    # Initial population
    pop_size = 30
    population = []
    
    # Seed with some structured initial guesses
    # 1. Central rod
    bits_rod = np.zeros(num_fundamental, dtype=int)
    for idx, (i, j) in enumerate(orbit_reps):
        # Check if this pixel is within radius 0.15 of center
        cx, cy = 0.5, 0.5
        px, py = (i + 0.5) / size, (j + 0.5) / size
        if (px - cx)**2 + (py - cy)**2 < 0.15**2:
            bits_rod[idx] = 1
    population.append(bits_rod)
    
    # 2. Cross shape
    bits_cross = np.zeros(num_fundamental, dtype=int)
    for idx, (i, j) in enumerate(orbit_reps):
        px, py = (i + 0.5) / size, (j + 0.5) / size
        if abs(px - 0.5) < 0.05 or abs(py - 0.5) < 0.05:
            bits_cross[idx] = 1
    population.append(bits_cross)
    
    # 3. Checkerboard-like
    bits_check = np.zeros(num_fundamental, dtype=int)
    for idx, (i, j) in enumerate(orbit_reps):
        if (i // 4 + j // 4) % 2 == 0:
            bits_check[idx] = 1
    population.append(bits_check)
    
    # Random individuals
    while len(population) < pop_size:
        bits = np.random.randint(0, 2, num_fundamental)
        population.append(bits)
    
    # Evaluate initial population
    scores = [evaluate_bits(bits) for bits in population]
    
    # Genetic algorithm
    elite_size = 4
    mutation_rate = 0.15
    crossover_rate = 0.7
    
    generation = 0
    while time.time() < deadline - 5:
        generation += 1
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        population = [population[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        # Elitism
        new_population = population[:elite_size]
        new_scores = scores[:elite_size]
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 5:
                break
            
            # Select parents using tournament selection
            def tournament_select(pop, scores, k=3):
                candidates = np.random.randint(0, len(pop), k)
                best_idx = candidates[np.argmax([scores[c] for c in candidates])]
                return pop[best_idx].copy()
            
            parent1 = tournament_select(population, scores)
            parent2 = tournament_select(population, scores)
            
            # Crossover
            if np.random.random() < crossover_rate:
                # Uniform crossover
                mask = np.random.randint(0, 2, num_fundamental)
                child = np.where(mask == 1, parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            if np.random.random() < mutation_rate:
                # Bit flip mutation
                num_flips = np.random.randint(1, 5)
                flip_indices = np.random.choice(num_fundamental, num_flips, replace=False)
                child[flip_indices] = 1 - child[flip_indices]
            
            # Evaluate
            score = evaluate_bits(child)
            new_population.append(child)
            new_scores.append(score)
        
        population = new_population
        scores = new_scores
    
    # Return best individual
    best_idx = np.argmax(scores)
    best_bits = population[best_idx]
    best_grid = bitmask_to_grid(best_bits)
    
    # Convert to nested list
    final_cell = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_grid[i, j]))
        final_cell.append(row)
    
    return final_cell
