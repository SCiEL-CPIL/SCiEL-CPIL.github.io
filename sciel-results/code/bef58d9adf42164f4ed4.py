"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell maximizing TM band-4/5 gap
    using a pixel-level genetic algorithm with fast plane-wave surrogate.
    """
    start_time = time.time()
    deadline = start_time + 200  # 200s safety margin
    
    size = problem["pixels"]
    half = size // 2
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Fast 5x5 plane-wave solver for TM mode
    def fast_tm_gap(cell_array, n_pw=5, n_kpts=8):
        """
        Compute approximate TM band gap ratio using reduced plane-wave basis.
        cell_array: 32x32 numpy array of 0/1
        Returns: gap ratio 2*(min_omega5 - max_omega4)/(min_omega5 + max_omega4)
        """
        # Compute Fourier coefficients of epsilon
        # For TM mode, we need 1/epsilon
        eps = np.where(cell_array == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients up to order n_pw
        # f_mn = (1/N^2) * sum_{i,j} f(i,j) * exp(-2*pi*i*(m*i + n*j)/N)
        N = size
        f_coeffs = np.zeros((2*n_pw+1, 2*n_pw+1), dtype=complex)
        for m in range(-n_pw, n_pw+1):
            for n in range(-n_pw, n_pw+1):
                # Use FFT for efficiency
                phase = np.exp(-2j * math.pi * (m * np.arange(N) / N + n * np.arange(N) / N))
                f_coeffs[m+n_pw, n+n_pw] = np.sum(inv_eps * phase[:, None] * phase[None, :]) / (N*N)
        
        # K-points in irreducible BZ for square lattice
        # Irreducible BZ: 0 <= kx <= ky, kx^2 + ky^2 <= 1 (in units of pi/a)
        # Sample key points
        k_points = [
            (0.0, 0.0),      # Gamma
            (0.5, 0.0),      # M/2
            (0.5, 0.5),      # M
            (0.25, 0.0),
            (0.25, 0.25),
            (0.4, 0.1),
            (0.3, 0.3),
            (0.45, 0.2),
        ]
        if n_kpts < len(k_points):
            k_points = k_points[:n_kpts]
        
        # For each k-point, solve the eigenvalue problem
        # H = D * G * D where D = diag(f_mn), G is the kinetic matrix
        # For TM: H_{mn} = sum_p f_{mn-p} * (k+p)^2 * delta-like structure
        # Actually for TM: eigenvalue problem is (k+p)^2 * (1/eps)_{mn} * (k+q)^2
        # Let's use the standard formulation:
        # A_{mn} = sum_p f_{mn-p} * |k+p|^2
        # where f_mn are Fourier coeffs of 1/eps
        
        min_omega5 = float('inf')
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            # Build the matrix A where A_{mn} = sum_p f_{mn-p} * |k+p|^2
            # p and q range over -n_pw to n_pw
            A = np.zeros((2*n_pw+1, 2*n_pw+1), dtype=complex)
            for m in range(-n_pw, n_pw+1):
                for n in range(-n_pw, n_pw+1):
                    # A_{mn} = sum_p f_{m-p, n-q} * |k+p, k+q|^2
                    # Actually the standard form: (A v)_m = sum_n A_{mn} v_n
                    # A_{mn} = sum_p f_{mn-p} * |k+p|^2 where p runs over 2D
                    val = 0.0
                    for px in range(-n_pw, n_pw+1):
                        for py in range(-n_pw, n_pw+1):
                            idx_m = m + n_pw
                            idx_n = n + n_pw
                            idx_p = (px + n_pw, py + n_pw)
                            # f_{mn - p}
                            f_idx_m = m - px + n_pw
                            f_idx_n = n - py + n_pw
                            if 0 <= f_idx_m < 2*n_pw+1 and 0 <= f_idx_n < 2*n_pw+1:
                                kpx = kx + px
                                kpy = ky + py
                                val += f_coeffs[f_idx_m, f_idx_n] * (kpx*kpx + kpy*kpy)
                    A[idx_m, idx_n] = val
            
            # Solve eigenvalue problem
            try:
                eigvals = np.linalg.eigvalsh(A)
                eigvals = np.sort(np.real(eigvals))
                # Take bands 4 and 5 (0-indexed: 3 and 4)
                if len(eigvals) >= 5:
                    omega4 = eigvals[3]
                    omega5 = eigvals[4]
                    if omega4 > 0 and omega5 > 0:
                        max_omega4 = max(max_omega4, omega4)
                        min_omega5 = min(min_omega5, omega5)
            except:
                pass
        
        if min_omega5 == float('inf') or max_omega4 == 0:
            return 0.0
        
        gap_ratio = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return gap_ratio
    
    # Convert cell to numpy array for fast computation
    def cell_to_array(cell):
        return np.array(cell, dtype=np.float64)
    
    # C4v-symmetric operations on the top-left quadrant
    def quadrant_to_cell(quadrant):
        """Convert 16x16 quadrant to full 32x32 cell with C4v symmetry."""
        cell = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(half):
            for j in range(half):
                val = quadrant[i][j]
                cell[i][j] = val
                cell[i][size - 1 - j] = val
                cell[size - 1 - i][j] = val
                cell[size - 1 - i][size - 1 - j] = val
        return cell
    
    def cell_to_quadrant(cell):
        """Extract top-left 16x16 quadrant from full cell."""
        return [row[:half] for row in cell[:half]]
    
    # Initialize population with diverse starting points
    def create_initial_population(pop_size):
        population = []
        
        # 1. Super-ellipse seed (current best)
        q1 = [[0 for _ in range(half)] for _ in range(half)]
        cx, cy = 7.5, 7.5
        a, n_exp = 5.0, 4.0
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                dx, dy = abs(x - cx)/a, abs(y - cy)/a
                if dx**n_exp + dy**n_exp < 1.0:
                    q1[i][j] = 1
        population.append(q1)
        
        # 2. Circle seed
        q2 = [[0 for _ in range(half)] for _ in range(half)]
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                if (x-cx)**2 + (y-cy)**2 < 4.0**2:
                    q2[i][j] = 1
        population.append(q2)
        
        # 3. Square seed
        q3 = [[0 for _ in range(half)] for _ in range(half)]
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                if abs(x-cx) < 4.0 and abs(y-cy) < 4.0:
                    q3[i][j] = 1
        population.append(q3)
        
        # 4. Concentric ring
        q4 = [[0 for _ in range(half)] for _ in range(half)]
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                r = math.sqrt((x-cx)**2 + (y-cy)**2)
                if 2.5 < r < 5.0:
                    q4[i][j] = 1
        population.append(q4)
        
        # 5. Cross pattern
        q5 = [[0 for _ in range(half)] for _ in range(half)]
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                if abs(x-cx) < 2.0 or abs(y-cy) < 2.0:
                    q5[i][j] = 1
        population.append(q5)
        
        # 6-8. Random perturbations of seed
        import random
        random.seed(42)
        for _ in range(3):
            q = [row[:] for row in q1]
            for i in range(half):
                for j in range(half):
                    if random.random() < 0.1:
                        q[i][j] = 1 - q[i][j]
            population.append(q)
        
        return population
    
    # Crossover: swap random regions between two quadrants
    def crossover(q1, q2, rate=0.3):
        child1 = [row[:] for row in q1]
        child2 = [row[:] for row in q2]
        for i in range(half):
            for j in range(half):
                if random.random() < rate:
                    child1[i][j], child2[i][j] = child2[i][j], child1[i][j]
        return child1, child2
    
    # Mutation: flip random pixels
    def mutate(quadrant, rate=0.05):
        q = [row[:] for row in quadrant]
        for i in range(half):
            for j in range(half):
                if random.random() < rate:
                    q[i][j] = 1 - q[i][j]
        return q
    
    # Evaluate a quadrant
    def evaluate_quadrant(quadrant):
        cell = quadrant_to_cell(quadrant)
        cell_arr = cell_to_array(cell)
        return fast_tm_gap(cell_arr)
    
    # Main GA loop
    import random
    random.seed(123)
    
    pop_size = 12
    population = create_initial_population(pop_size)
    
    # Evaluate initial population
    scores = [evaluate_quadrant(q) for q in population]
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_quadrant = population[best_idx][:]
    
    generation = 0
    max_generations = 50
    
    while generation < max_generations:
        if time.time() > deadline - 30:  # Leave 30s for final refinement
            break
        
        # Selection: keep top half
        sorted_indices = np.argsort(scores)[::-1]
        elite = [population[i][:] for i in sorted_indices[:pop_size//2]]
        
        # Generate new population
        new_population = list(elite)
        
        while len(new_population) < pop_size:
            if time.time() > deadline - 30:
                break
            
            # Tournament selection
            p1_idx = sorted_indices[random.randint(0, min(5, len(sorted_indices)-1))]
            p2_idx = sorted_indices[random.randint(0, min(5, len(sorted_indices)-1))]
            
            c1, c2 = crossover(population[p1_idx], population[p2_idx])
            c1 = mutate(c1, rate=0.03)
            c2 = mutate(c2, rate=0.03)
            
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)
        
        population = new_population[:pop_size]
        scores = [evaluate_quadrant(q) for q in population]
        
        # Track best
        gen_best_idx = np.argmax(scores)
        if scores[gen_best_idx] > best_score:
            best_score = scores[gen_best_idx]
            best_quadrant = population[gen_best_idx][:]
        
        generation += 1
    
    # Final refinement: local search around best
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            
            q = [row[:] for row in best_quadrant]
            # Single pixel mutation
            i = random.randint(0, half-1)
            j = random.randint(0, half-1)
            q[i][j] = 1 - q[i][j]
            
            score = evaluate_quadrant(q)
            if score > best_score:
                best_score = score
                best_quadrant = q
    
    # Convert best quadrant to final cell
    best_cell = quadrant_to_cell(best_quadrant)
    
    # Verify symmetry
    for i in range(size):
        for j in range(size):
            assert best_cell[i][j] == best_cell[size-1-i][j], "Row symmetry failed"
            assert best_cell[i][j] == best_cell[i][size-1-j], "Col symmetry failed"
            assert best_cell[i][j] == best_cell[j][i], "Transpose symmetry failed"
    
    return best_cell
# EVOLVE-BLOCK-END
