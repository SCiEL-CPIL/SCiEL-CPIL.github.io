<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 symmetric photonic crystal unit cell to maximize the TM band gap
    between bands 4 and 5. Uses a genetic algorithm with a fast plane-wave solver.
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Time management
    start_time = time.time()
    total_time_budget = 110.0  # Leave margin before 120s limit
    
    # Symmetry: the cell must be invariant under:
    # 1. Row reversal: cell[i] == cell[size-1-i] reversed
    # 2. Column reversal: cell[i][j] == cell[i][size-1-j]
    # 3. Transpose: cell[i][j] == cell[j][i]
    #
    # This means the cell is determined by its upper-left quadrant (with proper handling of center lines)
    # For a 32x32 cell, the independent region is 16x16 (top-left quarter)
    # Actually, let's think more carefully:
    # - cell[i][j] == cell[j][i] (transpose symmetry)
    # - cell[i][j] == cell[size-1-i][size-1-j] (180-degree rotation, which follows from row+col reversal)
    # - cell[i][j] == cell[size-1-i][j] (row reversal)
    # - cell[i][j] == cell[i][size-1-j] (column reversal)
    #
    # The independent degrees of freedom are in the top-left 16x16 quadrant (indices 0..15 in both dims)
    # because:
    # - For i < 16 and j < 16: independent
    # - cell[i][j] for i>=16 or j>=16 can be derived from symmetry
    
    # Actually, let's be more precise. The symmetry group is the square group D4 applied about the center.
    # For a 32x32 grid, the center is at (15.5, 15.5).
    # The fundamental domain is the triangle in the top-left where i <= j and i < 16 (roughly).
    # But for simplicity, let's use the 16x16 top-left quadrant as the independent region.
    
    # Let's define the independent region as indices [0:16, 0:16]
    # And reconstruct the full cell from this.
    
    def reconstruct_from_quadrant(quadrant):
        """Reconstruct full 32x32 cell from 16x16 top-left quadrant."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(16):
            for j in range(16):
                val = quadrant[i][j]
                # Apply symmetries
                cell[i][j] = val
                cell[j][i] = val  # transpose
                cell[size-1-i][size-1-j] = val  # 180 rot
                cell[size-1-j][size-1-i] = val  # transpose of 180 rot
                cell[size-1-i][j] = val  # row reversal
                cell[i][size-1-j] = val  # col reversal
                cell[size-1-j][i] = val  # col reversal + transpose
                cell[j][size-1-i] = val  # row reversal + transpose
        return cell
    
    # Verify reconstruction works
    test_quad = np.random.randint(0, 2, (16, 16))
    test_cell = reconstruct_from_quadrant(test_quad)
    # Check symmetries
    assert np.array_equal(test_cell, test_cell[::-1, :]), "Row reversal failed"
    assert np.array_equal(test_cell, test_cell[:, ::-1]), "Col reversal failed"
    assert np.array_equal(test_cell, test_cell.T), "Transpose failed"
    
    # Fast plane-wave solver for TM bands
    def solve_tm_bands(cell, n_pw=13, n_bands=6, n_kpts=10):
        """
        Solve TM band structure using plane-wave method.
        Returns the gap ratio between band 4 and 5.
        """
        N = size
        eps = np.where(cell == 1, eps_high, eps_low)
        
        # Compute Fourier coefficients of 1/eps for TM polarization
        # TM: H_z satisfies -∇·(1/eps ∇H_z) = (omega/c)^2 H_z
        # We need the inverse permittivity: inv_eps = 1/eps
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients of inv_eps
        # Use a grid of G-vectors
        # For a 32x32 cell, we'll use G-vectors from -n_pw to n_pw in each direction
        G_range = list(range(-n_pw, n_pw + 1))
        G_vectors = np.array([(gx, gy) for gx in G_range for gy in G_range], dtype=float)
        n_G = len(G_vectors)
        
        # Compute Fourier coefficients: eps_inv_G = (1/N^2) * sum_{x,y} inv_eps(x,y) * exp(-i G·r)
        # r = (i/N, j/N) for pixel (i,j)
        # We'll compute this efficiently
        inv_eps_G = np.zeros(n_G, dtype=complex)
        for k, (gx, gy) in enumerate(G_vectors):
            # exp(-i * 2*pi * (gx*i/N + gy*j/N))
            phase_i = np.exp(-1j * 2 * np.pi * gx * np.arange(N) / N)
            phase_j = np.exp(-1j * 2 * np.pi * gy * np.arange(N) / N)
            # Sum over all pixels
            s = np.sum(inv_eps * np.outer(phase_i, phase_j))
            inv_eps_G[k] = s / (N * N)
        
        # K-points in irreducible Brillouin zone
        # The BZ is [-pi, pi] x [-pi, pi] (in units of 2*pi/a where a=1)
        # Irreducible zone: kx >= ky >= 0, kx + ky <= pi (triangle)
        # Let's sample several k-points
        k_points = []
        # High symmetry points and some interior points
        k_points.append((0.0, 0.0))  # Gamma
        k_points.append((np.pi, 0.0))  # X
        k_points.append((np.pi/2, np.pi/2))  # M
        k_points.append((np.pi/2, 0.0))
        k_points.append((np.pi/3, np.pi/3))
        k_points.append((2*np.pi/3, np.pi/3))
        k_points.append((np.pi/2, np.pi/6))
        k_points.append((2*np.pi/3, 0.0))
        k_points.append((np.pi/3, 0.0))
        k_points.append((np.pi/4, np.pi/4))
        
        # For each k-point, solve the eigenvalue problem
        # The matrix is: sum_G' inv_eps_{G-G'} * |K+G|^2 * delta_{G,G'}
        # Actually for TM: the equation is sum_{G'} inv_eps_{G-G'} (K+G)·(K+G') c_{G'} = omega^2 c_G
        # Wait, let me recall the correct form.
        
        # For TM (H_z), the equation is:
        # -∇·(1/eps ∇H_z) = (omega/c)^2 H_z
        # In plane wave: H_z = sum_G c_G exp(i(K+G)·r)
        # Then: sum_{G'} (1/eps)_{G-G'} (K+G)·(K+G') c_{G'} = (omega/c)^2 c_G
        
        # So the matrix M_{G,G'} = inv_eps_{G-G'} * (K+G)·(K+G')
        # And we solve M c = (omega/c)^2 c
        
        # Precompute (K+G)·(K+G') for all pairs
        # But this is O(n_G^2) per k-point, which might be slow.
        
        # Let's use a smaller n_pw for speed
        # With n_pw=8, we have 17x17=289 G-vectors, which is manageable
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            K = np.array([kx, ky])
            # Build the matrix
            KG = K + G_vectors  # shape (n_G, 2)
            KG_sq = np.sum(KG**2, axis=1)  # |K+G|^2
            
            # inv_eps_{G-G'}: we need the difference G-G'
            # Create a mapping from (gx, gy) to index
            g_to_idx = {}
            for idx, (gx, gy) in enumerate(G_vectors):
                g_to_idx[(gx, gy)] = idx
            
            M = np.zeros((n_G, n_G), dtype=complex)
            for i in range(n_G):
                for j in range(n_G):
                    dx = G_vectors[i][0] - G_vectors[j][0]
                    dy = G_vectors[i][1] - G_vectors[j][1]
                    diff_idx = g_to_idx.get((dx, dy), None)
                    if diff_idx is not None:
                        M[i, j] = inv_eps_G[diff_idx] * np.dot(KG[i], KG[j])
            
            # Solve eigenvalue problem
            # M is Hermitian if inv_eps is real (which it is)
            # But due to numerical errors, let's symmetrize
            M = (M + M.conj().T) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(M)
            eigenvalues = np.real(eigenvalues)
            # omega^2/c^2 = eigenvalue, so omega = sqrt(eigenvalue) * c
            # We just need relative values, so omega = sqrt(eigenvalue)
            eigenvalues = np.sort(eigenvalues)
            # Take the first n_bands
            if len(eigenvalues) >= n_bands:
                omegas = np.sqrt(np.maximum(eigenvalues[:n_bands], 0))
                if len(omegas) >= 5:
                    max_omega_4 = max(max_omega_4, omegas[3])  # 4th band (0-indexed: 3)
                    min_omega_5 = min(min_omega_5, omegas[4])  # 5th band (0-indexed: 4)
        
        if min_omega_5 > 0 and max_omega_4 > 0:
            gap_ratio = 2 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        else:
            gap_ratio = -1.0
        
        return gap_ratio
    
    # Genetic algorithm
    pop_size = 20
    n_generations = 30
    mutation_rate = 0.15
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        quad = np.random.randint(0, 2, (16, 16))
        population.append(quad)
    
    # Evaluate initial population
    scores = []
    for quad in population:
        cell = reconstruct_from_quadrant(quad)
        score = solve_tm_bands(cell, n_pw=8, n_bands=6, n_kpts=10)
        scores.append(score)
    
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_quad = population[best_idx].copy()
    
    for gen in range(n_generations):
        # Check time
        elapsed = time.time() - start_time
        if elapsed > total_time_budget - 5:
            break
        
        # Sort by score
        order = np.argsort(scores)[::-1]
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Elitism: keep top 5
        elite = [population[i].copy() for i in range(min(5, pop_size))]
        elite_scores = scores[:min(5, pop_size)]
        
        # Create new population
        new_population = elite[:]
        new_scores = elite_scores[:]
        
        while len(new_population) < pop_size:
            # Tournament selection
            t1 = np.random.randint(0, pop_size)
            t2 = np.random.randint(0, pop_size)
            if scores[t1] >= scores[t2]:
                parent1 = population[t1]
            else:
                parent1 = population[t2]
            
            t1 = np.random.randint(0, pop_size)
            t2 = np.random.randint(0, pop_size)
            if scores[t1] >= scores[t2]:
                parent2 = population[t1]
            else:
                parent2 = population[t2]
            
            # Crossover
            child = parent1.copy()
            mask = np.random.rand(16, 16) < 0.5
            child[mask] = parent2[mask]
            
            # Mutation
            mutate_mask = np.random.rand(16, 16) < mutation_rate
            child[mutate_mask] = 1 - child[mutate_mask]
            
            new_population.append(child)
            # Placeholder score, will be evaluated
            new_scores.append(-np.inf)
        
        # Evaluate new individuals
        for i in range(len(elite), len(new_population)):
            cell = reconstruct_from_quadrant(new_population[i])
            new_scores[i] = solve_tm_bands(cell, n_pw=8, n_bands=6, n_kpts=10)
            
            # Update best
            if new_scores[i] > best_score:
                best_score = new_scores[i]
                best_quad = new_population[i].copy()
        
        population = new_population
        scores = new_scores
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > total_time_budget - 10:
            break
    
    # Convert best quadrant to nested list
    best_cell = reconstruct_from_quadrant(best_quad)
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    
    return result

# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 symmetric photonic crystal unit cell to maximize the TM band gap
    between bands 4 and 5. Uses a structural search over symmetric patterns with
    a fast plane-wave solver.
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Time management
    start_time = time.time()
    total_time_budget = 210.0  # Leave margin before 230s limit
    
    # Symmetry constraints:
    # cell[i][j] == cell[j][i] (transpose)
    # cell[i][j] == cell[size-1-i][j] (row reversal)
    # cell[i][j] == cell[i][size-1-j] (col reversal)
    # This implies cell[i][j] == cell[size-1-i][size-1-j] (180 rot)
    # The independent region is the top-left 16x16 quadrant.
    
    def reconstruct_from_quadrant(quadrant):
        """Reconstruct full 32x32 cell from 16x16 top-left quadrant."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(16):
            for j in range(16):
                val = quadrant[i][j]
                # Apply all 8 symmetries of the square group
                cell[i][j] = val
                cell[j][i] = val
                cell[size-1-i][size-1-j] = val
                cell[size-1-j][size-1-i] = val
                cell[size-1-i][j] = val
                cell[i][size-1-j] = val
                cell[size-1-j][i] = val
                cell[j][size-1-i] = val
        return cell
    
    # Fast plane-wave solver for TM bands
    def solve_tm_bands(cell, n_pw=12, n_bands=6):
        """
        Solve TM band structure using plane-wave method.
        Returns the gap ratio between band 4 and 5.
        """
        N = size
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients of inv_eps using FFT
        # G-vectors from -n_pw to n_pw
        G_range = list(range(-n_pw, n_pw + 1))
        G_vectors = np.array([(gx, gy) for gx in G_range for gy in G_range], dtype=float)
        n_G = len(G_vectors)
        
        # Use FFT to compute Fourier coefficients efficiently
        # inv_eps is N x N, we need coefficients for G in [-n_pw, n_pw] x [-n_pw, n_pw]
        # FFT gives coefficients for G in [0, N-1], we need to shift
        inv_eps_fft = np.fft.fft2(inv_eps) / (N * N)
        
        # Extract the needed G components
        # FFT index for G=(gx, gy) is (gx mod N, gy mod N)
        inv_eps_G = np.zeros(n_G, dtype=complex)
        for k, (gx, gy) in enumerate(G_vectors):
            inv_eps_G[k] = inv_eps_fft[gx % N, gy % N]
        
        # K-points in irreducible Brillouin zone
        # BZ: [-pi, pi] x [-pi, pi]
        # Irreducible: kx >= ky >= 0, kx + ky <= pi
        k_points = [
            (0.0, 0.0),          # Gamma
            (np.pi, 0.0),        # X
            (np.pi/2, np.pi/2),  # M
            (np.pi/2, 0.0),
            (np.pi/3, np.pi/3),
            (2*np.pi/3, np.pi/3),
            (np.pi/2, np.pi/6),
            (2*np.pi/3, 0.0),
            (np.pi/3, 0.0),
            (np.pi/4, np.pi/4),
            (3*np.pi/8, np.pi/8),
            (5*np.pi/8, np.pi/8),
        ]
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        # Precompute G vector differences mapping
        g_to_idx = {}
        for idx, (gx, gy) in enumerate(G_vectors):
            g_to_idx[(gx, gy)] = idx
        
        for kx, ky in k_points:
            K = np.array([kx, ky])
            KG = K + G_vectors  # shape (n_G, 2)
            
            # Build matrix M_{G,G'} = inv_eps_{G-G'} * (K+G)·(K+G')
            M = np.zeros((n_G, n_G), dtype=complex)
            for i in range(n_G):
                KG_i = KG[i]
                for j in range(n_G):
                    dx = G_vectors[i][0] - G_vectors[j][0]
                    dy = G_vectors[i][1] - G_vectors[j][1]
                    diff_idx = g_to_idx.get((dx, dy), None)
                    if diff_idx is not None:
                        M[i, j] = inv_eps_G[diff_idx] * np.dot(KG_i, KG[j])
            
            # Solve eigenvalue problem
            M = (M + M.conj().T) / 2
            eigenvalues, _ = np.linalg.eigh(M)
            eigenvalues = np.real(eigenvalues)
            eigenvalues = np.sort(eigenvalues)
            
            if len(eigenvalues) >= n_bands:
                omegas = np.sqrt(np.maximum(eigenvalues[:n_bands], 0))
                if len(omegas) >= 5:
                    max_omega_4 = max(max_omega_4, omegas[3])
                    min_omega_5 = min(min_omega_5, omegas[4])
        
        if min_omega_5 > 0 and max_omega_4 > 0:
            gap_ratio = 2 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        else:
            gap_ratio = -1.0
        
        return gap_ratio
    
    # Generate candidate patterns based on structural knowledge
    # Key insight: multi-circle clusters can open wider gaps for higher-order bands
    candidates = []
    
    # Pattern 1: 4-circle cluster (from knowledge: measured 5.786)
    quad1 = np.zeros((16, 16), dtype=int)
    # Place 4 circles in the quadrant
    centers = [(4, 4), (10, 4), (4, 10), (10, 10)]
    for cx, cy in centers:
        for i in range(16):
            for j in range(16):
                if (i - cx)**2 + (j - cy)**2 < 4**2:  # radius 4
                    quad1[i][j] = 1
    candidates.append(quad1)
    
    # Pattern 2: Larger 4-circle cluster
    quad2 = np.zeros((16, 16), dtype=int)
    for cx, cy in centers:
        for i in range(16):
            for j in range(16):
                if (i - cx)**2 + (j - cy)**2 < 5**2:  # radius 5
                    quad2[i][j] = 1
    candidates.append(quad2)
    
    # Pattern 3: Smaller 4-circle cluster
    quad3 = np.zeros((16, 16), dtype=int)
    for cx, cy in centers:
        for i in range(16):
            for j in range(16):
                if (i - cx)**2 + (j - cy)**2 < 3**2:  # radius 3
                    quad3[i][j] = 1
    candidates.append(quad3)
    
    # Pattern 4: 8-circle cluster (denser)
    quad4 = np.zeros((16, 16), dtype=int)
    centers4 = [(3, 3), (11, 3), (3, 11), (11, 11), (7, 7), (9, 7), (7, 9), (9, 9)]
    for cx, cy in centers4:
        for i in range(16):
            for j in range(16):
                if (i - cx)**2 + (j - cy)**2 < 2**2:  # radius 2
                    quad4[i][j] = 1
    candidates.append(quad4)
    
    # Pattern 5: Ring-like structure
    quad5 = np.zeros((16, 16), dtype=int)
    for i in range(16):
        for j in range(16):
            dist = (i - 7.5)**2 + (j - 7.5)**2
            if 3**2 < dist < 6**2:
                quad5[i][j] = 1
    candidates.append(quad5)
    
    # Pattern 6: Checkerboard variant
    quad6 = np.zeros((16, 16), dtype=int)
    for i in range(16):
        for j in range(16):
            if (i % 4 < 2) == (j % 4 < 2):
                quad6[i][j] = 1
    candidates.append(quad6)
    
    # Evaluate all candidates
    best_score = -np.inf
    best_quad = candidates[0].copy()
    
    for quad in candidates:
        cell = reconstruct_from_quadrant(quad)
        score = solve_tm_bands(cell, n_pw=12, n_bands=6)
        if score > best_score:
            best_score = score
            best_quad = quad.copy()
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > total_time_budget - 30:
            break
    
    # If time remains, do local optimization around best candidate
    elapsed = time.time() - start_time
    if elapsed < total_time_budget - 60:
        # Local mutation search
        current_best = best_quad.copy()
        current_score = best_score
        
        for iteration in range(50):
            elapsed = time.time() - start_time
            if elapsed > total_time_budget - 30:
                break
            
            # Mutate a random subset of pixels
            mutated = current_best.copy()
            n_mutations = np.random.randint(1, 5)
            for _ in range(n_mutations):
                i = np.random.randint(0, 16)
                j = np.random.randint(0, 16)
                mutated[i][j] = 1 - mutated[i][j]
            
            cell = reconstruct_from_quadrant(mutated)
            score = solve_tm_bands(cell, n_pw=12, n_bands=6)
            
            if score > current_score:
                current_score = score
                current_best = mutated.copy()
                if score > best_score:
                    best_score = score
                    best_quad = mutated.copy()
    
    # Convert best quadrant to nested list
    best_cell = reconstruct_from_quadrant(best_quad)
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    
    return result

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
