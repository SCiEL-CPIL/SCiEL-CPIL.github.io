import time
import math
import numpy as np
import random

def design(problem):
    start_time = time.time()
    deadline = start_time + 215  # 15s margin for safety
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # --- Fast PWE Solver (5x5 = 25 plane waves) ---
    # This is significantly faster than the 11x11 or 13x13 solvers used in the current program.
    # It is less accurate but sufficient for ranking candidates in a GA.
    N_pw = 5
    half_N = N_pw // 2
    
    # Plane wave vectors (integer indices)
    pw_list = []
    for nx in range(-half_N, half_N + 1):
        for ny in range(-half_N, half_N + 1):
            pw_list.append((nx, ny))
    pw_list = np.array(pw_list)
    N_pw_actual = len(pw_list)
    
    # K-points in the irreducible Brillouin Zone
    # Gamma (0,0), M (0.5, 0), X (0.5, 0.5)
    # We sample a few points to get a decent estimate of the gap.
    k_points = [
        (0.0, 0.0),
        (0.25, 0.0),
        (0.5, 0.0),
        (0.5, 0.25),
        (0.5, 0.5),
        (0.25, 0.25)
    ]
    
    def compute_eps_inv_fft(cell):
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)
        fft_inv_eps = np.fft.fftshift(fft_inv_eps)
        return fft_inv_eps

    def solve_bands_fast(cell, kx, ky, num_bands=5):
        eps_inv_fft = compute_eps_inv_fft(cell)
        k_vectors = pw_list + np.array([kx, ky])
        
        H = np.zeros((N_pw_actual, N_pw_actual), dtype=complex)
        
        # Precompute indices for FFT lookup
        # q = k_i - k_j
        # Since k_i and k_j are from the same set, q is integer.
        # We need to map q to the shifted FFT array.
        
        for i in range(N_pw_actual):
            for j in range(N_pw_actual):
                qx = int(pw_list[i][0] - pw_list[j][0])
                qy = int(pw_list[i][1] - pw_list[j][1])
                
                # Check if within range
                if -half_N <= qx <= half_N and -half_N <= qy <= half_N:
                    idx_x = qx + half_N
                    idx_y = qy + half_N
                    coeff = eps_inv_fft[idx_y, idx_x]
                else:
                    coeff = 0.0
                
                k_diff_sq = (k_vectors[i][0] - k_vectors[j][0])**2 + (k_vectors[i][1] - k_vectors[j][1])**2
                H[i, j] = coeff * k_diff_sq
        
        vals, vecs = np.linalg.eigh(H)
        vals = vals[vals > 1e-6]
        
        if len(vals) < num_bands:
            return None, None
            
        return vals[:num_bands], vecs[:, :num_bands]

    def evaluate_gap_fast(cell):
        max_omega_4 = 0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            vals, _ = solve_bands_fast(cell, kx, ky, num_bands=5)
            if vals is None:
                continue
            omegas = np.sqrt(vals)
            if len(omegas) < 5:
                continue
            max_omega_4 = max(max_omega_4, omegas[3])
            min_omega_5 = min(min_omega_5, omegas[4])
            
        if min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # --- Symmetry Enforcement ---
    def enforce_symmetry(cell):
        # C4v symmetry:
        # 1. Central inversion: cell[i][j] == cell[N-1-i][N-1-j]
        # 2. Transpose: cell[i][j] == cell[j][i]
        # This implies 8-fold symmetry (dihedral group D4).
        
        # Average all 8 symmetric copies to ensure strict symmetry
        # Then threshold.
        c0 = cell
        c1 = cell[::-1, ::-1]
        c2 = cell.T
        c3 = cell[::-1, ::-1].T
        c4 = cell[:, ::-1]
        c5 = cell[::-1, :]
        c6 = cell[:, ::-1].T
        c7 = cell[::-1, :].T
        
        avg = (c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7) / 8.0
        new_cell = (avg >= 0.5).astype(int)
        
        # Final check/fix
        new_cell = (new_cell + new_cell[::-1, ::-1]) // 2
        new_cell = (new_cell + new_cell.T) // 2
        return new_cell

    # --- Initialization ---
    def make_initial_cell():
        cell = np.zeros((size, size), dtype=int)
        y, x = np.mgrid[0:size, 0:size]
        y = y + 0.5
        x = x + 0.5
        
        # Strategy: Random symmetric patterns
        # 1. Four rods in corners
        if random.random() < 0.5:
            centers = [
                (size/4, size/4),
                (size/4, 3*size/4),
                (3*size/4, size/4),
                (3*size/4, 3*size/4)
            ]
            r = random.uniform(3.0, 7.0)
            for cx, cy in centers:
                dist = (x - cx)**2 + (y - cy)**2
                cell[dist < r**2] = 1
        else:
            # 2. Checkerboard-like or random symmetric blocks
            # Generate a 4x4 fundamental domain and tile it with symmetry
            fd_size = size // 8
            fd = np.zeros((fd_size, fd_size), dtype=int)
            for i in range(fd_size):
                for j in range(fd_size):
                    if random.random() < 0.5:
                        fd[i, j] = 1
                        # Mirror within FD for symmetry? No, just tile.
            
            # Tile to 32x32 with symmetry
            # The fundamental domain for C4v is 1/8 of the cell.
            # Let's just generate random and enforce symmetry.
            cell = np.random.randint(0, 2, (size, size))
            cell = enforce_symmetry(cell)
            
            # Or try a specific topology: Air holes in dielectric
            if random.random() < 0.3:
                 cell = np.ones((size, size), dtype=int)
                 centers = [
                    (size/4, size/4),
                    (size/4, 3*size/4),
                    (3*size/4, size/4),
                    (3*size/4, 3*size/4)
                 ]
                 r = random.uniform(2.0, 5.0)
                 for cx, cy in centers:
                     dist = (x - cx)**2 + (y - cy)**2
                     cell[dist < r**2] = 0
                 cell = enforce_symmetry(cell)

        return cell

    # --- Genetic Algorithm ---
    pop_size = 20
    population = []
    
    # Seed with diverse initial cells
    for _ in range(pop_size):
        cell = make_initial_cell()
        population.append(cell)
    
    # Evaluate initial population
    scores = []
    for cell in population:
        score = evaluate_gap_fast(cell)
        scores.append(score)
        
    best_cell = population[np.argmax(scores)]
    best_score = max(scores)
    
    # Optimization Loop
    max_generations = 100
    mutation_rate = 0.1
    
    for gen in range(max_generations):
        if time.time() > deadline:
            break
            
        # Rank population
        order = np.argsort(scores)[::-1]
        ranked_pop = [population[i] for i in order]
        ranked_scores = [scores[i] for i in order]
        
        # Elitism: Keep the best 2
        new_population = [ranked_pop[0].copy(), ranked_pop[1].copy()]
        
        # Fill the rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def tournament(k=3):
                candidates = random.sample(range(pop_size), k)
                best_idx = max(candidates, key=lambda i: scores[i])
                return population[best_idx]
            
            parent1 = tournament()
            parent2 = tournament()
            
            # Crossover
            child = parent1.copy()
            # Random mask crossover
            mask = np.random.rand(size, size) > 0.5
            child[mask] = parent2[mask]
            
            # Mutation
            # Flip some pixels
            if random.random() < mutation_rate:
                # Select a random subset of pixels to flip
                n_mutations = int(size * size * 0.05)
                idx = np.random.choice(size * size, n_mutations, replace=False)
                rows, cols = np.unravel_index(idx, (size, size))
                child[rows, cols] = 1 - child[rows, cols]
            
            # Enforce symmetry
            child = enforce_symmetry(child)
            
            new_population.append(child)
        
        population = new_population
        
        # Evaluate new population
        new_scores = []
        for cell in population:
            score = evaluate_gap_fast(cell)
            new_scores.append(score)
        scores = new_scores
        
        # Update best
        current_best_idx = np.argmax(scores)
        if scores[current_best_idx] > best_score:
            best_score = scores[current_best_idx]
            best_cell = population[current_best_idx].copy()
            
    # Final refinement: 
    # The fast solver might have a local optimum that is not the global optimum for the exact solver.
    # We have time left, so let's try a few local moves on the best cell using a slightly larger solver?
    # Or just return the best cell from the GA.
    # Given the 5x5 solver is quite coarse, the exact 289-plane-wave solver might disagree.
    # However, the topology (location of dielectric) is usually robust.
    
    # Let's do a quick local search on the best cell with the fast solver to polish it.
    for step in range(20):
        if time.time() > deadline - 5:
            break
        neighbor = best_cell.copy()
        # Flip a random symmetric patch
        i = random.randint(0, size//2 - 1)
        j = random.randint(0, size//2 - 1)
        # Flip (i,j) and its symmetric counterparts
        neighbor[i, j] = 1 - neighbor[i, j]
        neighbor = enforce_symmetry(neighbor)
        
        score = evaluate_gap_fast(neighbor)
        if score > best_score:
            best_score = score
            best_cell = neighbor
    
    return best_cell.tolist()
