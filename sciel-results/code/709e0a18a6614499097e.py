import time
import numpy as np
from numpy.linalg import eigh

def _compute_gap_fast(cell, n_k=8, pw_radius=8):
    """
    Fast surrogate solver using a reduced plane-wave basis.
    Used for ranking candidates in the GA.
    """
    N = len(cell)
    inv_eps = 1.0 / (1.0 + 11.0 * np.array(cell, dtype=float))
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx*Gx + Gy*Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    size_d = 4 * pw_radius + 1
    offset_d = 2 * pw_radius
    inv_eps_full = np.zeros((size_d, size_d), dtype=complex)
    
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    # Precompute indices for the Toeplitz-like structure
    idx_qx = (Gx_diff + offset_d)
    idx_qy = (Gy_diff + offset_d)
    
    # Fill the matrix. Note: inv_eps_full is constant for a given cell.
    # We can fill it once.
    # Optimization: Only fill unique pairs if necessary, but for 8 radius it's small enough.
    for i in range(n_pw):
        for j in range(n_pw):
            dqx = int(Gx_diff[i, j])
            dqy = int(Gy_diff[i, j])
            inv_eps_full[dqx + offset_d, dqy + offset_d] = fft_inv[dqx % N, dqy % N]
            
    # Vectorized k-point evaluation
    # k_points in irreducible BZ (square lattice)
    # Gamma (0,0), X (0.5, 0), M (0.5, 0.5)
    # We sample a grid in the irreducible wedge: 0 <= ky <= kx <= 0.5
    k_points = []
    # Add high-symmetry points
    k_points.append((0.0, 0.0))
    k_points.append((0.5, 0.0))
    k_points.append((0.5, 0.5))
    # Add some interior points
    k_points.append((0.25, 0.0))
    k_points.append((0.25, 0.25))
    k_points.append((0.4, 0.1))
    k_points.append((0.45, 0.2))
    k_points.append((0.3, 0.15))
    
    max_w4 = 0.0
    min_w5 = float('inf')
    
    # Precompute kG and dots for all k-points? 
    # kG is (n_k, n_pw, 2). dots is (n_k, n_pw, n_pw).
    # M_k = dots_k * inv_eps_full
    # We can loop over k-points.
    
    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = kG @ kG.T
        
        M = dots * inv_eps_full
        M = (M + M.conj().T) / 2
        
        try:
            eigvals = eigh(M, eigvals_only=True)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                # Bands are 1-indexed in problem, 0-indexed in array.
                # Band 4 is index 3. Band 5 is index 4.
                max_w4 = max(max_w4, eigvals[3])
                min_w5 = min(min_w5, eigvals[4])
        except:
            pass
    
    if min_w5 == float('inf') or max_w4 <= 0:
        return -1.0
    return 2 * (min_w5 - max_w4) / (min_w5 + max_w4)

def _compute_gap_accurate(cell, n_k=12, pw_radius=10):
    """
    Higher fidelity solver for final polishing.
    """
    # Same as fast but with more PWs and k-points
    # Reuse the logic but with larger radius
    N = len(cell)
    inv_eps = 1.0 / (1.0 + 11.0 * np.array(cell, dtype=float))
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx*Gx + Gy*Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    size_d = 4 * pw_radius + 1
    offset_d = 2 * pw_radius
    inv_eps_full = np.zeros((size_d, size_d), dtype=complex)
    
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    for i in range(n_pw):
        for j in range(n_pw):
            dqx = int(Gx_diff[i, j])
            dqy = int(Gy_diff[i, j])
            inv_eps_full[dqx + offset_d, dqy + offset_d] = fft_inv[dqx % N, dqy % N]
            
    k_points = [
        (0.0, 0.0), (0.5, 0.0), (0.5, 0.5),
        (0.25, 0.0), (0.25, 0.25), (0.4, 0.1), (0.45, 0.2), (0.3, 0.15),
        (0.1, 0.0), (0.5, 0.1), (0.4, 0.4), (0.2, 0.1)
    ]
    
    max_w4 = 0.0
    min_w5 = float('inf')
    
    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = kG @ kG.T
        
        M = dots * inv_eps_full
        M = (M + M.conj().T) / 2
        
        try:
            eigvals = eigh(M, eigvals_only=True)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                max_w4 = max(max_w4, eigvals[3])
                min_w5 = min(min_w5, eigvals[4])
        except:
            pass
    
    if min_w5 == float('inf') or max_w4 <= 0:
        return -1.0
    return 2 * (min_w5 - max_w4) / (min_w5 + max_w4)

def _make_symmetric(cell):
    """Ensure cell is symmetric under transpose, row-reverse, col-reverse."""
    N = len(cell)
    c = np.array(cell, dtype=int)
    c_sym = (c + c.T + c[::-1, :] + c[:, ::-1]) / 4.0
    c_sym = (c_sym > 0.5).astype(int)
    # Enforce strict symmetry to avoid floating point issues in averaging
    c_sym = np.maximum(c_sym, c_sym.T)
    c_sym = np.maximum(c_sym, c_sym[::-1, :])
    c_sym = np.maximum(c_sym, c_sym[:, ::-1])
    return c_sym

def _gen_rod(r):
    size = 32
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < r*r:
                cell[i, j] = 1
    return cell

def _gen_hole(r):
    size = 32
    cell = np.ones((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < r*r:
                cell[i, j] = 0
    return cell

def _gen_square(r):
    size = 32
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < r and abs(y) < r:
                cell[i, j] = 1
    return cell

def _gen_cross(arm_width, arm_length):
    size = 32
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if (abs(x) < arm_width/2 and abs(y) < arm_length/2) or \
               (abs(y) < arm_width/2 and abs(x) < arm_length/2):
                cell[i, j] = 1
    return cell

def _crossover(parent1, parent2):
    """
    C4v-symmetric crossover.
    Operates on the fundamental domain (top-left quadrant) to ensure symmetry.
    """
    N = len(parent1)
    p1 = np.array(parent1)
    p2 = np.array(parent2)
    child = np.zeros((N, N), dtype=int)
    
    # Fundamental domain: i in [0, N/2), j in [0, N/2)
    # For each pixel in FD, pick from p1 or p2 randomly
    for i in range(N // 2):
        for j in range(N // 2):
            if np.random.random() < 0.5:
                val = p1[i, j]
            else:
                val = p2[i, j]
            # Mirror to all 4 quadrants
            child[i, j] = val
            child[N-1-i, j] = val
            child[i, N-1-j] = val
            child[N-1-i, N-1-j] = val
            
    return child

def _mutate(cell, mutation_rate=0.05):
    """
    C4v-symmetric mutation.
    """
    N = len(cell)
    c = np.array(cell)
    
    for i in range(N // 2):
        for j in range(N // 2):
            if np.random.random() < mutation_rate:
                val = 1 - c[i, j]
                c[i, j] = val
                c[N-1-i, j] = val
                c[i, N-1-j] = val
                c[N-1-i, N-1-j] = val
    return c

def _gen_random_symmetric(density):
    """
    Generate a random symmetric pattern.
    """
    size = 32
    cell = np.zeros((size, size), dtype=int)
    for i in range(size//2):
        for j in range(size//2):
            if np.random.random() < density:
                cell[i, j] = 1
                cell[size-1-i, j] = 1
                cell[i, size-1-j] = 1
                cell[size-1-i, size-1-j] = 1
    return cell

def design(problem):
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin
    
    size = problem["pixels"]
    
    # 1. Initialize Population
    # Diverse initial population of known good topologies
    initial_cells = []
    # Rods
    for r in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        initial_cells.append(_gen_rod(r))
    # Holes
    for r in [0.3, 0.4, 0.5, 0.6]:
        initial_cells.append(_gen_hole(r))
    # Squares
    for r in [0.15, 0.2, 0.25]:
        initial_cells.append(_gen_square(r))
    # Crosses
    for w in [0.1, 0.15]:
        for l in [0.3, 0.4]:
            initial_cells.append(_gen_cross(w, l))
            
    # Add some random ones
    for _ in range(10):
        initial_cells.append(_gen_random_symmetric(np.random.uniform(0.2, 0.5)))
        
    # Ensure symmetry and evaluate initial population
    population = []
    for cell in initial_cells:
        cell_sym = _make_symmetric(cell)
        score = _compute_gap_fast(cell_sym, n_k=6, pw_radius=8)
        population.append((cell_sym, score))
        
    # Sort by score descending
    population.sort(key=lambda x: x[1], reverse=True)
    
    # Keep top K
    pop_size = 20
    if len(population) > pop_size:
        population = population[:pop_size]
        
    best_cell = population[0][0]
    best_score = population[0][1]
    
    # 2. Genetic Algorithm Loop
    # Parameters
    elite_size = 4
    mutation_rate = 0.05
    crossover_rate = 0.8
    
    while time.time() < deadline - 30:
        new_population = []
        
        # Elitism
        for i in range(min(elite_size, len(population))):
            new_population.append(population[i])
            
        # Generate offspring to fill population
        while len(new_population) < pop_size and time.time() < deadline - 30:
            # Selection: Tournament selection
            # Pick 2 random parents
            p1_idx = np.random.randint(0, len(population))
            p2_idx = np.random.randint(0, len(population))
            
            # Ensure we have distinct parents if possible, or just use top 2 if tournament is weak
            # Let's do a simple tournament of 3 to be more selective
            candidates = np.random.choice(len(population), 3, replace=False)
            best_cand = max(candidates, key=lambda i: population[i][1])
            p1 = population[best_cand][0]
            
            candidates2 = np.random.choice(len(population), 3, replace=False)
            best_cand2 = max(candidates2, key=lambda i: population[i][1])
            p2 = population[best_cand2][0]
            
            # Crossover
            if np.random.random() < crossover_rate:
                child = _crossover(p1, p2)
            else:
                child = p1.copy()
                
            # Mutation
            child = _mutate(child, mutation_rate)
            
            # Symmetrize (though crossover/mutation should preserve it, just in case)
            child_sym = _make_symmetric(child)
            
            # Evaluate
            score = _compute_gap_fast(child_sym, n_k=6, pw_radius=8)
            new_population.append((child_sym, score))
            
        # Update population
        population = new_population
        population.sort(key=lambda x: x[1], reverse=True)
        
        if population[0][1] > best_score:
            best_score = population[0][1]
            best_cell = population[0][0]
            
        # Adaptive mutation rate?
        # If stuck, increase mutation rate
        # Simple heuristic: if no improvement in last 5 generations, bump mutation
        # For now, keep fixed.
        
    # 3. Final Polish
    # Take the best cell and do a local search with higher fidelity
    # Or just run the accurate solver on the top 5 candidates to pick the real best
    top_candidates = [c[0] for c in population[:5]]
    
    final_best_cell = best_cell
    final_best_score = best_score
    
    if time.time() < deadline - 10:
        for cell in top_candidates:
            if time.time() > deadline - 5:
                break
            try:
                # Use accurate solver for final check
                # Note: accurate solver is slower. Only do it for top few.
                acc_score = _compute_gap_accurate(cell, n_k=10, pw_radius=10)
                if acc_score > final_best_score:
                    final_best_score = acc_score
                    final_best_cell = cell
            except:
                pass
                
    # Ensure final output is a list of lists
    return final_best_cell.tolist()
