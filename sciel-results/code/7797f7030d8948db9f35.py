"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _fast_pwe_gap(cell, eps_low=1.0, eps_high=12.0, n_pw=9, n_kpoints=12):
    """
    Fast PWE surrogate using FFT to get Fourier coefficients.
    Returns the gap-midgap ratio for bands 4 and 5 (1-indexed, so indices 3 and 4).
    """
    N = len(cell)
    # Convert cell to epsilon array
    # cell is 0/1. 1 is dielectric (eps_high), 0 is air (eps_low)
    eps = np.array(cell, dtype=np.float64)
    eps = eps * (eps_high - eps_low) + eps_low
    
    # For TM mode, the operator involves 1/epsilon
    inv_eps = 1.0 / eps
    
    # Compute Fourier coefficients of 1/eps
    # f_mn = (1/N^2) * sum_{i,j} f(i,j) * exp(-2*pi*i*(m*i + n*j)/N)
    # We can use FFT. Note: numpy FFT uses exp(-2*pi*i*k/N)
    # The coefficients we need are for G = (2*pi*m/N, 2*pi*n/N)
    # f_mn is the coefficient of exp(i * G . r)
    # Inverse FFT gives sum f_mn exp(-i G r). 
    # Standard PWE: H_{mn} = (k+G_m) dot (k+G_n) * f_{mn-n}
    # Let's compute the DFT matrix.
    
    # Efficient way: 
    # F = FFT2(inv_eps)
    # The DFT coefficient F_{u,v} corresponds to sum_{i,j} f_{i,j} exp(-2*pi*i (u*i/N + v*j/N))
    # So f_{m,n} = F_{m mod N, n mod N} / N^2
    # We need f_{m,n} for m,n in [-n_pw, n_pw]
    
    F = np.fft.fft2(inv_eps) / (N * N)
    
    # Extract coefficients
    # Index mapping: index = (m + N) % N
    coeffs = np.zeros((2*n_pw+1, 2*n_pw+1), dtype=complex)
    for i, m in enumerate(range(-n_pw, n_pw+1)):
        for j, n in enumerate(range(-n_pw, n_pw+1)):
            idx_m = m % N
            idx_n = n % N
            coeffs[i, j] = F[idx_m, idx_n]
            
    # K-points in the irreducible Brillouin zone
    # Square lattice BZ: -pi/a <= kx, ky <= pi/a
    # Irreducible wedge: 0 <= kx <= ky, kx^2 + ky^2 <= 1 (in units of pi/a)
    # Actually, the first BZ is [-pi/a, pi/a] x [-pi/a, pi/a].
    # Irreducible part is 1/8 of the square.
    # Points: Gamma (0,0), M (pi/a, 0) -> (1,0) in units of pi/a? No, usually (1,0) is M/2?
    # Let's use normalized coordinates where G = 2*pi/n * (m,n).
    # k is in units of 2*pi/N? No, k is in physical units.
    # Let's work in units of 2*pi/N. Then G = (m, n).
    # BZ boundaries are at k = +/- pi/a = +/- N/2 * (2*pi/N) = +/- 16 in our units.
    # Irreducible BZ: 0 <= kx <= ky, kx^2 + ky^2 <= 16^2? No, BZ is square.
    # Square BZ: -16 <= kx, ky <= 16.
    # Irreducible: 0 <= kx <= ky, kx <= 16, ky <= 16.
    
    # Sample points
    k_points = [
        (0.0, 0.0),       # Gamma
        (16.0, 0.0),      # M
        (16.0, 16.0),     # X (corner)
        (8.0, 0.0),       # M/2
        (8.0, 8.0),       # X/2
        (12.0, 6.0),      # Interior
        (10.0, 10.0),     # Interior
        (14.0, 2.0),      # Near M
        (14.0, 14.0),     # Near X
        (5.0, 5.0),       # Near Gamma
        (16.0, 8.0),      # Boundary
        (4.0, 12.0)       # Boundary
    ]
    
    if n_kpoints < len(k_points):
        k_points = k_points[:n_kpoints]
        
    min_omega5 = float('inf')
    max_omega4 = 0.0
    
    # Precompute G vectors
    G_vals = range(-n_pw, n_pw+1)
    G_list = [(m, n) for m in G_vals for n in G_vals]
    G_idx = {g: i for i, g in enumerate(G_list)}
    
    for kx, ky in k_points:
        # Build matrix A
        # A_{mn} = sum_p f_{mn-p} * |k+p|^2 ? No.
        # The TM eigenvalue problem is:
        # sum_n (k+G_m) dot (k+G_n) f_{mn-n} c_n = omega^2 c_m
        # Let's construct the matrix H where H_{mn} = (k+G_m) dot (k+G_n) * f_{G_m - G_n}
        # Wait, the standard form is:
        # H_{mn} = (k+G_m) dot (k+G_n) * f_{mn}
        # where f_{mn} is the Fourier coeff of 1/eps for G_m - G_n.
        
        size_mat = len(G_list)
        H = np.zeros((size_mat, size_mat), dtype=complex)
        
        for i, (mx, my) in enumerate(G_list):
            kx_m = kx + mx
            ky_m = ky + my
            for j, (nx, ny) in enumerate(G_list):
                kx_n = kx + nx
                ky_n = ky + ny
                # Dot product
                dot = kx_m * kx_n + ky_m * ky_n
                # Fourier coeff index: G_m - G_n = (mx-nx, my-ny)
                dm = mx - nx
                dn = my - ny
                if -n_pw <= dm <= n_pw and -n_pw <= dn <= n_pw:
                    # Map dm, dn to coeffs array index
                    # coeffs is indexed by (m+n_pw, n+n_pw)
                    ci = dm + n_pw
                    cj = dn + n_pw
                    f_val = coeffs[ci, cj]
                    H[i, j] = dot * f_val
                else:
                    # If difference is outside computed range, assume 0 (truncation error)
                    H[i, j] = 0.0
                    
        # Solve eigenvalue problem
        # H is Hermitian if f is real (which it is for real eps) and symmetric.
        # But f_{mn} = conj(f_{nm}). Since eps is real, f is real and even?
        # 1/eps is real. F_{m,n} = conj(F_{-m,-n}).
        # H_{mn} = (k+G_m).(k+G_n) f_{mn}.
        # H_{nm} = (k+G_n).(k+G_m) f_{nm}.
        # Since dot is symmetric, H_{mn} = H_{nm} * (f_{mn}/f_{nm}).
        # If f is real, H is symmetric.
        
        try:
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(np.real(eigvals))
            # We want bands 4 and 5. Indices 3 and 4.
            if len(eigvals) >= 5:
                omega4 = eigvals[3]
                omega5 = eigvals[4]
                if omega4 > 0 and omega5 > 0:
                    if omega4 > max_omega4:
                        max_omega4 = omega4
                    if omega5 < min_omega5:
                        min_omega5 = omega5
        except:
            pass
            
    if min_omega5 == float('inf') or max_omega4 == 0:
        return 0.0
        
    gap_ratio = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
    return gap_ratio

def _quadrant_to_cell(quadrant, size=32):
    """
    Convert a 16x16 quadrant to a full 32x32 cell with C4v symmetry.
    quadrant[i][j] maps to:
    cell[i][j]
    cell[i][size-1-j]
    cell[size-1-i][j]
    cell[size-1-i][size-1-j]
    """
    half = size // 2
    cell = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(half):
        for j in range(half):
            val = quadrant[i][j]
            cell[i][j] = val
            cell[i][size - 1 - j] = val
            cell[size - 1 - i][j] = val
            cell[size - 1 - i][size - 1 - j] = val
    return cell

def _cell_to_quadrant(cell, size=32):
    half = size // 2
    return [row[:half] for row in cell[:half]]

def design(problem):
    start_time = time.time()
    deadline = start_time + 180  # Leave 20s margin
    
    size = problem["pixels"]
    half = size // 2
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Initialize random seed for reproducibility if needed, but random is fine
    random.seed(42)
    
    # Helper to evaluate a quadrant
    def evaluate_quadrant(q):
        cell = _quadrant_to_cell(q, size)
        return _fast_pwe_gap(cell, eps_low, eps_high, n_pw=9, n_kpoints=12)
    
    # Initial Population
    pop_size = 20
    population = []
    scores = []
    
    # 1. Current best: 4 rods
    q_seed = [[0 for _ in range(half)] for _ in range(half)]
    cx, cy = 7.5, 7.5
    radius = 5.0
    for i in range(half):
        for j in range(half):
            x, y = i + 0.5, j + 0.5
            if (x - cx)**2 + (y - cy)**2 < radius**2:
                q_seed[i][j] = 1
    population.append(q_seed)
    scores.append(evaluate_quadrant(q_seed))
    
    # 2. Other heuristics
    # Ring
    q_ring = [[0 for _ in range(half)] for _ in range(half)]
    for i in range(half):
        for j in range(half):
            x, y = i + 0.5, j + 0.5
            r = math.sqrt((x-cx)**2 + (y-cy)**2)
            if 3.0 < r < 5.5:
                q_ring[i][j] = 1
    population.append(q_ring)
    scores.append(evaluate_quadrant(q_ring))
    
    # Cross
    q_cross = [[0 for _ in range(half)] for _ in range(half)]
    for i in range(half):
        for j in range(half):
            x, y = i + 0.5, j + 0.5
            if abs(x-cx) < 2.0 or abs(y-cy) < 2.0:
                q_cross[i][j] = 1
    population.append(q_cross)
    scores.append(evaluate_quadrant(q_cross))
    
    # Randoms
    while len(population) < pop_size:
        q = [[random.randint(0,1) for _ in range(half)] for _ in range(half)]
        population.append(q)
        scores.append(evaluate_quadrant(q))
        
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_quadrant = population[best_idx]
    
    # GA Loop
    generations = 0
    max_generations = 100
    
    while generations < max_generations:
        if time.time() > deadline:
            break
            
        new_population = []
        new_scores = []
        
        # Elitism: Keep top 2
        sorted_indices = np.argsort(scores)[::-1]
        for i in range(2):
            idx = sorted_indices[i]
            new_population.append(population[idx])
            new_scores.append(scores[idx])
            
        # Fill rest
        while len(new_population) < pop_size:
            # Selection: Tournament of 3
            idxs = random.sample(range(pop_size), 3)
            best_in_tournament = max(idxs, key=lambda x: scores[x])
            
            # Crossover
            parent1 = population[best_in_tournament]
            # Pick another random parent
            parent2_idx = random.choice(range(pop_size))
            parent2 = population[parent2_idx]
            
            child = [row[:] for row in parent1]
            for i in range(half):
                for j in range(half):
                    if random.random() < 0.5:
                        child[i][j] = parent2[i][j]
                        
            # Mutation
            # Adaptive mutation rate
            mut_rate = 0.05 * (1.0 - generations/max_generations) + 0.01
            for i in range(half):
                for j in range(half):
                    if random.random() < mut_rate:
                        child[i][j] = 1 - child[i][j]
                        
            new_population.append(child)
            new_scores.append(evaluate_quadrant(child))
            
        population = new_population
        scores = new_scores
        
        current_best_idx = np.argmax(scores)
        if scores[current_best_idx] > best_score:
            best_score = scores[current_best_idx]
            best_quadrant = population[current_best_idx]
            
        generations += 1
        
    # Return best cell
    best_cell = _quadrant_to_cell(best_quadrant, size)
    return best_cell

# EVOLVE-BLOCK-END
