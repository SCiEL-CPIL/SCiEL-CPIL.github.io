"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """Local search over 4-fold symmetric 32x32 cells to maximize TM band-4/5 gap."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    deadline = time.time() + 115.0  # safety margin before 120s CPU budget
    
    # Precompute plane-wave vectors (289 = 17x17 grid, |G| <= 8.5)
    N = 17
    half = N // 2  # 8
    G_vectors = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            if gx * gx + gy * gy <= (half + 0.5) ** 2:  # |G| <= 8.5
                G_vectors.append((gx, gy))
    G_vectors = np.array(G_vectors, dtype=float)
    n_pw = len(G_vectors)
    
    # Precompute 30 k-points in irreducible BZ
    k_points = []
    # Along Gamma-X: kx from 0 to 0.5, ky=0
    for i in range(11):
        k_points.append((i * 0.5 / 10.0, 0.0))
    # Along X-M: kx=0.5, ky from 0 to 0.5
    for i in range(1, 11):
        k_points.append((0.5, i * 0.5 / 10.0))
    # Along M-Gamma: kx from 0.5 to 0, ky from 0.5 to 0
    for i in range(1, 11):
        t = i / 10.0
        k_points.append((0.5 * (1 - t), 0.5 * (1 - t)))
    # Remove duplicates
    unique_k = []
    for k in k_points:
        if not any(abs(k[0] - uk[0]) < 1e-9 and abs(k[1] - uk[1]) < 1e-9 for uk in unique_k):
            unique_k.append(k)
    k_points = np.array(unique_k[:30], dtype=float)
    
    # Precompute coefficient index matrix
    coeff_lookup = {}
    for idx, (gx, gy) in enumerate(G_vectors):
        coeff_lookup[(int(gx), int(gy))] = idx
        
    coeff_idx_mat = np.full((n_pw, n_pw), -1, dtype=int)
    for i in range(n_pw):
        for j in range(n_pw):
            dx = int(G_vectors[i, 0] - G_vectors[j, 0])
            dy = int(G_vectors[i, 1] - G_vectors[j, 1])
            key = (dx, dy)
            if key in coeff_lookup:
                coeff_idx_mat[i, j] = coeff_lookup[key]
                
    # Precompute cross product matrices for each k-point to save time
    cross_mats = []
    for kx, ky in k_points:
        kG_x = kx + G_vectors[:, 0]
        kG_y = ky + G_vectors[:, 1]
        cross = np.outer(kG_x, kG_y) - np.outer(kG_y, kG_x)
        cross_mats.append(cross)

    def compute_inv_eps_fourier(cell_arr):
        """Compute Fourier coefficients of 1/eps for the given cell."""
        eps = np.where(cell_arr == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_inv = np.fft.fft2(inv_eps)
        c = fft_inv / (size * size)
        coeffs = np.zeros(n_pw, dtype=complex)
        for idx, (gx, gy) in enumerate(G_vectors):
            ix = int(gx) % size
            iy = int(gy) % size
            coeffs[idx] = c[iy, ix]
        return np.real(coeffs)

    def score_cell(cell_arr):
        """Compute the gap-midgap ratio."""
        coeffs = compute_inv_eps_fourier(cell_arr)
        
        max_w4 = 0.0
        min_w5 = float('inf')
        
        for idx_k, cross in enumerate(cross_mats):
            # Build coefficient matrix using precomputed indices
            C = np.zeros((n_pw, n_pw), dtype=float)
            valid = coeff_idx_mat >= 0
            C[valid] = coeffs[coeff_idx_mat[valid]]
            
            M = C * cross
            M = (M + M.T) / 2.0
            
            eigvals = np.linalg.eigvalsh(M)
            
            if len(eigvals) >= 5:
                w4 = math.sqrt(max(0, eigvals[3]))
                w5 = math.sqrt(max(0, eigvals[4]))
                if w4 > max_w4: max_w4 = w4
                if w5 < min_w5: min_w5 = w5
        
        if min_w5 <= 0:
            return -1.0
        return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)

    # Precompute symmetry orbits
    orbits = []
    visited = np.zeros((size, size), dtype=bool)
    for i in range(size):
        for j in range(size):
            if not visited[i, j]:
                orbit = [(i, j), (j, i), (size-1-i, size-1-j), (size-1-j, size-1-i)]
                unique_orbit = []
                for p in orbit:
                    if not visited[p[0], p[1]]:
                        unique_orbit.append(p)
                        visited[p[0], p[1]] = True
                if unique_orbit:
                    orbits.append(unique_orbit)
    
    rng = np.random.default_rng(42)
    
    # Define multiple seed patterns
    def make_circle(radius, inverted=False):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < radius * radius:
                    cell[i, j] = 1 if not inverted else 0
                    if inverted: cell[i, j] = 0
                    else: cell[i, j] = 1
        return cell

    def make_square(half_width, inverted=False):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                if abs(x) < half_width and abs(y) < half_width:
                    cell[i, j] = 1 if not inverted else 0
                    if inverted: cell[i, j] = 0
                    else: cell[i, j] = 1
        return cell

    def make_checkerboard(block_size=8):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if (i // block_size + j // block_size) % 2 == 0:
                    cell[i, j] = 1
        return cell

    def make_plus(width=4):
        cell = np.zeros((size, size), dtype=int)
        center = size // 2
        for i in range(size):
            for j in range(size):
                if abs(i - center + 0.5) < width or abs(j - center + 0.5) < width:
                    cell[i, j] = 1
        return cell

    seeds = [
        make_circle(0.2, inverted=False),
        make_circle(0.2, inverted=True),
        make_square(0.2, inverted=False),
        make_square(0.2, inverted=True),
        make_checkerboard(8),
        make_plus(4)
    ]
    
    best_score = -1.0
    best_cell = seeds[0].copy()
    
    # Run local search for each seed
    # Total time budget is ~115s. Let's allocate ~15s per seed if we have 6 seeds.
    # Or just run until deadline, prioritizing the best current one?
    # Better: Run a fixed number of iterations per seed, then keep the best.
    
    # Let's estimate iterations.
    # One eval takes ~0.1-0.2s. 15s -> ~100-150 evals.
    
    for seed in seeds:
        if time.time() > deadline - 10.0: break
        
        cell_arr = seed.copy()
        local_best_score = score_cell(cell_arr)
        local_best_cell = cell_arr.copy()
        
        seed_deadline = time.time() + 15.0 # 15s per seed
        
        while time.time() < seed_deadline:
            # Pick a random orbit
            idx = rng.integers(0, len(orbits))
            orbit = orbits[idx]
            
            # Flip all pixels in the orbit
            for (i, j) in orbit:
                cell_arr[i, j] = 1 - cell_arr[i, j]
            
            new_score = score_cell(cell_arr)
            
            if new_score > local_best_score:
                local_best_score = new_score
                local_best_cell = cell_arr.copy()
            else:
                # Revert
                for (i, j) in orbit:
                    cell_arr[i, j] = 1 - cell_arr[i, j]
        
        if local_best_score > best_score:
            best_score = local_best_score
            best_cell = local_best_cell
    
    # If we have time left, refine the best cell further
    if time.time() < deadline - 5.0:
        cell_arr = best_cell.copy()
        local_best_score = best_score
        local_best_cell = best_cell.copy()
        
        # Simulated Annealing or just more Hill Climbing
        # Let's do a more aggressive search: flip 1-3 orbits at a time
        while time.time() < deadline - 2.0:
            # Randomly choose 1 to 3 orbits to flip
            n_flips = rng.integers(1, 4)
            flipped_orbits = []
            for _ in range(n_flips):
                idx = rng.integers(0, len(orbits))
                orbit = orbits[idx]
                flipped_orbits.append(orbit)
                for (i, j) in orbit:
                    cell_arr[i, j] = 1 - cell_arr[i, j]
            
            new_score = score_cell(cell_arr)
            
            if new_score > local_best_score:
                local_best_score = new_score
                local_best_cell = cell_arr.copy()
            else:
                # Revert
                for orbit in flipped_orbits:
                    for (i, j) in orbit:
                        cell_arr[i, j] = 1 - cell_arr[i, j]
                        
        if local_best_score > best_score:
            best_score = local_best_score
            best_cell = local_best_cell

    return [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
# EVOLVE-BLOCK-END
