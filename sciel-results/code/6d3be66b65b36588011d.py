import numpy as np
import time
import sys

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell to maximize the TM band gap between bands 4 and 5.
    Uses a local search (hill climbing) with multiple restarts and a fast plane-wave solver.
    """
    size = problem["pixels"] # 32
    eps_high = problem["eps_high"] # 12
    eps_low = problem["eps_low"] # 1
    
    # Time management
    start_time = time.time()
    time_limit = 200.0 # Safety margin for 230s kill switch
    
    # Precompute symmetric pixel indices
    # A pixel (i, j) is mapped to (size-1-i, j), (i, size-1-j), etc.
    # We only need to iterate over a fundamental domain.
    # For C4v symmetry, the fundamental domain is roughly 1/8 of the cell.
    # However, for simplicity and correctness, we can just define a list of unique "free" pixels.
    free_pixels = []
    seen = set()
    for i in range(size):
        for j in range(size):
            # Generate all symmetric equivalents
            equivs = [
                (i, j),
                (size-1-i, j),
                (i, size-1-j),
                (size-1-i, size-1-j),
                (j, i),
                (size-1-j, i),
                (j, size-1-i),
                (size-1-j, size-1-i)
            ]
            # Normalize to find a canonical representative
            # Sort and take the first
            equivs.sort()
            rep = equivs[0]
            if rep not in seen:
                seen.add(rep)
                free_pixels.append((i, j, equivs))
    
    # Number of free pixels is roughly size*size / 8
    # 32*32/8 = 128 free pixels. This is a small search space.
    
    # PWE Solver Setup
    # We need to solve for eigenvalues of the dielectric distribution.
    # The operator is (1/eps) * (-k^2).
    # In PWE, we expand fields in plane waves.
    # The matrix is A_{G', G} = delta_{G', G} * (k+G)^2 / eps_{G'-G}
    
    # Reciprocal lattice vectors
    # Square lattice, a=1. G = (n, m)
    # We need a cutoff. Let's try N=20 (approx 1200 waves? No, 2N+1 squared is too many).
    # Irreducible BZ sampling requires evaluating at specific k points.
    # For speed, we use a limited number of plane waves.
    # Let's use a hexagonal-ish cutoff in G-space to keep count low.
    # Or just a square cutoff.
    # N=15 -> 31x31 = 961 waves. Eigh is O(N^3). 961^3 is ~ 8e8. Too slow per eval?
    # 1000x1000 matrix eigh in numpy takes ~1s.
    # We have 200s. We can do ~200 evals if 1s each.
    # Let's try N=10 -> 21x21 = 441 waves. 441^3 ~ 8e7. ~0.1s.
    # 441 waves might be too coarse for accurate gap.
    # Let's try N=12 -> 25x25 = 625 waves.
    
    N = 12
    G_vectors = []
    for n in range(-N, N+1):
        for m in range(-N, N+1):
            G_vectors.append((n, m))
    G_vectors = np.array(G_vectors, dtype=float)
    num_G = len(G_vectors)
    
    # Precompute G^2
    G_sq = G_vectors[:, 0]**2 + G_vectors[:, 1]**2
    
    # Fourier coefficients of 1/eps
    # eps(x) is 0/1. 1/eps is 1/eps_high or 1/eps_low.
    # We compute FFT of the 1/eps map.
    # Note: The evaluator uses exact coefficients. We approximate with FFT of the pixel grid.
    # This is a standard approach for pixel-based PWE.
    
    # K-points in irreducible BZ
    # Gamma: (0,0)
    # M: (0.5, 0)
    # X (same as M in square): (0, 0.5)
    # K: (0.5, 0.5)
    # Midpoints: (0.25, 0), (0.25, 0.25), etc.
    # Let's sample a grid of k-points in the irreducible triangle.
    # Irreducible BZ: 0 <= kx <= ky, kx^2+ky^2 <= 0.5 (approx)
    # Actually, for square, irreducible is 1/8 of BZ.
    # kx in [0, 0.5], ky in [kx, 0.5-kx] ? No.
    # Boundary is kx=0, ky=0.5, and kx+ky=0.5.
    # So 0 <= kx <= 0.25, kx <= ky <= 0.5-kx.
    
    k_points = []
    # Gamma
    k_points.append((0.0, 0.0))
    # M
    k_points.append((0.5, 0.0))
    # X
    k_points.append((0.0, 0.5))
    # K (center of M-X edge? No, K is (0.5, 0.5) which is corner)
    k_points.append((0.5, 0.5))
    
    # Interior points
    for i in range(1, 4):
        for j in range(i, 4):
            kx = i / 8.0
            ky = j / 8.0
            if kx <= 0.25 and ky <= 0.5 - kx:
                k_points.append((kx, ky))
            # Also check symmetry equivalent if needed, but we just need to cover the domain.
            # The condition kx <= ky and kx+ky <= 0.5 covers the 1/8 sector.
            
    # Remove duplicates
    k_points = list(set(k_points))
    
    def compute_fourier_coeffs(cell):
        # cell is 32x32 list of 0/1
        # 1 = dielectric (eps_high), 0 = air (eps_low)
        # We need 1/eps
        inv_eps_map = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if cell[i][j] == 1:
                    inv_eps_map[i][j] = 1.0 / eps_high
                else:
                    inv_eps_map[i][j] = 1.0 / eps_low
        
        # FFT
        # The Fourier coefficient G_{nm} is integral of (1/eps) * exp(-i 2pi (nx+my))
        # FFT gives sum over pixels. We need to scale by pixel area (1/size^2).
        fft_inv_eps = np.fft.fft2(inv_eps_map)
        fft_inv_eps /= (size * size)
        
        # Map G_vectors (n, m) to FFT indices
        # FFT index (i, j) corresponds to frequency (i - N_fft//2, j - N_fft//2) if shifted?
        # np.fft.fft2 returns frequencies in order 0, 1, ..., N/2-1, -N/2, ..., -1.
        # Our G_vectors are in range [-N, N].
        # We need to extract coefficients for our specific G_vectors.
        
        coeffs = np.zeros(num_G, dtype=complex)
        for idx, (n, m) in enumerate(G_vectors):
            # Map n, m to FFT array indices
            # n is in [-N, N]. FFT size is size (32).
            # If n >= 0, idx_n = n. If n < 0, idx_n = size + n.
            idx_n = n if n >= 0 else size + n
            idx_m = m if m >= 0 else size + m
            coeffs[idx] = fft_inv_eps[idx_n, idx_m]
            
        return coeffs

    def solve_bands(cell, kx, ky):
        coeffs = compute_fourier_coeffs(cell)
        
        # Construct the matrix A
        # A_{G', G} = (k+G')^2 * delta_{G', G} / eps_{G'-G}
        # Wait, the standard PWE eigenvalue problem is:
        # sum_G  (1/eps_{G'-G}) * (k+G)^2 * c_G = omega^2 c_{G'}
        # So A_{G', G} = (k+G)^2 / eps_{G'-G}
        
        # We have coeffs for 1/eps.
        # Let H_{G'-G} = coeffs[G'-G]
        
        # We need to build the matrix.
        # For performance, we can precompute the indices.
        
        # k vector is (kx, ky)
        # (k+G)^2 = (kx + G_n)^2 + (ky + G_m)^2
        
        # Let's build the matrix row by row.
        # A is symmetric and real?
        # 1/eps is real and symmetric (since cell is real and symmetric).
        # coeffs are real? For symmetric real functions, FFT coeffs are real.
        # Let's assume real for speed.
        
        A = np.zeros((num_G, num_G))
        
        # Precompute (k+G)^2 for all G
        kG_sq = (kx + G_vectors[:, 0])**2 + (ky + G_vectors[:, 1])**2
        
        # We need H_{G'-G}.
        # Let's create a lookup for H.
        # H is indexed by difference vector.
        # We can just compute on the fly or precompute a 2D array of H.
        
        # Since num_G is small (~625), O(N^2) is fine.
        # But we need to map (G' - G) to the coefficient.
        
        # Let's create a dictionary or 2D array for coeffs.
        # coeffs_dict[(n, m)] = value
        # But G_vectors are unique.
        
        # Alternative:
        # For each G', we need to sum over G.
        # A[G', G] = kG_sq[G] * coeffs[G' - G]
        
        # Let's precompute a 2D array of coeffs indexed by (n, m) in range [-2N, 2N]?
        # No, just use the list.
        
        # To speed up, we can exploit that A is symmetric.
        # A[G', G] = A[G, G']?
        # (k+G)^2 * H_{G'-G} vs (k+G')^2 * H_{G-G'}
        # H is symmetric (H_{-q} = H_q^* = H_q).
        # But (k+G)^2 != (k+G')^2.
        # So A is NOT symmetric in the standard sense unless we transform the basis.
        # Wait, the operator is Hermitian.
        # The matrix in the plane wave basis is not symmetric, but it is similar to a symmetric matrix?
        # Actually, for TM, the matrix is:
        # M_{G', G} = (k+G)^2 / eps_{G'-G}
        # This matrix is not symmetric.
        # However, we can transform to a symmetric form by multiplying by sqrt(eps)?
        # Or just use eigh on the non-symmetric matrix? No, eigh requires symmetric.
        # We can use eigvals, but it's slower and less stable.
        
        # Let's check the symmetry.
        # The Hamiltonian is H = (1/eps) * (-nabla^2).
        # In PW basis: <G'| H |G> = (k+G)^2 / eps_{G'-G}.
        # Is this matrix symmetric?
        # M_{G', G} = (k+G)^2 H_{G'-G}
        # M_{G, G'} = (k+G')^2 H_{G-G'} = (k+G')^2 H_{G'-G}
        # They are not equal.
        
        # However, we can define a new basis vector v_G = c_G * (k+G).
        # Or simpler: The matrix B_{G', G} = sqrt((k+G')(k+G)) / eps_{G'-G} is symmetric.
        # The eigenvalues of A and B are the same?
        # A c = omega^2 c
        # Let c_G = d_G / (k+G)
        # sum_G (k+G)^2 / eps_{G'-G} * d_G / (k+G) = omega^2 d_{G'} / (k+G')
        # sum_G (k+G) / eps_{G'-G} d_G = omega^2 d_{G'} / (k+G')
        # Multiply by (k+G'):
        # sum_G (k+G') (k+G) / eps_{G'-G} d_G = omega^2 d_{G'}
        # Let B_{G', G} = (k+G') (k+G) / eps_{G'-G}
        # B is symmetric because (k+G')(k+G) is symmetric in G', G and eps is symmetric.
        # So we can solve B d = omega^2 d using eigh.
        
        B = np.zeros((num_G, num_G))
        
        # We need H_{G'-G}.
        # Let's create a 2D array H_2d of size (4N+1, 4N+1) to quickly lookup H_{n, m}.
        # Range of n, m in G'-G is [-2N, 2N].
        H_range = 2 * N
        H_2d = np.zeros((2*H_range+1, 2*H_range+1))
        for idx, (n, m) in enumerate(G_vectors):
            # Map n, m to 0-based index in H_2d
            # n in [-N, N]. Index = n + N.
            H_2d[n + N, m + N] = coeffs[idx].real # Assume real
        
        # Now fill B
        for i in range(num_G):
            kG_i = kx + G_vectors[i, 0]
            kG_i_sq_term = kG_i # part of (k+G')
            # Actually we need (k+G') for row i and (k+G) for col j
            # B[i, j] = (k+G_i) * (k+G_j) * H_{G_i - G_j}
            
            # Vectorized inner loop
            kG_j = kx + G_vectors[:, 0] # This is wrong, need ky too.
            
        # Let's do it properly with numpy broadcasting if possible, or loop.
        # num_G ~ 625. Loop is 625*625 = 390k ops. Fast.
        
        for i in range(num_G):
            Gi = G_vectors[i]
            kGi = kx + Gi[0]
            kGi_y = ky + Gi[1]
            # We need (k+Gi) magnitude? No, the term is (k+Gi)^2 in the original, 
            # but in the symmetric form B, it is (k+Gi) * (k+Gj) ?
            # Let's re-derive.
            # Original: sum (k+G)^2 / eps_{G'-G} c_G = w^2 c_{G'}
            # Let c_G = d_G / (k+G)
            # sum (k+G)^2 / eps * d_G / (k+G) = w^2 d_{G'} / (k+G')
            # sum (k+G) / eps d_G = w^2 d_{G'} / (k+G')
            # (k+G') sum (k+G) / eps d_G = w^2 d_{G'}
            # So B_{G', G} = (k+G') (k+G) / eps_{G'-G}
            # Wait, (k+G) is a scalar? No, (k+G)^2 was the eigenvalue of -nabla^2.
            # The term is (kx+Gn)^2 + (ky+Gm)^2.
            # So it is NOT separable as (k+G).
            # My transformation c_G = d_G / (k+G) only works if the operator was 1D or if (k+G) was a vector norm.
            # Here, the "mass" is (k+G)^2.
            # So let c_G = d_G / sqrt((k+G)^2) = d_G / |k+G|.
            # Then sum |k+G|^2 / eps * d_G / |k+G| = w^2 d_{G'} / |k+G'|
            # sum |k+G| / eps d_G = w^2 d_{G'} / |k+G'|
            # |k+G'| sum |k+G| / eps d_G = w^2 d_{G'}
            # So B_{G', G} = |k+G'| |k+G| / eps_{G'-G}
            
            # This B is symmetric.
            
            # Calculate |k+G|^2 for all G
            # kG_sq is already calculated as (kx+Gn)^2 + (ky+Gm)^2
            # So |k+G| = sqrt(kG_sq)
            
            sqrt_kG_sq = np.sqrt(kG_sq)
            
            for j in range(num_G):
                Gj = G_vectors[j]
                # Difference G' - G
                dn = Gi[0] - Gj[0]
                dm = Gi[1] - Gj[1]
                
                # Lookup H
                # H_2d index: dn + H_range, dm + H_range
                h_val = H_2d[dn + H_range, dm + H_range]
                
                B[i, j] = sqrt_kG_sq[i] * sqrt_kG_sq[j] * h_val
                
        # Solve eigenvalue problem
        # B is symmetric
        eigenvalues, _ = np.linalg.eigh(B)
        
        # We want bands 4 and 5 (1-indexed). So indices 3 and 4.
        # Ensure we have at least 5 bands.
        if len(eigenvalues) < 5:
            return -1.0, -1.0
        
        w4 = eigenvalues[3]
        w5 = eigenvalues[4]
        
        return w4, w5

    def evaluate_gap(cell):
        max_w4 = -np.inf
        min_w5 = np.inf
        
        for kx, ky in k_points:
            w4, w5 = solve_bands(cell, kx, ky)
            if w4 > max_w4:
                max_w4 = w4
            if w5 < min_w5:
                min_w5 = w5
                
        if max_w4 == -np.inf or min_w5 == np.inf:
            return -1.0
            
        gap = min_w5 - max_w4
        midgap = (min_w5 + max_w4) / 2.0
        if midgap == 0:
            return 0.0
        ratio = gap / midgap
        return ratio

    # Initial Designs
    def make_cell_from_pattern(pattern_func):
        cell = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                cell[i][j] = pattern_func(i, j)
        return cell

    # 1. Seed: Rod
    def pattern_rod(i, j):
        x = (i + 0.5) / size - 0.5
        y = (j + 0.5) / size - 0.5
        return 1 if x*x + y*y < 0.2**2 else 0

    # 2. Ring
    def pattern_ring(i, j):
        x = (i + 0.5) / size - 0.5
        y = (j + 0.5) / size - 0.5
        r = np.sqrt(x*x + y*y)
        return 1 if 0.15 < r < 0.3 else 0

    # 3. Cross
    def pattern_cross(i, j):
        x = (i + 0.5) / size - 0.5
        y = (j + 0.5) / size - 0.5
        # Thin cross
        if abs(x) < 0.05 or abs(y) < 0.05:
            return 1
        return 0

    # 4. Grid (Checkerboard-like but symmetric)
    def pattern_grid(i, j):
        # 4x4 blocks
        block_i = i // 8
        block_j = j // 8
        if (block_i + block_j) % 2 == 0:
            return 1
        return 0

    initial_patterns = [
        make_cell_from_pattern(pattern_rod),
        make_cell_from_pattern(pattern_ring),
        make_cell_from_pattern(pattern_cross),
        make_cell_from_pattern(pattern_grid)
    ]
    
    best_score = -1.0
    best_cell = initial_patterns[0]
    
    # Local Search
    # We iterate through free pixels and flip them if it improves the score.
    # To avoid cycling, we use a simple hill climb.
    
    def local_search(cell, max_iter=100):
        current_cell = [row[:] for row in cell]
        current_score = evaluate_gap(current_cell)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            # Shuffle free pixels to avoid bias
            np.random.shuffle(free_pixels)
            
            for i, j, equivs in free_pixels:
                if time.time() - start_time > time_limit - 5:
                    return current_cell, current_score
                    
                # Flip
                old_val = current_cell[i][j]
                new_val = 1 - old_val
                
                for ei, ej in equivs:
                    current_cell[ei][ej] = new_val
                    
                new_score = evaluate_gap(current_cell)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    for ei, ej in equivs:
                        current_cell[ei][ej] = old_val
                        
        return current_cell, current_score

    # Run search for each initial pattern
    for init_cell in initial_patterns:
        if time.time() - start_time > time_limit - 10:
            break
            
        cell, score = local_search(init_cell, max_iter=50)
        
        if score > best_score:
            best_score = score
            best_cell = cell
            
    # If time remains, do more random restarts
    while time.time() - start_time < time_limit - 20:
        # Random initial cell
        rand_cell = [[0]*size for _ in range(size)]
        for i, j, equivs in free_pixels:
            val = 1 if np.random.rand() > 0.5 else 0
            for ei, ej in equivs:
                rand_cell[ei][ej] = val
                
        cell, score = local_search(rand_cell, max_iter=20)
        
        if score > best_score:
            best_score = score
            best_cell = cell

    return best_cell

if __name__ == "__main__":
    # Dummy test
    problem = {"pixels": 32, "eps_low": 1, "eps_high": 12}
    result = design(problem)
    print("Design complete")
