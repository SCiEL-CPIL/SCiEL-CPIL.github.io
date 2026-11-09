import numpy as np
import time

def design(problem):
    """
    Optimizes a discrete, symmetry-constrained binary mask by directly manipulating
    pixels in the unique D4-orbit. Uses a fast, approximate plane-wave solver to
    score candidates and performs a local search (hill climbing) combined with
    random restarts to navigate the rugged landscape of the band gap objective.
    
    This approach avoids the continuous-to-binary mapping issues of the previous
    Fourier-based method and directly optimizes the discrete variables that the
    evaluator uses.
    """
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin before 230s kill (allow 40s buffer)
    
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # --- 1. Define the unique pixel orbits under D4 symmetry ---
    # The cell is 32x32. Symmetry about center (0.5, 0.5) and transpose.
    # We represent the cell by a set of "orbits". Each orbit is a set of indices (i,j)
    # that are equivalent under the symmetry group. We only need to store the value
    # for one representative pixel per orbit, and replicate it to all others.
    
    # The symmetry group D4 has 8 elements:
    # 1. Identity: (i, j)
    # 2. Rotate 90: (j, N-1-i)
    # 3. Rotate 180: (N-1-i, N-1-j)
    # 4. Rotate 270: (N-1-j, i)
    # 5. Reflect x (horizontal): (N-1-i, j)
    # 6. Reflect y (vertical): (i, N-1-j)
    # 7. Reflect diag 1: (j, i)
    # 8. Reflect diag 2: (N-1-j, N-1-i)
    
    # We will collect the canonical representative for each orbit.
    # The canonical representative is the one with the smallest (i, j) in lexicographic order.
    
    orbits = {}
    for i in range(size):
        for j in range(size):
            # Generate all symmetric equivalents
            equivs = [
                (i, j),
                (j, size - 1 - i),
                (size - 1 - i, size - 1 - j),
                (size - 1 - j, i),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (j, i),
                (size - 1 - j, size - 1 - i)
            ]
            # Find the canonical representative (min i, then min j)
            rep = min(equivs)
            if rep not in orbits:
                orbits[rep] = []
            orbits[rep].append((i, j))
    
    # Sort orbits by their representative for consistent indexing
    orbit_keys = sorted(orbits.keys())
    num_orbits = len(orbit_keys)
    
    # Precompute the list of indices for each orbit for fast reconstruction
    orbit_indices = [orbits[k] for k in orbit_keys]
    
    # --- 2. Fast Plane-Wave Solver ---
    # We approximate the evaluator's 289-PW method.
    # To be faster, we might use fewer k-points or a smaller PW cutoff if needed,
    # but 289 PWs is small enough. The bottleneck is the eigenvalue solve.
    # We will use the same 289 PWs (N=8) but optimize the matrix construction.
    
    N_pw = 8
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # Precompute G-G' differences and their indices
    # coeff_dict[(dx, dy)] = index into the FFT array
    # We need coeff for dx, dy in [-16, 16]
    
    # K-points: We need to cover the irreducible BZ.
    # The evaluator uses 30 points. We will use a similar set.
    # Irreducible BZ of square lattice: 0 <= kx <= ky <= 0.5? No.
    # Standard irreducible BZ for square lattice (Gamma-M-X):
    # 0 <= kx <= 0.5, 0 <= ky <= kx
    # Let's sample a grid in this triangle.
    k_points = []
    # We want ~30 points.
    # Let's use a grid of kx in [0, 0.5] and ky in [0, kx].
    # 7x7 grid in the triangle?
    # Or just specific points.
    # Let's use a dense enough set to capture the gap.
    # The gap is usually near the M or X points or along the lines.
    
    # Let's use a fixed set of 30 points that are well distributed.
    # Gamma (0,0), M (0.5, 0), X (0.5, 0.5)
    # And points along the lines and interior.
    
    # For speed, we can reduce k-points during the search and use more for final evaluation?
    # No, the score is based on the evaluator's 30 points. We should try to match that.
    # However, 30 eigenvalue solves of 289x289 matrices might take ~0.1-0.2s.
    # If we can do ~100 evaluations in 180s, that's 1.8s per eval. This is fine.
    
    # Let's generate 30 points in the irreducible BZ.
    # Triangle: 0 <= kx <= 0.5, 0 <= ky <= kx
    # We'll use a grid of 6x6 = 36 points and filter to unique/valid ones.
    kx_vals = np.linspace(0, 0.5, 6)
    ky_vals = np.linspace(0, 0.5, 6)
    for kx in kx_vals:
        for ky in ky_vals:
            if ky <= kx:
                k_points.append((kx, ky))
    
    # Ensure we have enough points. If <30, add more? Or just use what we have.
    # 6x6 triangle has 21 points. Let's use 8x8 grid.
    k_points = []
    kx_vals = np.linspace(0, 0.5, 8)
    ky_vals = np.linspace(0, 0.5, 8)
    for kx in kx_vals:
        for ky in ky_vals:
            if ky <= kx:
                k_points.append((kx, ky))
    # This gives 36 points. We'll use the first 30 or all of them.
    # Using all 36 is safer for finding the true min/max.
    
    def evaluate_cell_binary(mask):
        """
        mask: 1D array of 0/1 values for the unique orbits.
        Returns the gap-midgap ratio.
        """
        # Reconstruct the full 32x32 epsilon grid
        eps_grid = np.zeros((size, size))
        for idx, val in enumerate(mask):
            if val == 1:
                for i, j in orbit_indices[idx]:
                    eps_grid[i, j] = eps_high
            else:
                for i, j in orbit_indices[idx]:
                    eps_grid[i, j] = eps_low
        
        # Compute 1/epsilon
        inv_eps_grid = 1.0 / eps_grid
        
        # Compute FFT of 1/epsilon
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)
        
        # Extract coefficients for G-G' in [-16, 16]
        # We can precompute a lookup table for the FFT indices
        # But for now, direct access is fast enough.
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        # Precompute |k+G|^2 for all k-points and G-vectors?
        # k_points is small (36), G_vectors is 289.
        # 36 * 289 = 10k values. Small.
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            
            # Vectorized computation of |k+G|^2
            k_plus_G = G_vectors + k
            omega_sq = np.sum(k_plus_G ** 2, axis=1)
            
            # Build the matrix M
            # M[i, j] = coeff[G_i - G_j] * omega_sq[i]
            
            # We can build the coefficient matrix C where C[i, j] = coeff[G_i - G_j]
            # This is a Toeplitz-like matrix.
            
            # For speed, let's just loop over i and j.
            # 289x289 = 83521 operations.
            # 36 k-points * 83521 = 3 million ops.
            # Plus eigenvalue solve (289^3 ~ 24 million flops).
            # Total ~ 36 * 25 million = 900 million flops.
            # NumPy eig is highly optimized. This should be fast.
            
            M = np.empty((num_pw, num_pw), dtype=complex)
            
            for i in range(num_pw):
                Gi = G_vectors[i]
                for j in range(num_pw):
                    Gj = G_vectors[j]
                    dx = int(Gi[0] - Gj[0])
                    dy = int(Gi[1] - Gj[1])
                    # Get coeff
                    ix = dx % size
                    iy = dy % size
                    coeff = fft_inv_eps[ix, iy]
                    M[i, j] = coeff * omega_sq[i]
            
            # Solve eigenvalue problem
            # M is not Hermitian, but eigenvalues are real.
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                omega_4 = np.sqrt(real_eigvals[3])
                omega_5 = np.sqrt(real_eigvals[4])
                
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio

    # --- 3. Optimization Loop ---
    
    best_mask = None
    best_score = -np.inf
    
    # Initial guess: A solid dielectric block in the center?
    # Or a hole?
    # Let's try a few random initial masks.
    
    num_restarts = 5
    max_iterations_per_restart = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Initialize mask
        if restart == 0:
            # Try a "rod" shape: dielectric in center
            # Orbits near the center (16,16) etc.
            mask = np.zeros(num_orbits, dtype=int)
            # Set orbits with representative (i,j) such that i,j are in [8, 23]
            for idx, (i, j) in enumerate(orbit_keys):
                if 8 <= i <= 23 and 8 <= j <= 23:
                    mask[idx] = 1
        else:
            # Random mask
            mask = np.random.randint(0, 2, num_orbits)
        
        current_score = evaluate_cell_binary(mask)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = mask.copy()
        
        # Local search: Flip one orbit at a time
        # We will try flipping each orbit and keep the best improvement.
        # To speed up, we can use a "steepest ascent" approach.
        
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            best_flip_idx = -1
            best_flip_score = current_score
            
            # Try flipping each orbit
            for idx in range(num_orbits):
                # Flip
                mask[idx] = 1 - mask[idx]
                
                new_score = evaluate_cell_binary(mask)
                
                # Unflip
                mask[idx] = 1 - mask[idx]
                
                if new_score > best_flip_score:
                    best_flip_score = new_score
                    best_flip_idx = idx
            
            # If we found an improvement, apply it
            if best_flip_idx != -1:
                mask[best_flip_idx] = 1 - mask[best_flip_idx]
                current_score = best_flip_score
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask.copy()
            
            # If no single flip improves, we might be stuck in a local minimum.
            # We can try a random perturbation (flip 2-3 random orbits) and restart local search?
            # Or just break and do a new restart.
            # For now, let's just break.
            
            # To avoid infinite loops or too long searches, we limit the number of successful flips.
            # But the while loop already handles time.
            
            # Add a small probability of a random perturbation to escape local minima
            if np.random.random() < 0.1:
                # Flip 3 random orbits
                for _ in range(3):
                    idx = np.random.randint(num_orbits)
                    mask[idx] = 1 - mask[idx]
                current_score = evaluate_cell_binary(mask)
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask.copy()
                # Don't set improved = True here, let the next iteration of the while loop
                # check for single-flip improvements from this new state.
                # Actually, if we perturb, we should continue the local search from there.
                # But if the perturbation made it worse, we might want to revert?
                # No, let's just continue. If it's worse, the local search might find its way back.
                # Or we can just break and restart.
                # Let's just break if we've been stuck for a while?
                # Simpler: just break the local search loop and do a new restart.
                break

    # --- 4. Finalize ---
    
    if best_mask is None:
        # Fallback: return a simple symmetric pattern
        mask = np.zeros(num_orbits, dtype=int)
        for idx, (i, j) in enumerate(orbit_keys):
            if i == 15 or i == 16 or j == 15 or j == 16:
                mask[idx] = 1
        best_mask = mask
    
    # Reconstruct the final binary cell
    final_cell = [[0 for _ in range(size)] for _ in range(size)]
    for idx, val in enumerate(best_mask):
        if val == 1:
            for i, j in orbit_indices[idx]:
                final_cell[i][j] = 1
    
    return final_cell
