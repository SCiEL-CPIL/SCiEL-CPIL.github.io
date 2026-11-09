"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell for a 2D square-lattice photonic crystal
    that maximizes the gap between the 4th and 5th TM bands.
    
    Strategy: Use a local search starting from known good configurations for
    photonic crystals with large band gaps. The key insight is that for TM
    polarization in a square lattice, the optimal design often involves
    a specific pattern of dielectric and air that creates a large band gap.
    
    We use a time-budgeted local search with symmetry constraints.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    eps_low = problem.get("eps_low", 1)
    
    start_time = time.time()
    time_limit = 220  # Leave 10s safety margin before 230s kill
    
    # Precompute G vectors for 289 plane waves (17x17)
    N_PW = 8
    G_range = np.arange(-N_PW, N_PW + 1)
    Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    
    # Define the 30 k-points in the irreducible Brillouin zone
    # The irreducible BZ for square lattice is the region where 0 <= kx <= ky <= pi
    # We'll sample points along the boundary and interior
    k_points = []
    # Along kx = 0 (Gamma to X): ky from 0 to pi
    for i in range(8):
        ky = (i + 1) * np.pi / 8
        k_points.append((0.0, ky))
    # Along ky = kx (Gamma to M): kx = ky from 0 to pi/sqrt(2)
    for i in range(1, 8):
        k = (i) * np.pi / (8 * np.sqrt(2))
        k_points.append((k, k))
    # Along ky = pi - kx (X to M): kx from 0 to pi/2, ky from pi to pi/2
    for i in range(1, 8):
        kx = (i) * np.pi / 16
        ky = np.pi - kx
        k_points.append((kx, ky))
    # Interior points
    interior_k = [
        (0.1, 0.3), (0.2, 0.4), (0.3, 0.5),
        (0.15, 0.35), (0.25, 0.45), (0.35, 0.55),
        (0.1, 0.5), (0.2, 0.6), (0.3, 0.7),
        (0.4, 0.6), (0.5, 0.7), (0.15, 0.45),
        (0.25, 0.55), (0.35, 0.65), (0.45, 0.75),
    ]
    k_points.extend(interior_k)
    k_points = np.array(k_points)
    
    def symmetrize(cell):
        """Ensure 4-fold symmetry: cell == cell[::-1] rows, cols, and transpose."""
        cell = np.array(cell, dtype=float)
        # Average over all 4 symmetry operations
        # 1. Identity
        sym1 = cell.copy()
        # 2. Flip rows (vertical mirror)
        sym2 = cell[::-1, :].copy()
        # 3. Flip columns (horizontal mirror)
        sym3 = cell[:, ::-1].copy()
        # 4. Flip both (180 deg rotation)
        sym4 = cell[::-1, ::-1].copy()
        # 5. Transpose
        sym5 = cell.T.copy()
        # 6. Transpose then flip rows
        sym6 = cell.T[::-1, :].copy()
        # 7. Transpose then flip columns
        sym7 = cell.T[:, ::-1].copy()
        # 8. Transpose then flip both
        sym8 = cell.T[::-1, ::-1].copy()
        
        avg = (sym1 + sym2 + sym3 + sym4 + sym5 + sym6 + sym7 + sym8) / 8.0
        # Threshold to binary
        return (avg > 0.5).astype(int)
    
    def compute_fourier_coefficients(cell):
        """Compute exact Fourier coefficients of the pixel map."""
        cell = np.array(cell, dtype=float)
        eps = eps_low + (eps_high - eps_low) * cell
        # Fourier coefficients: eps_G = (1/N^2) sum_{i,j} eps_{i,j} exp(-i G . r_{i,j})
        # r_{i,j} = ((i+0.5)/N, (j+0.5)/N) in units of lattice constant
        # G = (Gx, Gy) * 2*pi
        coeff = np.zeros(len(G_vectors), dtype=complex)
        for idx, (Gx_val, Gy_val) in enumerate(G_vectors):
            # G . r = 2*pi * (Gx_val * (i+0.5)/N + Gy_val * (j+0.5)/N)
            phase_x = 2 * np.pi * Gx_val * (np.arange(size) + 0.5) / size
            phase_y = 2 * np.pi * Gy_val * (np.arange(size) + 0.5) / size
            # Use outer product for 2D phase
            phase_2d = np.outer(phase_x, np.ones(size)) + np.outer(np.ones(size), phase_y)
            # Actually, let's do it properly
            X, Y = np.meshgrid(phase_x, phase_y, indexing='ij')
            phase = X + Y
            coeff[idx] = np.sum(eps * np.exp(-1j * phase)) / (size * size)
        return coeff
    
    def evaluate_cell(cell):
        """Compute the gap-midgap ratio for the given cell."""
        cell = np.array(cell, dtype=int)
        # Ensure symmetry
        cell = symmetrize(cell)
        
        # Compute Fourier coefficients
        eps_coeff = compute_fourier_coefficients(cell)
        
        # For each k-point, solve the eigenvalue problem
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            # k is in units of 2*pi/a, so (k+G)^2 where G is integer
            # H_{GG'} = eps^{-1}_{G-G'} * (k+G')^2
            # Actually, for TM: H_{GG'} = (k+G)^2 * eps^{-1}_{G-G'}
            # Wait, the standard formulation:
            # det[(k+G)^2 delta_{GG'} - eps^{-1}_{G-G'}] omega^2 = 0
            # So the matrix is: M_{GG'} = (k+G)^2 delta_{GG'} - eps^{-1}_{G-G'}
            # We want eigenvalues omega^2
            
            # Build the matrix
            # First, compute (k+G)^2 for each G
            # k = (kx, ky) in units where reciprocal lattice = 2*pi
            # So k+G = (kx + Gx, ky + Gy) in units of 2*pi/a
            # (k+G)^2 = (kx + Gx)^2 + (ky + Gy)^2 (times (2*pi/a)^2, but this cancels)
            
            kG_sq = np.zeros(len(G_vectors))
            for idx, (Gx_val, Gy_val) in enumerate(G_vectors):
                kG_sq[idx] = (kx + Gx_val)**2 + (ky + Gy_val)**2
            
            # Build eps^{-1} matrix
            # eps^{-1}_{G-G'} = sum_{H} eps^{-1}_H delta_{G, G'+H}
            # We need the Fourier coefficients of 1/eps
            # Compute 1/eps in real space
            eps_real = eps_low + (eps_high - eps_low) * cell
            inv_eps_real = 1.0 / eps_real
            
            # Fourier coefficients of 1/eps
            inv_eps_coeff = np.zeros(len(G_vectors), dtype=complex)
            for idx, (Gx_val, Gy_val) in enumerate(G_vectors):
                phase_x = 2 * np.pi * Gx_val * (np.arange(size) + 0.5) / size
                phase_y = 2 * np.pi * Gy_val * (np.arange(size) + 0.5) / size
                X, Y = np.meshgrid(phase_x, phase_y, indexing='ij')
                phase = X + Y
                inv_eps_coeff[idx] = np.sum(inv_eps_real * np.exp(-1j * phase)) / (size * size)
            
            # Build the matrix M
            # M_{GG'} = (k+G)^2 delta_{GG'} - inv_eps_coeff_{G-G'}
            M = np.zeros((len(G_vectors), len(G_vectors)), dtype=complex)
            for i in range(len(G_vectors)):
                for j in range(len(G_vectors)):
                    diff = G_vectors[i] - G_vectors[j]
                    # Find index of diff in G_vectors
                    # Since G_vectors is a grid, we can compute the index
                    diff_idx_x = int(diff[0]) + N_PW
                    diff_idx_y = int(diff[1]) + N_PW
                    if 0 <= diff_idx_x < 2*N_PW+1 and 0 <= diff_idx_y < 2*N_PW+1:
                        diff_coeff_idx = diff_idx_x * (2*N_PW+1) + diff_idx_y
                        M[i, j] = kG_sq[i] * (1 if i == j else 0) - inv_eps_coeff[diff_coeff_idx]
                    else:
                        M[i, j] = 0
            
            # Make M Hermitian (it should be, but numerical errors may occur)
            M = (M + M.conj().T) / 2
            
            # Solve for eigenvalues
            try:
                eigvals = np.linalg.eigvalsh(M)
                eigvals = np.real(eigvals)
                # Sort eigenvalues
                eigvals = np.sort(eigvals)
                # We want the 4th and 5th bands (0-indexed: 3 and 4)
                if len(eigvals) >= 5:
                    omega4 = np.sqrt(max(eigvals[3], 0))
                    omega5 = np.sqrt(max(eigvals[4], 0))
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                continue
        
        if min_omega5 == np.inf or max_omega4 == 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        if midgap <= 0:
            return -1.0
        
        return 2 * gap / midgap
    
    # Initial design: start with a known good pattern
    # For square lattice TM band gap, a common good starting point is
    # a pattern with 4-fold symmetry that creates a large gap
    
    # Try a few initial patterns and pick the best
    initial_patterns = []
    
    # Pattern 1: Center cross
    cell1 = np.zeros((size, size), dtype=int)
    center = size // 2
    for i in range(size):
        cell1[i, center-1:center+1] = 1
        cell1[center-1:center+1, i] = 1
    initial_patterns.append(cell1)
    
    # Pattern 2: Diagonal cross
    cell2 = np.zeros((size, size), dtype=int)
    for i in range(size):
        cell2[i, i] = 1
        cell2[i, size-1-i] = 1
    initial_patterns.append(cell2)
    
    # Pattern 3: Ring
    cell3 = np.zeros((size, size), dtype=int)
    cx, cy = (size-1)/2, (size-1)/2
    for i in range(size):
        for j in range(size):
            r = np.sqrt((i-cx)**2 + (j-cy)**2)
            if 8 < r < 12:
                cell3[i, j] = 1
    initial_patterns.append(cell3)
    
    # Pattern 4: Checkerboard-like with symmetry
    cell4 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            # Create a pattern with 4-fold symmetry
            if (i % 8 < 4) == (j % 8 < 4):
                cell4[i, j] = 1
    initial_patterns.append(cell4)
    
    # Evaluate initial patterns
    best_cell = None
    best_score = -np.inf
    for pattern in initial_patterns:
        pattern = symmetrize(pattern)
        score = evaluate_cell(pattern)
        if score > best_score:
            best_score = score
            best_cell = pattern.copy()
    
    # Local search: flip individual pixels while maintaining symmetry
    # For each pixel, if flipping it (and its symmetric counterparts) improves the score, do it
    
    # Precompute symmetry groups for each pixel
    symmetry_groups = {}
    for i in range(size):
        for j in range(size):
            # All symmetric equivalents of (i, j)
            equivs = [(i, j), (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j),
                      (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)]
            # Remove duplicates
            equivs = list(set(equvs))
            symmetry_groups[(i, j)] = equivs
    
    # Local search with time budget
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Randomly sample pixels to try flipping
        num_trials = 200
        for _ in range(num_trials):
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random pixel
            i = np.random.randint(size)
            j = np.random.randint(size)
            
            # Get all symmetric equivalents
            equivs = symmetry_groups[(i, j)]
            
            # Try flipping all of them
            new_cell = best_cell.copy()
            for (ei, ej) in equivs:
                new_cell[ei, ej] = 1 - new_cell[ei, ej]
            
            # Ensure symmetry
            new_cell = symmetrize(new_cell)
            
            # Evaluate
            new_score = evaluate_cell(new_cell)
            
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        if not improved:
            no_improve_count += 10
        
        if no_improve_count > max_no_improve:
            # Try a different strategy: larger perturbations
            if time.time() - start_time > time_limit:
                break
            
            # Flip a random block
            block_size = 4
            i = np.random.randint(0, size - block_size + 1)
            j = np.random.randint(0, size - block_size + 1)
            
            new_cell = best_cell.copy()
            for di in range(block_size):
                for dj in range(block_size):
                    new_cell[i+di, j+dj] = 1 - new_cell[i+di, j+dj]
            
            # Apply symmetry
            new_cell = symmetrize(new_cell)
            
            new_score = evaluate_cell(new_cell)
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell
                no_improve_count = 0
            else:
                no_improve_count += 5
            
            if no_improve_count > max_no_improve * 2:
                # Reset and try a new random start
                np.random.seed(int(time.time() * 1000) % 10000)
                new_cell = np.random.randint(0, 2, (size, size))
                new_cell = symmetrize(new_cell)
                new_score = evaluate_cell(new_cell)
                if new_score > best_score:
                    best_score = new_score
                    best_cell = new_cell
                no_improve_count = 0
    
    # Convert to nested list
    result = best_cell.tolist()
    
    # Final symmetry check
    result = [[int(result[i][j]) for j in range(size)] for i in range(size)]
    
    return result

# EVOLVE-BLOCK-END
