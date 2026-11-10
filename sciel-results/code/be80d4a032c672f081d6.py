"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_fourier_coeffs_fast(cell, size, n_pw):
    """
    Compute Fourier coefficients of 1/eps for the cell.
    Returns a dictionary or a 2D array centered at (0,0) for easy indexing.
    """
    eps = np.where(np.array(cell, dtype=float) > 0.5, 12.0, 1.0)
    inv_eps = 1.0 / eps
    
    # FFT gives coefficients for G in 0..N-1.
    # We need G in -N//2 .. N//2.
    # fftshift moves 0 to center.
    fft_coeffs = np.fft.fftshift(np.fft.fft2(inv_eps)) / (size * size)
    
    # We only need the central (2*half+1) x (2*half+1) block
    half = n_pw // 2
    start = size // 2 - half
    end = size // 2 + half + 1
    # Extract the block. 
    # Note: fft_coeffs is complex.
    block = fft_coeffs[start:end, start:end]
    
    return block

def _band_structure_fast(cell, size, n_pw=13, n_bands=6):
    """
    Compute TM band structure at key points for fast evaluation.
    Uses a reduced set of k-points and PWs for speed.
    """
    coeffs_block = _compute_fourier_coeffs_fast(cell, size, n_pw)
    half = n_pw // 2
    N = n_pw * n_pw
    
    # Key k-points in the irreducible BZ (fractional coordinates)
    # Gamma (0,0), X (1,0), M (0.5,0.5)
    # Adding (0.5, 0) might help but let's start with 3 for speed
    k_points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, 0.5)
    ]
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    # Precompute G vectors
    Gx = np.arange(-half, half + 1)
    Gy = np.arange(-half, half + 1)
    GX, GY = np.meshgrid(Gx, Gy, indexing='ij')
    GX_flat = GX.flatten()
    GY_flat = GY.flatten()
    
    # Precompute G differences for matrix construction
    # C[i,j] = coeffs[(GX[i]-GX[j], GY[i]-GY[j])]
    # We can vectorize this if we map the differences to indices in coeffs_block
    # coeffs_block[i, j] corresponds to Gx = i - half, Gy = j - half
    # So for diff (dx, dy), index is (dx + half, dy + half)
    
    # Create difference matrices
    DGX = GX_flat[:, None] - GX_flat[None, :]
    DGY = GY_flat[:, None] - GY_flat[None, :]
    
    # Map differences to indices in coeffs_block
    # Valid range for dx, dy is [-half, half]
    # If out of range, coeff is 0.
    # We can use np.where to create a mask
    mask = (np.abs(DGX) <= half) & (np.abs(DGY) <= half)
    
    # Indices into coeffs_block
    # coeffs_block is (2*half+1, 2*half+1)
    # Index i = dx + half, j = dy + half
    idx_i = DGX + half
    idx_j = DGY + half
    
    C = np.zeros((N, N), dtype=complex)
    C[mask] = coeffs_block[idx_i[mask], idx_j[mask]]
    
    for kx, ky in k_points:
        # Build diagonal matrix D
        # D = (G+k)^2
        diag = ((GX_flat + kx)**2 + (GY_flat + ky)**2)
        
        # Solve D @ C^{-1} @ A = omega^2 A
        # We need C to be invertible. 
        # C is the matrix of 1/eps Fourier coeffs. It should be positive definite for physical eps.
        try:
            # Use solve instead of inv for numerical stability and speed
            # We want eigvals of D @ C^{-1}
            # This is equivalent to solving C @ X = D @ I ? No.
            # Standard approach: 
            # H = C^{-1/2} D C^{-1/2} is Hermitian.
            # But computing sqrt is expensive.
            # Alternative: Solve C @ Y = D @ I, then eigvals of Y? No, Y = C^{-1} D.
            # eigvals(C^{-1} D) are not necessarily real if C and D don't commute, but the generalized eigenproblem 
            # D A = w C A leads to C^{-1} D A = w A.
            # Since C is Hermitian PD and D is Real Diagonal (thus Hermitian), 
            # the matrix C^{-1} D is similar to C^{-1/2} D C^{-1/2} which is Hermitian.
            # So we can just compute C_inv_D = C^{-1} @ D (where D is diagonal)
            # Actually, D is diagonal, so C^{-1} @ D is just C^{-1} scaled by diag.
            
            # Let's compute C_inv = inv(C)
            # For N=169, inv is ~169^3 ~ 4.8M ops. Fast enough.
            C_inv = np.linalg.inv(C)
            M = C_inv @ np.diag(diag)
            
            # Enforce Hermiticity to remove numerical noise
            M = (M + M.conj().T) / 2.0
            
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= n_bands:
                # Filter out negative eigenvalues which might arise from numerical errors
                # though they should be positive for TM.
                valid_eigs = eigvals[eigvals > 1e-6]
                if len(valid_eigs) >= n_bands:
                    omega4 = np.sqrt(valid_eigs[3])
                    omega5 = np.sqrt(valid_eigs[4])
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
        except:
            continue
    
    if min_omega5 == float('inf') or max_omega4 == 0.0:
        return 0.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2
    if midgap <= 0:
        return 0.0
    return gap / midgap

def _generate_initial_cell(size, r=0.2, shape='circle'):
    """Generate a symmetric initial cell."""
    cell = [[0]*size for _ in range(size)]
    center = (size - 1) / 2.0
    
    for i in range(size):
        for j in range(size):
            x = (i - center) / size
            y = (j - center) / size
            dist = np.sqrt(x*x + y*y)
            
            if shape == 'circle':
                is_dielectric = dist < r
            elif shape == 'square':
                is_dielectric = abs(x) < r and abs(y) < r
            elif shape == 'superellipse':
                # n=4 superellipse approximates rounded square
                n = 4
                val = abs(x)**n + abs(y)**n
                is_dielectric = val < r**n
            else:
                is_dielectric = dist < r
                
            cell[i][j] = 1 if is_dielectric else 0
            
    return cell

def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # 1. Try a few different initial shapes
    candidates = []
    
    # Circle radii
    for r in [0.15, 0.2, 0.25, 0.3]:
        candidates.append(_generate_initial_cell(size, r, 'circle'))
        
    # Superellipses
    for r in [0.2, 0.25]:
        candidates.append(_generate_initial_cell(size, r, 'superellipse'))
        
    # Square
    for r in [0.15, 0.2]:
        candidates.append(_generate_initial_cell(size, r, 'square'))

    best_cell = None
    best_score = -1.0
    
    # Evaluate candidates
    for c in candidates:
        if time.time() > deadline:
            break
        score = _band_structure_fast(c, size, n_pw=13, n_bands=6)
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in c]
            
    if best_cell is None:
        # Fallback
        best_cell = _generate_initial_cell(size, 0.2, 'circle')
        best_score = _band_structure_fast(best_cell, size, n_pw=13, n_bands=6)
        
    # 2. Local Search
    cell = [row[:] for row in best_cell]
    current_score = best_score
    
    # Parameters for search
    n_pw_search = 13
    n_bands = 6
    
    # We will do random flips
    # To speed up, we can batch some checks or just run as many as fit.
    
    while time.time() < deadline - 5: # 5s safety buffer
        # Randomly select a pixel
        i = np.random.randint(0, size)
        j = np.random.randint(0, size)
        
        # Symmetric positions
        positions = [
            (i, j),
            (size-1-i, size-1-j),
            (j, i),
            (size-1-j, size-1-i)
        ]
        
        # Unique positions
        unique_positions = list(set(positions))
        
        # Get current value
        val = cell[i][j]
        new_val = 1 - val
        
        # Apply flip
        for pi, pj in unique_positions:
            cell[pi][pj] = new_val
            
        # Evaluate
        score = _band_structure_fast(cell, size, n_pw=n_pw_search, n_bands=n_bands)
        
        if score > current_score:
            current_score = score
            best_cell = [row[:] for row in cell]
        else:
            # Revert
            for pi, pj in unique_positions:
                cell[pi][pj] = val
                
    # Final verification with more k-points? 
    # The fast scorer uses 3 k-points. The evaluator uses 30.
    # The gap might be narrower or wider. 
    # If time permits, we could run a slightly more accurate check on the best cell.
    # But the fast scorer is a good proxy.
    
    return best_cell
# EVOLVE-BLOCK-END
