<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _enforce_symmetry(cell, size):
    """Enforce 4-fold rotational and mirror symmetry about center."""
    sym = np.zeros((size, size), dtype=int)
    for i in range(size // 2 + 1):
        for j in range(size // 2 + 1):
            val = cell[i][j]
            # 4-fold rotational symmetry + mirror symmetry
            sym[i][j] = val
            sym[size - 1 - i][j] = val
            sym[i][size - 1 - j] = val
            sym[size - 1 - i][size - 1 - j] = val
    return sym

def _quick_score(cell, size, eps_high):
    """Fast approximate score using reduced plane-wave basis for local optimization."""
    # Use a smaller PW basis for speed (this is an approximation)
    Npw = 9  # 3x3 PW basis for quick evaluation
    half = (Npw - 1) // 2
    
    # Compute Fourier coefficients of epsilon
    eps = np.array(cell, dtype=float)
    eps_inv = np.zeros_like(eps)
    eps_inv[eps > 0.5] = 1.0 / eps_high
    eps_inv[eps <= 0.5] = 1.0
    
    # Get unique symmetric pixel positions
    unique_positions = set()
    for i in range(size // 2 + 1):
        for j in range(size // 2 + 1):
            unique_positions.add((i, j))
    
    # Precompute G vectors for the small basis
    G_vectors = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            G_vectors.append((gx, gy))
    N = len(G_vectors)
    
    # Compute epsilon Fourier coefficients
    eps_F = np.zeros((N, N), dtype=complex)
    for idx_g, (gx, gy) in enumerate(G_vectors):
        for idx_h, (hx, hy) in enumerate(G_vectors):
            dgx = hx - gx
            dgy = hy - gy
            # Sum over pixels
            s = 0.0
            for (i, j) in unique_positions:
                # Account for symmetry: each unique pixel represents 4 pixels (except edges)
                weight = 1
                if i == 0 or i == size // 2 or j == 0 or j == size // 2:
                    # Edge pixels have fewer symmetric copies
                    if i == 0 and j == 0:
                        weight = 1
                    elif i == 0 or j == 0:
                        weight = 2
                    else:
                        weight = 4
                else:
                    weight = 4
                
                phase = np.exp(2j * np.pi * (dgx * (i + 0.5) / size + dgy * (j + 0.5) / size))
                s += weight * eps_inv[i][j] * phase
            s /= (size * size)
            eps_F[idx_g][idx_h] = s
    
    # Define wave vectors in irreducible BZ
    # Square lattice: BZ is [-pi, pi] x [-pi, pi], irreducible: 0 <= ky <= kx, kx <= pi
    k_points = []
    # Gamma point
    k_points.append((0.0, 0.0))
    # X point
    k_points.append((1.0, 0.0))
    # M point
    k_points.append((1.0, 1.0))
    # Midpoint Gamma-X
    k_points.append((0.5, 0.0))
    # Midpoint Gamma-M
    k_points.append((0.5, 0.5))
    # Midpoint X-M
    k_points.append((1.0, 0.5))
    
    min_omega5 = float('inf')
    max_omega4 = 0.0
    
    for (kx, ky) in k_points:
        # Build the TM eigenvalue problem
        # For TM: eigenvalue is omega^2, matrix is (G+k)^2 * inv(eps_{G-G'})
        # Actually for TM polarization, the eigenvalue problem is:
        # sum_{G'} (G+K)^2 / eps(G-G') * c_{G'} = omega^2 * c_G
        # where (G+K)^2 is the scalar (Gx+Kx)^2 + (Gy+Ky)^2
        
        Kx = 2 * np.pi * kx
        Ky = 2 * np.pi * ky
        
        A = np.zeros((N, N), dtype=complex)
        for idx_g, (gx, gy) in enumerate(G_vectors):
            kgx = gx + kx
            kgy = gy + ky
            kg2 = kgx * kgx + kgy * kgy
            for idx_h, (hx, hy) in enumerate(G_vectors):
                A[idx_g][idx_h] = kg2 * eps_F[idx_h][idx_g]
        
        # Solve eigenvalue problem
        try:
            eigvals = np.linalg.eigvalsh(A)
            eigvals = np.sort(eigvals)
            # Take the 5th and 4th eigenvalues (0-indexed: index 3 and 4)
            if len(eigvals) >= 5:
                omega4 = np.sqrt(eigvals[3])
                omega5 = np.sqrt(eigvals[4])
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            pass
    
    if max_omega4 > 0 and min_omega5 > 0:
        gap = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return gap
    return -1.0

def design(problem):
    """Local search optimization for photonic crystal unit cell."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin before 120s limit
    
    # Initial design: circle with radius ~0.2
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.2 ** 2:
                cell[i][j] = 1
    
    # Get unique symmetric positions
    unique_positions = []
    for i in range(size // 2 + 1):
        for j in range(size // 2 + 1):
            unique_positions.append((i, j))
    
    best_cell = cell.copy()
    best_score = _quick_score(best_cell, size, 12)
    
    # Local search: flip individual symmetric pixels
    random.seed(42)
    pos_idx = list(range(len(unique_positions)))
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Pick a random unique position
        idx = random.choice(pos_idx)
        i, j = unique_positions[idx]
        
        # Try flipping this pixel
        old_val = cell[i][j]
        cell[i][j] = 1 - old_val
        
        # Enforce symmetry
        cell = _enforce_symmetry(cell, size)
        
        # Score
        score = _quick_score(cell, size, 12)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
        else:
            # Revert
            cell[i][j] = old_val
            cell = _enforce_symmetry(cell, size)
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # Convert back to nested list
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    return result
# EVOLVE-BLOCK-END
=======
"""Physics-driven inverse design: optimize Fourier coefficients of dielectric map."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _enforce_c4v(cell, size):
    """Enforce C4v symmetry (4-fold rotation + mirrors) about cell center."""
    sym = np.zeros((size, size), dtype=float)
    for i in range(size // 2 + 1):
        for j in range(size // 2 + 1):
            val = cell[i][j]
            sym[i][j] = val
            sym[size - 1 - i][j] = val
            sym[i][size - 1 - j] = val
            sym[size - 1 - i][size - 1 - j] = val
    return sym

def _compute_eps_fourier(cell, size, n_g):
    """Compute Fourier coefficients of 1/eps for plane-wave expansion.
    
    Args:
        cell: size x size array of 0/1 (1=dielectric)
        size: cell size
        n_g: number of G-vectors in each direction (odd)
    
    Returns:
        eps_F: (n_g, n_g) array of Fourier coefficients of 1/eps
    """
    eps = np.array(cell, dtype=float)
    eps_inv = np.ones((size, size), dtype=float)
    eps_inv[eps > 0.5] = 1.0 / 12.0
    
    half = (n_g - 1) // 2
    G_indices = np.arange(-half, half + 1)
    
    # Precompute pixel positions
    i_pix = np.arange(size)
    j_pix = np.arange(size)
    I, J = np.meshgrid(i_pix, j_pix, indexing='ij')
    
    eps_F = np.zeros((n_g, n_g), dtype=complex)
    for idx_g, gx in enumerate(G_indices):
        for idx_h, hx in enumerate(G_indices):
            dg = hx - gx
            # Compute sum over all pixels: eps_inv[i,j] * exp(2*pi*i*dg/size)
            # Vectorized
            phase_x = np.exp(2j * np.pi * dg * I / size)
            # For 2D, we need phase_x * phase_y
            # But we need to do this for both x and y directions
            # Let's compute it properly
            s = 0.0
            for gy in G_indices:
                for hy in G_indices:
                    if (gy == G_indices[idx_g] and hy == G_indices[idx_h]):
                        dgx = hy - gy
                        # Actually let me redo this more carefully
                        pass
            
            # Simpler: iterate over G differences
            pass
    
    # Let me rewrite this more cleanly
    eps_F = np.zeros((n_g, n_g), dtype=complex)
    for idx_g in range(n_g):
        gx = G_indices[idx_g]
        for idx_h in range(n_g):
            hx = G_indices[idx_h]
            dgx = hx - gx
            # We need to sum over all (i,j) pixels
            # eps_inv[i,j] * exp(2*pi*i*(dgx/size))
            # But this is 2D, so we need both x and y components
            # Actually the Fourier coefficient is:
            # eps_F[g,h] = (1/N^2) * sum_{i,j} eps_inv[i,j] * exp(2*pi*i*(G_h - G_g)/N)
            # where G_h - G_g is a 2D vector
            
            # Let me reconsider: for a 2D lattice, G = (m, n) where m,n are integers
            # The Fourier coefficient is:
            # eps_F[(m1,n1),(m2,n2)] = (1/N^2) sum_{i,j} eps_inv[i,j] * exp(2*pi*i*((m2-m1)/N) + 2*pi*j*((n2-n1)/N))
            
            # For our case, G_vectors are (gx, gy) pairs
            # Let me restructure to use 2D G vectors
            
            pass
    
    # Complete rewrite with proper 2D G vectors
    G_vectors = []
    for gx in G_indices:
        for gy in G_indices:
            G_vectors.append((gx, gy))
    n_pw = len(G_vectors)
    
    eps_F = np.zeros((n_pw, n_pw), dtype=complex)
    for idx_g, (gx, gy) in enumerate(G_vectors):
        for idx_h, (hx, hy) in enumerate(G_vectors):
            dgx = hx - gx
            dgy = hy - gy
            # Compute sum over all pixels
            phase = np.exp(2j * np.pi * (dgx * I / size + dgy * J / size))
            s = np.sum(eps_inv * phase)
            eps_F[idx_g][idx_h] = s / (size * size)
    
    return eps_F, G_vectors, n_pw

def _solve_bands_fast(eps_F, G_vectors, n_pw, k_points, n_bands=5):
    """Fast band structure calculation using reduced PW basis.
    
    Args:
        eps_F: Fourier coefficients of 1/eps
        G_vectors: list of (gx, gy) G-vectors
        n_pw: number of plane waves
        k_points: list of (kx, ky) wave vectors (in units of 1, not 2*pi)
        n_bands: number of bands to compute
    
    Returns:
        min_omega5: minimum of 5th band frequency
        max_omega4: maximum of 4th band frequency
    """
    min_omega5 = float('inf')
    max_omega4 = 0.0
    
    for (kx, ky) in k_points:
        # Build TM eigenvalue matrix
        # A[g,h] = (G_g + K)^2 * eps_F[h,g]
        A = np.zeros((n_pw, n_pw), dtype=complex)
        for idx_g, (gx, gy) in enumerate(G_vectors):
            kgx = gx + kx
            kgy = gy + ky
            kg2 = kgx * kgx + kgy * kgy
            for idx_h in range(n_pw):
                A[idx_g][idx_h] = kg2 * eps_F[idx_h][idx_g]
        
        # Make Hermitian (should already be, but ensure numerical stability)
        A = (A + A.conj().T) / 2.0
        
        try:
            eigvals = np.linalg.eigvalsh(A)
            eigvals = np.sort(np.real(eigvals))
            if len(eigvals) >= n_bands:
                omega4 = np.sqrt(max(eigvals[n_bands - 2], 0))
                omega5 = np.sqrt(max(eigvals[n_bands - 1], 0))
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            pass
    
    if max_omega4 > 0 and min_omega5 > 0:
        gap = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return gap, min_omega5, max_omega4
    return -1.0, 0, 0

def _cell_to_fourier(cell, size, n_g):
    """Convert cell to Fourier coefficient representation with C4v symmetry."""
    eps_F, G_vectors, n_pw = _compute_eps_fourier(cell, size, n_g)
    return eps_F, G_vectors, n_pw

def _fourier_to_cell(eps_F, G_vectors, n_pw, size):
    """Convert Fourier coefficients back to pixel cell."""
    # Inverse Fourier transform
    cell_float = np.zeros((size, size), dtype=float)
    I, J = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    
    for idx_g, (gx, gy) in enumerate(G_vectors):
        for idx_h, (hx, hy) in enumerate(G_vectors):
            dgx = hx - gx
            dgy = hy - gy
            phase = np.exp(-2j * np.pi * (dgx * I / size + dgy * J / size))
            cell_float += eps_F[idx_g][idx_h] * phase
    
    # cell_float should be proportional to eps_inv
    # Threshold to get binary cell
    eps_inv_est = np.real(cell_float)
    # Where eps_inv is close to 1/12, it's dielectric; where close to 1, it's air
    # eps_inv = 1/12 for dielectric, 1 for air
    cell = np.zeros((size, size), dtype=int)
    cell[eps_inv_est < 0.5] = 1  # 1/12 < 0.5, so dielectric
    cell[eps_inv_est >= 0.5] = 0  # 1 >= 0.5, so air
    
    # Enforce C4v symmetry
    cell = _enforce_c4v(cell, size).astype(int)
    
    return cell

def design(problem):
    """Inverse design: optimize Fourier coefficients of dielectric map."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    # Use a moderate PW basis for the optimizer
    n_g = 7  # 7x7 = 49 plane waves
    half = (n_g - 1) // 2
    
    # Define k-points in irreducible BZ
    k_points = []
    # Gamma
    k_points.append((0.0, 0.0))
    # X
    k_points.append((1.0, 0.0))
    # M
    k_points.append((1.0, 1.0))
    # Midpoints
    k_points.append((0.5, 0.0))
    k_points.append((0.5, 0.5))
    k_points.append((1.0, 0.5))
    # Additional points for better sampling
    k_points.append((0.25, 0.0))
    k_points.append((0.75, 0.0))
    k_points.append((0.25, 0.25))
    k_points.append((0.75, 0.75))
    k_points.append((1.0, 0.25))
    k_points.append((1.0, 0.75))
    k_points.append((0.25, 0.5))
    k_points.append((0.5, 0.25))
    k_points.append((0.5, 0.75))
    k_points.append((0.75, 0.5))
    
    # Initial design: circle
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.2 ** 2:
                cell[i][j] = 1
    
    best_cell = cell.copy()
    best_score, _, _ = _solve_bands_fast(*_cell_to_fourier(best_cell, size, n_g), k_points)
    
    # Inverse design: perturb Fourier coefficients and convert back to cell
    random_seed = 42
    rng = np.random.RandomState(random_seed)
    
    # Get initial Fourier representation
    eps_F, G_vectors, n_pw = _cell_to_fourier(best_cell, size, n_g)
    
    # Optimize by perturbing Fourier coefficients
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        if time.time() > deadline - 10.0:
            break
        
        # Perturb Fourier coefficients
        eps_F_new = eps_F.copy()
        for idx_g in range(n_pw):
            for idx_h in range(n_pw):
                # Only perturb upper triangle to maintain Hermitian property
                if idx_h >= idx_g:
                    delta = rng.normal(0, step_size)
                    eps_F_new[idx_g][idx_h] += delta
                    if idx_g != idx_h:
                        eps_F_new[idx_h][idx_g] += np.conj(delta)
        
        # Convert back to cell
        cell_new = _fourier_to_cell(eps_F_new, G_vectors, n_pw, size)
        
        # Score
        eps_F_check, G_vectors_check, n_pw_check = _cell_to_fourier(cell_new, size, n_g)
        score, _, _ = _solve_bands_fast(eps_F_check, G_vectors_check, n_pw_check, k_points)
        
        if score > best_score:
            best_score = score
            best_cell = cell_new.copy()
            eps_F = eps_F_check
            no_improve_count = 0
            # Reduce step size on improvement
            step_size *= 0.95
        else:
            no_improve_count += 1
            # Increase step size on no improvement
            step_size = min(step_size * 1.05, 0.2)
        
        # Periodic check
        if time.time() > deadline - 5.0:
            break
    
    # Final local search refinement
    if time.time() < deadline - 5.0:
        # Get unique symmetric positions
        unique_positions = []
        for i in range(size // 2 + 1):
            for j in range(size // 2 + 1):
                unique_positions.append((i, j))
        
        local_rng = np.random.RandomState(43)
        while time.time() < deadline - 2.0:
            idx = local_rng.randint(len(unique_positions))
            i, j = unique_positions[idx]
            
            old_val = best_cell[i][j]
            best_cell[i][j] = 1 - old_val
            best_cell = _enforce_c4v(best_cell, size).astype(int)
            
            eps_F_local, G_vectors_local, n_pw_local = _cell_to_fourier(best_cell, size, n_g)
            score_local, _, _ = _solve_bands_fast(eps_F_local, G_vectors_local, n_pw_local, k_points)
            
            if score_local > best_score:
                best_score = score_local
            else:
                best_cell[i][j] = old_val
                best_cell = _enforce_c4v(best_cell, size).astype(int)
    
    # Convert to nested list
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    return result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
