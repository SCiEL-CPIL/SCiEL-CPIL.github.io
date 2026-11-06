"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _apply_symmetry(cell):
    """Enforce 4-fold symmetry: flip rows, flip cols, transpose. Use minimum to preserve dielectric."""
    cell = np.minimum(cell, cell[::-1, :].copy())
    cell = np.minimum(cell, cell[:, ::-1].copy())
    cell = np.minimum(cell, cell.T.copy())
    return cell

def _make_four_rod_cell(radius=6.0, centers=None):
    """Create a 32x32 cell with 4 circular rods."""
    size = 32
    if centers is None:
        centers = [(8, 8), (8, 23), (23, 8), (23, 23)]
    r2 = radius * radius
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            for (cx, cy) in centers:
                dx = i - cx
                dy = j - cy
                if dx * dx + dy * dy <= r2:
                    cell[i, j] = 1
                    break
    return cell

def _compute_eps_ft_fast(cell, n_pw=17):
    """
    Compute the Fourier coefficients of epsilon for plane-wave expansion.
    For TM mode, we need 1/epsilon in the eigenvalue equation.
    We compute eps[k] = sum_{ij} eps_ij * exp(-i k . r_ij) / N
    where k = (m, n) * 2pi/L, L=32.
    
    Returns array of shape (n_pw, n_pw) for indices -8..8 in each direction.
    """
    size = cell.shape[0]
    indices = np.arange(-n_pw // 2, n_pw - n_pw // 2)
    # indices: -8, -7, ..., 7, 8 for n_pw=17
    # Actually for n_pw=17, range is -8 to 8 inclusive
    eps_ft = np.zeros((n_pw, n_pw), dtype=complex)
    
    # Precompute exponential factors
    # r_i = i * (2pi/size), so exp(-i * m * 2pi * i / size)
    # We can use FFT for efficiency
    # eps_ft[m, n] = (1/N) * sum_{i,j} eps[i,j] * exp(-i * (m*2pi*i + n*2pi*j)/size)
    # This is essentially the 2D DFT of the pixel map, normalized.
    
    # Use np.fft.fft2
    # The DFT gives F[m, n] = sum_{i,j} f[i,j] * exp(-i * 2pi * (m*i + n*j)/N)
    # We want indices from -8 to 8.
    # np.fft.fftshift(np.fft.fft2(cell)) gives the shifted spectrum.
    # But we need to be careful about normalization and indexing.
    
    # Let's use the direct FFT approach:
    # F = fft2(cell) / (size*size)
    # F[m, n] corresponds to frequency (m, n) where m, n are in 0..size-1
    # We need frequencies -8..8.
    
    F = np.fft.fft2(cell) / (size * size)
    
    # Map indices -8..8 to FFT indices
    # FFT index for frequency f: if f >= 0, idx = f; if f < 0, idx = size + f
    for im, m in enumerate(indices):
        for in_, n in enumerate(indices):
            idx_m = m % size
            idx_n = n % size
            eps_ft[im, in_] = F[idx_m, idx_n]
    
    return eps_ft

def _tm_eigenvalues_fast(cell, n_pw=17, kx=0.0, ky=0.0, bands=6):
    """
    Solve the TM eigenvalue problem for a given k-point.
    Eigenvalue equation: sum_n [ (k+G_m)^2 / eps_{mn} ] c_n = omega^2 c_m
    For TM: (k+G)^2 / eps = omega^2
    Matrix element: A_{mn} = (k+G_m)^2 * (1/eps)_{m-n}
    Wait, the standard form is:
    sum_n (1/eps)_{m-n} (k+G_m)^2 c_n = omega^2 c_m
    Or: sum_n eps_{m-n}^{-1} (k+G_m)^2 c_n = omega^2 c_m
    
    Actually, the plane-wave expansion for TM is:
    H = sum_n c_n exp(i (k+G_n) . r)
    (k+G_m)^2 / eps = omega^2
    eps = sum_l eps_l exp(i G_l . r)
    1/eps = sum_l (1/eps)_l exp(i G_l . r)
    
    The eigenvalue equation is:
    sum_n [(1/eps)_{m-n} (k+G_m)^2] c_n = omega^2 c_m
    
    So the matrix M_{mn} = (1/eps)_{m-n} * (k+G_m)^2
    And we solve M c = omega^2 c
    
    But we need 1/eps Fourier coefficients.
    We have eps Fourier coefficients. We can compute 1/eps in real space and then FFT.
    """
    size = cell.shape[0]
    indices = np.arange(-n_pw // 2, n_pw - n_pw // 2)
    
    # Compute 1/eps in real space
    eps_real = np.where(cell > 0, 12.0, 1.0)
    inv_eps_real = 1.0 / eps_real
    
    # Compute Fourier coefficients of 1/eps
    inv_eps_ft = np.fft.fft2(inv_eps_real) / (size * size)
    
    # Map to our index range
    inv_eps = np.zeros((n_pw, n_pw), dtype=complex)
    for im, m in enumerate(indices):
        for in_, n in enumerate(indices):
            idx_m = m % size
            idx_n = n % size
            inv_eps[im, in_] = inv_eps_ft[idx_m, idx_n]
    
    # Build the eigenvalue matrix
    # M_{mn} = inv_eps[m-n] * (k+G_m)^2
    # G_m = (indices[m] * 2pi/size, indices[n] * 2pi/size)
    # (k+G_m)^2 = (kx + indices[m]*2pi/size)^2 + (ky + indices[n]*2pi/size)^2
    
    Gx = indices * 2 * np.pi / size
    Gy = indices * 2 * np.pi / size
    
    # Build Gx, Gy arrays
    Gx2d, Gy2d = np.meshgrid(Gx, Gy, indexing='ij')
    
    # (k+G)^2
    kg2 = (kx + Gx2d)**2 + (ky + Gy2d)**2
    
    # Build matrix M
    M = np.zeros((n_pw*n_pw, n_pw*n_pw), dtype=complex)
    for im in range(n_pw):
        for in_ in range(n_pw):
            row = im * n_pw + in_
            kg2_val = kg2[im, in_]
            for jm in range(n_pw):
                for jn in range(n_pw):
                    col = jm * n_pw + jn
                    # inv_eps[m-n] where m-n means (im-jm, in_-jn)
                    dm = im - jm
                    dn = in_ - jn
                    # Map to index in inv_eps array
                    # inv_eps is indexed by indices[im] - indices[jm] etc.
                    # Actually, inv_eps[im, in_] corresponds to frequency (indices[im], indices[in_])
                    # We need inv_eps at frequency (indices[im]-indices[jm], indices[in_]-indices[jn])
                    # This is inv_eps[dm_idx, dn_idx] where dm_idx is the index in indices array
                    # such that indices[dm_idx] = indices[im] - indices[jm]
                    # Since indices are consecutive integers, dm_idx = im - jm
                    # But we need to handle wrap-around if out of range
                    if -n_pw//2 <= dm < n_pw - n_pw//2 and -n_pw//2 <= dn < n_pw - n_pw//2:
                        # Find the index in the indices array
                        dm_idx = dm + n_pw//2
                        dn_idx = dn + n_pw//2
                        if 0 <= dm_idx < n_pw and 0 <= dn_idx < n_pw:
                            M[row, col] += inv_eps[dm_idx, dn_idx] * kg2_val
                    # For out-of-range, the coefficient is 0 (truncation)
    
    # Solve eigenvalue problem
    # M is Hermitian (should be real symmetric if inv_eps is real and even)
    # Let's take the real part since it should be real
    M_real = np.real(M)
    
    # Ensure symmetry
    M_real = (M_real + M_real.T) / 2.0
    
    # Get eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(M_real)
    except:
        eigvals = np.zeros(bands)
    
    # Return the first 'bands' eigenvalues (omega^2)
    return eigvals[:bands]

def _quick_gap_estimate(cell, n_pw=17):
    """
    Estimate the gap between bands 4 and 5 for a given cell.
    Evaluates at several k-points in the irreducible BZ.
    """
    # Irreducible BZ for square lattice: 0 <= kx <= ky, 0 <= kx, ky <= pi/a
    # In units of 2pi/L where L=32, the BZ boundary is at k = 0.5 * 2pi/L * 32 = pi
    # So kx, ky range from 0 to 1 in units of 2pi/L (where 1 = pi)
    # Actually, let's use k in units of 2pi/L. The BZ goes from -0.5 to 0.5 in each direction.
    # Irreducible wedge: 0 <= kx <= ky <= 0.5
    
    k_points = [
        (0.0, 0.0),      # Gamma
        (0.25, 0.0),     # M/2
        (0.5, 0.0),      # M
        (0.5, 0.25),     
        (0.5, 0.5),      # K
        (0.25, 0.25),    # center of wedge
        (0.1, 0.0),
        (0.2, 0.0),
        (0.3, 0.0),
        (0.4, 0.0),
        (0.45, 0.0),
        (0.0, 0.25),
        (0.1, 0.1),
        (0.2, 0.2),
        (0.3, 0.3),
        (0.4, 0.4),
        (0.45, 0.45),
        (0.3, 0.4),
        (0.2, 0.4),
        (0.1, 0.4),
    ]
    
    # Convert k to actual values: k_actual = k * 2pi/size
    size = cell.shape[0]
    k_factor = 2 * np.pi / size
    
    max_omega4 = 0.0
    min_omega5 = np.inf
    
    for kx, ky in k_points:
        kx_actual = kx * k_factor
        ky_actual = ky * k_factor
        eigvals = _tm_eigenvalues_fast(cell, n_pw=n_pw, kx=kx_actual, ky=ky_actual, bands=6)
        if len(eigvals) >= 5:
            omega4 = np.sqrt(max(eigvals[3], 0))
            omega5 = np.sqrt(max(eigvals[4], 0))
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
    
    if min_omega5 == np.inf or max_omega4 == 0:
        return -1.0
    
    gap_ratio = 2 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
    return gap_ratio

def _random_symmetric_cell(size=32):
    """Generate a random symmetric binary cell."""
    # Generate a quarter cell (16x16) and mirror
    quarter = np.random.randint(0, 2, (size//2, size//2))
    # Mirror horizontally
    half = np.hstack([quarter, quarter[:, ::-1]])
    # Mirror vertically
    cell = np.vstack([half, half[::-1, :]])
    return cell

def _mutate_cell(cell, mutation_rate=0.1):
    """Randomly flip some pixels while preserving symmetry."""
    cell_mut = cell.copy()
    size = cell.shape[0]
    # Flip random pixels in the quarter cell
    for _ in range(int(mutation_rate * size * size // 4)):
        i = np.random.randint(0, size//2)
        j = np.random.randint(0, size//2)
        val = 1 - cell_mut[i, j]
        # Apply to all symmetric positions
        cell_mut[i, j] = val
        cell_mut[i, size-1-j] = val
        cell_mut[size-1-i, j] = val
        cell_mut[size-1-i, size-1-j] = val
    return cell_mut

def design(problem):
    """
    Design a 32x32 unit cell with a 2x2 basis of dielectric rods.
    This structure enhances the (2,0) and (0,2) Fourier components,
    which are critical for opening higher-order band gaps in square lattices.
    """
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for evaluator
    
    # Start with the known good four-rod design
    best_cell = _make_four_rod_cell(radius=6.0)
    best_score = _quick_gap_estimate(best_cell, n_pw=17)
    
    # Local search with random restarts
    current_cell = best_cell.copy()
    current_score = best_score
    
    iterations = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        iterations += 1
        
        # Mutate the current cell
        mutation_rate = 0.05 if no_improve_count < 20 else 0.15
        candidate = _mutate_cell(current_cell, mutation_rate=mutation_rate)
        
        # Evaluate
        candidate_score = _quick_gap_estimate(candidate, n_pw=17)
        
        if candidate_score > current_score:
            current_cell = candidate
            current_score = candidate_score
            no_improve_count = 0
            
            if candidate_score > best_score:
                best_cell = candidate.copy()
                best_score = candidate_score
        else:
            no_improve_count += 1
        
        # Random restart if stuck
        if no_improve_count > max_no_improve:
            # Try a random restart
            if time.time() > deadline - 20.0:
                break
            current_cell = _random_symmetric_cell(size)
            current_score = _quick_gap_estimate(current_cell, n_pw=17)
            no_improve_count = 0
        
        # Also try some structured variations
        if iterations % 10 == 0:
            # Try different radii
            for r in [5.0, 5.5, 6.5, 7.0]:
                candidate = _make_four_rod_cell(radius=r)
                score = _quick_gap_estimate(candidate, n_pw=17)
                if score > best_score:
                    best_cell = candidate
                    best_score = score
    
    return best_cell.tolist()
# EVOLVE-BLOCK-END
