"""
Photonic crystal design for maximizing TM band gap between 4th and 5th bands.
Uses a 2x2 supercell arrangement with dielectric rods in air background.
Implements local search with a fast band-structure solver.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh

def _apply_symmetry(pixels):
    """Apply square symmetry to a 16x32 independent pixel array to produce a 32x32 cell."""
    size = 32
    cell = np.zeros((size, size), dtype=int)
    
    # Independent region: top-left triangular area where i < j and i < 16
    # Symmetry operations: 
    # 1. Horizontal flip: (i, j) -> (size-1-i, j)
    # 2. Vertical flip: (i, j) -> (i, size-1-j)  
    # 3. Diagonal: (i, j) -> (j, i)
    # 4. Anti-diagonal: (i, j) -> (size-1-j, size-1-i)
    
    # First, fill the primary quadrant (i in 0..15, j in 0..31)
    # Actually, let's think more carefully about the symmetry group D4
    
    # The cell must satisfy:
    # cell[i][j] == cell[size-1-i][j]  (horizontal reflection)
    # cell[i][j] == cell[i][size-1-j]  (vertical reflection)
    # cell[i][j] == cell[j][i]          (transpose/diagonal)
    # cell[i][j] == cell[size-1-j][size-1-i]  (anti-diagonal)
    
    # These imply that the fundamental domain is i in [0, 15], j in [i, 31-i]
    # Let's verify: if i > 15, then by horizontal reflection, cell[i][j] = cell[31-i][j]
    # where 31-i < 16. If j < i, then by diagonal, cell[i][j] = cell[j][i] where j < i.
    # If j > 31-i, then by vertical reflection, cell[i][j] = cell[i][31-j] where 31-j < i.
    
    # So the independent pixels are at positions (i, j) where 0 <= i <= 15 and i <= j <= 31-i
    
    for i in range(16):
        for j in range(i, 32 - i):
            val = pixels[i, j]
            cell[i, j] = val
            cell[i, 31 - j] = val
            cell[31 - i, j] = val
            cell[31 - i, 31 - j] = val
            cell[j, i] = val
            cell[31 - j, i] = val
            cell[j, 31 - i] = val
            cell[31 - j, 31 - i] = val
    
    return cell

def _compute_epsilon_fourier(cell, n_plane_waves):
    """
    Compute the Fourier coefficients of 1/epsilon for the given cell.
    cell: 32x32 array of 0s and 1s (0=air, 1=dielectric)
    n_plane_waves: number of plane waves in each dimension (odd)
    """
    size = 32
    eps_low = 1.0
    eps_high = 12.0
    
    # epsilon matrix
    eps = np.where(cell == 1, eps_high, eps_low)
    
    # 1/epsilon
    inv_eps = 1.0 / eps
    
    # Compute FFT to get Fourier coefficients
    # The Fourier coefficients are:
    # g_{mn} = (1/N^2) * sum_{i,j} inv_eps[i,j] * exp(-2*pi*i*(m*i/N + n*j/N))
    # This is just the 2D FFT scaled appropriately
    
    # We need plane waves with indices from -N/2 to N/2-1 (for odd N)
    # n_plane_waves should be odd, e.g., 17 means indices -8 to 8
    
    N = n_plane_waves
    if N % 2 == 0:
        N += 1  # Make it odd
    
    half_N = N // 2
    
    # Compute full 2D FFT
    fft_2d = np.fft.fft2(inv_eps)
    fft_2d = np.fft.fftshift(fft_2d)  # Center zero frequency
    
    # Extract the relevant Fourier coefficients
    # The indices in the shifted FFT correspond to frequencies from -size/2 to size/2-1
    # We need frequencies from -half_N to half_N-1... wait, for N=17, we need -8 to 8
    
    # Actually, let's think about this differently.
    # The plane wave basis is exp(i * k * r) where k = 2*pi*(m*a_x + n*a_y) for integer m, n
    # The Fourier coefficients of 1/epsilon are:
    # g_{mn} = (1/a^2) * integral (1/epsilon(r)) exp(-i * 2*pi*(m*x/a + n*y/a)) d^2r
    # In discrete form:
    # g_{mn} = (1/N_pixel^2) * sum_{i,j} (1/epsilon[i,j]) * exp(-i * 2*pi*(m*i/N_pixel + n*j/N_pixel))
    
    # This is exactly (1/N_pixel^2) * FFT[inv_eps] at indices (m, n)
    # But we need to handle the index wrapping properly
    
    # For the plane wave expansion, we use G vectors: G_{mn} = 2*pi*(m, n)/a
    # where m, n are integers. The basis functions are exp(i*G_{mn}*r)
    
    # The matrix element <exp(i*G_{m'n'}*r) | 1/eps | exp(i*G_{mn}*r)>
    # = g_{m'-m, n'-n}
    
    # So we need g_{pq} for p, q in the range determined by the plane wave indices
    
    # Let's use a simpler approach: compute the convolution directly
    
    # The eigenvalue problem is:
    # sum_{n'} g_{m-m', n-n'} * |k + G_{m'|}^2 * c_{n'} = omega^2 * c_{m}
    
    # Where g_{pq} are the Fourier coefficients of 1/epsilon
    
    # Let's compute g_{pq} for all needed (p, q)
    
    max_g = 4  # Maximum G index to use (adjustable)
    
    g_coeffs = {}
    for p in range(-max_g, max_g + 1):
        for q in range(-max_g, max_g + 1):
            # g_{pq} = (1/N^2) * sum_{i,j} inv_eps[i,j] * exp(-i*2*pi*(p*i/N + q*j/N))
            phase_p = np.exp(-2j * np.pi * p * np.arange(size) / size)
            phase_q = np.exp(-2j * np.pi * q * np.arange(size) / size)
            
            # Outer product for 2D
            phase_2d = np.outer(phase_p, phase_q)
            g_coeffs[(p, q)] = np.sum(inv_eps * phase_2d) / (size * size)
    
    return g_coeffs

def _solve_band_structure(cell, k_points, n_bands=5, max_g=4):
    """
    Solve the band structure for the given cell at the specified k-points.
    Returns arrays of omega values for each band at each k-point.
    """
    size = 32
    eps_low = 1.0
    eps_high = 12.0
    
    # Compute Fourier coefficients
    g_coeffs = _compute_epsilon_fourier(cell, n_bands)
    
    # Define G vectors
    G_list = []
    for m in range(-max_g, max_g + 1):
        for n in range(-max_g, max_g + 1):
            G_list.append((m, n))
    
    N_pw = len(G_list)
    
    # Store omega values
    omegas = []
    
    for k in k_points:
        kx, ky = k
        # Build the eigenvalue matrix
        # A_{mn} = |k + G_m|^2 * delta_{mn} + g_{G_m - G_n} * |k + G_n|^2
        # Wait, let me get this right.
        
        # The TM mode equation is:
        # sum_{n'} (k + G_n')^2 * g_{m - m', n - n'} * c_{m', n'} = omega^2 * c_{m, n}
        
        # Where g_{pq} are Fourier coefficients of 1/epsilon
        
        # Let's build the matrix
        A = np.zeros((N_pw, N_pw), dtype=complex)
        
        for idx_m, (m, n) in enumerate(G_list):
            kmx = kx + 2 * np.pi * m
            kmy = ky + 2 * np.pi * n
            km_sq = kmx * kmx + kmy * kmy
            
            for idx_n, (mp, np_) in enumerate(G_list):
                # G_m - G_n = (m - mp, n - np_)
                dp = m - mp
                dq = n - np_
                
                g_val = g_coeffs.get((dp, dq), 0.0)
                
                # k + G_n
                knx = kx + 2 * np.pi * mp
                kny = ky + 2 * np.pi * np_
                kn_sq = knx * knx + kny * kny
                
                A[idx_m, idx_n] += g_val * kn_sq
        
        # Solve eigenvalue problem
        # A * c = omega^2 * c
        # A is Hermitian (should be real symmetric if g is real and symmetric)
        
        # Take the real part if imaginary part is small
        if np.max(np.abs(A.imag)) < 1e-10 * np.max(np.abs(A.real)):
            A_real = A.real
            eigenvalues, _ = eigh(A_real)
            omega_vals = np.sqrt(np.maximum(eigenvalues, 0))
        else:
            eigenvalues, _ = np.linalg.eigh(A)
            omega_vals = np.sqrt(np.maximum(np.real(eigenvalues), 0))
        
        # Sort and take the first n_bands
        omega_vals = np.sort(omega_vals)[:n_bands]
        omegas.append(omega_vals)
    
    return np.array(omegas)

def _compute_gap_ratio(cell, k_points, n_bands=5):
    """
    Compute the gap-midgap ratio between bands 4 and 5 (indices 3 and 4).
    """
    omegas = _solve_band_structure(cell, k_points, n_bands)
    
    # omega[band_idx, k_point_idx]
    # Band 4 is index 3, Band 5 is index 4
    
    max_omega_4 = np.max(omegas[3])
    min_omega_5 = np.min(omegas[4])
    
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    
    if midgap == 0:
        return 0.0
    
    return 2.0 * gap / midgap

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell for TM band gap.
    """
    size = 32
    eps_low = 1.0
    eps_high = 12.0
    
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin
    
    # K-points in the irreducible Brillouin zone
    # The BZ is a square from -pi to pi in both x and y (in units of 2*pi/a)
    # Irreducible wedge: 0 <= kx <= ky <= pi
    # Key points: Gamma(0,0), X(pi,0), M(pi,pi), and points along X-M and Gamma-M
    
    k_points = []
    # Gamma
    k_points.append((0.0, 0.0))
    # X
    k_points.append((np.pi, 0.0))
    # M
    k_points.append((np.pi, np.pi))
    
    # Along Gamma to X: (t, 0) for t in [0, pi]
    for t in np.linspace(0.1, 0.9, 5):
        k_points.append((t * np.pi, 0.0))
    
    # Along X to M: (pi, t) for t in [0, pi]
    for t in np.linspace(0.1, 0.9, 5):
        k_points.append((np.pi, t * np.pi))
    
    # Along Gamma to M: (t, t) for t in [0, pi]
    for t in np.linspace(0.1, 0.9, 5):
        k_points.append((t * np.pi, t * np.pi))
    
    # Interior points
    k_points.append((0.5 * np.pi, 0.25 * np.pi))
    k_points.append((0.75 * np.pi, 0.25 * np.pi))
    k_points.append((0.5 * np.pi, 0.5 * np.pi))
    k_points.append((0.75 * np.pi, 0.5 * np.pi))
    k_points.append((0.25 * np.pi, 0.25 * np.pi))
    
    # Initialize with a 2x2 rod pattern
    # Rods centered at (8, 8), (8, 24), (24, 8), (24, 24) in a 32x32 cell
    # This is a common configuration for higher-order gaps
    
    # Independent pixels: i in 0..15, j in i..31-i
    best_indep = np.zeros((16, 32), dtype=int)
    
    # Set up initial rod pattern
    # Rod centers in the full 32x32 cell
    rod_centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
    rod_radius = 3
    
    for cx, cy in rod_centers:
        for i in range(16):
            for j in range(i, 32 - i):
                # Check if pixel (i, j) is within radius of any rod center
                # Due to symmetry, we only need to check the primary center
                # but let's check all for safety
                for rcx, rcy in rod_centers:
                    dist_sq = (i - rcx) ** 2 + (j - rcy) ** 2
                    if dist_sq < rod_radius ** 2:
                        best_indep[i, j] = 1
                        break
    
    # Also add a central feature
    best_indep[15, 15] = 1
    best_indep[15, 16] = 1
    
    best_cell = _apply_symmetry(best_indep)
    best_score = _compute_gap_ratio(best_cell, k_points)
    
    # Local search: flip random independent pixels
    np.random.seed(42)
    
    num_steps = 300
    for step in range(num_steps):
        if time.time() > deadline:
            break
        
        # Pick a random independent pixel
        i = np.random.randint(0, 16)
        j_min = i
        j_max = 31 - i
        if j_min > j_max:
            continue
        j = np.random.randint(j_min, j_max + 1)
        
        # Flip it
        old_val = best_indep[i, j]
        best_indep[i, j] = 1 - old_val
        
        new_cell = _apply_symmetry(best_indep)
        new_score = _compute_gap_ratio(new_cell, k_points)
        
        if new_score > best_score:
            best_score = new_score
            # Accept the change
        else:
            # Revert
            best_indep[i, j] = old_val
    
    return best_cell.tolist()

# EVOLVE-BLOCK-END
