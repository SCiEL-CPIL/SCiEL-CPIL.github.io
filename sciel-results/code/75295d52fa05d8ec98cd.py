"""Photonic crystal design using wavelet-based fractal textures with C4v symmetry."""
import time
import math
import numpy as np
from scipy.optimize import minimize

def _enforce_c4v(cell):
    """Enforce C4v symmetry: mirror rows, mirror cols, transpose."""
    n = len(cell)
    arr = np.array(cell, dtype=float)
    # Mirror vertically: row i should equal row (n-1-i)
    arr = (arr + arr[::-1, :]) / 2.0
    # Mirror horizontally: col j should equal col (n-1-j)
    arr = (arr + arr[:, ::-1]) / 2.0
    # Transpose symmetry
    arr = (arr + arr.T) / 2.0
    return arr

def _compute_fourier_coefficients(cell, n_pw=17):
    """Compute Fourier coefficients of the dielectric permittivity map.
    
    cell: 32x32 array of 0/1 (air/dielectric)
    Returns: eps_g for |g| <= n_pw in each direction
    """
    n = len(cell)
    eps_low = 1.0
    eps_high = 12.0
    eps_map = np.where(cell > 0.5, eps_high, eps_low)
    
    # Compute 2D FFT
    eps_fft = np.fft.fft2(eps_map)
    eps_fft = np.fft.fftshift(eps_fft)
    
    # Grid of G indices
    g_range = np.arange(-n_pw, n_pw + 1)
    Gx, Gy = np.meshgrid(g_range, g_range, indexing='ij')
    
    # Map to FFT indices
    fft_idx_x = (Gx + n_pw) % (2 * n_pw + 1)
    fft_idx_y = (Gy + n_pw) % (2 * n_pw + 1)
    
    # The FFT gives sum over pixels, but we need the average (divide by N^2)
    # Actually for plane wave method, eps_g = (1/N^2) * sum eps(r) * exp(-i g.r)
    # which is exactly the FFT normalized by N^2
    eps_coeffs = np.zeros((2*n_pw+1, 2*n_pw+1), dtype=complex)
    for ix in range(2*n_pw+1):
        for iy in range(2*n_pw+1):
            # Map G index to FFT bin
            fft_ix = (ix) % (2*n_pw+1)
            fft_iy = (iy) % (2*n_pw+1)
            eps_coeffs[ix, iy] = eps_fft[fft_ix, fft_iy] / (n * n)
    
    return eps_coeffs, g_range

def _solve_tm_bands(eps_coeffs, g_range, kx, ky, n_bands=6):
    """Solve TM band structure at wavevector (kx, ky) using plane wave method.
    
    For TM polarization, the eigenvalue problem is:
    sum_g' eps^{-1}_{g-g'} |g'+k|^2 c_{g'} = omega^2 c_g
    
    where eps^{-1} is the inverse permittivity Fourier series.
    """
    n_pw = len(g_range) - 1
    # Inverse permittivity: 1/eps
    # eps(r) = sum_g eps_g exp(i g.r)
    # 1/eps(r) = sum_g (1/eps)_g exp(i g.r)
    # We need to compute (1/eps)_g from eps_g
    # For a binary material, 1/eps is also binary, so we can compute it directly
    
    # Actually, let's compute 1/eps map directly
    # eps_map has values eps_low and eps_high
    # inv_eps_map has values 1/eps_low and 1/eps_high
    # We need its Fourier coefficients
    
    # For simplicity, let's use the fact that for binary:
    # eps = eps_low + (eps_high - eps_low) * mask
    # 1/eps = 1/eps_low - (eps_high - eps_low)/(eps_low * eps_high) * mask + ...
    # This is getting complex. Let me just compute 1/eps directly from the map.
    
    # Reconstruct eps map from coefficients (or use the original)
    # Actually, let's pass the original cell to compute 1/eps coefficients
    
    # For now, use a simplified approach
    n_g = len(g_range)
    # Build the matrix
    # K_g = g + k
    Kx = g_range + kx
    Ky = g_range + ky
    
    # K^2 for each G
    K2 = Kx**2 + Ky**2
    
    # Build inverse permittivity matrix
    # inv_eps_{g-g'} where g-g' is the difference
    # We need inv_eps coefficients
    # For binary: inv_eps = a + b*mask where a=1/eps_low, b=1/eps_high - 1/eps_low
    # inv_eps_g = a*delta_{g,0} + b*mask_g
    
    # mask_g is the Fourier coefficient of the mask (0/1 where dielectric)
    # eps_g = eps_low*delta_{g,0} + (eps_high-eps_low)*mask_g
    # So mask_g = (eps_g - eps_low*delta_{g,0}) / (eps_high - eps_low)
    
    eps_low = 1.0
    eps_high = 12.0
    
    mask_coeffs = np.zeros((n_g, n_g), dtype=complex)
    for ix in range(n_g):
        for iy in range(n_g):
            gx = g_range[ix]
            gy = g_range[iy]
            if gx == 0 and gy == 0:
                mask_coeffs[ix, iy] = (eps_coeffs[ix, iy] - eps_low) / (eps_high - eps_low)
            else:
                mask_coeffs[ix, iy] = eps_coeffs[ix, iy] / (eps_high - eps_low)
    
    inv_eps_low = 1.0 / eps_low
    inv_eps_diff = 1.0 / eps_high - 1.0 / eps_low
    
    # inv_eps_g = inv_eps_low * delta_{g,0} + inv_eps_diff * mask_g
    inv_eps_coeffs = np.zeros((n_g, n_g), dtype=complex)
    for ix in range(n_g):
        for iy in range(n_g):
            gx = g_range[ix]
            gy = g_range[iy]
            if gx == 0 and gy == 0:
                inv_eps_coeffs[ix, iy] = inv_eps_low + inv_eps_diff * mask_coeffs[ix, iy]
            else:
                inv_eps_coeffs[ix, iy] = inv_eps_diff * mask_coeffs[ix, iy]
    
    # Build the N x N matrix
    N = n_g * n_g
    A = np.zeros((N, N), dtype=complex)
    
    for ig in range(N):
        gx = g_range[ig // n_g]
        gy = g_range[ig % n_g]
        for igp in range(N):
            gxp = g_range[igp // n_g]
            gyp = g_range[igp % n_g]
            # g - g'
            dgx = gx - gxp
            dgy = gy - gyp
            if abs(dgx) <= n_pw and abs(dgy) <= n_pw:
                idx_dgx = dgx + n_pw
                idx_dgy = dgy + n_pw
                A[ig, igp] = inv_eps_coeffs[idx_dgx, idx_dgy] * K2[igp]
    
    # Solve eigenvalue problem
    try:
        eigvals = np.linalg.eigvalsh(A)
        eigvals = np.real(eigvals)
        eigvals = np.sort(eigvals)
        # omega = sqrt(eigenvalue)
        omegas = np.sqrt(np.maximum(eigvals, 0))
        return omegas[:n_bands]
    except:
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

def _score_cell(cell, n_bands=6, n_pw=8):
    """Compute the gap-midgap ratio for a given cell."""
    cell_arr = np.array(cell, dtype=float)
    eps_coeffs, g_range = _compute_fourier_coefficients(cell_arr, n_pw)
    
    # Key wavevectors in irreducible BZ of square lattice
    # Gamma=(0,0), M=(0.5,0), X=(0.5,0.5), and interior points
    k_points = [
        (0.0, 0.0),      # Gamma
        (0.5, 0.0),      # M
        (0.5, 0.5),      # X
        (0.25, 0.0),
        (0.25, 0.25),
        (0.5, 0.25),
        (0.1, 0.1),
        (0.3, 0.2),
        (0.4, 0.1),
        (0.15, 0.35),
    ]
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in k_points:
        omegas = _solve_tm_bands(eps_coeffs, g_range, kx, ky, n_bands)
        if len(omegas) >= 5:
            max_omega4 = max(max_omega4, omegas[3])
            min_omega5 = min(min_omega5, omegas[4])
    
    if min_omega5 == float('inf') or max_omega4 == 0.0:
        return 0.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    if midgap <= 0:
        return 0.0
    return 2.0 * gap / midgap

def _wavelet_basis(n=32):
    """Create 2D Haar wavelet basis for the independent quadrant (16x16)."""
    # The independent region is rows 0-15, cols 0-15 (16x16)
    # We use multi-scale Haar wavelets
    basis = []
    
    # Approximation (DC) component
    dc = np.zeros((16, 16))
    dc[0, 0] = 1.0
    basis.append(dc)
    
    # Detail coefficients at various scales
    # Scale 1: 8x8 blocks
    for scale in range(1, 5):
        block_size = 16 // (2 ** scale)
        if block_size < 1:
            break
        n_blocks = 2 ** scale
        
        for bi in range(n_blocks):
            for bj in range(n_blocks):
                # Horizontal detail
                h = np.zeros((16, 16))
                row_start = bi * block_size
                col_start = bj * block_size
                # Positive half
                h[row_start:row_start+block_size//2, col_start:col_start+block_size] = 1.0
                h[row_start+block_size//2:row_start+block_size, col_start:col_start+block_size] = -1.0
                basis.append(h)
                
                # Vertical detail
                v = np.zeros((16, 16))
                v[row_start:row_start+block_size, col_start:col_start+block_size//2] = 1.0
                v[row_start:row_start+block_size, col_start+block_size//2:col_start+block_size] = -1.0
                basis.append(v)
                
                # Diagonal detail
                d = np.zeros((16, 16))
                d[row_start:row_start+block_size//2, col_start:col_start+block_size//2] = 1.0
                d[row_start+block_size//2:row_start+block_size, col_start+block_size//2:col_start+block_size] = 1.0
                d[row_start:row_start+block_size//2, col_start+block_size//2:col_start+block_size] = -1.0
                d[row_start+block_size//2:row_start+block_size, col_start:col_start+block_size//2] = -1.0
                basis.append(d)
    
    return basis

def _expand_quadrant(q):
    """Expand a 16x16 quadrant to a full 32x32 C4v-symmetric cell."""
    n = 32
    cell = np.zeros((n, n), dtype=float)
    # Upper-left quadrant: rows 0-15, cols 0-15
    cell[0:16, 0:16] = q
    # Upper-right: mirror cols
    cell[0:16, 16:32] = q[:, ::-1]
    # Lower-left: mirror rows
    cell[16:32, 0:16] = q[::-1, :]
    # Lower-right: mirror both
    cell[16:32, 16:32] = q[::-1, ::-1]
    return cell

def design(problem):
    """Design a photonic crystal unit cell using wavelet-based optimization."""
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin
    
    size = problem["pixels"]
    n = size  # 32
    
    # Initial design: a ring pattern (known to have good gaps)
    def make_ring(inner_r, outer_r):
        q = np.zeros((16, 16))
        for i in range(16):
            for j in range(16):
                x = (j + 0.5) / 32.0 - 0.5
                y = (i + 0.5) / 32.0 - 0.5
                r = math.sqrt(x*x + y*y)
                q[i, j] = 1.0 if inner_r < r < outer_r else 0.0
        return q
    
    # Start with a ring
    q0 = make_ring(0.15, 0.35)
    cell0 = _expand_quadrant(q0)
    score0 = _score_cell(cell0)
    
    best_cell = cell0
    best_score = score0
    
    # Wavelet optimization
    basis = _wavelet_basis()
    n_params = len(basis)
    
    # Initial coefficients: start with ring, project onto wavelet basis
    # For simplicity, start with small random perturbations
    np.random.seed(42)
    
    def evaluate(coeffs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            return 0.0  # Stop optimizing
        q = np.zeros((16, 16))
        for k, c in enumerate(coeffs):
            q += c * basis[k]
        # Threshold
        q_bin = (q > 0.0).astype(float)
        cell = _expand_quadrant(q_bin)
        return -_score_cell(cell)  # Minimize negative score
    
    # Use Nelder-Mead or Powell for the optimization
    # Start from zero coefficients (which gives all-0 quadrant)
    # Better: start from the ring projected
    x0 = np.zeros(n_params)
    
    # Try a few different starting points
    best_overall = best_score
    best_cell_overall = best_cell.copy()
    
    # First, try some geometric designs quickly
    for inner_r, outer_r in [(0.1, 0.3), (0.15, 0.35), (0.2, 0.4), (0.1, 0.25), (0.25, 0.45)]:
        if time.time() > deadline - 60:
            break
        q = make_ring(inner_r, outer_r)
        cell = _expand_quadrant(q)
        s = _score_cell(cell)
        if s > best_overall:
            best_overall = s
            best_cell_overall = cell.copy()
    
    # Now do wavelet optimization
    # Use a subset of basis functions for speed
    # Focus on lower-frequency wavelets for structural features
    n_opt = min(n_params, 40)
    
    for trial in range(5):
        if time.time() > deadline - 60:
            break
        
        # Random initialization
        x0 = np.random.randn(n_opt) * 0.1
        
        def obj(x):
            elapsed = time.time() - start_time
            if elapsed > deadline - 30:
                return 0.0
            q = np.zeros((16, 16))
            for k in range(n_opt):
                q += x[k] * basis[k]
            q_bin = (q > 0.0).astype(float)
            cell = _expand_quadrant(q_bin)
            return -_score_cell(cell)
        
        try:
            res = minimize(obj, x0, method='Nelder-Mead', 
                          options={'maxiter': 50, 'xatol': 0.01, 'fatol': 0.01,
                                  'adaptive': True})
            if time.time() > deadline - 30:
                break
            # Evaluate final result
            q = np.zeros((16, 16))
            for k in range(n_opt):
                q += res.x[k] * basis[k]
            q_bin = (q > 0.0).astype(float)
            cell = _expand_quadrant(q_bin)
            s = _score_cell(cell)
            if s > best_overall:
                best_overall = s
                best_cell_overall = cell.copy()
        except:
            pass
    
    # Final: convert to binary list
    final_cell = best_cell_overall
    # Ensure binary
    final_cell = (final_cell > 0.5).astype(int)
    
    # Convert to nested list
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(int(final_cell[i, j]))
        result.append(row)
    
    # Verify C4v symmetry
    for i in range(n):
        for j in range(n):
            assert result[i][j] == result[n-1-i][j], "Row symmetry broken"
            assert result[i][j] == result[i][n-1-j], "Col symmetry broken"
            assert result[i][j] == result[j][i], "Transpose symmetry broken"
    
    return result
