"""Rewritten design program: Fourier-harmonic parameterization with local search."""
import time
import math
import numpy as np
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START

def _compute_few_band_gap(cell, max_pws=11, num_bands=6, k_points=8):
    """
    Compute the gap ratio between bands 4 and 5 (0-indexed bands 3 and 4)
    using a reduced plane-wave method for speed.
    """
    size = len(cell)
    eps_low = 1.0
    eps_high = 12.0
    
    # Discretize epsilon on the pixel grid
    eps_grid = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if cell[i][j] == 1:
                eps_grid[i, j] = eps_high
            else:
                eps_grid[i, j] = eps_low
    
    # Compute Fourier coefficients of 1/epsilon for the supercell
    # We use DFT on the pixel grid
    inv_eps = 1.0 / eps_grid
    # Fourier coefficients
    G_max = max_pws  # max |Gx| or |Gy|
    G_vals = list(range(-G_max, G_max + 1))
    num_G = len(G_vals)
    
    # Build list of G vectors
    G_list = []
    for gx in G_vals:
        for gy in G_vals:
            G_list.append((gx, gy))
    
    num_Gs = len(G_list)
    
    # Compute Fourier coefficients of inv_eps
    # Using the pixel grid as a discrete approximation
    # f(G) = (1/N^2) * sum_{i,j} f(x_i,y_j) * exp(-i*(Gx*x_i + Gy*y_j))
    # For pixel (i,j), x_i = (i+0.5)/size, y_j = (j+0.5)/size
    
    fourier_inv_eps = np.zeros((num_Gs, num_Gs), dtype=complex)
    
    # Precompute pixel positions
    x_pixels = np.array([(j + 0.5) / size for j in range(size)])
    y_pixels = np.array([(i + 0.5) / size for i in range(size)])
    
    # For efficiency, compute DFT-like transform
    # Actually, let's use a simpler approach: compute the Fourier coefficients
    # using the fact that the grid is uniform.
    
    # For each G pair (Gx, Gy), the Fourier coefficient is:
    # f(Gx, Gy) = (1/size^2) * sum_i sum_j inv_eps[i,j] * exp(-2*pi*i*(Gx*(j+0.5)/size + Gy*(i+0.5)/size))
    #            = (1/size^2) * sum_i sum_j inv_eps[i,j] * exp(-2*pi*i*(Gx*j/size + Gy*i/size)) * exp(-2*pi*i*(Gx+Gy)/(2*size))
    
    # This is essentially a 2D DFT of inv_eps with a phase factor.
    
    # Let's use numpy's FFT for efficiency
    # First, zero-pad if needed (size=32 is a power of 2, so FFT is efficient)
    
    inv_eps_fft = np.fft.fft2(inv_eps) / (size * size)
    
    # Map G indices to FFT indices
    # FFT index k corresponds to G = k if k <= size/2, else G = k - size
    def g_to_fft_idx(g):
        if g >= 0:
            return g % size
        else:
            return size + g % size
    
    # Build the matrix A where A[m,n] = inv_eps(G_m - G_n)
    # For TM polarization: (k-G_m)^T * M * (k-G_n) where M has 1/eps in it
    # Actually for TM: the eigenvalue problem is (k - G_m) . (1/eps) . (k - G_n) * E_n = omega^2 E_m
    # where the dot product is a matrix multiplication with the 1/eps Fourier matrix.
    
    # Let's use a simplified approach: build the matrix and diagonalize
    # For each k-point, build the matrix:
    # H[m,n] = (k_x - Gx_m)(k_x - Gx_n) * inv_eps(G_m - G_n) + 
    #         (k_y - Gy_m)(k_y - Gy_n) * inv_eps(G_m - G_n)
    #         + [(k_x - Gx_m)(k_y - Gy_n) + (k_y - Gy_m)(k_x - Gx_n)] * inv_eps(G_m - G_n)
    
    # Wait, the TM eigenvalue equation is:
    # sum_n [(k - G_m) . (k - G_n)] * (1/eps)(G_m - G_n) * E_n = omega^2 E_m
    # where (k-G_m).(k-G_n) = (kx-Gx_m)(kx-Gx_n) + (ky-Gy_m)(ky-Gy_n)
    
    # Let's precompute the G vectors
    Gx_arr = np.array([g[0] for g in G_list])
    Gy_arr = np.array([g[1] for g in G_list])
    
    # Precompute inv_eps Fourier matrix
    inv_eps_mat = np.zeros((num_Gs, num_Gs), dtype=complex)
    for i, (gx1, gy1) in enumerate(G_list):
        for j, (gx2, gy2) in enumerate(G_list):
            dgx = gx1 - gx2
            dgy = gy1 - gy2
            # Get from FFT
            idx_x = g_to_fft_idx(dgx)
            idx_y = g_to_fft_idx(dgy)
            inv_eps_mat[i, j] = inv_eps_fft[idx_y, idx_x]  # Note: FFT axis order
            # Actually fft2 uses row=i (y), col=j (x), so inv_eps_fft[i,j] = f(Gx=j, Gy=i)
            # Let me fix: inv_eps_fft[gy_fft, gx_fft]
    
    # Fix the indexing
    inv_eps_mat = np.zeros((num_Gs, num_Gs), dtype=complex)
    for i, (gx1, gy1) in enumerate(G_list):
        for j, (gx2, gy2) in enumerate(G_list):
            dgx = gx1 - gx2
            dgy = gy1 - gy2
            idx_x = dgx % size
            idx_y = dgy % size
            inv_eps_mat[i, j] = inv_eps_fft[idx_y, idx_x]
    
    # Now sweep over k-points in the irreducible BZ
    # Irreducible BZ of square lattice: 0 <= ky <= kx, 0 <= kx <= 1 (in units of 2*pi/a)
    # Actually, let's use kx, ky in [0, 1] with 0 <= ky <= kx for the 1/8 sector
    # Or use the standard irreducible zone: triangle from (0,0) to (0.5,0) to (0.5,0.5)
    
    k_points = []
    # Add key points
    k_points.append((0.0, 0.0))      # Gamma
    k_points.append((0.5, 0.0))      # M
    k_points.append((0.5, 0.5))      # X
    k_points.append((0.25, 0.0))
    k_points.append((0.25, 0.25))
    k_points.append((0.125, 0.125))
    k_points.append((0.375, 0.125))
    k_points.append((0.375, 0.375))
    
    min_band5 = float('inf')
    max_band4 = 0.0
    
    for kx, ky in k_points:
        # Build the Hamiltonian matrix
        H = np.zeros((num_Gs, num_Gs), dtype=complex)
        for m in range(num_Gs):
            kmx = kx - Gx_arr[m]
            kmy = ky - Gy_arr[m]
            for n in range(num_Gs):
                knx = kx - Gx_arr[n]
                kny = ky - Gy_arr[n]
                dot = kmx * knx + kmy * kny
                H[m, n] = dot * inv_eps_mat[m, n]
        
        # Diagonalize
        eigenvalues = np.linalg.eigvalsh(H)
        eigenvalues = np.sort(eigenvalues.real)
        
        # Take the first num_bands bands
        if len(eigenvalues) >= num_bands:
            # Bands are 0-indexed, we want bands 3 and 4 (4th and 5th)
            band4 = eigenvalues[3]
            band5 = eigenvalues[4]
            if band4 > 0 and band5 > 0:
                min_band5 = min(min_band5, band5)
                max_band4 = max(max_band4, band4)
    
    if max_band4 <= 0:
        return -1.0
    
    # Gap ratio: 2*(min_omega5 - max_omega4) / (min_omega5 + max_omega4)
    # This is in terms of omega^2, so we need to be careful
    # Actually the score uses omega, not omega^2
    # Let's compute in terms of omega
    if min_band5 <= 0 or max_band4 <= 0:
        return -1.0
    
    omega_min5 = math.sqrt(min_band5)
    omega_max4 = math.sqrt(max_band4)
    
    gap_ratio = 2.0 * (omega_min5 - omega_max4) / (omega_min5 + omega_max4)
    return gap_ratio


def _fourier_basis_cell(a20, a02, a22, phase20, phase02, phase22, size=32):
    """
    Generate a symmetric 32x32 cell from low-order Fourier harmonics.
    Modes: (2,0), (0,2), (2,2)
    """
    cell = np.zeros((size, size), dtype=int)
    x = np.array([(j + 0.5) / size for j in range(size)])
    y = np.array([(i + 0.5) / size for i in range(size)])
    
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # Fourier modes
    # cos(2*pi*(2*X + phase20)) * a20
    # cos(2*pi*(2*Y + phase02)) * a02
    # cos(2*pi*(2*X + 2*Y + phase22)) * a22
    
    field = (a20 * np.cos(2 * np.pi * (2 * X + phase20)) +
             a02 * np.cos(2 * np.pi * (2 * Y + phase02)) +
             a22 * np.cos(2 * np.pi * (2 * X + 2 * Y + phase22)))
    
    # Threshold to binary
    threshold = 0.0
    cell = (field > threshold).astype(int)
    
    # Enforce symmetry: cell == cell[::-1], cell == cell[:, ::-1], cell == cell.T
    # The basis functions should be symmetric by construction if phases are chosen appropriately
    # But let's enforce it
    cell = (cell + cell[::-1, :]) // 2
    cell = (cell + cell[:, ::-1]) // 2
    cell = (cell + cell.T) // 2
    cell = (cell + cell[::-1, ::-1]) // 2
    
    return cell


def _local_search(initial_params, deadline, max_iters=50):
    """
    Local search over Fourier parameters.
    params = [a20, a02, a22, phase20, phase02, phase22]
    """
    best_params = initial_params.copy()
    best_score = _compute_few_band_gap(_fourier_basis_cell(*best_params))
    
    elapsed = lambda: time.time() - _local_search_start_time
    
    iters = 0
    while iters < max_iters and elapsed() < deadline - 10:
        iters += 1
        improved = False
        for i in range(len(best_params)):
            # Try small perturbations
            for delta in [-0.1, 0.1, -0.05, 0.05]:
                new_params = best_params.copy()
                new_params[i] += delta
                
                # Clip phases to [0, 1)
                if i >= 3:
                    new_params[i] = new_params[i] % 1.0
                # Clip amplitudes
                new_params[i] = np.clip(new_params[i], -1.0, 1.0)
                
                cell = _fourier_basis_cell(*new_params)
                score = _compute_few_band_gap(cell)
                
                if score > best_score:
                    best_params = new_params
                    best_score = score
                    improved = True
                    break
        if not improved:
            break
    
    return best_params, best_score


def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell maximizing the 4-5 TM band gap
    using Fourier-harmonic parameterization with local search.
    """
    global _local_search_start_time
    _local_search_start_time = time.time()
    deadline = 150.0  # Safety margin: evaluator kills at 230s, we target 150s
    
    size = problem.get("pixels", 32)
    
    # Start with the known good four-rod configuration as baseline
    # Then try Fourier parameterization
    
    # Baseline: four rods at (0.25,0.25), (0.75,0.25), (0.25,0.75), (0.75,0.75) with radius 0.18
    # This corresponds to a specific Fourier profile
    
    # Try several initial parameter sets
    initial_candidates = [
        [0.5, 0.5, 0.3, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.3, 0.25, 0.25, 0.0],
        [0.4, 0.4, 0.4, 0.0, 0.0, 0.25],
        [0.6, 0.2, 0.2, 0.0, 0.0, 0.0],
        [0.2, 0.6, 0.2, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.3, 0.3, 0.6, 0.0, 0.0, 0.5],
    ]
    
    best_params = None
    best_score = -1.0
    
    for init in initial_candidates:
        if time.time() - _local_search_start_time > deadline - 20:
            break
        cell = _fourier_basis_cell(*init)
        score = _compute_few_band_gap(cell)
        if score > best_score:
            best_score = score
            best_params = init.copy()
    
    # Local search from the best initial candidate
    if best_params is not None:
        best_params, best_score = _local_search(best_params, deadline)
    
    # Also try the known good four-rod design
    if time.time() - _local_search_start_time < deadline - 30:
        cell_4rod = np.zeros((size, size), dtype=int)
        centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
        radius = 0.18
        for i in range(size):
            y = (i + 0.5) / size
            for j in range(size):
                x = (j + 0.5) / size
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy < radius * radius:
                        cell_4rod[i, j] = 1
                        break
        
        score_4rod = _compute_few_band_gap(cell_4rod)
        if score_4rod > best_score:
            # Use the four-rod design
            cell_4rod = (cell_4rod + cell_4rod[::-1, :]) // 2
            cell_4rod = (cell_4rod + cell_4rod[:, ::-1]) // 2
            cell_4rod = (cell_4rod + cell_4rod.T) // 2
            return cell_4rod.tolist()
    
    # Generate final cell from best Fourier parameters
    if best_params is not None:
        cell = _fourier_basis_cell(*best_params)
    else:
        # Fallback: four rods
        cell = np.zeros((size, size), dtype=int)
        centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
        radius = 0.18
        for i in range(size):
            y = (i + 0.5) / size
            for j in range(size):
                x = (j + 0.5) / size
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy < radius * radius:
                        cell[i, j] = 1
                        break
    
    # Final symmetry enforcement
    cell = (cell + cell[::-1, :]) // 2
    cell = (cell + cell[:, ::-1]) // 2
    cell = (cell + cell.T) // 2
    cell = (cell + cell[::-1, ::-1]) // 2
    
    return cell.tolist()

# EVOLVE-BLOCK-END
