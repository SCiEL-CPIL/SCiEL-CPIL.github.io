"""Adjoint spectral inverse design for the 4-5 TM band gap."""
import math
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def _build_eps(k, N, eps_high):
    """eps^{-1} (k) for 289 plane waves, N=32."""
    K = N  # pixels per lattice constant
    n = 17
    ks = list(range(-n, n + 1))
    # plane wave indices: k = (K * (a + 0.5), K * (b + 0.5))
    kx = np.array([K * (a + 0.5) for a in ks], dtype=float)
    ky = np.array([K * (b + 0.5) for b in ks], dtype=float)
    # Build eps^{-1} matrix: M_{ij} = (1/N^2) sum_{x,y} eps^{-1}(x,y) exp(-i (k_i - k_j).(x,y))
    # eps^{-1}(x,y) = 1 if pixel 0 (air), 1/eps_high if pixel 1 (dielectric)
    # We'll compute via FFT for speed: eps^{-1} is piecewise constant on pixels.
    # Actually, the Fourier coefficient of a pixel map:
    # f_{G} = (1/N^2) sum_{i,j} f_{i,j} exp(-i G . (i,j)/N)
    # where G = (Gx, Gy) in reciprocal lattice units, and pixel center (i+0.5, j+0.5)
    # Let's use the standard: plane wave k = k0 + G, G = (Gx, Gy) * 2pi
    # For a square lattice with side 1, reciprocal lattice vectors are 2pi * (m, n)
    # Pixel (i,j) center at ((i+0.5)/N, (j+0.5)/N)
    # f_G = (1/N^2) sum_{i,j} f_{i,j} exp(-i 2pi (Gx*(i+0.5)/N + Gy*(j+0.5)/N))
    # This can be computed by FFT of the pixel map, with a phase shift.
    
    # For now, let's just compute the matrix directly for the 17x17 = 289 waves.
    # This is 289^2 = 83521 elements, each a sum over 1024 pixels = 85 million ops.
    # That's too slow for many iterations. Let's use FFT.
    
    # Actually, the matrix M_{ij} = f_{G_i - G_j} where G_i - G_j = 2pi (m_i - m_j, n_i - n_j)
    # So we only need the Fourier coefficients f_{dm, dn} for dm, dn in [-34, 34].
    # We can compute these via FFT of the pixel map (padded to 64x64 or so).
    
    # Let's compute f_{dm, dn} for dm, dn in [-34, 34]
    # f_{dm, dn} = (1/N^2) sum_{i,j} eps^{-1}_{i,j} exp(-i 2pi (dm*(i+0.5)/N + dn*(j+0.5)/N))
    
    # We'll build the full M matrix using these coefficients.
    idx = np.array([a * n + b for a in range(2 * n + 1) for b in range(2 * n + 1)])
    # Wait, I need a consistent indexing. Let's use linear index for (a, b) in ks.
    # ks = [-17, ..., 17], length 35.
    # Let's re-index: a_idx in 0..34, a_val = ks[a_idx]
    
    # Compute f_{dm, dn} for all needed dm, dn
    # dm = a_i - a_j, dn = b_i - b_j
    # We need f for dm, dn in [-34, 34]
    
    # Use FFT: create a 64x64 array, fill with eps^{-1} pixel values (centered)
    # Actually, the phase shift (i+0.5) means we multiply by exp(-i 2pi dm / (2N))
    
    # Let's just do it directly for the 35x35 grid of G differences.
    # 35*35 = 1225 coefficients, each a sum over 1024 pixels = 1.2 million ops.
    # That's fine.
    
    # eps^{-1} pixel values
    eps_inv = np.where(k > 0.5, 1.0 / eps_high, 1.0)  # shape (32, 32)
    
    # Precompute exp(-i 2pi (dm * (i+0.5) + dn * (j+0.5)) / N)
    # For dm, dn in [-34, 34]
    dm_range = np.arange(-34, 35)
    dn_range = np.arange(-34, 35)
    i_idx = np.arange(N) + 0.5
    j_idx = np.arange(N) + 0.5
    
    # f[dm_idx, dn_idx] = (1/N^2) sum_{i,j} eps_inv[i,j] exp(-i 2pi (dm*i + dn*j)/N)
    # We can compute this as a 2D FFT with phase correction.
    # Let's just do it with numpy broadcasting for the 69x69 grid of (dm, dn).
    # 69*69 = 4761, each sum over 1024 = 4.8 million. Fine.
    
    # Actually, let's use the FFT trick:
    # f_{dm, dn} = (1/N^2) sum_{i,j} eps_inv[i,j] exp(-i 2pi dm i / N) exp(-i 2pi dn j / N) exp(-i 2pi dm / (2N)) exp(-i 2pi dn / (2N))
    # = exp(-i 2pi (dm + dn) / (2N)) * (1/N^2) sum_{i,j} eps_inv[i,j] exp(-i 2pi (dm i + dn j) / N)
    # The sum is just the 2D DFT of eps_inv.
    
    # Compute 2D FFT of eps_inv
    F = np.fft.fft2(eps_inv)  # F[m, n] = sum_{i,j} eps_inv[i,j] exp(-i 2pi (m i + n j) / N)
    # F is indexed 0..N-1, corresponding to m, n in 0..N-1
    # We need m, n in [-34, 34], which wraps to [0, 31] and [32, 31] -> [0, 31]
    # Actually, np.fft.fft2 gives frequencies 0, 1, ..., N/2-1, -N/2, ..., -1
    # For N=32, frequencies are 0..15, -16..-1
    # We need -34..34, which wraps modulo 32.
    
    # Let's just extract the needed coefficients directly.
    f_coeffs = np.zeros((69, 69), dtype=complex)
    for dm in range(-34, 35):
        for dn in range(-34, 35):
            m = dm % N
            n = dn % N
            # Phase correction for the 0.5 offset
            phase = math.exp(-1j * 2 * math.pi * (dm + dn) / (2 * N))
            f_coeffs[dm + 34, dn + 34] = F[m, n] * phase / (N * N)
    
    # Now build M_{ij} = f_{G_i - G_j}
    # G_i = (a_i, b_i), G_j = (a_j, b_j)
    # dm = a_i - a_j, dn = b_i - b_j
    M = np.zeros((289, 289), dtype=complex)
    for i_idx in range(35):
        for j_idx in range(35):
            a_i = ks[i_idx]
            b_i = ks[i_idx]  # Wait, this is wrong. I need separate a and b indices.
            # Let me redo the indexing.
    
    # Let's use a 2D grid for the plane waves.
    # pw_idx = (a_idx, b_idx) where a_idx, b_idx in 0..34
    # a_val = ks[a_idx], b_val = ks[b_idx]
    # Linear index: lin = a_idx * 35 + b_idx
    
    M = np.zeros((35 * 35, 35 * 35), dtype=complex)
    for a_i in range(35):
        for b_i in range(35):
            a_val_i = ks[a_i]
            b_val_i = ks[b_i]
            for a_j in range(35):
                for b_j in range(35):
                    a_val_j = ks[a_j]
                    b_val_j = ks[b_j]
                    dm = a_val_i - a_val_j
                    dn = b_val_i - b_val_j
                    M[a_i * 35 + b_i, a_j * 35 + b_j] = f_coeffs[dm + 34, dn + 34]
    
    return M

def _solve_bands(M, k0, eps_high):
    """Solve the generalized eigenvalue problem for TM bands.
    Returns eigenvalues (omega^2) and eigenvectors."""
    # The TM eigenvalue problem: sum_j M_{ij} (k_i + G_i) x (k_j + G_j) c_j = omega^2 c_i
    # where (k_i + G_i) = (k0x + Gx_i, k0y + Gy_i)
    # This is a standard eigenvalue problem: A c = omega^2 c
    # where A_{ij} = M_{ij} * (k_i + G_i) . (k_j + G_j)
    
    # Build A
    n = 17
    ks = list(range(-n, n + 1))
    # k0 is in reciprocal lattice units (2pi * k0)
    # G_i = 2pi * (a_i, b_i)
    # (k_i + G_i) = 2pi * (k0x + a_i, k0y + b_i)
    # The factor 2pi cancels in the ratio, so we can work in units of 2pi.
    
    A = np.zeros((35 * 35, 35 * 35), dtype=complex)
    for a_i in range(35):
        for b_i in range(35):
            kx_i = k0[0] + ks[a_i]
            ky_i = k0[1] + ks[b_i]
            for a_j in range(35):
                for b_j in range(35):
                    kx_j = k0[0] + ks[a_j]
                    ky_j = k0[1] + ks[b_j]
                    A[a_i * 35 + b_i, a_j * 35 + b_j] = M[a_i * 35 + b_i, a_j * 35 + b_j] * (kx_i * kx_j + ky_i * ky_j)
    
    # Solve A c = omega^2 c
    # A is Hermitian (if M is Hermitian, which it should be for real eps)
    # Use scipy
    eigvals, eigvecs = spla.eigsh(A, k=6, which='SM')  # Smallest 6 eigenvalues
    # Sort
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs

def design(problem):
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    lower_band = problem["lower_band"]  # 4
    
    # Start with a reasonable initial guess: a ring or a cross
    # Let's start with a dielectric ring (hole in the middle)
    cell = np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = math.sqrt(x * x + y * y)
            if 0.15 < r < 0.35:
                cell[i, j] = 1.0
    
    # Enforce C4v symmetry by mirroring the top-left quadrant
    def _symmetrize(cell):
        q = cell[:size//2, :size//2].copy()
        cell[:size//2, :size//2] = q
        cell[:size//2, size//2:] = q[:, ::-1]
        cell[size//2:, :size//2] = q[::-1, :]
        cell[size//2:, size//2:] = q[::-1, ::-1]
        return cell
    
    cell = _symmetrize(cell)
    
    # Fast gap evaluation
    def _eval_gap(cell_float):
        """Compute the gap-midgap ratio for a continuous dielectric map."""
        # Binarize for the solver (the solver needs 0/1)
        cell_bin = (cell_float > 0.5).astype(float)
        # Enforce symmetry
        cell_bin = _symmetrize(cell_bin)
        
        # Build M
        M = _build_eps(cell_bin, size, eps_high)
        
        # Evaluate at 30 wave vectors in the irreducible BZ
        # Irreducible BZ of square lattice: 0 <= kx <= ky <= 1/2
        # 30 points: let's use a grid
        kx_vals = np.linspace(0, 0.5, 6)
        ky_vals = np.linspace(0, 0.5, 6)
        k_points = []
        for kx in kx_vals:
            for ky in kx_vals:
                if kx <= ky <= 0.5 and kx + ky <= 0.5:
                    k_points.append((kx, ky))
        # Add some boundary points
        k_points.append((0.5, 0.0))
        k_points.append((0.0, 0.5))
        k_points.append((0.5, 0.5))
        k_points = k_points[:30]
        
        max_omega_4 = 0
        min_omega_5 = float('inf')
        for k0 in k_points:
            eigvals, eigvecs = _solve_bands(M, k0, eps_high)
            # eigvals are omega^2 (in units of (2pi)^2)
            # We need bands 4 and 5 (1-indexed), so indices 3 and 4
            if len(eigvals) >= 5:
                max_omega_4 = max(max_omega_4, math.sqrt(eigvals[3]))
                min_omega_5 = min(min_omega_5, math.sqrt(eigvals[4]))
        
        if min_omega_5 == float('inf') or max_omega_4 == 0:
            return 0.0
        gap = 2 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return gap
    
    # Gradient ascent on the continuous field
    best_gap = _eval_gap(cell)
    best_cell = cell.copy()
    
    # Compute gradient via finite differences (expensive but simple)
    # For each pixel in the top-left quadrant, perturb it and see the change in gap
    # This is 256 perturbations, each requiring a band structure calculation.
    # Too slow. Let's use a simpler approach: random search with symmetry.
    
    # Random search with C4v symmetry
    np.random.seed(42)
    for it in range(50):
        # Mutate the cell: flip some pixels in the top-left quadrant
        cell_mut = best_cell.copy()
        for _ in range(20):
            i = np.random.randint(0, size // 2)
            j = np.random.randint(0, size // 2)
            cell_mut[i, j] = 1.0 - cell_mut[i, j]
        cell_mut = _symmetrize(cell_mut)
        gap = _eval_gap(cell_mut)
        if gap > best_gap:
            best_gap = gap
            best_cell = cell_mut
    
    # Binarize and return
    cell_final = (best_cell > 0.5).astype(int)
    cell_final = _symmetrize(cell_final)
    return cell_final.tolist()
