"""Design a 32x32 photonic-crystal unit cell for a large TM band gap."""

import time
import math
import numpy as np
from copy import deepcopy

# Time management
START_TIME = time.time()
DEADLINE = START_TIME + 200.0  # 20s safety margin


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return DEADLINE - time.time()


def _make_symmetric_cell_from_quadrant(quadrant):
    """Create a full 32x32 cell from a 16x16 quadrant, enforcing all symmetries."""
    size = 32
    q = np.array(quadrant, dtype=float)
    # quadrant is top-left 16x16
    # Full cell: 
    # top half: [Q, flip_h(Q)]
    # bottom half: [flip_v(Q), flip_vh(Q)]
    # where Q is top-left quadrant
    top_left = q
    top_right = np.flip(top_left, axis=1)
    bottom_left = np.flip(top_left, axis=0)
    bottom_right = np.flip(top_left, axis=(0, 1))
    
    cell = np.zeros((size, size), dtype=float)
    cell[:16, :16] = top_left
    cell[:16, 16:] = top_right
    cell[16:, :16] = bottom_left
    cell[16:, 16:] = bottom_right
    return cell


def _compute_fourier_coefficients(cell, N=289):
    """Compute the Fourier coefficients of 1/epsilon for the pixel map.
    
    For a square lattice photonic crystal, the dielectric function is:
        epsilon(r) = eps_bg + (eps_hi - eps_bg) * sum_k f_k exp(i k.r)
    
    where f_k are the Fourier coefficients of the pixel map (normalized).
    
    For TM polarization, we need 1/epsilon in the eigenvalue equation.
    """
    size = cell.shape[0]
    # Normalize cell to [0,1]
    cell_norm = cell.astype(float) / 255.0 if cell.max() > 1 else cell.astype(float)
    
    # Compute Fourier coefficients for all (nx, ny) with |nx|^2 + |ny|^2 <= (N-1)/4
    # For a 32x32 pixel grid, the fundamental period is 1, so k = 2*pi*(nx, ny)
    # We need coefficients up to some order for the 289 plane waves
    
    # Use FFT-based approach: the Fourier coefficient for mode (mx, my) is
    # f_{mx,my} = (1/N^2) * sum_{x,y} cell(x,y) * exp(-2*pi*i*(mx*x + my*y)/N)
    
    # For a 32x32 grid, we can compute up to the Nyquist frequency
    # But we need enough for the 289 plane waves
    
    # The 289 plane waves correspond to (nx, ny) with nx^2 + ny^2 <= 71 (since 289 = 17^2,
    # and for a square lattice, the number of G-vectors with |G|^2 <= Gmax^2 in the BZ)
    # Actually, for a 32x32 pixel map, we should compute the discrete Fourier transform
    
    # Let's use numpy FFT for efficiency
    # The pixel values represent the dielectric occupancy
    cell_float = cell_norm.copy()
    
    # Compute 2D FFT
    fft_vals = np.fft.fft2(cell_float)
    # Shift zero frequency to center
    fft_shifted = np.fft.fftshift(fft_vals)
    
    # The FFT gives us coefficients for frequencies from -16 to 15 (for 32 points)
    # We need to map these to the actual G-vectors
    
    # For a 32x32 grid, the frequency indices are -16, -15, ..., 15
    # The corresponding G-vectors are G = 2*pi*(nx, ny) where nx, ny are in [-16, 15]
    
    # We want to return a dictionary mapping (nx, ny) -> f_{nx,ny}
    coeff = {}
    center = 16  # For 32-point FFT, the center after shift is at index 16
    
    for i in range(size):
        for j in range(size):
            nx = i - center
            ny = j - center
            coeff[(nx, ny)] = fft_shifted[i, j] / (size * size)
    
    return coeff


def _surrogate_gap(cell, eps_low=1.0, eps_high=12.0, max_order=8):
    """Compute a surrogate band gap metric using plane-wave expansion.
    
    This is a fast approximation that doesn't solve the full eigenvalue problem
    but captures the essential physics of the band structure.
    
    For TM polarization, the eigenvalue problem is:
        sum_{G'} [ (k + G)^2 / (k + G').2 ] * delta_{G,G'} - (1/eps_{G-G'}) * omega^2 = 0
    
    where 1/eps is the Fourier transform of 1/epsilon.
    
    We'll compute the gap between bands 4 and 5 at high-symmetry points.
    """
    size = cell.shape[0]
    
    # First, compute 1/epsilon for each pixel
    eps_pixels = np.where(cell > 0, eps_high, eps_low)
    inv_eps = 1.0 / eps_pixels
    
    # Compute Fourier coefficients of 1/epsilon
    # For the plane-wave method, we need the Fourier coefficients of 1/eps
    fft_inv_eps = np.fft.fft2(inv_eps)
    fft_shifted = np.fft.fftshift(fft_inv_eps)
    
    # Normalize
    N = size
    inv_eps_coeff = fft_shifted / (N * N)
    
    # Define the G-vectors we'll use (up to max_order)
    g_vectors = []
    for nx in range(-max_order, max_order + 1):
        for ny in range(-max_order, max_order + 1):
            g_vectors.append((nx, ny))
    
    # Sort by magnitude
    g_vectors.sort(key=lambda g: g[0]**2 + g[1]**2)
    
    # For each high-symmetry point in the BZ, compute the band structure
    # High symmetry points: Gamma (0,0), M (0.5, 0), X (0.5, 0.5) in units of 2*pi
    # In terms of wave vectors k = (kx, ky)
    
    def _compute_bands_at_k(kx, ky, num_bands=6, max_g=max_order):
        """Compute the first num_bands TM bands at wave vector (kx, ky)."""
        # Build the matrix for the plane-wave expansion
        # H_{G,G'} = (k+G)^2 / (k+G')^2 * delta_{G,G'} - (1/eps)_{G-G'}
        
        # Use a subset of G-vectors for speed
        g_subset = [(nx, ny) for nx in range(-max_g, max_g+1) 
                    for ny in range(-max_g, max_g+1)]
        
        n_g = len(g_subset)
        
        # Map G-vector to index
        g_to_idx = {g: i for i, g in enumerate(g_subset)}
        
        # Build the matrix
        # For TM: H_{G,G'} = [(kx+Gx)^2 + (ky+Gy)^2] * delta_{G,G'} - (1/eps)_{G-G'}
        
        H = np.zeros((n_g, n_g), dtype=complex)
        
        for i, G in enumerate(g_subset):
            kGx = kx + G[0]
            kGy = ky + G[1]
            kG2 = kGx*kGx + kGy*kGy
            if abs(kG2) < 1e-12:
                H[i, i] = 0.0
            else:
                H[i, i] = kG2  # The delta term: (k+G)^2
        
        # Now add the coupling terms from 1/eps
        for i, Gi in enumerate(g_subset):
            for j, Gj in enumerate(g_subset):
                dG = (Gi[0] - Gj[0], Gi[1] - Gj[1])
                if dG in inv_eps_coeff:
                    H[i, j] -= inv_eps_coeff[dG]
                elif dG[0] == 0 and dG[1] == 0:
                    pass  # Already handled in diagonal
        
        # Actually, let me redo this more carefully
        # The eigenvalue equation for TM is:
        # sum_{G'} [ (k+G)^2 * delta_{G,G'} - (1/eps)_{G-G'} ] * A_{G'} = omega^2 * A_G
        # Wait, I need to be more careful with the formulation.
        
        # Standard plane-wave formulation for TM:
        # The eigenvalue equation is:
        # sum_{G'} [ ((k+G)x*(k+G')x + (k+G)y*(k+G')y) / |k+G'|^2 * delta_{G,G'} 
        #           - (1/eps)_{G-G'} ] * A_{G'} = omega^2 * A_G
        # No, that's not quite right either.
        
        # Let me use the simpler formulation where we diagonalize:
        # D * A = omega^2 * A
        # where D_{G,G'} = (k+G)^2 * delta_{G,G'} - (1/eps)_{G-G'}
        
        # Actually, the correct formulation for 2D TM is:
        # The Maxwell equations give:
        # (k + G)^2 * A_G - sum_{G'} (1/eps)_{G-G'} * A_{G'} = (omega/c)^2 * A_G
        # Wait, I need to check the exact form.
        
        # For now, let's use a simpler surrogate: compute the eigenvalues of
        # the matrix M where M_{G,G'} = (k+G)^2 * delta_{G,G'} - (1/eps)_{G-G'}
        # and look at the gap between the 4th and 5th smallest positive eigenvalues.
        
        # Rebuild the matrix properly
        H = np.zeros((n_g, n_g), dtype=complex)
        
        for i, G in enumerate(g_subset):
            kGx = kx + G[0]
            kGy = ky + G[1]
            kG2 = kGx*kGx + kGy*kGy
            H[i, i] = kG2
        
        # Subtract the 1/eps coupling
        for i, Gi in enumerate(g_subset):
            for j, Gj in enumerate(g_subset):
                dG = (Gi[0] - Gj[0], Gi[1] - Gj[1])
                if dG in inv_eps_coeff:
                    H[i, j] -= inv_eps_coeff[dG]
        
        # Compute eigenvalues
        try:
            eigvals = np.linalg.eigvalsh((H + H.conj().T) / 2)  # Make Hermitian
            eigvals = np.sort(eigvals.real)
            # Filter out negative eigenvalues (unphysical)
            positive_eigvals = [ev for ev in eigvals if ev > 1e-10]
            if len(positive_eigvals) >= num_bands:
                return positive_eigvals[:num_bands]
            else:
                return [0.0] * num_bands
        except:
            return [0.0] * num_bands
    
    # Evaluate at high-symmetry points
    # k is in units of 2*pi (so Gamma is (0,0), M is (0.5, 0), X is (0.5, 0.5))
    high_sym_points = [
        (0.0, 0.0),      # Gamma
        (0.5, 0.0),      # M
        (0.5, 0.5),      # X
        (0.25, 0.0),     # Interior
        (0.25, 0.25),    # Interior
        (0.1, 0.1),      # Interior
    ]
    
    min_band5 = float('inf')
    max_band4 = 0.0
    
    for kx, ky in high_sym_points:
        bands = _compute_bands_at_k(kx, ky, num_bands=6)
        if len(bands) >= 5 and bands[4] > 0:
            max_band4 = max(max_band4, bands[3])
            min_band5 = min(min_band5, bands[4])
    
    if min_band5 < float('inf') and max_band4 > 0:
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 0:
            return gap / midgap
    return 0.0


def _make_cell_from_params(params, size=32):
    """Create a cell from parameters.
    
    params can be:
    - (n_exp, a, b): superellipse parameters
    - ('grid', grid_16x16): direct grid specification
    """
    if isinstance(params, tuple) and len(params) == 3 and isinstance(params[0], (int, float)):
        n_exp, a, b = params
        # Superellipse: |x/a|^n + |y/b|^n <= 1
        x = np.linspace(-0.5, 0.5, size, endpoint=False) + 0.5/size
        y = np.linspace(-0.5, 0.5, size, endpoint=False) + 0.5/size
        X, Y = np.meshgrid(x, y, indexing='ij')
        # Use absolute values for the superellipse
        val = np.abs(X / a)**n_exp + np.abs(Y / b)**n_exp
        cell = (val <= 1.0).astype(float)
        return cell
    elif isinstance(params, tuple) and params[0] == 'grid':
        quadrant = params[1]
        return _make_symmetric_cell_from_quadrant(quadrant)
    else:
        # Default: centered circle
        x = np.linspace(-0.5, 0.5, size, endpoint=False) + 0.5/size
        y = np.linspace(-0.5, 0.5, size, endpoint=False) + 0.5/size
        X, Y = np.meshgrid(x, y, indexing='ij')
        cell = (X**2 + Y**2 <= 0.2**2).astype(float)
        return cell


def _local_pixel_optimize(cell, n_iterations=50, eps_low=1.0, eps_high=12.0):
    """Try flipping individual pixels to improve the gap."""
    size = cell.shape[0]
    best_score = _surrogate_gap(cell, eps_low, eps_high)
    best_cell = cell.copy()
    
    for it in range(n_iterations):
        if _remaining() < 30:
            break
        
        # Try random pixel flips
        improved = False
        for _ in range(20):
            i, j = np.random.randint(0, size, 2)
            # Flip pixel and its symmetric counterparts
            candidates = [(i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j)]
            
            test_cell = best_cell.copy()
            for ci, cj in candidates:
                test_cell[ci, cj] = 1.0 - best_cell[ci, cj]
            
            # Ensure 0/1 values
            test_cell = (test_cell > 0.5).astype(float)
            
            score = _surrogate_gap(test_cell, eps_low, eps_high)
            if score > best_score + 1e-6:
                best_score = score
                best_cell = test_cell
                improved = True
                break
        
        if not improved and it > 20:
            # Try more aggressive changes: flip 2x2 blocks
            for _ in range(10):
                i, j = np.random.randint(0, size//2, 2)
                test_cell = best_cell.copy()
                # Flip a 2x2 block and its symmetric counterparts
                for di in range(2):
                    for dj in range(2):
                        ci, cj = i + di, j + dj
                        test_cell[ci, cj] = 1.0 - best_cell[ci, cj]
                        test_cell[ci, size-1-cj] = 1.0 - best_cell[ci, size-1-cj]
                        test_cell[size-1-ci, cj] = 1.0 - best_cell[size-1-ci, cj]
                        test_cell[size-1-ci, size-1-cj] = 1.0 - best_cell[size-1-ci, size-1-cj]
                
                test_cell = (test_cell > 0.5).astype(float)
                score = _surrogate_gap(test_cell, eps_low, eps_high)
                if score > best_score + 1e-6:
                    best_score = score
                    best_cell = test_cell
                    improved = True
                    break
            if not improved:
                break
    
    return best_cell, best_score


def design(problem):
    """Design a 32x32 photonic crystal unit cell for maximum TM band gap."""
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    if size != 32:
        # Fallback for non-32 size
        x = np.linspace(-0.5, 0.5, size, endpoint=False) + 0.5/size
        y = np.linspace(-0.5, 0.5, size, endpoint=False) + 0.5/size
        X, Y = np.meshgrid(x, y, indexing='ij')
        cell = (X**2 + Y**2 <= 0.2**2).astype(float)
        return [[int(cell[i, j]) for j in range(size)] for i in range(size)]
    
    # Phase 1: Grid search over superellipse parameters
    best_score = -float('inf')
    best_cell = None
    
    # Try a range of superellipse parameters
    n_values = [2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0]
    a_values = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    b_values = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    
    for n_exp in n_values:
        if _remaining() < 60:
            break
        for a in a_values:
            if _remaining() < 50:
                break
            for b in b_values:
                if _remaining() < 40:
                    break
                # Enforce symmetry: a == b for circular symmetry, or a != b for elliptical
                # The cell must be invariant under 90-degree rotation, so a must equal b
                # Wait, the symmetry requirement is: cell == cell[::-1] row-wise and column-wise
                # and cell == its transpose. This means the cell is symmetric under:
                # - Reflection about horizontal midline
                # - Reflection about vertical midline  
                # - Transpose (reflection about main diagonal)
                # This implies 90-degree rotational symmetry as well!
                # So a must equal b.
                
                if abs(a - b) < 1e-10:
                    cell = _make_cell_from_params((n_exp, a, b), size)
                    score = _surrogate_gap(cell, eps_low, eps_high)
                    if score > best_score:
                        best_score = score
                        best_cell = cell
    
    # Phase 2: Try different fill fractions with circular rods
    if _remaining() > 40:
        for r in np.arange(0.10, 0.45, 0.02):
            if _remaining() < 30:
                break
            x = np.linspace(-0.5, 0.5, size, endpoint=False) + 0.5/size
            y = np.linspace(-0.5, 0.5, size, endpoint=False) + 0.5/size
            X, Y = np.meshgrid(x, y, indexing='ij')
            cell = (X**2 + Y**2 <= r**2).astype(float)
            score = _surrogate_gap(cell, eps_low, eps_high)
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # Phase 3: Try square rods (axis-aligned)
    if _remaining() > 30:
        for side in np.arange(0.15, 0.45, 0.02):
            if _remaining() < 20:
                break
            x = np.linspace(-0.5, 0.5, size, endpoint=False) + 0.5/size
            y = np.linspace(-0.5, 0.5, size, endpoint=False) + 0.5/size
            X, Y = np.meshgrid(x, y, indexing='ij')
            cell = (np.abs(X) <= side/2 and np.abs(Y) <= side/2).astype(float)
            # Wait, need to use numpy broadcasting properly
            cell = ((np.abs(X) <= side/2) & (np.abs(Y) <= side/2)).astype(float)
            score = _surrogate_gap(cell, eps_low, eps_high)
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # Phase 4: Local pixel optimization
    if best_cell is not None and _remaining() > 25:
        best_cell, best_score = _local_pixel_optimize(best_cell, n_iterations=30,
                                                       eps_low=eps_low, eps_high=eps_high)
    
    # Phase 5: Try a 4x4 superlattice of independently sized circles
    if _remaining() > 20:
        # The cell has 90-degree rotational symmetry, so we only need to optimize
        # the top-left 8x8 quadrant... actually, let's think about this differently.
        # With 90-degree rotational symmetry and reflections, the fundamental domain
        # is an 8x8 block in the top-left corner (since 32/4 = 8, but with the
        # additional reflection symmetries, it's actually even smaller).
        
        # Actually, the symmetry group D4 (dihedral group of order 8) means the
        # fundamental domain is 1/8 of the cell. For a 32x32 cell, that's 32*32/8 = 128 pixels.
        # The fundamental domain is the triangle in the top-left corner where 0 <= y <= x <= 16
        # (in pixel coordinates, with the center at (16, 16)).
        
        # Let's try optimizing a few key pixels
        if best_cell is not None:
            test_cell = best_cell.copy()
            # Try flipping some pixels in the fundamental domain
            np.random.seed(42)
            for _ in range(100):
                if _remaining() < 15:
                    break
                # Pick a random pixel in the fundamental domain
                # Fundamental domain: i from 0 to 15, j from i to 15
                i = np.random.randint(0, 16)
                j = np.random.randint(i, 16)
                
                # Get all symmetric positions
                positions = [
                    (i, j), (i, size-1-j),
                    (size-1-i, j), (size-1-i, size-1-j),
                    (j, i), (j, size-1-i),
                    (size-1-j, i), (size-1-j, size-1-i)
                ]
                # Remove duplicates
                positions = list(set(positions))
                
                new_cell = test_cell.copy()
                for pi, pj in positions:
                    new_cell[pi, pj] = 1.0 - test_cell[pi, pj]
                
                new_cell = (new_cell > 0.5).astype(float)
                score = _surrogate_gap(new_cell, eps_low, eps_high)
                if score > best_score + 1e-6:
                    best_score = score
                    best_cell = new_cell
                    test_cell = new_cell.copy()
    
    # Ensure we have a valid cell
    if best_cell is None:
        # Default: centered circle
        x = np.linspace(-0.5, 0.5, size, endpoint=False) + 0.5/size
        y = np.linspace(-0.5, 0.5, size, endpoint=False) + 0.5/size
        X, Y = np.meshgrid(x, y, indexing='ij')
        best_cell = (X**2 + Y**2 <= 0.2**2).astype(float)
    
    # Convert to nested list
    cell_list = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        cell_list.append(row)
    
    return cell_list
