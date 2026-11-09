"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _make_cell_from_quadrant(quadrant):
    """Expand a 16x16 quadrant to a full 32x32 symmetric cell."""
    size = 32
    half = 16
    cell = np.zeros((size, size), dtype=int)
    # Top-left quadrant
    cell[:half, :half] = quadrant
    # Top-right: mirror horizontally
    cell[:half, half:] = quadrant[:, ::-1]
    # Bottom-left: mirror vertically
    cell[half:, :half] = quadrant[::-1, :]
    # Bottom-right: mirror both
    cell[half:, half:] = quadrant[::-1, ::-1]
    return cell


def _compute_fourier_coefficients(cell, n_pw=17):
    """
    Compute the inverse dielectric Fourier coefficients for the plane-wave method.
    For TM polarization, we need 1/epsilon in real space.
    
    cell: 32x32 array of 0/1 (1 = dielectric eps=12, 0 = air eps=1)
    n_pw: half-width of the plane-wave basis (total (2*n_pw+1)^2 = 289 waves)
    Returns: dict mapping (gx, gy) index to coefficient, or 2D array of shape (2*n_pw+1, 2*n_pw+1)
    """
    size = 32
    eps_low = 1.0
    eps_high = 12.0
    
    # Inverse dielectric in real space (pixel values)
    inv_eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if cell[i][j] == 1:
                inv_eps[i, j] = 1.0 / eps_high
            else:
                inv_eps[i, j] = 1.0 / eps_low
    
    # Compute Fourier coefficients: epsilon_G = (1/N^2) sum_{i,j} (1/eps_{i,j}) exp(-i G . r_{i,j})
    # where r_{i,j} = ((i+0.5)/size, (j+0.5)/size)
    # G = (Gx, Gy) with Gx, Gy integers (in units of 2*pi)
    
    # We need coefficients for G in range [-n_pw, n_pw] in each direction
    n = 2 * n_pw + 1  # 17
    coeffs = np.zeros((n, n), dtype=complex)
    
    for gx_idx in range(n):
        gx = gx_idx - n_pw
        for gy_idx in range(n):
            gy = gy_idx - n_pw
            # Sum over all pixels
            s = 0.0 + 0.0j
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size
                    y = (j + 0.5) / size
                    phase = -2.0 * math.pi * (gx * x + gy * y)
                    s += inv_eps[i, j] * (math.cos(phase) + 1j * math.sin(phase))
            coeffs[gx_idx, gy_idx] = s / (size * size)
    
    return coeffs


def _compute_fourier_coefficients_fast(cell, n_pw=17):
    """
    Faster computation using vectorized numpy operations.
    """
    size = 32
    eps_low = 1.0
    eps_high = 12.0
    
    inv_eps = np.zeros((size, size))
    inv_eps[cell == 1] = 1.0 / eps_high
    inv_eps[cell == 0] = 1.0 / eps_low
    
    n = 2 * n_pw + 1
    coeffs = np.zeros((n, n), dtype=complex)
    
    # Precompute pixel positions
    xs = (np.arange(size) + 0.5) / size
    ys = (np.arange(size) + 0.5) / size
    
    for gx_idx in range(n):
        gx = gx_idx - n_pw
        for gy_idx in range(n):
            gy = gy_idx - n_pw
            # Phase matrix
            phase_x = -2.0 * math.pi * gx * xs
            phase_y = -2.0 * math.pi * gy * ys
            # Build full phase: phase[i,j] = phase_x[i] + phase_y[j]
            phase = phase_x[:, np.newaxis] + phase_y[np.newaxis, :]
            coeffs[gx_idx, gy_idx] = np.sum(inv_eps * np.exp(1j * phase)) / (size * size)
    
    return coeffs


def _tm_band_energies(coeffs, kx, ky, n_pw=17):
    """
    Compute TM band energies (omega^2) for a given k-point.
    
    For TM polarization, the eigenvalue problem is:
    sum_{G'} [ (k+G) . (k+G') ] / epsilon_{G-G'} delta_{G,G'} ... 
    
    Actually, for TM: the matrix element is (k+G) dot (k+G') * epsilon_{G-G'}
    Wait, let me recall the standard formulation.
    
    For 2D square lattice photonic crystal, TM polarization (E perpendicular to plane):
    The eigenvalue equation is:
    sum_{G'} (k+G) . (k+G') / epsilon_{G-G'} * c_{G'} = omega^2 * c_G
    
    So the matrix H has elements:
    H[G, G'] = (k+G) . (k+G') / epsilon_{G-G'}
    
    But we have 1/epsilon Fourier coefficients. Let inv_eps_G = Fourier coeff of 1/eps.
    Then 1/epsilon_{G-G'} = inv_eps_{G-G'}.
    
    So H[G, G'] = (k+G) . (k+G') * inv_eps_{G-G'}
    
    Where G, G' are reciprocal lattice vectors. For a square lattice with lattice constant a=1:
    G = (Gx, Gy) where Gx, Gy are integers.
    k is in the first BZ.
    
    (k+G) . (k+G') = (kx+Gx)(kx+Gx') + (ky+Gy)(ky+Gy')
    """
    n = 2 * n_pw + 1
    size_mat = n * n
    
    # Build the matrix
    H = np.zeros((size_mat, size_mat), dtype=complex)
    
    # Map (gx_idx, gy_idx) to flat index
    def flat_idx(gx_idx, gy_idx):
        return gx_idx * n + gy_idx
    
    for i in range(n):
        gx = i - n_pw
        for j in range(n):
            gy = j - n_pw
            # k + G
            kgx = kx + gx
            kgy = ky + gy
            for ip in range(n):
                gxp = ip - n_pw
                for jp in range(n):
                    gyp = jp - n_pw
                    # k + G'
                    kgxp = kx + gxp
                    kgyp = ky + gyp
                    # dot product
                    dot = kgx * kgxp + kgy * kgyp
                    # G - G'
                    dGx = gx - gxp
                    dGy = gy - gyp
                    # inv_eps index
                    idx_dx = dGx + n_pw
                    idx_dy = dGy + n_pw
                    if 0 <= idx_dx < n and 0 <= idx_dy < n:
                        eps_inv = coeffs[idx_dx, idx_dy]
                    else:
                        eps_inv = 0.0 + 0.0j
                    H[flat_idx(i, j), flat_idx(ip, jp)] = dot * eps_inv
    
    # Solve eigenvalue problem
    # H should be Hermitian, take real part
    H_real = np.real(H)
    # Symmetrize
    H_real = (H_real + H_real.T) / 2.0
    
    eigenvalues = np.linalg.eigvalsh(H_real)
    return eigenvalues


def _generate_k_points(n_per_segment=6):
    """
    Generate 30 wave vectors covering the boundary and interior of the irreducible BZ.
    Irreducible BZ of square lattice: 0 <= ky <= kx <= pi (in units where lattice constant = 1, so BZ boundary at pi)
    Actually, for a=1, the BZ is [-pi, pi] x [-pi, pi]. The irreducible wedge is 0 <= ky <= kx <= pi.
    """
    k_points = []
    # Boundary: kx from 0 to pi, ky = 0
    for i in range(n_per_segment):
        kx = math.pi * i / (n_per_segment - 1)
        k_points.append((kx, 0.0))
    # Boundary: kx = ky, from (0,0) to (pi, pi)
    for i in range(1, n_per_segment):
        kx = math.pi * i / (n_per_segment - 1)
        k_points.append((kx, kx))
    # Boundary: ky from 0 to pi, kx = pi
    for i in range(1, n_per_segment - 1):
        ky = math.pi * i / (n_per_segment - 1)
        k_points.append((math.pi, ky))
    # Interior points
    # Add a few interior points
    k_points.append((math.pi/3, math.pi/6))
    k_points.append((math.pi/2, math.pi/4))
    k_points.append((2*math.pi/3, math.pi/3))
    k_points.append((math.pi/4, math.pi/8))
    
    # Trim to exactly 30
    if len(k_points) > 30:
        k_points = k_points[:30]
    elif len(k_points) < 30:
        # Add more boundary points
        extra = 30 - len(k_points)
        for i in range(extra):
            t = (i + 1) / (extra + 1)
            kx = math.pi * t
            ky = math.pi * t * 0.5
            k_points.append((kx, ky))
    
    return k_points[:30]


def _evaluate_cell(cell):
    """
    Evaluate the gap-midgap ratio for a given cell.
    Returns the score (can be negative).
    """
    n_pw = 17
    coeffs = _compute_fourier_coefficients_fast(cell, n_pw)
    k_points = _generate_k_points(6)
    
    max_omega4_sq = -np.inf
    min_omega5_sq = np.inf
    
    for kx, ky in k_points:
        eigs = _tm_band_energies(coeffs, kx, ky, n_pw)
        # Bands are indexed 1-based in the problem, so band 4 is index 3, band 5 is index 4
        if len(eigs) >= 5:
            omega4_sq = eigs[3]  # 4th band
            omega5_sq = eigs[4]  # 5th band
            max_omega4_sq = max(max_omega4_sq, omega4_sq)
            min_omega5_sq = min(min_omega5_sq, omega5_sq)
    
    if max_omega4_sq == -np.inf or min_omega5_sq == np.inf:
        return -1.0
    
    # Convert to omega (square root)
    max_omega4 = math.sqrt(max(0, max_omega4_sq))
    min_omega5 = math.sqrt(max(0, min_omega5_sq))
    
    if min_omega5 + max_omega4 < 1e-10:
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    ratio = 2.0 * gap / (min_omega5 + max_omega4)
    
    return ratio


def _flip_pixel_in_quadrant(quadrant, qi, qj):
    """Flip a pixel in the quadrant and return the new quadrant."""
    new_quad = quadrant.copy()
    new_quad[qi, qj] = 1 - new_quad[qi, qj]
    return new_quad


def design(problem):
    """
    Design a symmetric 32x32 unit cell for maximum TM band gap between bands 4 and 5.
    Uses local search (hill climbing) with pixel flips in the 16x16 quadrant.
    """
    start_time = time.time()
    deadline = start_time + 210.0  # Leave 20s margin
    
    size = 32
    half = 16
    
    # Start with a reasonable initial design: a dielectric disk
    quadrant = np.zeros((half, half), dtype=int)
    for i in range(half):
        for j in range(half):
            x = (i + 0.5) / size
            y = (j + 0.5) / size
            if x * x + y * y < 0.25 ** 2:
                quadrant[i, j] = 1
    
    cell = _make_cell_from_quadrant(quadrant)
    best_score = _evaluate_cell(cell)
    best_quadrant = quadrant.copy()
    
    # Local search: try flipping pixels in the quadrant
    # Order pixels by some heuristic (center first)
    pixels = []
    for i in range(half):
        for j in range(half):
            # Distance from center of the full cell (16, 16)
            # In quadrant coords, center is at (7.5, 7.5) approximately
            dist = math.sqrt((i - 7.5) ** 2 + (j - 7.5) ** 2)
            pixels.append((dist, i, j))
    pixels.sort()
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        for dist, i, j in pixels:
            if time.time() > deadline:
                break
            
            new_quad = _flip_pixel_in_quadrant(best_quadrant, i, j)
            new_cell = _make_cell_from_quadrant(new_quad)
            score = _evaluate_cell(new_cell)
            
            if score > best_score:
                best_score = score
                best_quadrant = new_quad
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        if not improved and no_improve_count > max_no_improve:
            # Try a random restart with a small perturbation
            if time.time() > deadline - 30:
                break
            # Perturb the best quadrant
            for _ in range(10):
                ri = np.random.randint(half)
                rj = np.random.randint(half)
                best_quadrant[ri, rj] = 1 - best_quadrant[ri, rj]
            best_cell = _make_cell_from_quadrant(best_quadrant)
            best_score = _evaluate_cell(best_cell)
            no_improve_count = 0
    
    return best_quadrant.tolist() if False else _make_cell_from_quadrant(best_quadrant).tolist()

# EVOLVE-BLOCK-END
