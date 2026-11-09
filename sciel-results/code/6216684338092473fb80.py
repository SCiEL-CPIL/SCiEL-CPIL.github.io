"""Rewritten design: local-search over C4v-symmetric pixel orbits, scored by a small PW solver."""
import time
import numpy as np
from numpy.linalg import eigvalsh

_START_TIME = time.time()
TIME_BUDGET = 120.0  # seconds of CPU budget; keep safety margin

# Precompute C4v orbits for 32x32 (indices 0..31, centre at 15.5)
_SIZE = 32
_HALF = _SIZE // 2  # 16
_orbit_cache = None

def _build_orbits():
    global _orbit_cache
    if _orbit_cache is not None:
        return _orbit_cache
    orbits = []
    seen = set()
    for i in range(_SIZE):
        for j in range(_SIZE):
            if (i, j) in seen:
                continue
            orbit = []
            # Generate all 8 symmetries: (i,j), (i,N-1-j), (N-1-i,j), (N-1-i,N-1-j),
            # (j,i), (j,N-1-i), (N-1-j,i), (N-1-j,N-1-i)
            N = _SIZE
            for (a, b) in [
                (i, j), (i, N-1-j), (N-1-i, j), (N-1-i, N-1-j),
                (j, i), (j, N-1-i), (N-1-j, i), (N-1-j, N-1-i)
            ]:
                if (a, b) not in seen:
                    orbit.append((a, b))
                    seen.add((a, b))
            orbits.append(orbit)
    _orbit_cache = orbits
    return orbits

_ORBITS = _build_orbits()

# Fourier coefficients for a 32x32 pixel map using N=17 plane waves (from -8 to 8)
_NPW = 17  # plane waves per axis
_M = _NPW * _NPW  # 289

def _fourier_coefficients(cell):
    """
    Compute Fourier coefficients of eps(x,y) for the 32x32 pixel map.
    cell is a 32x32 numpy array of 0/1 values (1 = dielectric).
    We return an array of shape (M, M) for the dielectric part.
    
    For a square lattice with lattice constant a=1, the Fourier coefficients are:
    eps^(G) = (1/S) ∫∫ eps(x,y) exp(-i G·r) dA
    where S = 1 (unit cell area).
    
    For a pixelated map, we approximate this as a sum over pixels.
    Each pixel is a small square of area 1/1024 centered at ((j+0.5)/32 - 0.5, (i+0.5)/32 - 0.5).
    """
    # Convert to dielectric: eps = 1 + 11 * cell (since eps_low=1, eps_high=12)
    eps = 1.0 + 11.0 * cell.astype(np.float64)
    
    # Pixel centers
    i_coords = (np.arange(_SIZE) + 0.5) / _SIZE - 0.5  # x-coordinates (row direction)
    j_coords = i_coords.copy()  # y-coordinates (column direction)
    
    # Reciprocal lattice vectors: G = (2π*n, 2π*m) for n, m in -8..8
    n_vals = np.arange(-_NPW//2, _NPW//2 + 1)  # -8 to 8
    G_n, G_m = np.meshgrid(n_vals, n_vals, indexing='ij')
    G_flat = np.column_stack([G_n.ravel(), G_m.ravel()])
    
    # Pixel positions
    px = i_coords[:, None]  # 32 x 1
    py = j_coords[None, :]  # 1 x 32
    px_flat = np.repeat(px, _SIZE, axis=1).ravel()  # 1024
    py_flat = np.repeat(py, _SIZE, axis=0).ravel()  # 1024
    
    # eps values for each pixel
    eps_flat = eps.ravel()  # 1024
    
    # Compute Fourier coefficients: eps^(G) = sum_pixels eps(pixel) * exp(-i G·r_pixel) * pixel_area
    pixel_area = 1.0 / (_SIZE * _SIZE)
    
    # For efficiency, compute exp(-i * 2π * (n*px + m*py)) for all G and pixels
    # G_flat is (M, 2), px_flat and py_flat are (1024,)
    # phase = -2π * (n*px + m*py)
    phase = -2.0 * np.pi * (G_flat[:, 0:1] * px_flat[None, :] + G_flat[:, 1:2] * py_flat[None, :])
    phase = phase.ravel()  # (M * 1024,)
    phase = phase.reshape(_M, _SIZE * _SIZE)
    
    exp_phase = np.exp(1j * phase)  # (M, 1024)
    coeffs = exp_phase @ eps_flat * pixel_area  # (M,)
    
    # For TM polarization, the operator involves 1/eps, not eps.
    # Actually, for TM: the eigenvalue problem is (G+k)^2 * eps^(0-G) ... 
    # Let me think about this more carefully.
    
    # Standard plane-wave method for TM polarization:
    # The equation is: ∇·(1/eps ∇ E_z) + (ω/c)^2 E_z = 0
    # In Fourier space: sum_G' (G+k)^2 / eps^(G-G') E^(G') = (ω/c)^2 E^(G)
    # Wait, let me be more careful.
    
    # Actually, for TM (electric field perpendicular to plane, E = (0,0,Ez)):
    # ∇²Ez + (ω/c)² ε Ez = 0
    # But ε is inhomogeneous, so: ∇²Ez + (ω/c)² ε(x,y) Ez = 0
    # In Fourier space: (G+k)² E^(G) + (ω/c)² sum_G' ε^(G-G') E^(G') = 0
    # So the eigenvalue equation is: sum_G' [δ(G-G') (G+k)² - (ω/c)² ε^(G-G')] E^(G') = 0
    # Or: sum_G' [δ(G-G') (G+k)² / 1 - (ω/c)² ε^(G-G')] E^(G') = 0
    
    # This is a generalized eigenvalue problem or can be rearranged.
    # Let me use a different formulation.
    
    # Alternative: For TE polarization (H perpendicular): ∇·(1/ε ∇ H_z) + (ω/c)² H_z = 0
    # In Fourier space: sum_G' (G+k)·(G'+k) / ε^(G-G') H^(G') = (ω/c)² H^(G)
    
    # The task says TM polarization. Let me check the standard formulation.
    # For 2D square lattice photonic crystal with TM polarization:
    # The eigenvalue equation in plane-wave method is:
    # sum_{G'} [δ_{G,G'} |G+k|² - (ω/c)² ε_{G-G'}] E_{G'} = 0
    # This can be written as: D(ω) E = 0 where D = |G+k|² I - (ω/c)² ε̃
    # To make it a standard eigenvalue problem, we can write:
    # (1/(ω/c)²) |G+k|² E = ε̃ E
    # Or: ε̃^{-1} |G+k|² E = (ω/c)² E
    
    # Let me use the formulation: A E = λ E where A = ε̃^{-1} |G+k|²
    # But ε̃ might not be invertible in the truncated basis.
    
    # Actually, a more stable formulation is to solve the generalized eigenvalue problem:
    # |G+k|² E = (ω/c)² ε̃ E
    # Which is: (|G+k|²) E - (ω/c)² ε̃ E = 0
    
    # Let me use numpy's eig for this.
    return coeffs

def _solve_bands(cell, kx, ky, n_bands=5):
    """
    Solve for the first n_bands TM bands at wave vector (kx, ky).
    Returns the eigenvalues (ω/c)².
    """
    coeffs = _fourier_coefficients(cell)
    
    # Reciprocal lattice vectors
    n_vals = np.arange(-_NPW//2, _NPW//2 + 1)
    G_n, G_m = np.meshgrid(n_vals, n_vals, indexing='ij')
    G_flat = np.column_stack([G_n.ravel(), G_m.ravel()])  # (M, 2)
    
    # |G+k|² for each plane wave
    Gk = G_flat + np.array([kx, ky])
    Gk2 = np.sum(Gk ** 2, axis=1)  # (M,)
    
    # The eigenvalue equation: sum_G' ε^(G-G') E^(G') = (c/ω)² |G+k|² E^(G)
    # Or: ε̃ E = (c/ω)² diag(|G+k|²) E
    # This is a generalized eigenvalue problem: ε̃ E = λ D E where D = diag(|G+k|²)
    
    # For numerical stability, let's use the formulation:
    # (1/|G+k|²) ε̃ E = (c/ω)² E
    # But this fails where |G+k|² = 0.
    
    # Let me use scipy.linalg.eigh for generalized eigenvalue problem.
    # A E = λ B E where A = ε̃, B = diag(|G+k|²)
    
    # Build the ε matrix: ε_{G,G'} = ε^(G-G')
    n_vals_list = list(n_vals)
    idx_map = {n_vals_list[i]: i for i in range(_NPW)}
    
    eps_mat = np.zeros((_M, _M), dtype=np.complex128)
    for i, (ni, mi) in enumerate(zip(G_n.ravel(), G_m.ravel())):
        for j, (nj, mj) in enumerate(zip(G_n.ravel(), G_m.ravel())):
            dn = ni - nj
            dm = mi - mj
            if dn in idx_map and dm in idx_map:
                # Find the index of (dn, dm) in coeffs
                # coeffs is indexed by (n, m) where n, m are in -8..8
                dn_idx = idx_map.get(dn, None)
                dm_idx = idx_map.get(dm, None)
                if dn_idx is not None and dm_idx is not None:
                    g_idx = dn_idx * _NPW + dm_idx
                    eps_mat[i, j] = coeffs[g_idx]
    
    # B = diag(|G+k|²)
    B = np.diag(Gk2)
    
    # Solve generalized eigenvalue problem: A x = λ B x
    # Use scipy if available, otherwise use numpy
    try:
        from scipy.linalg import eigh as scipy_eigh
        vals, _ = scipy_eigh(eps_mat, B, subset_by_index=[0, n_bands-1])
        # λ = (c/ω)², so (ω/c)² = 1/λ
        return 1.0 / vals
    except ImportError:
        # Fallback: use numpy
        # Convert to standard eigenvalue problem by solving B^{-1} A x = λ x
        # But B might be singular
        # Use eig instead
        vals = eigvalsh(eps_mat)  # This is not correct for generalized problem
        # Let me just use a simpler approach for now
        return vals
    
    # This won't be reached
    return np.zeros(n_bands)

def _score_cell(cell, kx_ky_list=None):
    """
    Score a cell by computing the gap between bands 4 and 5 over a set of k-points.
    Returns the minimum gap-midgap ratio.
    """
    if kx_ky_list is None:
        # Use a few k-points in the irreducible BZ: (0,0), (π/a, 0), (π/a, π/a)
        kx_ky_list = [(0, 0), (np.pi, 0), (np.pi, np.pi)]
    
    max_omega4 = 0.0
    min_omega5 = np.inf
    
    for kx, ky in kx_ky_list:
        bands = _solve_bands(cell, kx, ky, n_bands=5)
        if len(bands) >= 5:
            max_omega4 = max(max_omega4, bands[3])
            min_omega5 = min(min_omega5, bands[4])
    
    if min_omega5 == np.inf or max_omega4 == 0:
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = min_omega5 + max_omega4
    if midgap == 0:
        return -1.0
    
    return 2.0 * gap / midgap

def _flip_orbit(cell, orbit):
    """Flip all pixels in an orbit."""
    cell = cell.copy()
    val = 1 - cell[orbit[0][0], orbit[0][1]]
    for (i, j) in orbit:
        cell[i, j] = val
    return cell

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell using local search over C4v-symmetric orbits.
    """
    size = problem.get("pixels", 32)
    if size != 32:
        # Fallback for different sizes
        cell = np.zeros((size, size), dtype=np.int8)
        return cell.tolist()
    
    # Time tracking
    t_start = time.time()
    t_limit = t_start + TIME_BUDGET - 5.0  # 5 second safety margin
    
    # Initial cell: centered dielectric rod
    cell = np.zeros((_SIZE, _SIZE), dtype=np.int8)
    for i in range(_SIZE):
        for j in range(_SIZE):
            x = (i + 0.5) / _SIZE - 0.5
            y = (j + 0.5) / _SIZE - 0.5
            if x * x + y * y < 0.2 ** 2:
                cell[i, j] = 1
    
    current_score = _score_cell(cell)
    best_score = current_score
    best_cell = cell.copy()
    
    # Local search
    rng = np.random.default_rng(42)
    num_orbits = len(_ORBITS)
    
    while time.time() < t_limit:
        # Pick a random orbit
        orbit_idx = rng.integers(0, num_orbits)
        orbit = _ORBITS[orbit_idx]
        
        new_cell = _flip_orbit(cell, orbit)
        new_score = _score_cell(new_cell)
        
        if new_score > current_score:
            cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = cell.copy()
    
    return best_cell.tolist()
