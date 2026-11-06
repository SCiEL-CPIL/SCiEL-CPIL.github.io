"""Photonic-crystal 4-5 TM band-gap designer.

Strategy:
- Model the dielectric region as a central shape defined by a radial function r(theta).
- Use harmonics compatible with square symmetry (m=4, 8) to maintain C4v symmetry.
- Precompute Fourier coefficients of 1/epsilon for the pixel map.
- Use a robust, vectorized plane-wave Galerkin solver.
- Optimize the few radial parameters using Nelder-Mead.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 220.0  # wall-clock safety margin


def _symmetrize(m):
    """Enforce square symmetry about the cell centre."""
    m = np.asarray(m, dtype=np.float64)
    # The problem requires: cell == cell[::-1] row-wise and column-wise and cell == its transpose
    # This is D4 symmetry (square symmetry group).
    m = (m + m.T + m[::-1, ::-1] + m.T[::-1, ::-1]) / 4.0
    return m


def _make_cell_from_radial(params, size):
    """
    Build a symmetric 0/1 cell from radial parameters.
    params = [R0, A4, A8]
    R0: base radius
    A4: amplitude of cos(4*theta)
    A8: amplitude of cos(8*theta)
    """
    R0 = params[0]
    A4 = params[1]
    A8 = params[2]
    
    # Pixel centers
    i2, j2 = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    # Coordinates centered at 0, in units of cell size (0 to 1)
    # Cell is [-0.5, 0.5] x [-0.5, 0.5]
    x = (i2 + 0.5) / size - 0.5
    y = (j2 + 0.5) / size - 0.5
    
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Radial function
    # We want the dielectric (1) inside the boundary r(theta)
    # r(theta) = R0 + A4*cos(4*theta) + A8*cos(8*theta)
    r_boundary = R0 + A4 * np.cos(4 * theta) + A8 * np.cos(8 * theta)
    
    # Mask: 1 if r < r_boundary
    mask = (r < r_boundary).astype(np.float64)
    
    # Apply symmetry to fix any pixel discretization errors
    mask = _symmetrize(mask)
    
    # Binarize: > 0.5 is dielectric (1), else air (0)
    # Ensure strict 0/1
    cell = (mask > 0.5).astype(int)
    
    return cell


def _fourier_coeffs(eps, N, size):
    """
    Return array of Fourier coefficients of 1/eps for |mx|,|my| <= N.
    Shape: (2*N+1, 2*N+1)
    Index offset: N
    c[mx+N, my+N] = sum_{i,j} (1/eps_{ij}) exp(-i 2 pi (mx*i + my*j)/size) / size^2
    """
    inv = 1.0 / eps.astype(np.float64)
    # Use FFT for speed
    # fft2 computes sum exp(-i 2 pi (m i + n j)/size)
    # We need to divide by size^2
    fft_vals = np.fft.fft2(inv) / (size * size)
    
    # Extract the central (2N+1)x(2N+1) block
    # The FFT output has indices 0..size-1.
    # Frequency 0 is at index 0.
    # Frequency -N is at index size-N.
    # We want frequencies from -N to N.
    
    # Construct the grid of desired frequencies
    freqs = np.arange(-N, N+1)
    
    # Map desired freqs to FFT indices
    # idx = freq % size
    indices = freqs % size
    
    # Extract sub-array
    # Note: fft2 output is ordered such that index 0 is freq 0, 1 is freq 1, ..., size/2 is Nyquist, etc.
    # For even size, freq -1 is at size-1.
    # So we can just index using the modded values.
    
    c_full = fft_vals[np.ix_(indices, indices)]
    
    # The result should be real (for real input), but due to numerical noise, take real part.
    # Also, for a symmetric real function, the coefficients are real and symmetric.
    return c_full.real


def _band_structure_tm(eps, kx, ky, N, n_bands, size, coeffs):
    """
    Return the lowest n_bands TM eigenfrequencies.
    coeffs: precomputed (2N+1, 2N+1) array of Fourier coeffs of 1/eps.
    """
    m = np.arange(-N, N + 1)
    # G vectors
    Gx = m[:, None] * np.ones((1, len(m)))
    Gy = np.ones((len(m), 1)) * m[None, :]
    
    # Flatten G vectors
    G = np.stack([Gx.ravel(), Gy.ravel()], axis=-1) # (M, 2)
    M = len(G)
    
    kvec = np.array([kx, ky])
    K = G + kvec
    K2 = np.sum(K * K, axis=1)
    
    # Build the matrix D
    # D_{ij} = (K_i . K_j) * c_{G_i - G_j}
    
    # We need c_{G_i - G_j}
    # G_i - G_j = (Gx_i - Gx_j, Gy_i - Gy_j)
    # The coefficient c is indexed by (mx, my) in the range [-N, N].
    # If the difference is outside [-N, N], the coefficient is 0 (assuming bandlimiting).
    
    # Create meshgrid of G components
    Gx_flat = G[:, 0]
    Gy_flat = G[:, 1]
    
    # Differences
    dGx = Gx_flat[:, None] - Gx_flat[None, :] # (M, M)
    dGy = Gy_flat[:, None] - Gy_flat[None, :] # (M, M)
    
    # Map differences to indices in coeffs array
    # coeffs index for mx is mx + N
    idx_x = dGx.astype(int) + N
    idx_y = dGy.astype(int) + N
    
    # Create a mask for valid indices
    valid = (idx_x >= 0) & (idx_x < 2*N + 1) & (idx_y >= 0) & (idx_y < 2*N + 1)
    
    # Initialize c_matrix with zeros
    c_matrix = np.zeros((M, M), dtype=np.float64)
    
    # Fill valid entries
    # coeffs is (2N+1, 2N+1)
    c_matrix[valid] = coeffs[np.ix_(idx_x[valid], idx_y[valid])]
    
    # Dot products K_i . K_j
    K_dot = np.dot(K, K.T)
    
    # D matrix
    D = K_dot * c_matrix
    
    # Ensure symmetry
    D = (D + D.T) / 2.0
    
    # Solve
    # omega^2 = 0.5 * eigenvalue(D)
    # We want omega, so sqrt(eigenvalue/2)
    try:
        vals = np.linalg.eigvalsh(D)
    except np.linalg.LinAlgError:
        # Fallback: add small jitter to diagonal if it fails
        D_reg = D + np.eye(M) * 1e-12
        vals = np.linalg.eigvalsh(D_reg)
        
    vals = np.sort(vals)
    # omega = sqrt(max(val, 0) / 2)
    omegas = np.sqrt(np.maximum(vals[:n_bands], 0.0) / 2.0)
    return omegas


def _gap_ratio(eps, size, N=13):
    """
    Calculate the 4-5 TM band gap ratio.
    """
    coeffs = _fourier_coeffs(eps, N, size)
    
    # K-points in the irreducible Brillouin zone
    # Square lattice, 2pi/a units.
    # Irreducible wedge: 0 <= ky <= kx, 0 <= kx <= 0.5
    # Plus the X-M edge: kx=0.5, 0<=ky<=0.5
    
    kpts = []
    # Gamma to X
    for t in np.linspace(0.0, 0.5, 8):
        kpts.append((t, 0.0))
    # X to M
    for t in np.linspace(0.0, 0.5, 8):
        kpts.append((0.5, t))
    # M to Gamma
    for t in np.linspace(0.0, 0.5, 8):
        kpts.append((t, t))
    # Interior
    for t in [0.1, 0.2, 0.3, 0.4]:
        for s in [0.05, 0.15, 0.25, 0.35]:
            if s <= t:
                kpts.append((t, s))
                
    max4 = -np.inf
    min5 = np.inf
    
    for kx, ky in kpts:
        w = _band_structure_tm(eps, kx, ky, N, 5, size, coeffs)
        if len(w) < 5:
            continue
        max4 = max(max4, w[3])
        min5 = min(min5, w[4])
        
    if not np.isfinite(min5) or not np.isfinite(max4) or (min5 + max4) <= 0:
        return -1.0
        
    return 2.0 * (min5 - max4) / (min5 + max4)


def design(problem):
    global _DEADLINE
    t0 = time.time()
    size = int(problem["pixels"])
    eps_low = float(problem.get("eps_low", 1.0))
    eps_high = float(problem.get("eps_high", 12.0))
    
    # Parameters: R0, A4, A8
    # R0 in units of cell size (0.5 is half width)
    # Typical dielectric fraction ~ 0.5
    # Area ~ pi * R0^2. If R0=0.4, area ~ 0.5.
    
    best_cell = None
    best_score = -np.inf
    
    def objective(params):
        if time.time() > _DEADLINE - 10:
            return 1e9
        cell = _make_cell_from_radial(params, size)
        score = _gap_ratio(cell, size)
        return -score
    
    # Initial guess
    # Try a few starting points
    starts = [
        [0.40, 0.05, 0.02],
        [0.45, 0.08, 0.03],
        [0.35, 0.06, 0.02],
        [0.50, 0.10, 0.04],
    ]
    
    for p0 in starts:
        if time.time() > _DEADLINE - 30:
            break
            
        try:
            res = minimize(objective, p0, method="Nelder-Mead",
                           options={"maxiter": 100, "xatol": 0.005, "fatol": 1e-4,
                                    "adaptive": True})
            if res.x is not None and np.isfinite(res.fun):
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_cell = _make_cell_from_radial(res.x, size)
        except Exception:
            continue
            
    # If no valid cell found, return a default symmetric cell
    if best_cell is None:
        # Simple disk
        p_default = [0.40, 0.0, 0.0]
        best_cell = _make_cell_from_radial(p_default, size)
        
    # Ensure strict symmetry and 0/1 integers
    best_cell = _symmetrize(best_cell)
    best_cell = (best_cell > 0.5).astype(int)
    
    # Return as nested list
    return [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
