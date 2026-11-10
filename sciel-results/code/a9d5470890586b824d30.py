"""Design a 32x32 symmetric photonic-crystal unit cell maximizing the TM 4-5 band gap."""
import time
import math
import numpy as np
from itertools import product

_DEADLINE_MARGIN = 30  # seconds to spare before the 230s kill
_START = None


def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START


def _remaining():
    return 230.0 - _elapsed() - _DEADLINE_MARGIN


def _symmetrize(cell):
    """Enforce 4-fold (D4) symmetry about the cell center via averaging, then threshold."""
    c = np.array(cell, dtype=float)
    # 90-degree rotations about center
    r90 = np.rot90(c, 1)
    r180 = np.rot90(c, 2)
    r270 = np.rot90(c, 3)
    # Transpose (reflection about anti-diagonal)
    t = c.T
    # Compose with rotations to get all 8 D4 elements
    sym = [c, r90, r180, r270, t, np.rot90(t, 1), np.rot90(t, 2), np.rot90(t, 3)]
    avg = np.mean(np.stack(sym), axis=0)
    return (avg >= 0.5).astype(int)


def _fourier_eps(cell, N):
    """Compute Fourier coefficients of the permittivity map on an N x N grid."""
    c = np.array(cell, dtype=float)
    eps = 1.0 + 11.0 * c  # eps_low=1, eps_high=12
    # 2D FFT gives coefficients; normalize by N^2
    F = np.fft.fft2(eps) / (N * N)
    # Reorder to zero-frequency center for clarity (fftshift)
    F = np.fft.fftshift(F)
    return F


def _band_structure(cell, N=17, kpts=None, eps_low=1.0, eps_high=12.0):
    """
    Fast plane-wave surrogate: compute TM bands via eigenvalue problem.
    Returns list of band values (omega*a/2pi) for each k-point.
    """
    if kpts is None:
        kpts = _default_kpts()

    # Fourier coefficients of 1/eps
    c = np.array(cell, dtype=float)
    eps = eps_low + (eps_high - eps_low) * c
    inv_eps = 1.0 / eps
    F = np.fft.fftshift(np.fft.fft2(inv_eps) / (N * N))

    # Build reciprocal lattice vectors for the N x N plane-wave set
    # Indices from -(N-1)//2 to (N-1)//2
    half = N // 2
    idx = np.arange(-half, half + 1)
    Gx, Gy = np.meshgrid(idx, idx, indexing='ij')
    G2 = Gx * Gx + Gy * Gy
    nvec = N * N

    bands = []
    for kx, ky in kpts:
        # (k + G)^2 for each plane wave
        kg2 = (kx + Gx) ** 2 + (ky + Gy) ** 2
        # Build the matrix: for TM, the eigenvalue problem is
        # sum_G'  (1/eps)(G - G') * (k + G')^2 * c_{G'} = omega^2 c_G
        # In normalized units, we solve det(...) = 0.
        # We construct the matrix M where M[G,G'] = F[G-G'] * (k+G')^2
        # and multiply by (k+G)^2 on the diagonal for the eigenvalue.
        # Actually the standard form: for each G, (k+G)^2 c_G - sum_{G'} F[G-G'] (k+G')^2 c_{G'} = omega^2 c_G
        # => sum_{G'} [ (k+G)^2 delta_{G,G'} - F[G-G'] (k+G')^2 ] c_{G'} = omega^2 c_G
        # This is a generalized problem but we can form a symmetric one.
        # Let's build the matrix A = diag(1/(k+G)^2) @ M where M[G,G'] = F[G-G'] (k+G')^2
        # Then A c = omega^2 c, but A is not symmetric.
        # Better: use the form where we solve for omega^2 directly.
        # Construct M[G, G'] = F[G - G'] * (k + G')^2
        # Then the eigenvalue problem is: sum_{G'} M[G,G'] c_{G'} = (k+G)^2 c_G * omega^2
        # => [ (k+G)^2 I - M ] c = omega^2 (k+G)^2 c ... still generalized.
        # Standard trick: define d_G = c_G / (k+G), then it becomes a standard eigenvalue problem.
        # Let's just build the matrix and use a symmetric formulation.
        # M[G,G'] = F[G-G'] * (k+G')^2 ; we want eigenvalues of (k+G)^(-2) * M
        # Define B = diag(1/(k+G)^2) @ M ; B is similar to a symmetric matrix.
        # Simpler: build the matrix directly as the "folded" operator.
        kg2_flat = kg2.ravel()
        # Avoid division by zero (shouldn't happen for our kpts)
        kg2_safe = np.where(kg2_flat > 1e-12, kg2_flat, 1e-12)

        # Build M: for each G (row) and G' (col), M[G,G'] = F[G - G'] * (k+G')^2
        # F is indexed with shifted indices; F[i,j] corresponds to G = (idx[i], idx[j])
        # G - G' has index (i - i', j - j') in the shifted array.
        # We need F at (i - i', j - j'). With fftshift, index 0 is at center.
        # The coefficient for G - G' is F[ (i - i' + half) % N, (j - j' + half) % N ]
        # Build using broadcasting: for row (i,j) and col (i',j'), we need F[(i-i'+half)%N, (j-j'+half)%N]
        ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        # For each (i,j) as G and (i',j') as G', the difference index
        # We'll build a matrix D where D[i,j,i',j'] = (i-i', j-j')
        # This is 4D and large; for N=17, 17^4 = 83521, manageable.
        # But let's do it more efficiently by computing the convolution structure.
        # M[G, G'] = F[G-G'] * kg2[G']
        # This is: for each G', multiply F shifted by G' by kg2[G'], then sum... 
        # Actually M is: M[G,G'] = F[G-G'] * kg2[G']
        # We can build M by: for each offset (di,dj), F[di,dj] contributes to M[G, G'] where G' = G - (di,dj)
        # Let's just build it directly with a loop over the 17x17=289 pairs... that's 289*289 = 83521 operations, fine.
        M = np.zeros((nvec, nvec))
        for ig in range(nvec):
            igx, igy = ig % N, ig // N
            for igp in range(nvec):
                igpx, igpy = igp % N, igp // N
                # G - G' in shifted index:
                dif_i = (igx - igpx + half) % N
                dif_j = (igy - igpy + half) % N
                M[ig, igp] = F[dif_i, dif_j] * kg2_flat[igp]
        # Now the eigenvalue problem: sum_{G'} M[G,G'] c_{G'} = (k+G)^2 * omega^2 * c_G
        # => M c = diag((k+G)^2) * omega^2 * c
        # => diag(1/(k+G)^2) M c = omega^2 c
        # Let A = diag(1/kg2_safe) @ M
        A = M * (1.0 / kg2_safe)[:, np.newaxis]
        # A is not symmetric, but it has real eigenvalues (similar to a symmetric matrix).
        # Use eigvalsh on a symmetrized version or just eigvals.
        # For safety, symmetrize: A_sym = (A + A.T) / 2 if close to symmetric.
        # Actually A should be similar to a symmetric matrix, so eigenvalues are real.
        # Use np.linalg.eigvalsh on (A + A.T)/2 as an approximation, or eigvals.
        try:
            eigvals = np.linalg.eigvalsh((A + A.T) / 2)
        except np.linalg.LinAlgError:
            eigvals = np.sort(np.real(np.linalg.eigvals(A)))
        eigvals = np.sort(eigvals)
        # omega^2 = eigvals, so omega = sqrt(eigvals)
        # In normalized units, omega here is in units of 1/a where a is lattice constant.
        # The actual omega*a/2pi = sqrt(eigval) / (2*pi) * ... 
        # For the plane-wave method with G in units of 2pi/a, omega is in units of 2pi/a.
        # So omega*a/(2pi) = sqrt(eigval) / (2pi) * 2pi = sqrt(eigval).
        # Wait, let me reconsider. The eigenvalue lambda = omega^2 * (a/2pi)^2 or similar.
        # For simplicity, the relative band positions are what matter for the gap ratio.
        # The gap ratio is scale-invariant, so we can use sqrt(eigval) directly.
        bands.append(np.sqrt(np.clip(eigvals, 0, None)))

    return bands


def _default_kpts():
    """30 k-points in the irreducible BZ of a square lattice."""
    # Irreducible BZ: 0 <= ky <= kx, 0 <= kx <= pi/a (in units of 2pi/a, so 0 to 0.5)
    # Boundary points and interior
    kpts = []
    # Gamma
    kpts.append((0.0, 0.0))
    # M (pi/2, pi/2) in 2pi/a units => (0.5, 0.5)
    kpts.append((0.5, 0.5))
    # X (0.5, 0)
    kpts.append((0.5, 0.0))
    # Along Gamma-M: (t/2, t/2) for t in [0,1]
    for t in np.linspace(0.1, 0.9, 5):
        kpts.append((t / 2, t / 2))
    # Along M-X: (0.5, t/2) for t in [0,1]
    for t in np.linspace(0.1, 0.9, 5):
        kpts.append((0.5, t / 2))
    # Along X-Gamma: (t/2, 0) for t in [0,1]
    for t in np.linspace(0.1, 0.9, 5):
        kpts.append((t / 2, 0.0))
    # Interior points
    for t1 in [0.2, 0.3, 0.4]:
        for t2 in [0.1, 0.2, 0.3]:
            if t2 <= t1:
                kpts.append((t1 / 2, t2 / 2))
    # Deduplicate and limit to 30
    seen = set()
    unique = []
    for k in kpts:
        key = (round(k[0], 4), round(k[1], 4))
        if key not in seen:
            seen.add(key)
            unique.append(k)
    return unique[:30]


def _gap_ratio(bands, lower_band=4, upper_band=5):
    """Compute the gap-midgap ratio between bands lower_band and upper_band."""
    # bands is a list of arrays, one per k-point
    # For each k-point, get the band values
    # max omega_4 = max over kpts of bands[kpt][lower_band-1]
    # min omega_5 = min over kpts of bands[kpt][upper_band-1]
    max_lower = max(b[lower_band - 1] for b in bands)
    min_upper = min(b[upper_band - 1] for b in bands)
    gap = min_upper - max_lower
    midgap = min_upper + max_lower
    if midgap < 1e-12:
        return 0.0
    return 2.0 * gap / midgap


def _random_symmetric_cell(size=32, density=0.5, rng=None):
    """Generate a random D4-symmetric binary cell."""
    if rng is None:
        rng = np.random.default_rng()
    # Generate a fundamental domain (1/8 of the cell) and reflect
    # For 32x32, the fundamental domain under D4 is an 8x8 triangle-ish region.
    # Simpler: generate full random, then symmetrize.
    c = (rng.random((size, size)) < density).astype(int)
    return _symmetrize(c)


def _optimize_cell(initial_cell, size=32, max_iters=500, n_samples=200, rng=None):
    """
    Local search / hill-climbing on the binary cell to maximize the gap ratio.
    Uses a fast surrogate (N=17 plane waves) for evaluation.
    """
    if rng is None:
        rng = np.random.default_rng()

    best_cell = initial_cell.copy()
    best_score = _gap_ratio(_band_structure(best_cell, N=17))

    for it in range(max_iters):
        if _remaining() < 5:
            break

        # Pick a random pixel in the fundamental domain to flip
        # To preserve symmetry, flip a whole orbit
        i = rng.integers(0, size)
        j = rng.integers(0, size)

        # Compute the orbit under D4
        c = np.array(best_cell)
        new_c = c.copy()
        orbit = set()
        # Identity
        orbit.add((i, j))
        # Rotations
        ri, rj = (size - 1 - j, i)  # 90 deg
        orbit.add((ri, rj))
        ri, rj = (size - 1 - i, size - 1 - j)  # 180
        orbit.add((ri, rj))
        ri, rj = (j, size - 1 - i)  # 270
        orbit.add((ri, rj))
        # Transpose
        orbit.add((j, i))
        ri, rj = (size - 1 - i, j)  # rot90(transpose)
        orbit.add((ri, rj))
        ri, rj = (size - 1 - j, size - 1 - i)  # rot180(transpose)
        orbit.add((ri, rj))
        ri, rj = (i, size - 1 - j)  # rot270(transpose)
        orbit.add((ri, rj))

        # Flip all in orbit
        for oi, oj in orbit:
            new_c[oi, oj] = 1 - new_c[oi, oj]

        # Evaluate
        score = _gap_ratio(_band_structure(new_c, N=17))

        if score > best_score:
            best_cell = new_c
            best_score = score

    return best_cell, best_score


def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell maximizing the TM 4-5 band gap."""
    global _START
    _START = time.time()

    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    lower_band = problem.get("lower_band", 4)

    rng = np.random.default_rng(42)

    # Start with a reasonable initial guess: a ring or cross pattern
    # Try a few initial structures
    candidates = []

    # 1. Central dielectric disk
    cell1 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.15 ** 2:
                cell1[i, j] = 1
    candidates.append(cell1)

    # 2. Air disk in dielectric background (inverse)
    cell2 = np.ones((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.2 ** 2:
                cell2[i, j] = 0
    candidates.append(cell2)

    # 3. Cross pattern
    cell3 = np.zeros((size, size), dtype=int)
    center = size // 2
    for i in range(size):
        for j in range(size):
            if abs(i - center + 0.5) < 2.5 or abs(j - center + 0.5) < 2.5:
                cell3[i, j] = 1
    candidates.append(cell3)

    # 4. Random symmetric
    candidates.append(_random_symmetric_cell(size, density=0.4, rng=rng))
    candidates.append(_random_symmetric_cell(size, density=0.5, rng=rng))

    # Evaluate all candidates with the surrogate
    best_cell = None
    best_score = -np.inf
    for c in candidates:
        c = _symmetrize(c)
        score = _gap_ratio(_band_structure(c, N=17), lower_band=lower_band)
        if score > best_score:
            best_score = score
            best_cell = c.copy()

    # Local search from the best candidate
    if _remaining() > 10:
        best_cell, best_score = _optimize_cell(
            best_cell, size=size, max_iters=300, n_samples=200, rng=rng
        )

    # Try a few more random restarts if time permits
    restarts = 3
    for r in range(restarts):
        if _remaining() < 15:
            break
        c = _random_symmetric_cell(size, density=rng.uniform(0.3, 0.6), rng=rng)
        c = _symmetrize(c)
        score = _gap_ratio(_band_structure(c, N=17), lower_band=lower_band)
        if score > best_score:
            best_score = score
            best_cell = c.copy()
        if _remaining() > 10:
            best_cell, best_score = _optimize_cell(
                best_cell, size=size, max_iters=100, n_samples=100, rng=rng
            )

    # Ensure the output is a nested list of 0/1 ints
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    return result
