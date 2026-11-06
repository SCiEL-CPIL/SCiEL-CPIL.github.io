"""
Photonic crystal band gap optimization.

Combines:
- The structural seed idea from the inspiration (four-quadrant rods, i.e. a
  four-hole / four-rod superlattice motif) which is a different design family
  from the single-centre shapes in the current program.
- The refinement machinery of the current program (fast PWE scoring + greedy
  pixel-flip local search with D4 symmetry enforcement).

Key structural change: start from a four-quadrant pattern and use a circular
plane-wave basis (more modes, better convergence) for the internal solver.
"""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _enforce_d4_symmetry(cell):
    """
    Enforce D4 symmetry (90-degree rotational symmetry + reflections) on a
    square array.  Returns the element-wise maximum over all 8 group elements,
    which is equivalent to majority voting for binary cells.
    """
    c = np.array(cell, dtype=np.int8)
    r0   = c
    r90  = np.rot90(c, 1)
    r180 = np.rot90(c, 2)
    r270 = np.rot90(c, 3)
    rx   = c[:, ::-1]
    ry   = c[::-1, :]
    rd   = c.T
    rad  = rd[:, ::-1]
    return (r0 + r90 + r180 + r270 + rx + ry + rd + rad >= 4).astype(np.int8)


def _make_four_quadrant(rod_radius, exponent=2.0, size=32, offset=0.25):
    """
    Create a cell with four rods (or four holes, depending on interpretation)
    centred in each quadrant.  This is a structurally different seed from a
    single central rod.
    """
    cell = np.zeros((size, size), dtype=np.int8)
    coords = (np.arange(size) + 0.5) / size - 0.5
    centers = [(-offset, -offset), (offset, -offset),
               (-offset,  offset), (offset,  offset)]
    xx, yy = np.meshgrid(coords, coords, indexing='ij')
    for cx, cy in centers:
        dx = xx - cx
        dy = yy - cy
        if rod_radius > 1e-9:
            val = np.abs(dx / rod_radius) ** exponent + np.abs(dy / rod_radius) ** exponent
        else:
            val = np.zeros_like(xx)
        cell[val <= 1.0 + 1e-9] = 1
    return _enforce_d4_symmetry(cell)


def _make_single_rod(radius, size=32):
    """Single central rod for comparison."""
    cell = np.zeros((size, size), dtype=np.int8)
    center = size / 2.0
    for i in range(size):
        for j in range(size):
            x = i - center + 0.5
            y = j - center + 0.5
            if x * x + y * y < radius * radius:
                cell[i, j] = 1
    return _enforce_d4_symmetry(cell)


def _compute_gap(cell, size=32, n_pw=13, eps_high=12.0, eps_low=1.0):
    """
    Compute the gap-midgap ratio between TM bands 4 and 5.
    Uses a circular plane-wave basis and FFT-based Fourier coefficients.
    """
    cell = np.array(cell, dtype=float)
    eps = eps_low + (eps_high - eps_low) * cell
    inv_eps = 1.0 / eps
    N = size

    # Fourier coefficients of 1/eps via FFT
    fft_coeffs = np.fft.fft2(inv_eps) / (N * N)

    # Circular plane-wave basis: include G with |G| <= (half + 0.5)
    half = n_pw // 2
    g_indices = []
    for m in range(-half, half + 1):
        for n in range(-half, half + 1):
            if m * m + n * n <= half * half + half:
                g_indices.append((m, n))
    n_total = len(g_indices)

    # Precompute S matrix: S[i,j] = inv_eps(G_i - G_j)
    S = np.zeros((n_total, n_total), dtype=complex)
    for i in range(n_total):
        mi, ni = g_indices[i]
        for j in range(n_total):
            gm = (mi - g_indices[j][0]) % N
            gn = (ni - g_indices[j][1]) % N
            S[i, j] = fft_coeffs[gm, gn]

    # k-points along the irreducible BZ boundary (in units of 2*pi/a)
    # Irreducible triangle: Gamma(0,0) - X(0.5,0) - M(0.5,0.5)
    k_points = []
    n_seg = 10
    for t in np.linspace(0.0, 0.5, n_seg + 1):
        k_points.append((t, 0.0))
    for t in np.linspace(0.0, 0.5, n_seg + 1):
        k_points.append((0.5, t))
    for t in np.linspace(0.5, 0.0, n_seg + 1):
        k_points.append((t, t))

    # Remove duplicates
    k_unique = []
    for kp in k_points:
        if not any(abs(kp[0] - k2[0]) < 1e-10 and abs(kp[1] - k2[1]) < 1e-10
                   for k2 in k_unique):
            k_unique.append(kp)

    max_omega4 = 0.0
    min_omega5 = np.inf

    for kx, ky in k_unique:
        Gk = np.array([(m + kx, n + ky) for m, n in g_indices], dtype=float)
        Gk_sq = Gk[:, 0] ** 2 + Gk[:, 1] ** 2

        # For TM: eigenvalue problem is (k+G)^2 * (1/eps) matrix
        # A[i,j] = (k+Gi).(k+Gj) * inv_eps(Gi-Gj)
        # This is a standard (not generalized) eigenvalue problem if we
        # absorb 1/eps into the matrix.
        A = S * Gk_sq[None, :]
        # Symmetrize
        A = (A + A.conj().T) / 2.0

        try:
            eigvals = np.linalg.eigvalsh(A)
            eigvals = np.real(eigvals)
            eigvals = eigvals[eigvals > 1e-10]
        except Exception:
            continue

        if len(eigvals) < 5:
            continue

        b4 = eigvals[3]
        b5 = eigvals[4]

        if b4 > max_omega4:
            max_omega4 = b4
        if b5 < min_omega5:
            min_omega5 = b5

    if min_omega5 == np.inf or max_omega4 <= 0:
        return -1.0

    midgap = 0.5 * (min_omega5 + max_omega4)
    if midgap <= 0:
        return -1.0

    return 2.0 * (min_omega5 - max_omega4) / midgap


def design(problem):
    """
    Design the photonic crystal unit cell.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)

    start_time = time.time()
    deadline = start_time + 195.0

    # --- Phase 1: Evaluate a diverse set of geometric seeds ---
    candidates = []

    # Four-quadrant rods with various radii and exponents
    for r in [0.06, 0.08, 0.10, 0.12, 0.14, 0.16]:
        for exp in [2.0, 2.5, 3.0]:
            candidates.append(_make_four_quadrant(r, exp, size))

    # Single central rods for comparison
    for r in [0.10, 0.15, 0.20, 0.25]:
        candidates.append(_make_single_rod(r * size, size))

    best_cell = None
    best_score = -1.0

    for cell in candidates:
        if time.time() > deadline - 60:
            break
        try:
            score = _compute_gap(cell, size, n_pw=13, eps_high=eps_high, eps_low=eps_low)
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best_cell = cell

    if best_cell is None:
        best_cell = _make_four_quadrant(0.12, 2.0, size)

    # --- Phase 2: Local search via pixel flips ---
    current_cell = np.array(best_cell, dtype=np.int8)
    current_score = best_score
    rng = np.random.default_rng(42)

    while time.time() < deadline - 5:
        # Mutation: flip 1-3 random pixels
        mut_cell = current_cell.copy()
        n_flips = int(rng.integers(1, 4))
        for _ in range(n_flips):
            i = int(rng.integers(0, size))
            j = int(rng.integers(0, size))
            mut_cell[i, j] = 1 - mut_cell[i, j]

        sym_cell = _enforce_d4_symmetry(mut_cell)

        # Skip if unchanged
        if np.array_equal(sym_cell, current_cell):
            continue

        try:
            score = _compute_gap(sym_cell, size, n_pw=13, eps_high=eps_high, eps_low=eps_low)
        except Exception:
            score = -1.0

        if score > current_score:
            current_cell = sym_cell
            current_score = score

    return current_cell.tolist()

# EVOLVE-BLOCK-END
