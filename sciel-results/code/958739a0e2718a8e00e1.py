"""Design a 32x32 symmetric photonic crystal unit cell for maximum TM band gap."""
import time
import math
import numpy as np

_DEADLINE = 220.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _enforce_d4(cell):
    """Enforce full D4 symmetry on the cell."""
    size = len(cell)
    result = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            positions = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, i),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i),
            ]
            vals = [cell[pi][pj] for pi, pj in positions]
            result[i][j] = 1 if sum(vals) >= 4 else 0
    return result


def _make_cell_disk(r):
    size = 32
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < r * r:
                cell[i][j] = 1
    return _enforce_d4(cell)


def _make_cell_ring(r_in, r_out):
    size = 32
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = math.sqrt(x * x + y * y)
            if r_in <= r < r_out:
                cell[i][j] = 1
    return _enforce_d4(cell)


def _make_cell_combined(r_disk, r_ring_in, r_ring_out):
    size = 32
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = math.sqrt(x * x + y * y)
            if r < r_disk or r_ring_in <= r < r_ring_out:
                cell[i][j] = 1
    return _enforce_d4(cell)


def _make_cell_cross(bar_w, cross_l):
    size = 32
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < bar_w / 2.0 and abs(y) < cross_l / 2.0:
                cell[i][j] = 1
            if abs(y) < bar_w / 2.0 and abs(x) < cross_l / 2.0:
                cell[i][j] = 1
    return _enforce_d4(cell)


def _make_cell_four_rods(r, offset):
    """Four rods at (±offset, 0) and (0, ±offset)."""
    size = 32
    cell = [[0] * size for _ in range(size)]
    centers = [
        (offset, 0.0),
        (-offset, 0.0),
        (0.0, offset),
        (0.0, -offset),
    ]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy < r * r:
                    cell[i][j] = 1
                    break
    return _enforce_d4(cell)


def _make_cell_diamond(r):
    """Diamond-shaped (rotated square) dielectric region."""
    size = 32
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) + abs(y) < r:
                cell[i][j] = 1
    return _enforce_d4(cell)


def _make_cell_hexagon(r):
    """Hexagonal dielectric region (approximated)."""
    size = 32
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Hexagon: max(|x|, |y|/sqrt(3)) < r approximately
            # Use a simpler diamond+circle hybrid
            r_x = abs(x)
            r_y = abs(y)
            # Hexagonal norm approximation
            h = max(r_x, r_y * 0.866, (r_x + r_y) * 0.5)
            if h < r:
                cell[i][j] = 1
    return _enforce_d4(cell)


def _make_cell_stripe_h(period, width):
    """Horizontal stripes."""
    size = 32
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            y = (j + 0.5) / size
            if (y % period) < width:
                cell[i][j] = 1
    return _enforce_d4(cell)


def _make_cell_checker(block):
    """Checkerboard pattern with block size."""
    size = 32
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (i // block + j // block) % 2 == 0:
                cell[i][j] = 1
    return _enforce_d4(cell)


def _compute_tm_gap(cell, N_pw=13, n_kpts=15):
    """
    Compute the TM band gap between bands 4 and 5.
    Uses plane-wave expansion with exact Fourier coefficients.
    """
    size = len(cell)
    eps_high = 12.0
    eps_low = 1.0

    # Precompute 1/eps for each pixel
    inv_eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            inv_eps[i][j] = 1.0 / (eps_high if cell[i][j] == 1 else eps_low)

    # Compute Fourier coefficients using FFT for speed
    fft_coeffs = np.fft.fft2(inv_eps)
    N = size

    def get_coeff(m, n):
        idx_m = m % N
        idx_n = n % N
        return fft_coeffs[idx_m, idx_n] / (N * N)

    # G vectors: m, n in range [-N_pw+1, N_pw-1]
    g_indices = []
    for m in range(-N_pw + 1, N_pw):
        for n in range(-N_pw + 1, N_pw):
            g_indices.append((m, n))

    npw = len(g_indices)

    # Precompute S matrix: S[i,j] = c_{m_i - m_j, n_i - n_j}
    S = np.zeros((npw, npw), dtype=float)
    for i, (mi, ni) in enumerate(g_indices):
        for j, (mj, nj) in enumerate(g_indices):
            gm = mi - mj
            gn = ni - nj
            S[i, j] = get_coeff(gm, gn).real

    # k-points in the irreducible BZ
    # Square lattice BZ: 0 <= ky <= kx, kx + ky <= pi
    # Using k in units of pi/a: Gamma=(0,0), X=(1,0), M=(1,1)
    # Irreducible: 0 <= ky <= kx, kx + ky <= 1
    kpts = []
    # Along kx axis (Gamma to X)
    for i in range(n_kpts // 3 + 1):
        kx = i / (n_kpts // 3) * 1.0
        kpts.append((kx, 0.0))
    # Along kx=ky diagonal (Gamma to M)
    for i in range(1, n_kpts // 3 + 1):
        kx = i / (n_kpts // 3) * 0.5
        kpts.append((kx, kx))
    # Along the line kx+ky=1 (X to M)
    for i in range(1, n_kpts // 3):
        kx = 1.0 - i / (n_kpts // 3) * 0.5
        ky = i / (n_kpts // 3) * 0.5
        if ky >= 0 and kx >= ky:
            kpts.append((kx, ky))
    # Add some interior points
    kpts.append((0.5, 0.25))
    kpts.append((0.75, 0.25))
    kpts.append((0.6, 0.3))

    # Remove duplicates
    unique_kpts = []
    seen = set()
    for kx, ky in kpts:
        if ky < 0:
            ky = -ky
        if kx < ky:
            kx, ky = ky, kx
        if kx + ky > 1.0:
            scale = 1.0 / (kx + ky)
            kx *= scale
            ky *= scale
        key = (round(kx, 6), round(ky, 6))
        if key not in seen:
            seen.add(key)
            unique_kpts.append((kx, ky))

    kpts = unique_kpts[:n_kpts]

    max_omega4 = 0.0
    min_omega5 = float('inf')

    for kx, ky in kpts:
        # k in units of pi/a, so G = 2(m,n) in same units
        # |G_j + k|^2 = (2*m_j + kx)^2 + (2*n_j + ky)^2
        km2 = np.zeros(npw)
        for idx, (m, n) in enumerate(g_indices):
            km2[idx] = (2.0 * m + kx) ** 2 + (2.0 * n + ky) ** 2

        # A = S * diag(km2)
        A = S * km2[np.newaxis, :]

        # Solve eigenvalue problem A * E = omega^2 * E
        try:
            eigenvalues = np.linalg.eigvalsh(A)
            eigenvalues = np.real(eigenvalues)
            eigenvalues = np.sort(eigenvalues)

            if len(eigenvalues) >= 5:
                omega4_sq = eigenvalues[3]
                omega5_sq = eigenvalues[4]

                if omega4_sq > 0 and omega5_sq > 0:
                    omega4 = np.sqrt(omega4_sq)
                    omega5 = np.sqrt(omega5_sq)

                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
        except:
            continue

    if min_omega5 == float('inf') or max_omega4 == 0:
        return 0.0

    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0

    if midgap <= 0:
        return 0.0

    return 2.0 * gap / midgap


def _compute_tm_gap_fast(cell, N_pw=13, n_kpts=12):
    """
    Fast TM band gap computation with fewer k-points for optimization loops.
    """
    size = len(cell)
    eps_high = 12.0
    eps_low = 1.0

    inv_eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            inv_eps[i][j] = 1.0 / (eps_high if cell[i][j] == 1 else eps_low)

    fft_coeffs = np.fft.fft2(inv_eps)
    N = size

    def get_coeff(m, n):
        idx_m = m % N
        idx_n = n % N
        return fft_coeffs[idx_m, idx_n] / (N * N)

    g_indices = []
    for m in range(-N_pw + 1, N_pw):
        for n in range(-N_pw + 1, N_pw):
            g_indices.append((m, n))

    npw = len(g_indices)

    S = np.zeros((npw, npw), dtype=float)
    for i, (mi, ni) in enumerate(g_indices):
        for j, (mj, nj) in enumerate(g_indices):
            gm = mi - mj
            gn = ni - nj
            S[i, j] = get_coeff(gm, gn).real

    # Key high-symmetry k-points
    kpts = [
        (0.0, 0.0),    # Gamma
        (0.5, 0.0),    # Mid X
        (1.0, 0.0),    # X
        (0.25, 0.25),  # Mid M
        (0.5, 0.5),    # M
        (0.75, 0.0),   # 3/4 X
        (0.5, 0.25),   # Interior
        (0.75, 0.25),  # Interior
    ]

    max_omega4 = 0.0
    min_omega5 = float('inf')

    for kx, ky in kpts:
        km2 = np.zeros(npw)
        for idx, (m, n) in enumerate(g_indices):
            km2[idx] = (2.0 * m + kx) ** 2 + (2.0 * n + ky) ** 2

        A = S * km2[np.newaxis, :]

        try:
            eigenvalues = np.linalg.eigvalsh(A)
            eigenvalues = np.real(eigenvalues)
            eigenvalues = np.sort(eigenvalues)

            if len(eigenvalues) >= 5:
                omega4_sq = eigenvalues[3]
                omega5_sq = eigenvalues[4]

                if omega4_sq > 0 and omega5_sq > 0:
                    omega4 = np.sqrt(omega4_sq)
                    omega5 = np.sqrt(omega5_sq)

                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
        except:
            continue

    if min_omega5 == float('inf') or max_omega4 == 0:
        return 0.0

    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0

    if midgap <= 0:
        return 0.0

    return 2.0 * gap / midgap


def _get_field_intensities(cell, band_idx, N_pw=13):
    """
    Get the spatial intensity pattern of a specific band at Gamma point.
    Returns a 32x32 array of intensities.
    """
    size = len(cell)
    eps_high = 12.0
    eps_low = 1.0

    inv_eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            inv_eps[i][j] = 1.0 / (eps_high if cell[i][j] == 1 else eps_low)

    fft_coeffs = np.fft.fft2(inv_eps)
    N = size

    def get_coeff(m, n):
        idx_m = m % N
        idx_n = n % N
        return fft_coeffs[idx_m, idx_n] / (N * N)

    g_indices = []
    for m in range(-N_pw + 1, N_pw):
        for n in range(-N_pw + 1, N_pw):
            g_indices.append((m, n))

    npw = len(g_indices)

    S = np.zeros((npw, npw), dtype=float)
    for i, (mi, ni) in enumerate(g_indices):
        for j, (mj, nj) in enumerate(g_indices):
            gm = mi - mj
            gn = ni - nj
            S[i, j] = get_coeff(gm, gn).real

    # At Gamma: k = (0,0)
    km2 = np.zeros(npw)
    for idx, (m, n) in enumerate(g_indices):
        km2[idx] = (2.0 * m) ** 2 + (2.0 * n) ** 2

    A = S * km2[np.newaxis, :]

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        eigenvalues = np.real(eigenvalues)
        # Sort
        idx_sort = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx_sort]
        eigenvectors = eigenvectors[:, idx_sort]

        if band_idx < len(eigenvalues):
            vec = eigenvectors[:, band_idx].real
            # Reconstruct field: E(r) = sum_G c_G exp(i(G+k)·r)
            # At Gamma, k=0
            field = np.zeros((size, size), dtype=complex)
            for idx, (m, n) in enumerate(g_indices):
                field += vec[idx] * np.exp(2j * np.pi * (m * np.arange(size) / size[:, np.newaxis] + n * np.arange(size) / size[np.newaxis, :]))

            intensity = np.abs(field) ** 2
            # Normalize
            if intensity.max() > 0:
                intensity = intensity / intensity.max()
            return intensity
    except:
        return None


def _topological_optimize(cell, max_iters=50, N_pw=13):
    """
    Topological optimization using band field intensities.
    Add dielectric where band 4 has high intensity, remove where band 5 has high intensity.
    """
    size = len(cell)
    cell_arr = np.array(cell, dtype=int)
    best_cell = cell_arr.copy()
    best_score = _compute_tm_gap_fast(cell, N_pw=N_pw)

    for it in range(max_iters):
        if _remaining() < 25:
            break

        # Get field intensities for bands 4 and 5
        int4 = _get_field_intensities(cell, 3, N_pw=N_pw)  # band 4 (0-indexed: 3)
        int5 = _get_field_intensities(cell, 4, N_pw=N_pw)  # band 5 (0-indexed: 4)

        if int4 is None or int5 is None:
            break

        # Compute gradient: where to add/remove dielectric
        gradient = int4 - int5

        # Find pixels to flip
        flips = []
        for i in range(size):
            for j in range(size):
                if cell_arr[i, j] == 0 and gradient[i, j] > 0.25:
                    flips.append((i, j, 1))
                elif cell_arr[i, j] == 1 and gradient[i, j] < -0.25:
                    flips.append((i, j, 0))

        if not flips:
            break

        # Sort by gradient magnitude
        flips.sort(key=lambda f: abs(gradient[f[0], f[1]]), reverse=True)

        # Try flipping top N pixels at a time
        n_flip = min(6, len(flips))
        for n_try in range(1, n_flip + 1):
            new_cell = cell_arr.copy()
            for k in range(n_try):
                i, j, val = flips[k]
                new_cell[i, j] = val

            # Enforce D4 symmetry
            new_cell_list = new_cell.tolist()
            new_cell_list = _enforce_d4(new_cell_list)

            score = _compute_tm_gap_fast(new_cell_list, N_pw=N_pw)
            if score > best_score:
                best_score = score
                best_cell = np.array(new_cell_list)
                cell = new_cell_list
                break

        if _remaining() < 25:
            break

    return best_cell.tolist(), best_score


def design(problem):
    """
    Design a 32x32 symmetric photonic crystal unit cell.
    Returns a nested list of 0/1 pixels.
    """
    global _start_time
    _start_time = time.time()

    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    polarization = problem.get("polarization", "TM")

    best_cell = None
    best_score = -float('inf')

    # Phase 1: Generate diverse initial candidates
    candidates = []

    # Central disks
    for r in [0.1, 0.15, 0.2, 0.25, 0.3]:
        candidates.append(_make_cell_disk(r))

    # Rings
    for r_in, r_out in [(0.1, 0.2), (0.15, 0.25), (0.2, 0.3), (0.25, 0.35), (0.3, 0.4)]:
        candidates.append(_make_cell_ring(r_in, r_out))

    # Combined disk + ring
    for r_disk, r_ring_in, r_ring_out in [
        (0.05, 0.15, 0.25), (0.1, 0.2, 0.3), (0.15, 0.25, 0.35), (0.2, 0.3, 0.4)
    ]:
        candidates.append(_make_cell_combined(r_disk, r_ring_in, r_ring_out))

    # Cross shapes
    for bar_w, cross_l in [(0.1, 0.5), (0.15, 0.6), (0.2, 0.7), (0.1, 0.8), (0.2, 0.9)]:
        candidates.append(_make_cell_cross(bar_w, cross_l))

    # Four rods
    for r, offset in [(0.08, 0.15), (0.1, 0.2), (0.12, 0.25), (0.15, 0.3), (0.1, 0.35)]:
        candidates.append(_make_cell_four_rods(r, offset))

    # Diamonds
    for r in [0.15, 0.2, 0.25, 0.3]:
        candidates.append(_make_cell_diamond(r))

    # Hexagons
    for r in [0.15, 0.2, 0.25]:
        candidates.append(_make_cell_hexagon(r))

    # Checkers
    for block in [4, 6, 8]:
        candidates.append(_make_cell_checker(block))

    # Score all candidates using fast solver
    for cell in candidates:
        if _remaining() < 50:
            break
        score = _compute_tm_gap_fast(cell, N_pw=13)
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in cell]

    # Phase 2: Topological optimization from best candidate
    if best_cell is not None and _remaining() > 50:
        opt_cell, opt_score = _topological_optimize(best_cell, max_iters=40, N_pw=13)
        if opt_score > best_score:
            best_score = opt_score
            best_cell = opt_cell

    # Phase 3: Try topological optimization from a few other good candidates
    if _remaining() > 35:
        # Re-score to get top 3 candidates
        scored = []
        for cell in candidates[:20]:
            if _remaining() < 35:
                break
            score = _compute_tm_gap_fast(cell, N_pw=13)
            scored.append((score, cell))

        scored.sort(key=lambda x: x[0], reverse=True)

        for score, cell in scored[:3]:
            if _remaining() < 25:
                break
            if score >= best_score - 0.01:  # Only optimize if close to best
                opt_cell, opt_score = _topological_optimize(cell, max_iters=20, N_pw=13)
                if opt_score > best_score:
                    best_score = opt_score
                    best_cell = opt_cell

    # Phase 4: Final verification with more accurate solver
    if best_cell is not None and _remaining() > 15:
        final_score = _compute_tm_gap(best_cell, N_pw=13, n_kpts=15)
        if final_score > best_score:
            best_score = final_score

    # Fallback
    if best_cell is None:
        best_cell = _make_cell_disk(0.2)

    return best_cell
