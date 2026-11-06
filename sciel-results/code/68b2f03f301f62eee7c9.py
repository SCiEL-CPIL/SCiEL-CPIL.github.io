"""Spectral-domain inverse design for a 32x32 D4-symmetric TM band-gap cell."""
import time
import math
import numpy as np

_DEADLINE = 215.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _enforce_d4(cell):
    size = len(cell)
    c = np.array(cell, dtype=np.int8)
    # D4 orbit: 8 transforms. Average then threshold (majority -> 0/1)
    trans = [
        c,
        c[::-1, :],          # vertical flip
        c[:, ::-1],          # horizontal flip
        c[::-1, ::-1],       # 180 rot
        c.T,                 # transpose (main diagonal)
        c[::-1, ::-1].T,     # 180 + transpose = anti-diagonal
        c.T[::-1, :],        # transpose + vertical
        c.T[:, ::-1],        # transpose + horizontal
    ]
    avg = np.mean(trans, axis=0)
    return (avg >= 0.5).astype(np.int8)


def _fourier_coeffs(cell, N=17):
    """Fourier coefficients of 1/eps on a 2N x 2N grid via FFT.
    Returns array indexed [m+N, n+N] for m,n in [-N+1, N-1]."""
    size = len(cell)
    eps_high = 12.0
    inv_eps = np.where(cell == 1, 1.0 / eps_high, 1.0)
    # 2D DFT: C[m,n] = sum_{i,j} f[i,j] exp(-2pi i (m i + n j)/size)
    # np.fft.fft2 gives C[m,n] for m,n in [0, size); wrap to negative via fftshift
    C = np.fft.fft2(inv_eps) / (size * size)
    C = np.fft.fftshift(C)  # now C[m+N, n+N] = C[m,n] for m,n in [-N, N-1]
    # We need m,n in [-N+1, N-1] -> indices [1, 2N-1] in shifted array
    return C[1:2 * N - 1, 1:2 * N - 1]


def _g_indices(N):
    """Return list of (m, n) for |m|<=N-1, |n|<=N-1 with m^2+n^2 <= (N-1)^2 * 1.1 (disk)."""
    glist = []
    maxr2 = (N - 1) ** 2 * 1.1
    for m in range(-(N - 1), N):
        for n in range(-(N - 1), N):
            if m * m + n * n <= maxr2:
                glist.append((m, n))
    return glist


def _solve_at_k(kx, ky, coeffs, glist, N, bands=(3, 4)):
    """Solve eigenproblem at (kx,ky). Return (omega4, omega5, E4, E5) or None."""
    npw = len(glist)
    # Precompute |k+G|^2
    km2 = np.zeros(npw)
    for idx, (m, n) in enumerate(glist):
        gx = 2 * np.pi * m
        gy = 2 * np.pi * n
        km2[idx] = (kx + gx) ** 2 + (ky + gy) ** 2

    # Build matrix A[mi, ni] = c_{m_i - m_n, n_i - n_n} * km2[mi]
    # c index: (dm + N - 1, dn + N - 1) in coeffs array (size 2N-1)
    A = np.zeros((npw, npw), dtype=float)
    for mi in range(npw):
        m_i, n_i = glist[mi]
        for ni in range(npw):
            m_n, n_n = glist[ni]
            dm = m_i - m_n
            dn = n_i - n_n
            ci = dm + N - 1
            cj = dn + N - 1
            if 0 <= ci < 2 * N - 1 and 0 <= cj < 2 * N - 1:
                A[mi, ni] = coeffs[ci, cj] * km2[mi]

    try:
        # A is Hermitian (real symmetric since c_{-m,-n}=conj(c_{m,n}) and we take real part)
        A = 0.5 * (A + A.T)
        vals, vecs = np.linalg.eigh(A)
        vals = np.real(vals)
        # Take top bands (highest omega^2). We want bands 4 and 5 (1-indexed) = indices 3,4
        # But we only need 5 bands; sort ascending and pick indices 3,4
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        if len(vals) < 5:
            return None
        omega4_sq = vals[3]
        omega5_sq = vals[4]
        if omega4_sq <= 0 or omega5_sq <= 0:
            return None
        omega4 = np.sqrt(omega4_sq)
        omega5 = np.sqrt(omega5_sq)
        E4 = vecs[:, 3]
        E5 = vecs[:, 4]
        return omega4, omega5, E4, E5
    except:
        return None


def _field_energy(E, glist, N, size=32):
    """Compute |E(x,y)|^2 on the 32x32 pixel grid from plane-wave coefficients.
    E(x,y) = sum_n E_n exp(i (k+G_n) . (x,y))  -- but for the band structure the
    field is expanded in G only (k is the crystal momentum). The envelope at a
    point r within the cell is sum_n E_n exp(i G_n . r)."""
    # r = (i+0.5)/size, (j+0.5)/size
    xs = (np.arange(size) + 0.5) / size
    ys = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    field = np.zeros((size, size), dtype=complex)
    for n, (m, nn) in enumerate(glist):
        if abs(E[n]) < 1e-12:
            continue
        gx = 2 * np.pi * m
        gy = 2 * np.pi * nn
        field += E[n] * np.exp(1j * (gx * X + gy * Y))
    return np.abs(field) ** 2


def _compute_gap(cell, coeffs, glist, N, kpts):
    """Return (gap_ratio, max_omega4, min_omega5, gap_k, E4, E5) for the cell."""
    max_o4 = 0.0
    min_o5 = float('inf')
    best_k = None
    best_E4 = None
    best_E5 = None
    for kx, ky in kpts:
        res = _solve_at_k(kx, ky, coeffs, glist, N)
        if res is None:
            continue
        o4, o5, E4, E5 = res
        if o4 > max_o4:
            max_o4 = o4
            best_k = (kx, ky)
            best_E4 = E4
            best_E5 = E5
        if o5 < min_o5:
            min_o5 = o5
            best_k = (kx, ky)
            best_E4 = E4
            best_E5 = E5
    if min_o5 == float('inf') or max_o4 <= 0:
        return -1.0, 0.0, float('inf'), None, None, None
    gap = min_o5 - max_o4
    mid = (min_o5 + max_o4) / 2.0
    ratio = 2.0 * gap / mid if mid > 0 else -1.0
    return ratio, max_o4, min_o5, best_k, best_E4, best_E5


def _kpts_irreducible(n=20):
    """Generate k-points in the irreducible BZ of the square lattice.
    Irreducible BZ: 0 <= ky <= kx, kx + ky <= pi."""
    kpts = []
    # Grid in the triangle
    for i in range(n):
        for j in range(n):
            kx = (i + 0.5) / n * np.pi
            ky = (j + 0.5) / n * np.pi
            if ky <= kx and kx + ky <= np.pi:
                kpts.append((kx, ky))
    # Add high-symmetry points
    kpts.append((0.0, 0.0))
    kpts.append((np.pi / 2, 0.0))
    kpts.append((np.pi / 2, np.pi / 2))
    kpts.append((np.pi / 4, np.pi / 4))
    return kpts


def _random_cell(size=32, fill=0.5, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    cell = (rng.rand(size, size) < fill).astype(np.int8)
    return _enforce_d4(cell)


def _flip_pixel(cell, i, j, new_val):
    """Flip a pixel and all its D4 orbit to new_val."""
    size = len(cell)
    c = np.array(cell, dtype=np.int8)
    positions = [
        (i, j), (size - 1 - i, j), (i, size - 1 - j), (size - 1 - i, size - 1 - j),
        (j, i), (size - 1 - j, i), (j, size - 1 - i), (size - 1 - j, size - 1 - i),
    ]
    for pi, pj in positions:
        c[pi, pj] = new_val
    return c


def _local_search(cell, coeffs, glist, N, kpts, E4, E5, time_budget=5.0):
    """Local search: flip single orbits to improve gap."""
    size = len(cell)
    c = np.array(cell, dtype=np.int8)
    current_ratio, *_ = _compute_gap(c, coeffs, glist, N, kpts)
    # Compute field energies
    e4 = _field_energy(E4, glist, N, size)
    e5 = _field_energy(E5, glist, N, size)
    # Candidate pixels: those in the D4 orbit of a single pixel
    # For each unique orbit representative, try flipping
    seen_orbits = set()
    t0 = time.time()
    for i in range(size):
        for j in range(size):
            if time.time() - t0 > time_budget:
                break
            # Orbit key: minimal representative
            orbit = sorted([
                (i, j), (size - 1 - i, j), (i, size - 1 - j), (size - 1 - i, size - 1 - j),
                (j, i), (size - 1 - j, i), (j, size - 1 - i), (size - 1 - j, size - 1 - i),
            ])
            key = tuple(orbit[0])
            if key in seen_orbits:
                continue
            seen_orbits.add(key)
            new_val = 1 - c[i, j]
            c_new = _flip_pixel(c, i, j, new_val)
            # Quick score: change in field energy difference
            # If new_val=1 (adding dielectric): 1/eps decreases -> band energies shift down
            # Heuristic: flip if it reduces e4 and increases e5 at that location
            delta_e = (e4[i, j] - e5[i, j]) * (1 if new_val == 1 else -1)
            if delta_e < 0:
                # Likely improves gap; verify with full compute occasionally
                if _remaining() < 10:
                    break
                # Full verify
                coeffs_new = _fourier_coeffs(c_new, N)
                ratio_new, *_ = _compute_gap(c_new, coeffs_new, glist, N, kpts)
                if ratio_new > current_ratio:
                    c = c_new
                    coeffs = coeffs_new
                    current_ratio = ratio_new
                    e4 = _field_energy(E4, glist, N, size)
                    e5 = _field_energy(E5, glist, N, size)
    return c, current_ratio


def design(problem):
    global _start_time
    _start_time = time.time()

    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    polarization = problem.get("polarization", "TM")

    N = 17  # plane-wave truncation
    glist = _g_indices(N)
    kpts = _kpts_irreducible(n=15)  # ~30-40 k-points

    # Initial random cell
    cell = _random_cell(size, fill=0.5, seed=42)
    coeffs = _fourier_coeffs(cell, N)
    ratio, max_o4, min_o5, best_k, E4, E5 = _compute_gap(cell, coeffs, glist, N, kpts)
    best_cell = [row[:] for row in cell.tolist()]
    best_ratio = ratio

    # Iterative inverse design
    t0 = time.time()
    max_iters = 200
    for it in range(max_iters):
        if _remaining() < 15:
            break
        if best_k is None or E4 is None or E5 is None:
            # Re-evaluate
            ratio, max_o4, min_o5, best_k, E4, E5 = _compute_gap(cell, coeffs, glist, N, kpts)
            if best_k is None:
                break

        # Compute field energies
        e4 = _field_energy(E4, glist, N, size)
        e5 = _field_energy(E5, glist, N, size)

        # Gradient-free update: flip pixels where e4 > e5 (add dielectric) or e4 < e5 (remove)
        # Add dielectric where band-4 field is strong (lowers omega4)
        # Remove dielectric where band-5 field is strong (lowers omega5? No, we want to RAISE omega5)
        # Actually: to OPEN gap, lower omega4 and raise omega5.
        # Adding dielectric (1/eps decreases) lowers ALL bands. So we should add dielectric
        # where band-4 energy is HIGH (to lower omega4 more) and remove where band-5 energy is HIGH.
        # But this is a heuristic. Let's use: flip to 1 where e4 - e5 > threshold, flip to 0 where e5 - e4 > threshold.
        diff = e4 - e5
        # Find best single orbit flip
        best_flip_ratio = ratio
        best_flip_cell = None
        best_flip_coords = None

        # Sample candidate flips (limited for speed)
        candidates = []
        for i in range(size):
            for j in range(size):
                if diff[i, j] > 0.1 * np.max(np.abs(diff)):
                    candidates.append((i, j, 1))  # add dielectric
                elif diff[i, j] < -0.1 * np.max(np.abs(diff)):
                    candidates.append((i, j, 0))  # remove dielectric
        # Limit candidates
        if len(candidates) > 50:
            candidates = candidates[:50]

        for i, j, new_val in candidates:
            if _remaining() < 15:
                break
            if c_val := cell[i][j] == new_val:
                continue  # already in that state
            c_new = _flip_pixel(np.array(cell), i, j, new_val)
            coeffs_new = _fourier_coeffs(c_new, N)
            ratio_new, *_ = _compute_gap(c_new, coeffs_new, glist, N, kpts)
            if ratio_new > best_flip_ratio:
                best_flip_ratio = ratio_new
                best_flip_cell = c_new
                best_flip_coords = (i, j, new_val)

        if best_flip_cell is not None:
            cell = best_flip_cell
            coeffs = _fourier_coeffs(cell, N)
            ratio = best_flip_ratio
            if ratio > best_ratio:
                best_ratio = ratio
                best_cell = [row[:] for row in cell.tolist()]
                # Update E4, E5
                _, _, _, best_k, E4, E5 = _compute_gap(cell, coeffs, glist, N, kpts)
        else:
            # No improvement; try random perturbation (simulated annealing)
            if it > 10:
                i = np.random.randint(size)
                j = np.random.randint(size)
                new_val = 1 - cell[i][j]
                c_new = _flip_pixel(np.array(cell), i, j, new_val)
                coeffs_new = _fourier_coeffs(c_new, N)
                ratio_new, *_ = _compute_gap(c_new, coeffs_new, glist, N, kpts)
                # Accept with probability exp(-delta/T)
                delta = ratio - ratio_new
                T = max(0.01, 0.1 * (1 - it / max_iters))
                if delta < 0 or np.random.rand() < math.exp(-delta / T):
                    cell = c_new
                    coeffs = coeffs_new
                    ratio = ratio_new
                    _, _, _, best_k, E4, E5 = _compute_gap(cell, coeffs, glist, N, kpts)

    # Final local search
    if _remaining() > 20 and best_k is not None and E4 is not None and E5 is not None:
        cell, ratio = _local_search(cell, coeffs, glist, N, kpts, E4, E5,
                                     time_budget=min(10, _remaining() - 20))
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = [row[:] for row in cell.tolist()]

    # Ensure best_cell is valid
    if best_cell is None:
        best_cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < 0.2 * 0.2:
                    best_cell[i][j] = 1
        best_cell = _enforce_d4(best_cell).tolist()

    return best_cell
