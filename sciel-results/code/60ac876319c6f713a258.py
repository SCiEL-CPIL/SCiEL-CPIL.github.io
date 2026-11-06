import numpy as np
import time

def _apply_symmetry(cell_quarter, size=32):
    """Apply full D4 symmetry about the cell centre to a top-left quarter."""
    cell = np.zeros((size, size), dtype=int)
    half = size // 2
    for r in range(half):
        for c in range(half):
            val = int(cell_quarter[r, c])
            cell[r, c] = val
            cell[r, size - 1 - c] = val
            cell[size - 1 - r, c] = val
            cell[size - 1 - r, size - 1 - c] = val
            cell[c, r] = val
            cell[c, size - 1 - r] = val
    return cell

def _make_cell(params, size=32):
    """Build a symmetric cell from 4-rod parameters (dx, dy, R, aspect)."""
    dx, dy, R, aspect = params
    coords = np.arange(size)
    x = (coords + 0.5) / size - 0.5
    y = (coords + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    cell = np.zeros((size, size), dtype=int)
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            d = np.sqrt(((X - sx * dx) / aspect) ** 2 + (Y - sy * dy) ** 2)
            cell[d <= R] = 1
    return cell

def _compute_gap(cell, n_pw=13, n_bands=5, kpts=None, n_k=14):
    """Gap-midgap ratio of TM bands 4 and 5 using plane-wave method.

    Returns the raw ratio 2*(min_w5 - max_w4)/(min_w5 + max_w4).
    """
    try:
        size = cell.shape[0]
        eps_high, eps_low = 12.0, 1.0
        n_half = n_pw // 2
        G_indices = np.arange(-n_half, n_half + 1)
        gx = G_indices * (2.0 * np.pi)
        gy = G_indices * (2.0 * np.pi)
        GX, GY = np.meshgrid(gx, gy, indexing='ij')
        Gx_f = GX.flatten()
        Gy_f = GY.flatten()
        N = n_pw * n_pw

        # Fourier coefficients of 1/epsilon via 2D FFT
        inv_eps = np.where(cell == 1, 1.0 / eps_high, 1.0 / eps_low)
        fft2 = np.fft.fft2(inv_eps)
        f_G = np.zeros((n_pw, n_pw), dtype=complex)
        for i, m in enumerate(G_indices):
            for j, n in enumerate(G_indices):
                f_G[i, j] = fft2[m % size, n % size] / (size * size)

        # Index map for G-difference (m,n) -> row,col in f_G
        idx_map = {}
        for i, m in enumerate(G_indices):
            for j, n in enumerate(G_indices):
                idx_map[(int(m), int(n))] = (i, j)

        if kpts is None:
            # Irreducible BZ of square lattice (0<=kx<=ky, kx+ky<=1, in 2pi units)
            kpts = []
            for t in np.linspace(0.0, 1.0, n_k):
                kpts.append((t, 0.0))        # Gamma -> X
                kpts.append((t / 2.0, t / 2.0))  # Gamma -> M
            kpts.append((1.0, 1.0))

        min_w5 = np.inf
        max_w4 = -np.inf

        for kx, ky in kpts:
            kx2 = kx * (2.0 * np.pi)
            ky2 = ky * (2.0 * np.pi)
            kpx = kx2 + Gx_f
            kpy = ky2 + Gy_f
            kp2 = kpx * kpx + kpy * kpy
            H = np.zeros((N, N), dtype=complex)
            for a in range(N):
                Ga_i = (a // n_pw)
                Ga_j = (a % n_pw)
                for b in range(N):
                    dm = G_indices[Ga_i] - G_indices[Gb_i]
                    dn = G_indices[Ga_j] - G_indices[Gb_j]
                    key = (int(dm), int(dn))
                    if key in idx_map:
                        ii, jj = idx_map[key]
                        H[a, b] = kp2[a] * f_G[ii, jj]
            w2 = np.linalg.eigvalsh((H + H.conj().T) * 0.5)
            w2 = w2.real
            w2 = w2[w2 > 0]
            if len(w2) >= n_bands:
                w = np.sqrt(w2[:n_bands])
                min_w5 = min(min_w5, float(w[4]))
                max_w4 = max(max_w4, float(w[3]))

        if not np.isfinite(min_w5) or not np.isfinite(max_w4):
            return -1.0
        denom = min_w5 + max_w4
        if denom <= 0:
            return -1.0
        return 2.0 * (min_w5 - max_w4) / denom
    except Exception:
        return -1.0

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell.

    Air background with 4 dielectric rods (2x2 arrangement) targeting the
    4th-5th TM band gap. Uses a coarse spectral/parameter local search with
    an epigraph-style smooth gap objective and a final high-precision polish.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin under 230s wall-clock

    size = problem.get("pixels", 32)

    # Baseline design (matches the current program's geometry)
    base_params = [0.25, 0.25, 0.12, 1.0]
    best_cell = _make_cell(base_params, size)
    best_score = _compute_gap(best_cell, n_pw=13, n_k=14)

    def elapsed_ok():
        return time.time() < deadline

    # ---- Phase 1: coarse local search over the 4-rod parameter family ----
    # Parameters: dx, dy, R, aspect. Search a small neighbourhood of the
    # baseline with a coarse step, tracking the best gap.
    cur = list(base_params)
    coarse_steps = {0: 0.03, 1: 0.03, 2: 0.015, 3: 0.10}
    iters = 0
    max_iters = 400
    while iters < max_iters and elapsed_ok():
        iters += 1
        # pick a random parameter and direction
        p = np.random.randint(0, 4)
        d = np.random.choice([-1, 1])
        cand = cur[:]
        cand[p] += d * coarse_steps[p]
        # clamp to sensible ranges
        cand[0] = min(max(cand[0], 0.05), 0.45)
        cand[1] = min(max(cand[1], 0.05), 0.45)
        cand[2] = min(max(cand[2], 0.03), 0.25)
        cand[3] = min(max(cand[3], 0.5), 1.6)
        cell = _make_cell(cand, size)
        score = _compute_gap(cell, n_pw=11, n_k=10)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            cur = cand
        else:
            # small probability of accepting (annealing-like) to escape
            if np.random.rand() < 0.05 and score > -0.5:
                cur = cand
    if not elapsed_ok():
        return best_cell.tolist()

    # ---- Phase 2: spectral (Fourier) refinement of the continuous field ----
    # Reconstruct the dielectric field, modulate low-frequency Fourier content
    # to shift band edges, then threshold to 0/1 while preserving D4 symmetry.
    if elapsed_ok():
        # Build a continuous field from the best cell (smoothed)
        field = best_cell.astype(float)
        # low-pass to get a smooth surrogate
        fft = np.fft.fft2(field)
        n = size
        mask = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                # keep low frequencies
                fi = min(i, n - i)
                fj = min(j, n - j)
                if fi <= 3 and fj <= 3:
                    mask[i, j] = 1.0
        fft = fft * mask
        base_field = np.real(np.fft.ifft2(fft))
        # small spectral perturbation search
        spec_best = best_cell.copy()
        spec_best_score = best_score
        for trial in range(20):
            if not elapsed_ok():
                break
            # random low-frequency phase/amplitude tweak
            f2 = np.fft.fft2(base_field)
            for (ii, jj) in [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]:
                f2[ii, jj] *= (1.0 + np.random.uniform(-0.25, 0.25))
                f2[n - ii, n - jj] = f2[ii, jj].conjugate()
            f2[0, 0] = f2[0, 0].real
            f2 = f2 + f2.conj().T * 0.0  # ensure hermitian-ish
            f2 = (f2 + f2.conj().T) / 2.0
            f2 = f2 + (f2 - f2.conj().T).conj() * 0.0
            fld = np.real(np.fft.ifft2(f2))
            thr = 0.5
            cand_cell = (fld >= thr).astype(int)
            # enforce D4 symmetry by mirroring the top-left quarter
            cand_cell = _apply_symmetry(cand_cell[:n // 2, :n // 2], n)
            sc = _compute_gap(cand_cell, n_pw=11, n_k=10)
            if sc > spec_best_score:
                spec_best_score = sc
                spec_best = cand_cell.copy()
        if spec_best_score > best_score:
            best_score = spec_best_score
            best_cell = spec_best.copy()

    # ---- Phase 3: high-precision polish on the best design ----
    if elapsed_ok():
        final_score = _compute_gap(best_cell, n_pw=17, n_k=20, n_bands=5)
        if np.isfinite(final_score):
            best_score = max(best_score, final_score)

    # Ensure exact D4 symmetry of the returned cell
    best_cell = _apply_symmetry(best_cell[:size // 2, :size // 2], size)

    return best_cell.tolist()
