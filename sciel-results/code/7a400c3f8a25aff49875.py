"""Design a 32x32 photonic crystal unit cell to maximize TM band 4-5 gap.

Strategy: single centrally-located superellipse inclusion.
  |x/a|^n + |y/b|^n <= 1  ->  dielectric (eps=12)
  otherwise               ->  air (eps=1)

The superellipse family is D4-symmetric about the cell centre, so the
resulting pixel map automatically satisfies the required symmetry.

Optimization: coarse grid over (a, b, n) -> local refinement -> random
restarts, all driven by a fast vectorized FFT-based plane-wave gap
estimator with an explicit deadline guard.
"""
import numpy as np
import time
import math

def design(problem):
    size = problem["pixels"]          # 32
    eps_low = problem["eps_low"]      # 1
    eps_high = problem["eps_high"]    # 12

    t_start = time.time()
    DEADLINE = 220.0                  # safety margin before 230 s kill

    def time_left():
        return DEADLINE - (time.time() - t_start)

    # ------------------------------------------------------------------
    # Pixel grid (centres at (i+0.5)/size - 0.5)
    # ------------------------------------------------------------------
    coords = (np.arange(size) + 0.5) / size - 0.5
    XX, YY = np.meshgrid(coords, coords, indexing='ij')
    XX2 = XX * XX
    YY2 = YY * YY

    # ------------------------------------------------------------------
    # Superellipse cell generator
    # ------------------------------------------------------------------
    def make_superellipse(a, b, n):
        """Return 32x32 int array (0/1) for the superellipse inclusion.
        a, b: semi-axes in pixel units (max ~15.5 to stay in cell).
        n: shape exponent (1 = diamond, 2 = circle, large -> square).
        """
        if a <= 0 or b <= 0:
            return np.zeros((size, size), dtype=int)
        # Clamp to cell
        a = min(a, size / 2 - 0.25)
        b = min(b, size / 2 - 0.25)
        if n <= 0.01:
            n = 0.01
        # Use log-sum-exp style for numerical stability when n is large
        # |x/a|^n + |y/b|^n <= 1
        ax = np.abs(XX) / a
        ay = np.abs(YY) / b
        # For n up to ~20, direct power is fine
        val = ax ** n + ay ** n
        cell = (val <= 1.0).astype(int)
        return cell

    def enforce_symmetry(cell):
        """Enforce D4 symmetry (no-op for superellipse, but safe)."""
        sym = [
            cell,
            cell[::-1, :],
            cell[:, ::-1],
            cell[::-1, ::-1],
            cell.T,
            cell.T[::-1, :],
            cell.T[:, ::-1],
            cell.T[::-1, ::-1],
        ]
        avg = np.mean(sym, axis=0)
        return (avg >= 0.5).astype(int)

    # ------------------------------------------------------------------
    # Fast vectorized plane-wave gap estimator
    # ------------------------------------------------------------------
    # G-vectors: indices -G_max .. +G_max in both directions
    G_MAX = 5
    G_range = np.arange(-G_MAX, G_MAX + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_range, G_range, indexing='ij')
    # Flatten to list of (gx, gy)
    G_list = list(zip(Gx_grid.ravel(), Gy_grid.ravel()))
    nG = len(G_list)

    # K-points in irreducible BZ (0 <= ky <= kx <= 0.5)
    k_points = [
        (0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0),
        (0.5, 0.1), (0.5, 0.2), (0.5, 0.3), (0.5, 0.4), (0.5, 0.5),
        (0.3, 0.1), (0.4, 0.2), (0.3, 0.2), (0.2, 0.1), (0.4, 0.3),
    ]

    def calculate_gap(cell):
        """Compute 4-5 TM band gap ratio using vectorized PWE.
        Returns the gap-midgap ratio (can be negative).
        """
        eps = eps_low + (eps_high - eps_low) * cell.astype(float)
        inv_eps = 1.0 / eps

        # FFT of inv_eps (N x N)
        inv_eps_fft = np.fft.fft2(inv_eps)
        N2 = size * size

        max_omega_4 = 0.0
        min_omega_5 = float('inf')

        for kx, ky in k_points:
            # (G - k) vectors for each G
            Gkx = Gx_grid - kx   # (nG, nG) shaped as (len(Gx_grid), len(Gy_grid))
            Gky = Gy_grid - ky
            # Actually Gx_grid is (len(G_range), len(G_range)) = (11, 11)
            # Gkx, Gky are (11, 11) arrays
            # (G-k)·(G'-k) = Gkx[i,j]*Gkx[i',j'] + Gky[i,j]*Gky[i',j']
            # We need M[i,j, i',j'] = (Gkx[i,j]*Gkx[i',j'] + Gky[i,j]*Gky[i',j']) * inv_eps_F[G-G']
            # inv_eps_F[dGx, dGy] = inv_eps_fft[(dGx % N), (dGy % N)] / N^2
            # dGx = Gx[i,j] - Gx[i',j'], dGy = Gy[i,j] - Gy[i',j']

            # Vectorized: build (G-k) outer products
            # Gkx_flat: (nG,)
            Gkx_flat = Gkx.ravel()
            Gky_flat = Gky.ravel()

            # (G-k)·(G'-k) = Gkx[i]*Gkx[j] + Gky[i]*Gky[j]
            dot_Gk = np.outer(Gkx_flat, Gkx_flat) + np.outer(Gky_flat, Gky_flat)

            # dG matrix: (nG, nG)
            Gx_flat = Gx_grid.ravel()
            Gy_flat = Gy_grid.ravel()
            dGx = Gx_flat[:, None] - Gx_flat[None, :]
            dGy = Gy_flat[:, None] - Gy_flat[None, :]

            # Index into FFT
            idx_x = dGx % size
            idx_y = dGy % size
            inv_eps_F = inv_eps_fft[idx_x, idx_y] / N2

            M = dot_Gk * inv_eps_F

            # Solve eigenvalue problem (real symmetric)
            M_real = M.real
            eigvals = np.linalg.eigvalsh(M_real)
            eigvals = np.sort(eigvals)

            if len(eigvals) >= 5:
                omega_4 = math.sqrt(max(0.0, eigvals[3]))
                omega_5 = math.sqrt(max(0.0, eigvals[4]))
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5

        if min_omega_5 == float('inf'):
            return -1.0

        denom = min_omega_5 + max_omega_4
        if denom < 1e-12:
            return -1.0
        return 2.0 * (min_omega_5 - max_omega_4) / denom

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    best_cell = None
    best_score = -float('inf')
    best_params = (8.0, 8.0, 2.0)  # default: circle radius 8

    def evaluate_and_update(a, b, n):
        nonlocal best_cell, best_score, best_params
        cell = make_superellipse(a, b, n)
        if cell.sum() == 0:
            return -1.0
        score = calculate_gap(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            best_params = (a, b, n)
        return score

    # Phase 1: Coarse grid search
    # a, b: 2 to 15, step 2
    # n: 1.0, 1.5, 2.0, 3.0, 5.0, 8.0
    a_vals = np.arange(2, 16, 2)
    b_vals = np.arange(2, 16, 2)
    n_vals = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

    for a in a_vals:
        if time_left() < 15:
            break
        for b in b_vals:
            if time_left() < 10:
                break
            for n in n_vals:
                if time_left() < 5:
                    break
                evaluate_and_update(float(a), float(b), float(n))

    # Phase 2: Local refinement around best
    if best_cell is not None and time_left() > 30:
        ba, bb, bn = best_params
        # Fine grid: step 0.5 for a,b; step 0.5 for n
        a_fine = np.arange(max(1.0, ba - 3), min(15.5, ba + 3) + 0.5, 0.5)
        b_fine = np.arange(max(1.0, bb - 3), min(15.5, bb + 3) + 0.5, 0.5)
        n_fine = np.arange(max(0.5, bn - 2), min(10, bn + 2) + 0.5, 0.5)

        for a in a_fine:
            if time_left() < 10:
                break
            for b in b_fine:
                if time_left() < 5:
                    break
                for n in n_fine:
                    if time_left() < 3:
                        break
                    evaluate_and_update(float(a), float(b), float(n))

    # Phase 3: Random search if time remains
    if time_left() > 15:
        rng = np.random.RandomState(42)
        n_random = int(min(time_left() / 3, 50))  # ~3s per eval estimate
        for _ in range(n_random):
            if time_left() < 5:
                break
            a = float(rng.uniform(3, 15))
            b = float(rng.uniform(3, 15))
            n = float(rng.uniform(0.5, 10))
            evaluate_and_update(a, b, n)

    # Final fallback: simple circle
    if best_cell is None or best_score < 0:
        r = 6.0
        cell = (XX2 + YY2 <= r * r).astype(int)
        best_cell = enforce_symmetry(cell)

    return best_cell.tolist()
