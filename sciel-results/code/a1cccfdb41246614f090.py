"""Combined photonic-crystal band-gap optimizer."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    start_time = time.time()
    deadline = start_time + 215.0  # safety margin before 230s kill

    def enforce_symmetry(cell):
        """Enforce 4-fold rotational + mirror symmetry about the cell centre."""
        sym = np.zeros((size, size), dtype=int)
        for i in range(size // 2 + 1):
            for j in range(size // 2 + 1):
                val = cell[i, j]
                sym[i, j] = val
                sym[size - 1 - i, j] = val
                sym[i, size - 1 - j] = val
                sym[size - 1 - i, size - 1 - j] = val
        return sym

    def make_superellipse_cell(radius, n_order, offset, inv=False):
        """Create a symmetric cell with four superellipse blobs near the corners."""
        cell = np.zeros((size, size), dtype=int)
        if inv:
            cell[:] = 1
        centers = [
            (size * 0.25 + offset, size * 0.25 + offset),
            (size * 0.75 - offset, size * 0.25 + offset),
            (size * 0.25 + offset, size * 0.75 - offset),
            (size * 0.75 - offset, size * 0.75 - offset),
        ]
        for i in range(size):
            for j in range(size):
                y_pix = i + 0.5
                x_pix = j + 0.5
                for cx, cy in centers:
                    dx = x_pix - cx
                    dy = y_pix - cy
                    if radius > 0:
                        val = abs(dx / radius) ** n_order + abs(dy / radius) ** n_order
                        if not inv:
                            if val < 1.0:
                                cell[i, j] = 1
                                break
                        else:
                            if val < 1.0:
                                cell[i, j] = 0
                                break
        return enforce_symmetry(cell)

    def compute_fft_inv_eps(cell):
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        return np.fft.fft2(inv_eps) / (size * size)

    def solve_bands(cell, n_pw=12, k_pts=None):
        """Fast TM band solver using plane-wave expansion with FFT-based coefficients."""
        fft_2d = compute_fft_inv_eps(cell)

        def get_coeff(m, n):
            return fft_2d[m % size, n % size]

        if k_pts is None:
            k_pts = [
                (0.0, 0.0),
                (0.5, 0.0),
                (0.5, 0.5),
                (0.25, 0.0),
                (0.25, 0.25),
                (0.5, 0.25),
                (0.1, 0.0),
                (0.1, 0.1),
                (0.4, 0.0),
                (0.4, 0.4),
                (0.3, 0.1),
            ]

        g_pairs = []
        for m in range(-n_pw, n_pw + 1):
            for n in range(-n_pw, n_pw + 1):
                if m * m + n * n <= n_pw * n_pw:
                    g_pairs.append((m, n))
        n_pw_actual = len(g_pairs)

        max_omega4 = -1e10
        min_omega5 = 1e10

        for kx, ky in k_pts:
            M = np.zeros((n_pw_actual, n_pw_actual), dtype=complex)
            for i, (mi, ni) in enumerate(g_pairs):
                for j, (mj, nj) in enumerate(g_pairs):
                    diff_m = mi - mj
                    diff_n = ni - nj
                    eps_val = get_coeff(diff_m, diff_n)
                    kx_g = kx - mj
                    ky_g = ky - nj
                    M[i, j] = eps_val * (kx_g * kx_g + ky_g * ky_g)

            H = (M + M.conj().T) / 2.0

            try:
                eigenvalues = np.linalg.eigvalsh(H)
                eigenvalues = np.sort(np.real(eigenvalues))
                if len(eigenvalues) >= 5:
                    omega4 = eigenvalues[3]
                    omega5 = eigenvalues[4]
                    if omega4 > max_omega4:
                        max_omega4 = omega4
                    if omega5 < min_omega5:
                        min_omega5 = omega5
            except:
                pass

        if max_omega4 <= 0 or min_omega5 <= 0 or min_omega5 <= max_omega4:
            return -1.0

        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0
        return gap / midgap

    # Phase 1: Coarse search over superellipse parameters, both orientations
    best_score = -1e10
    best_cell = None
    best_params = None

    # Try both: dielectric dots in air (inv=False) and air holes in dielectric (inv=True)
    for inv in [False, True]:
        radii = [3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0]
        orders = [2.0, 3.0, 4.0, 6.0, 8.0]
        offsets = [-2.0, -1.0, 0.0, 1.0, 2.0]

        for r in radii:
            for n_ord in orders:
                for off in offsets:
                    if time.time() > deadline - 30:
                        break
                    cell = make_superellipse_cell(r, n_ord, off, inv)
                    score = solve_bands(cell, n_pw=10)
                    if score > best_score:
                        best_score = score
                        best_cell = cell.copy()
                        best_params = (r, n_ord, off, inv)
                if time.time() > deadline - 30:
                    break
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break

    # Phase 2: Fine search around best parameters
    if best_params is not None and time.time() < deadline - 40:
        r_best, n_best, off_best, inv_best = best_params
        for _ in range(3):
            step_r = 0.5
            step_n = 0.5
            step_off = 0.5
            improved = False
            for dr in [-step_r, 0, step_r]:
                for dn in [-step_n, 0, step_n]:
                    for do in [-step_off, 0, step_off]:
                        if time.time() > deadline - 40:
                            break
                        r = r_best + dr
                        n_ord = n_best + dn
                        off = off_best + do
                        if r < 2 or r > 12 or n_ord < 1.5 or n_ord > 10:
                            continue
                        cell = make_superellipse_cell(r, n_ord, off, inv_best)
                        score = solve_bands(cell, n_pw=12)
                        if score > best_score:
                            best_score = score
                            best_cell = cell.copy()
                            best_params = (r, n_ord, off, inv_best)
                            r_best, n_best, off_best = r, n_ord, off
                            improved = True
                    if time.time() > deadline - 40:
                        break
                if time.time() > deadline - 40:
                    break
            if not improved:
                break

    # Phase 3: Local pixel search to refine the best cell
    if best_cell is not None and time.time() < deadline - 20:
        cell = best_cell.copy()
        current_score = best_score

        # Get unique symmetric positions
        unique_positions = []
        for i in range(size // 2 + 1):
            for j in range(size // 2 + 1):
                unique_positions.append((i, j))

        import random
        rng = random.Random(42)

        # Try flipping pixels, starting from the most impactful regions
        # First try all unique positions in a deterministic order
        positions_sorted = sorted(unique_positions, key=lambda p: (p[0] + p[1], p[0]))

        # Multiple passes with decreasing step size
        for pass_num in range(3):
            if time.time() > deadline - 20:
                break

            # Shuffle for diversity in later passes
            if pass_num > 0:
                positions_sorted = list(unique_positions)
                rng.shuffle(positions_sorted)

            for i, j in positions_sorted:
                if time.time() > deadline - 20:
                    break

                old_val = cell[i, j]
                cell[i, j] = 1 - old_val
                cell = enforce_symmetry(cell)

                score = solve_bands(cell, n_pw=10)

                if score > current_score:
                    current_score = score
                    best_cell = cell.copy()
                else:
                    cell[i, j] = old_val
                    cell = enforce_symmetry(cell)

    # Final result
    if best_cell is None:
        # Fallback: simple dielectric square in center
        best_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) / size - 0.5
                y = (i + 0.5) / size - 0.5
                if abs(x) < 0.2 and abs(y) < 0.2:
                    best_cell[i, j] = 1
        best_cell = enforce_symmetry(best_cell)

    result = [[int(best_cell[i, j]) for j in range(size)] for i in range(size)]
    return result
# EVOLVE-BLOCK-END
