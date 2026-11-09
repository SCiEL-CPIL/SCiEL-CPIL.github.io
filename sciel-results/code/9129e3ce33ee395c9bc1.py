"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    """
    Designs a 32x32 unit cell using local search over pixel flips with 4-fold symmetry,
    starting from a good initial configuration. Uses a fast plane-wave band solver.
    """
    start_time = time.time()
    deadline = start_time + 215.0  # Safety margin before 230s kill

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # Precompute the 8x8 fundamental region indices (1/8 of 32x32 with 4-fold symmetry)
    # For 4-fold symmetry (D4), the fundamental region is 1/8 of the cell.
    # We'll work with the full 32x32 but only flip pixels that are in the fundamental region.
    
    def enforce_symmetry(cell):
        """Enforce 4-fold symmetry (D4) about the center of the 32x32 cell."""
        cell = np.array(cell, dtype=int)
        # Symmetry: cell[i][j] == cell[31-i][j] == cell[i][31-j] == cell[31-i][31-j]
        # Also: cell[i][j] == cell[j][i] (transpose symmetry)
        # So the fundamental region is rows 0-15, cols 0-15, but with transpose constraint
        # Actually for D4 of a square: the fundamental domain is 1/8 of the cell
        # Let's just enforce: for each (i,j), set all 8 symmetric equivalents to the same value
        # The 8 symmetric positions of (i,j) under D4:
        # (i,j), (31-i,j), (i,31-j), (31-i,31-j),
        # (j,i), (j,31-i), (31-j,i), (31-j,31-i)
        
        # To enforce, we iterate over the fundamental region and set all symmetric pixels
        # Fundamental region: i in [0,15], j in [0,15] with j <= i (to account for transpose)
        # Actually, let's just do it properly:
        new_cell = np.zeros_like(cell)
        for i in range(size):
            for j in range(size):
                # Get the canonical representative
                # The 8 symmetric points
                pts = [
                    (i, j),
                    (31 - i, j),
                    (i, 31 - j),
                    (31 - i, 31 - j),
                    (j, i),
                    (j, 31 - i),
                    (31 - j, i),
                    (31 - j, 31 - i),
                ]
                # Use the first one as canonical
                val = cell[pts[0][0], pts[0][1]]
                for pi, pj in pts:
                    new_cell[pi, pj] = val
        return new_cell

    def get_fundamental_pixels():
        """Get list of (i,j) positions that form the fundamental region (one representative per orbit)."""
        fundamental = []
        visited = set()
        for i in range(size):
            for j in range(size):
                if (i, j) in visited:
                    continue
                pts = [
                    (i, j),
                    (31 - i, j),
                    (i, 31 - j),
                    (31 - i, 31 - j),
                    (j, i),
                    (j, 31 - i),
                    (31 - j, i),
                    (31 - j, 31 - i),
                ]
                for p in pts:
                    visited.add(p)
                fundamental.append((i, j))
        return fundamental

    fundamental_pixels = get_fundamental_pixels()
    n_fundamental = len(fundamental_pixels)

    def make_initial_cell():
        """
        Create an initial cell: dielectric background with 4 circular air holes
        at quarter positions. This is a known good starting point for photonic crystals.
        """
        cell = np.ones((size, size), dtype=int)  # Start with all dielectric
        # Add 4 circular air holes (value 0) at quarter positions
        centers = [
            (8.0, 8.0),    # (0.25*32, 0.25*32)
            (24.0, 8.0),   # (0.75*32, 0.25*32)
            (8.0, 24.0),   # (0.25*32, 0.75*32)
            (24.0, 24.0),  # (0.75*32, 0.75*32)
        ]
        radius = 3.5
        for i in range(size):
            for j in range(size):
                y_pix = i + 0.5
                x_pix = j + 0.5
                for cx, cy in centers:
                    dx = x_pix - cx
                    dy = y_pix - cy
                    if dx * dx + dy * dy < radius * radius:
                        cell[i, j] = 0
                        break
        return enforce_symmetry(cell)

    def compute_fft_inv_eps(cell):
        """Compute 2D FFT of 1/epsilon for fast convolution."""
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_2d = np.fft.fft2(inv_eps) / (size * size)
        return fft_2d

    def solve_bands(cell, n_pw=12, k_pts=None):
        """
        TM band solver using plane-wave expansion.
        Returns gap-midgap ratio for bands 4 and 5 (1-indexed, so indices 3 and 4).
        """
        fft_2d = compute_fft_inv_eps(cell)

        def get_coeff(m, n):
            m_idx = m % size
            n_idx = n % size
            return fft_2d[m_idx, n_idx]

        if k_pts is None:
            # Key points in the irreducible Brillouine zone
            k_pts = [
                (0.0, 0.0),    # Gamma
                (0.25, 0.0),
                (0.5, 0.0),    # X/2
                (0.75, 0.0),
                (1.0, 0.0),    # X
                (0.5, 0.25),
                (0.5, 0.5),    # M/2 (Gamma to M midpoint)
                (0.5, 0.75),
                (1.0, 0.5),    # X-M midpoint
                (1.0, 1.0),    # M
                (0.25, 0.25),
                (0.75, 0.75),
                (0.25, 0.5),
                (0.75, 0.5),
            ]

        # Build list of plane waves within circular cutoff
        g_pairs = []
        for m in range(-n_pw, n_pw + 1):
            for n in range(-n_pw, n_pw + 1):
                if m * m + n * n <= n_pw * n_pw:
                    g_pairs.append((m, n))

        n_pw_actual = len(g_pairs)

        max_omega4 = -1e10
        min_omega5 = 1e10

        for kx, ky in k_pts:
            # Build Hermitian matrix for TM polarization
            # H[G, G'] = eps_inv[G - G'] * |k - G'|^2
            # This is not Hermitian in general, so we need to symmetrize
            M = np.zeros((n_pw_actual, n_pw_actual), dtype=complex)

            for i in range(n_pw_actual):
                mi, ni = g_pairs[i]
                for j in range(n_pw_actual):
                    mj, nj = g_pairs[j]
                    diff_m = mi - mj
                    diff_n = ni - nj
                    eps_val = get_coeff(diff_m, diff_n)
                    kx_g = kx - mj
                    ky_g = ky - nj
                    M[i, j] = eps_val * (kx_g * kx_g + ky_g * ky_g)

            # Symmetrize to get Hermitian matrix
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

    # Start with initial cell
    cell = make_initial_cell()
    best_score = solve_bands(cell, n_pw=10)
    best_cell = cell.copy()

    # Local search: flip pixels in the fundamental region
    # For speed, use a fast solver with fewer k-points for exploration
    def solve_bands_fast(cell, n_pw=8, k_pts=None):
        """Faster solver for exploration."""
        fft_2d = compute_fft_inv_eps(cell)

        def get_coeff(m, n):
            m_idx = m % size
            n_idx = n % size
            return fft_2d[m_idx, n_idx]

        if k_pts is None:
            k_pts = [
                (0.0, 0.0),
                (0.5, 0.0),
                (1.0, 0.0),
                (0.5, 0.5),
                (1.0, 1.0),
                (1.0, 0.5),
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
            for i in range(n_pw_actual):
                mi, ni = g_pairs[i]
                for j in range(n_pw_actual):
                    mj, nj = g_pairs[j]
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

    # Local search with simulated annealing-like approach
    current_cell = cell.copy()
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 50

    # Get list of fundamental pixel indices for flipping
    flip_candidates = list(range(n_fundamental))

    while time.time() < deadline - 20:
        # Pick a random fundamental pixel to flip
        idx = flip_candidates[np.random.randint(n_fundamental)]
        fi, fj = fundamental_pixels[idx]
        
        # Try flipping this pixel
        old_val = current_cell[fi, fj]
        new_val = 1 - old_val
        
        # Apply flip to all symmetric positions
        pts = [
            (fi, fj),
            (31 - fi, fj),
            (fi, 31 - fj),
            (31 - fi, 31 - fj),
            (fj, fi),
            (fj, 31 - fi),
            (31 - fj, fi),
            (31 - fj, 31 - fi),
        ]
        
        for pi, pj in pts:
            current_cell[pi, pj] = new_val
        
        new_score = solve_bands_fast(current_cell, n_pw=8)
        
        # Accept if better or with some probability (simulated annealing)
        if new_score > current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_cell = current_cell.copy()
        else:
            # Reject: undo the flip
            for pi, pj in pts:
                current_cell[pi, pj] = old_val
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Reset to best and try a different strategy
                current_cell = best_cell.copy()
                current_score = best_score
                no_improve_count = 0
                max_no_improve = min(max_no_improve + 20, 100)

    # Final evaluation with more accurate solver
    final_score = solve_bands(best_cell, n_pw=12)
    if final_score > best_score:
        best_score = final_score

    # Convert to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        result.append(row)

    return result
# EVOLVE-BLOCK-END
