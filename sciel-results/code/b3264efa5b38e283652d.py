"""Photonic-crystal unit-cell design: maximize TM band-4/5 gap-midgap ratio."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    """Design a 32x32 symmetric unit cell to maximize TM band-4/5 gap-midgap ratio."""
    start_time = time.time()
    deadline = start_time + 210.0  # safety margin before 230s kill

    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    N = size  # 32

    # D4 symmetry: cell[i][j] = cell[j][i] = cell[N-1-i][j] = cell[i][N-1-j] = ...
    # Fundamental domain: upper triangle of top-left 16x16 quadrant
    # Independent pixels: (i,j) for i in [0,15], j in [i,15] -> 136 pixels

    def get_fundamental_indices():
        """Get indices of the 136 independent pixels in the 16x16 top-left quadrant."""
        indices = []
        for i in range(16):
            for j in range(i, 16):
                indices.append((i, j))
        return indices

    fd_indices = get_fundamental_indices()
    n_fd = len(fd_indices)  # 136

    def fd_to_cell(fd_bits):
        """Expand 136-bit fundamental domain to full 32x32 D4-symmetric cell."""
        cell = np.zeros((N, N), dtype=np.int8)
        for idx, (i, j) in enumerate(fd_indices):
            val = fd_bits[idx]
            # All 8 symmetric positions
            positions = [
                (i, j), (j, i),
                (N-1-i, j), (N-1-j, i),
                (i, N-1-j), (j, N-1-i),
                (N-1-i, N-1-j), (N-1-j, N-1-i)
            ]
            for (r, c) in positions:
                if 0 <= r < N and 0 <= c < N:
                    cell[r, c] = val
        return cell

    def cell_to_fd(cell):
        """Extract 136-bit fundamental domain from a 32x32 cell."""
        fd = np.zeros(n_fd, dtype=np.int8)
        for idx, (i, j) in enumerate(fd_indices):
            fd[idx] = cell[i, j]
        return fd

    def compute_fourier_coeffs(cell):
        """Compute Fourier coefficients of 1/eps for the cell."""
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps

        # Use FFT to compute Fourier coefficients
        # c_G = (1/N^2) * sum_{i,j} inv_eps[i,j] * exp(-i 2pi (gx*i + gy*j)/N)
        # This is just the 2D DFT scaled
        c = np.fft.fft2(inv_eps) / (N * N)
        return c

    def tm_bands_fast(cell, n_pw=13, n_kpts=12):
        """Fast TM band solver using FFT-computed Fourier coefficients."""
        c = compute_fourier_coeffs(cell)

        # Plane wave indices: from -(n_pw-1)//2 to (n_pw-1)//2
        half = (n_pw - 1) // 2
        g_idx = list(range(-half, half + 1))
        n = len(g_idx)  # n_pw

        # Map (mx, my) to linear index
        n2 = n * n

        # Precompute k-points in irreducible BZ
        # Path: (0,0) -> (pi,0) -> (pi,pi)
        kpts = []
        for t in np.linspace(0, 1, 6):
            kpts.append((np.pi * t, 0.0))
        for t in np.linspace(1/6, 1, 6):
            kpts.append((np.pi, np.pi * t))

        max_omega4 = 0.0
        min_omega5 = float('inf')

        # Precompute G vectors
        Gx = np.array([2 * np.pi * g for g in g_idx])
        Gy = np.array([2 * np.pi * g for g in g_idx])

        for kx, ky in kpts:
            # Build matrix M[i,j] = (k+Gi)·(k+Gj) * c_{Gi-Gj}
            # i = (a*n + b) where Gi = (Gx[a], Gy[b])
            M = np.zeros((n2, n2), dtype=float)

            for a in range(n):
                for b in range(n):
                    i = a * n + b
                    kGi_x = kx + Gx[a]
                    kGi_y = ky + Gy[b]
                    for a2 in range(n):
                        for b2 in range(n):
                            j = a2 * n + b2
                            kGj_x = kx + Gx[a2]
                            kGj_y = ky + Gy[b2]
                            dgx = g_idx[a] - g_idx[a2]
                            dgy = g_idx[b] - g_idx[b2]
                            # c_{dgx, dgy}: index in c array
                            # c is computed via fft2, so c[mx, my] for mx,my in [0, N)
                            # But we need c for negative indices too. Use wrap-around.
                            cidx_x = dgx % N
                            cidx_y = dgy % N
                            cval = c[cidx_x, cidx_y]
                            # For real symmetric eps, c is real and symmetric
                            M[i, j] = (kGi_x * kGj_x + kGi_y * kGj_y) * cval

            # M should be symmetric (real) for our case
            M = (M + M.T) / 2.0

            try:
                eigvals = np.linalg.eigvalsh(M)
                eigvals = np.sort(eigvals)
                # First 6 bands
                if len(eigvals) >= 6:
                    omega4 = np.sqrt(max(eigvals[3], 0))
                    omega5 = np.sqrt(max(eigvals[4], 0))
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                pass

        if min_omega5 == float('inf') or max_omega4 <= 0:
            return -1.0

        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

    def evaluate_fd(fd_bits):
        """Evaluate a fundamental domain configuration."""
        cell = fd_to_cell(fd_bits)
        return tm_bands_fast(cell, n_pw=13, n_kpts=12)

    # Initialize with a good starting point: dielectric circle
    def make_circle_fd(radius):
        cell = np.zeros((N, N), dtype=np.int8)
        for i in range(N):
            for j in range(N):
                x = (i + 0.5) / N - 0.5
                y = (j + 0.5) / N - 0.5
                if x*x + y*y < radius*radius:
                    cell[i, j] = 1
        return cell_to_fd(cell)

    # Try several initial configurations
    np.random.seed(42)
    best_fd = None
    best_score = -float('inf')

    initial_configs = [
        make_circle_fd(0.2),
        make_circle_fd(0.25),
        make_circle_fd(0.3),
    ]

    # Also try a square
    def make_square_fd(half_w):
        cell = np.zeros((N, N), dtype=np.int8)
        for i in range(N):
            for j in range(N):
                x = (i + 0.5) / N - 0.5
                y = (j + 0.5) / N - 0.5
                if abs(x) < half_w and abs(y) < half_w:
                    cell[i, j] = 1
        return cell_to_fd(cell)

    initial_configs.append(make_square_fd(0.2))
    initial_configs.append(make_square_fd(0.25))

    # Try air holes (dielectric background with circular holes)
    def make_holes_fd(hole_radius, n_holes=2):
        cell = np.ones((N, N), dtype=np.int8)
        for i in range(N):
            for j in range(N):
                x = (i + 0.5) / N
                y = (j + 0.5) / N
                for mx in range(n_holes):
                    for my in range(n_holes):
                        cx = (mx + 0.5) / n_holes
                        cy = (my + 0.5) / n_holes
                        if (x-cx)**2 + (y-cy)**2 < hole_radius**2:
                            cell[i, j] = 0
                            break
                    else:
                        continue
                    break
        return cell_to_fd(cell)

    initial_configs.append(make_holes_fd(0.1, 2))
    initial_configs.append(make_holes_fd(0.12, 2))

    # Evaluate initial configs
    for fd in initial_configs:
        if time.time() > deadline - 60:
            break
        score = evaluate_fd(fd)
        if score > best_score:
            best_score = score
            best_fd = fd.copy()

    # Simulated annealing on the fundamental domain
    if best_fd is not None and time.time() < deadline - 30:
        T_init = 0.1
        T_final = 0.001
        n_sa_iters = 500
        current_fd = best_fd.copy()
        current_score = best_score
        best_fd_sa = best_fd.copy()
        best_score_sa = best_score

        for it in range(n_sa_iters):
            if time.time() > deadline - 20:
                break

            # Cooling schedule
            T = T_init * (T_final / T_init) ** (it / n_sa_iters)

            # Pick a random independent pixel to flip
            idx = np.random.randint(0, n_fd)
            new_fd = current_fd.copy()
            new_fd[idx] = 1 - new_fd[idx]

            new_score = evaluate_fd(new_fd)
            delta = new_score - current_score

            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_fd = new_fd
                current_score = new_score
                if current_score > best_score_sa:
                    best_score_sa = current_score
                    best_fd_sa = current_fd.copy()

        if best_score_sa > best_score:
            best_score = best_score_sa
            best_fd = best_fd_sa.copy()

    # Final local search: try flipping each pixel and keep improvements
    if best_fd is not None and time.time() < deadline - 15:
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for idx in range(n_fd):
                if time.time() > deadline - 15:
                    break
                new_fd = best_fd.copy()
                new_fd[idx] = 1 - new_fd[idx]
                new_score = evaluate_fd(new_fd)
                if new_score > best_score:
                    best_fd = new_fd
                    best_score = new_score
                    improved = True

    # Convert to list of lists
    cell = fd_to_cell(best_fd)
    result = [[int(cell[i, j]) for j in range(N)] for i in range(N)]

    return result
# EVOLVE-BLOCK-END
