"""Photonic crystal design: maximize TM band 4-5 gap in 32x32 symmetric cell."""
import time
import math
import numpy as np

def design(problem):
    """Design a 32x32 symmetric unit cell maximizing the TM 4-5 band gap."""
    t0 = time.time()
    DEADLINE = t0 + 215  # safety margin before 230s kill

    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12)
    eps_low = problem.get("eps_low", 1)

    def make_cell(params):
        """Create a symmetric 32x32 cell from set of (i,j) positions that are dielectric.
        Symmetry: cell[i][j] = cell[j][i] = cell[N-1-i][N-1-j] = cell[N-1-j][N-1-i]
        """
        N = size
        cell = [[0]*N for _ in range(N)]
        for (i, j) in params:
            cell[i][j] = 1
            cell[j][i] = 1
            cell[N-1-i][N-1-j] = 1
            cell[N-1-j][N-1-i] = 1
        return cell

    def pw_solve_tm(kx, ky, eps_2d, max_n, n_bands=6):
        """Solve TM eigenvalue problem at wavevector (kx,ky).
        Returns first n_bands eigenvalues (omega^2 in arbitrary units).
        """
        N = size
        # Fourier coefficients of 1/eps
        inv_eps = 1.0 / eps_2d
        # 2D DFT
        fft_inv_eps = np.fft.fft2(inv_eps) / (N * N)
        
        # G-vector indices
        gvecs = []
        for nx in range(-max_n, max_n + 1):
            for ny in range(-max_n, max_n + 1):
                gvecs.append((nx, ny))
        n_g = len(gvecs)
        gvec_arr = np.array(gvecs, dtype=float)
        
        K = np.array([kx, ky])
        # (G_j - K)^2
        diff = gvec_arr - K
        GK_sq = np.sum(diff**2, axis=1)
        
        # Build matrix A[i,j] = (G_j - K)^2 * eps_coeff[G_i - G_j]
        A = np.zeros((n_g, n_g), dtype=complex)
        for i in range(n_g):
            for j in range(n_g):
                dnx = gvecs[i][0] - gvecs[j][0]
                dny = gvecs[i][1] - gvecs[j][1]
                # Wrap to [0, N)
                idx_nx = dnx % N
                idx_ny = dny % N
                eps_coeff = fft_inv_eps[idx_nx, idx_ny]
                A[i, j] = GK_sq[j] * eps_coeff
        
        # Hermitian
        A = (A + A.conj().T) / 2
        A_real = np.real(A)
        A_real = (A_real + A_real.T) / 2
        
        try:
            eigvals = np.linalg.eigvalsh(A_real)
        except:
            eigvals = np.linalg.eigvalsh((A_real + A_real.T)/2)
        
        eigvals = np.sort(eigvals)
        return eigvals[:n_bands]

    def score_cell(cell, kpoints, max_n=3):
        """Score cell by 4-5 band gap ratio."""
        eps_2d = np.array([[eps_high if cell[i][j] else eps_low for j in range(size)] for i in range(size)], dtype=float)
        
        max_omega4_sq = 0
        min_omega5_sq = float('inf')
        
        for (kx, ky) in kpoints:
            eigvals = pw_solve_tm(kx, ky, eps_2d, max_n, n_bands=6)
            if len(eigvals) < 5:
                return -1
            w4 = eigvals[3]
            w5 = eigvals[4]
            if w4 > max_omega4_sq:
                max_omega4_sq = w4
            if w5 < min_omega5_sq:
                min_omega5_sq = w5
        
        if min_omega5_sq <= 0 or max_omega4_sq <= 0:
            return -1
        gap = min_omega5_sq - max_omega4_sq
        mid = (min_omega5_sq + max_omega4_sq) / 2
        if mid <= 0:
            return -1
        return gap / mid * 2

    # K-points in irreducible BZ (0 <= kx <= ky, kx+ky <= 0.5)
    kpoints_opt = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.25, 0.0), (0.25, 0.25), (0.5, 0.25)]
    
    # Fundamental domain: i in [0, 15], j in [i, 31]
    fd_positions = []
    for i in range(16):
        for j in range(i, 32):
            fd_positions.append((i, j))
    
    # Initial: dielectric background with central air square
    initial_params = set()
    for (i, j) in fd_positions:
        # Air hole: central region
        if 13 <= i <= 15 and 13 <= j <= 15:
            pass  # air
        else:
            initial_params.add((i, j))
    
    best_params = list(initial_params)
    best_cell = make_cell(best_params)
    best_score = score_cell(best_cell, kpoints_opt, max_n=3)
    
    # Hill climbing
    iteration = 0
    while time.time() - t0 < DEADLINE - 5 and iteration < 3000:
        improved = False
        n_fd = len(fd_positions)
        n_try = min(40, n_fd)
        idxs = np.random.choice(n_fd, size=n_try, replace=False)
        
        for idx in idxs:
            pos = fd_positions[idx]
            new_set = set(best_params)
            if pos in new_set:
                new_set.remove(pos)
            else:
                new_set.add(pos)
            new_cell = make_cell(new_set)
            new_score = score_cell(new_cell, kpoints_opt, max_n=3)
            if new_score > best_score:
                best_score = new_score
                best_params = list(new_set)
                improved = True
            if time.time() - t0 > DEADLINE - 5:
                break
        
        if not improved:
            # Multi-flip
            n_flip = np.random.randint(2, 6)
            idxs = np.random.choice(n_fd, size=n_flip, replace=False)
            new_set = set(best_params)
            for idx in idxs:
                pos = fd_positions[idx]
                if pos in new_set:
                    new_set.remove(pos)
                else:
                    new_set.add(pos)
            new_cell = make_cell(new_set)
            new_score = score_cell(new_cell, kpoints_opt, max_n=3)
            if new_score > best_score:
                best_score = new_score
                best_params = list(new_set)
        
        iteration += 1
    
    return make_cell(best_params)
