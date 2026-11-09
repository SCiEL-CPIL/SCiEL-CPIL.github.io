import numpy as np
import time

def design(problem):
    start_time = time.time()
    deadline = start_time + 115
    
    size = 32
    eps_low = 1.0
    eps_high = 12.0
    
    # Fundamental domain: (i,j) with i in 0..15, j in 0..i
    fd_indices = [(i, j) for i in range(16) for j in range(i + 1)]
    n_fd = len(fd_indices)  # 136
    
    # G vectors for coarse solver (5x5 = 25 plane waves)
    G_coarse = np.array([[m, n] for m in range(-2, 3) for n in range(-2, 3)], dtype=float)
    N_coarse = len(G_coarse)  # 25
    
    # G vectors for fine solver (17x17 = 289 plane waves)
    G_fine = np.array([[m, n] for m in range(-8, 9) for n in range(-8, 9)], dtype=float)
    N_fine = len(G_fine)  # 289
    
    # K-points
    k_points_coarse = np.array([
        [0, 0], [0.25, 0], [0.5, 0], [0.5, 0.25], [0.5, 0.5],
        [0.25, 0.25], [0.125, 0], [0.375, 0], [0.5, 0.125], [0.5, 0.375],
        [0.25, 0.125], [0.375, 0.25], [0.125, 0.25], [0.25, 0.375],
        [0.375, 0.125], [0.125, 0.375], [0.375, 0.375], [0.125, 0.125],
        [0.25, 0.5], [0.5, 0.25], [0.125, 0.5], [0.5, 0.125],
        [0.375, 0.5], [0.5, 0.375], [0.25, 0.375], [0.375, 0.25],
        [0.125, 0.25], [0.25, 0.125], [0.375, 0.125], [0.125, 0.375]
    ])
    
    k_points_fine = np.array([
        [0, 0], [0.1, 0], [0.2, 0], [0.3, 0], [0.4, 0], [0.5, 0],
        [0.5, 0.1], [0.5, 0.2], [0.5, 0.3], [0.5, 0.4], [0.5, 0.5],
        [0.4, 0.4], [0.3, 0.3], [0.2, 0.2], [0.1, 0.1],
        [0.25, 0.25], [0.3, 0.1], [0.4, 0.2], [0.15, 0.15], [0.35, 0.05],
        [0.45, 0.1], [0.1, 0.45], [0.45, 0.45], [0.05, 0.35],
        [0.2, 0.3], [0.3, 0.2], [0.4, 0.1], [0.1, 0.4],
        [0.45, 0.25], [0.25, 0.45]
    ])
    
    def build_cell(fd_values):
        """Build 32x32 cell from fundamental domain values."""
        cell = np.zeros((32, 32), dtype=int)
        for idx, (i, j) in enumerate(fd_indices):
            v = fd_values[idx]
            # All 8 symmetry-equivalent positions
            positions = [
                (i, j), (i, 31-j), (31-i, j), (31-i, 31-j),
                (j, i), (j, 31-i), (31-j, i), (31-j, 31-i)
            ]
            for (ri, cj) in positions:
                cell[ri, cj] = v
        return cell
    
    def compute_fourier(cell_2d, G_vecs):
        """Compute Fourier coefficients of 1/eps(r)."""
        inv_eps = np.where(cell_2d == 1, 1.0/eps_high, 1.0/eps_low)
        F = np.fft.fft2(inv_eps)
        f_G = np.zeros(len(G_vecs), dtype=complex)
        for k in range(len(G_vecs)):
            m, n = int(G_vecs[k][0]), int(G_vecs[k][1])
            m_idx = m % size
            n_idx = n % size
            F_val = F[m_idx, n_idx]
            phase = np.exp(-1j * np.pi * (m + n))
            f_G[k] = F_val * phase / (size * size)
        return f_G
    
    def solve_bands(cell_2d, G_vecs, k_pts):
        """Solve bands and return gap ratio."""
        N = len(G_vecs)
        inv_eps_G = compute_fourier(cell_2d, G_vecs)
        
        # Precompute diff indices
        diff_m = G_vecs[:, 0:1] - G_vecs[0:1, :]
        diff_n = G_vecs[:, 1:2] - G_vecs[0:1, :]
        
        # Create lookup
        G_dict = {}
        for k in range(N):
            G_dict[(int(G_vecs[k][0]), int(G_vecs[k][1]))] = inv_eps_G[k]
        
        G_sq = np.sum(G_vecs**2, axis=1)
        
        max_b4 = 0.0
        min_b5 = float('inf')
        
        for kx, ky in k_pts:
            Gk = G_vecs + np.array([kx, ky])
            Gk_sq = np.sum(Gk**2, axis=1)
            
            M = np.zeros((N, N), dtype=complex)
            for i in range(N):
                mi, ni = int(G_vecs[i][0]), int(G_vecs[i][1])
                for j in range(N):
                    mj, nj = int(G_vecs[j][0]), int(G_vecs[j][1])
                    key = (mi - mj, ni - nj)
                    if key in G_dict:
                        M[i, j] = Gk_sq[i] * G_dict[key]
            
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                continue
            
            if len(eigvals) < 5:
                continue
            
            w4 = np.sqrt(max(eigvals[3], 0))
            w5 = np.sqrt(max(eigvals[4], 0))
            max_b4 = max(max_b4, w4)
            min_b5 = min(min_b5, w5)
        
        if min_b5 == float('inf') or min_b5 <= max_b4:
            return -1.0
        return 2 * (min_b5 - max_b4) / (min_b5 + max_b4)
    
    # Initial design: centered rod
    fd = np.zeros(n_fd, dtype=int)
    for idx, (i, j) in enumerate(fd_indices):
        x = (i + 0.5) / 32 - 0.5
        y = (j + 0.5) / 32 - 0.5
        if x*x + y*y < 0.2**2:
            fd[idx] = 1
    
    cell = build_cell(fd)
    best_score = solve_bands(cell, G_coarse, k_points_coarse)
    best_fd = fd.copy()
    
    # Also try a few other initializations
    initializations = [
        "rod",  # current
        "four_rod",
        "square_hole",
        "checkerboard"
    ]
    
    # Four rod: rods at (4,4), (4,27), (27,4), (27,27)
    fd2 = np.zeros(n_fd, dtype=int)
    rod_centers = [(4, 4)]
    for idx, (i, j) in enumerate(fd_indices):
        x = (i + 0.5) / 32 - 0.5
        y = (j + 0.5) / 32 - 0.5
        for (cx, cy) in rod_centers:
            dx = x - (cx + 0.5) / 32 + 0.5
            dy = y - (cy + 0.5) / 32 + 0.5
            if dx*dx + dy*dy < 0.15**2:
                fd2[idx] = 1
                break
    cell2 = build_cell(fd2)
    score2 = solve_bands(cell2, G_coarse, k_points_coarse)
    if score2 > best_score:
        best_score = score2
        best_fd = fd2.copy()
    
    # Hill climbing with coarse solver
    rng = np.random.RandomState(42)
    stagnant = 0
    max_stagnant = 50
    
    while time.time() < deadline - 20:  # Leave 20s for final evaluation
        idx = rng.randint(0, n_fd)
        fd[idx] = 1 - fd[idx]
        cell = build_cell(fd)
        score = solve_bands(cell, G_coarse, k_points_coarse)
        
        if score > best_score:
            best_score = score
            best_fd = fd.copy()
            stagnant = 0
        else:
            fd[idx] = 1 - fd[idx]
            stagnant += 1
        
        if stagnant > max_stagnant:
            # Restart with random perturbation
            for _ in range(5):
                idx = rng.randint(0, n_fd)
                fd[idx] = 1 - fd[idx]
            stagnant = 0
    
    # Final evaluation with fine solver
    best_cell = build_cell(best_fd)
    final_score = solve_bands(best_cell, G_fine, k_points_fine)
    
    return best_cell.tolist()
