import numpy as np
import time
import random

def design(problem):
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    deadline = time.time() + 110
    
    N = size
    
    # Symmetry orbits
    pixel_to_idx = np.zeros((N, N), dtype=int)
    visited = np.zeros((N, N), dtype=bool)
    var_pixels = []
    for i in range(N):
        for j in range(N):
            if visited[i, j]:
                continue
            orbit = set()
            orbit.add((i, j))
            orbit.add((N-1-i, N-1-j))
            orbit.add((j, i))
            orbit.add((N-1-j, N-1-i))
            pixels = []
            for (oi, oj) in orbit:
                visited[oi, oj] = True
                pixel_to_idx[oi, oj] = len(var_pixels)
                pixels.append((oi, oj))
            var_pixels.append(pixels)
    n_free = len(var_pixels)
    
    # Fast PW setup
    n_pw = 7
    gxs = np.arange(-n_pw, n_pw + 1) * 2 * np.pi
    gys = np.arange(-n_pw, n_pw + 1) * 2 * np.pi
    nGx = len(gxs)
    nGy = len(gys)
    total_G = nGx * nGy
    GX_flat, GY_flat = np.meshgrid(gxs, gys, indexing='ij')
    GX_flat = GX_flat.ravel()
    GY_flat = GY_flat.ravel()
    ia = np.arange(total_G) // nGy
    ib = np.arange(total_G) % nGy
    ia_diff_idx = (ia[:, None] - ia[None, :] + nGx // 2) % nGx
    ib_diff_idx = (ib[:, None] - ib[None, :] + nGy // 2) % nGy
    
    xs = (np.arange(N) + 0.5) / N
    ys = (np.arange(N) + 0.5) / N
    phase_x = np.exp(-1j * np.outer(xs, gxs))
    phase_y = np.exp(-1j * np.outer(ys, gys))
    
    k_points = []
    for t in np.linspace(0, 1, 12):
        k_points.append((np.pi * t, 0.0))
    for t in np.linspace(0, 1, 12):
        k_points.append((np.pi, np.pi * t))
    for t in np.linspace(0, 1, 12):
        k_points.append((np.pi * t, np.pi * t))
    k_points = np.array(k_points)
    
    def fast_gap(cell):
        inv_eps = np.where(cell == 1, 1.0 / eps_high, 1.0 / eps_low)
        A = phase_x.T @ inv_eps.T
        inv_eps_F = A @ phase_y
        inv_eps_F /= N * N
        
        min_omega5 = np.inf
        max_omega4 = -np.inf
        
        for (kx, ky) in k_points:
            kGx = kx + GX_flat
            kGy = ky + GY_flat
            dot = np.outer(kGx, kGx) + np.outer(kGy, kGy)
            M = dot * inv_eps_F[ia_diff_idx, ib_diff_idx]
            M = (M + M.T) / 2
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals)
            eigvals = eigvals[eigvals > 1e-10]
            if len(eigvals) >= 5:
                omega4 = np.sqrt(eigvals[3])
                omega5 = np.sqrt(eigvals[4])
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == np.inf or max_omega4 == -np.inf:
            return -1.0
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
    
    def make_cell(free_vals):
        cell = np.zeros((N, N), dtype=int)
        for idx, pixels in enumerate(var_pixels):
            v = free_vals[idx]
            for (i, j) in pixels:
                cell[i, j] = v
        return cell
    
    # Initial: circle
    free_init = np.zeros(n_free, dtype=int)
    for i in range(N):
        for j in range(N):
            x = (i + 0.5) / N - 0.5
            y = (j + 0.5) / N - 0.5
            if x * x + y * y < 0.2 ** 2:
                free_init[pixel_to_idx[i, j]] = 1
    
    best_cell = make_cell(free_init)
    best_score = fast_gap(best_cell)
    
    # SA
    T = 0.15
    T_min = 0.003
    random.seed(42)
    
    while time.time() < deadline - 5:
        idx_flip = random.randrange(n_free)
        old_val = free_init[idx_flip]
        free_init[idx_flip] = 1 - old_val
        cell = make_cell(free_init)
        score = fast_gap(cell)
        
        if score > best_score or random.random() < np.exp((score - best_score) / T):
            best_score = score
            best_cell = cell.copy()
        else:
            free_init[idx_flip] = old_val
        
        T = max(T_min, T * 0.9995)
    
    # Local refinement
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for idx in range(n_free):
            old_val = free_init[idx]
            free_init[idx] = 1 - old_val
            cell = make_cell(free_init)
            score = fast_gap(cell)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
                improved = True
            else:
                free_init[idx] = old_val
    
    result = [[int(best_cell[i, j]) for j in range(N)] for i in range(N)]
    return result
