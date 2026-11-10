import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 115.0

    # Fast 7x7 plane-wave setup
    N_G = 3
    g_range = np.arange(-N_G, N_G + 1)
    Gx, Gy = np.meshgrid(g_range, g_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_vectors)
    
    # K points in irreducible BZ
    k_points = []
    for i in range(8):
        t = i / 7.0 * 0.5
        k_points.append([t, 0.0])
        k_points.append([0.5, t])
        k_points.append([t, t])
    k_points = np.array(k_points)
    
    # Precompute diff indices
    dx = G_vectors[:, 0][:, None] - G_vectors[:, 0][None, :]
    dy = G_vectors[:, 1][:, None] - G_vectors[:, 1][None, :]
    valid_diffs = (np.abs(dx) <= N_G) & (np.abs(dy) <= N_G)
    diff_indices = np.where(valid_diffs, (dx + N_G) * (2*N_G+1) + (dy + N_G), 0)
    valid_count = np.sum(valid_diffs)
    
    def get_fourier_eps_inv(cell_arr):
        eps = np.where(cell_arr == 1, eps_high, eps_low)
        eps_inv = 1.0 / eps
        coeffs_2d = np.fft.fft2(eps_inv) / (size * size)
        Gx_grid = G_vectors[:, 0] % size
        Gy_grid = G_vectors[:, 1] % size
        return coeffs_2d[Gx_grid, Gy_grid]
    
    def evaluate(cell_arr):
        eps_inv_coeffs = get_fourier_eps_inv(cell_arr)
        
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for k in k_points:
            k_vec = np.array(k)
            k_plus_G = k_vec[None, :] + G_vectors
            k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
            
            H = eps_inv_coeffs[diff_indices] * k_plus_G_sq[None, :]
            H[~valid_diffs] = 0.0
            
            evals = np.linalg.eigvalsh(H)
            evals = np.sort(evals)
            
            w4 = evals[3]
            w5 = evals[4]
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap
    
    def symmetrize(cell):
        """Enforce all four-fold symmetry about cell center."""
        # Transpose
        cell = np.maximum(cell, cell.T)
        # Reverse rows
        cell = np.maximum(cell, cell[::-1, :])
        # Reverse columns
        cell = np.maximum(cell, cell[:, ::-1])
        return cell
    
    # Initialize with a centered dielectric rod
    cell = np.zeros((size, size), dtype=int)
    cx, cy = size // 2 - 0.5, size // 2 - 0.5
    for i in range(size):
        for j in range(size):
            if (i - cx)**2 + (j - cy)**2 < (size * 0.15)**2:
                cell[i, j] = 1
    cell = symmetrize(cell)
    
    best_score = evaluate(cell)
    best_cell = cell.copy()
    
    # Simulated annealing with direct pixel flips
    # Work in the fundamental domain (top-left quadrant) and mirror to enforce symmetry
    # The fundamental domain is rows 0..15, cols 0..15 (for 32x32)
    half = size // 2
    
    # Precompute which pixels in the full cell correspond to which fundamental pixel
    # For a 32x32 cell with 4-fold symmetry about center (16,16):
    # Pixel (i,j) maps to fundamental pixel (min(i, 31-i), min(j, 31-j))
    # But we need to be careful: the symmetry is about the cell center
    
    # Actually, let's think about this more carefully.
    # The cell is 32x32. The center is at (16, 16) in pixel coordinates.
    # Symmetry: cell[i,j] == cell[31-i, j] == cell[i, 31-j] == cell[31-i, 31-j]
    # So the fundamental domain is rows 0..15, cols 0..15
    
    # For each fundamental pixel (fi, fj) where fi in 0..15, fj in 0..15:
    # It corresponds to full-cell pixels: (fi, fj), (fi, 31-fj), (31-fi, fj), (31-fi, 31-fj)
    
    def flip_fundamental_pixel(cell, fi, fj):
        """Flip the fundamental pixel at (fi, fj) and all its symmetric counterparts."""
        new_cell = cell.copy()
        val = 1 - new_cell[fi, fj]
        new_cell[fi, fj] = val
        new_cell[fi, size-1-fj] = val
        new_cell[size-1-fi, fj] = val
        new_cell[size-1-fi, size-1-fj] = val
        return new_cell
    
    # Simulated annealing
    T = 0.05
    T_min = 0.001
    cooling_rate = 0.9995
    rng = np.random.RandomState(42)
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        # Pick a random fundamental pixel
        fi = rng.randint(half)
        fj = rng.randint(half)
        
        new_cell = flip_fundamental_pixel(cell, fi, fj)
        new_score = evaluate(new_cell)
        
        # Accept or reject
        if new_score > best_score:
            best_score = new_score
            best_cell = new_cell.copy()
            no_improve = 0
            T = max(T * 0.95, T_min)  # Cool faster on improvement
        elif new_score > best_score - T:
            # Accept with some probability
            if rng.random() < np.exp((new_score - best_score) / T):
                cell = new_cell
                no_improve += 1
        else:
            no_improve += 1
        
        # Cool
        T = max(T * cooling_rate, T_min)
        
        # Restart if stuck
        if no_improve > max_no_improve:
            cell = best_cell.copy()
            T = 0.05
            no_improve = 0
            max_no_improve = int(max_no_improve * 0.7)
    
    return best_cell.tolist()
