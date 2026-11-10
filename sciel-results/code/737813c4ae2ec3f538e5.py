import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 230s wall-clock kill
    
    # PWM Setup
    N_G = 15
    g_range = np.arange(-N_G//2, N_G//2 + 1)
    Gx, Gy = np.meshgrid(g_range, g_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_vectors)
    
    # K points in irreducible BZ (30 points)
    k_points = []
    for i in range(10):
        t = i / 9.0 * 0.5
        k_points.append([t, 0.0])
        k_points.append([0.5, t])
        k_points.append([t, t])
    k_points.append([0.25, 0.1])
    k_points.append([0.35, 0.2])
    k_points.append([0.45, 0.3])
    k_points = np.array(k_points)
    
    # Precompute diff indices for H matrix
    dx = G_vectors[:, 0][:, None] - G_vectors[:, 0][None, :]
    dy = G_vectors[:, 1][:, None] - G_vectors[:, 1][None, :]
    valid_diffs = (np.abs(dx) <= N_G//2) & (np.abs(dy) <= N_G//2)
    diff_indices = np.where(valid_diffs, (dx + N_G//2) * N_G + (dy + N_G//2), 0)
    
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
            
            evals = np.linalg.eigvals(H)
            evals = np.sort(np.real(evals))
            
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
    
    # Initialize with a centered dielectric rod
    cell = np.zeros((size, size), dtype=int)
    cx, cy = size // 2 - 0.5, size // 2 - 0.5
    for i in range(size):
        for j in range(size):
            if (i - cx)**2 + (j - cy)**2 < (size * 0.15)**2:
                cell[i, j] = 1
    
    best_score = evaluate(cell)
    best_cell = cell.copy()
    
    # Fourier-domain topology optimization
    # Work with low-order Fourier coefficients of the dielectric function
    # This allows smooth, global changes to the topology
    
    # Get initial Fourier coefficients of epsilon
    eps_initial = np.where(cell == 1, eps_high, eps_low)
    eps_fft = np.fft.fft2(eps_initial) / (size * size)
    
    # Keep only low-order coefficients (|nx| <= 4, |ny| <= 4)
    max_order = 4
    fourier_modes = []
    for nx in range(-max_order, max_order + 1):
        for ny in range(-max_order, max_order + 1):
            if nx == 0 and ny == 0:
                continue  # Skip DC component
            fourier_modes.append((nx, ny))
    
    # Extract current amplitudes
    current_coeffs = []
    for (nx, ny) in fourier_modes:
        idx = (nx % size, ny % size)
        current_coeffs.append(eps_fft[idx].copy())
    
    # Coordinate ascent in Fourier space
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        # Try perturbing one Fourier mode
        mode_idx = np.random.randint(len(fourier_modes))
        nx, ny = fourier_modes[mode_idx]
        
        # Try a few step sizes and directions
        found_improvement = False
        for direction in [1, -1]:
            for scale in [step_size, step_size * 0.5, step_size * 2]:
                # Create new epsilon with modified Fourier coefficient
                new_eps_fft = eps_fft.copy()
                idx = (nx % size, ny % size)
                new_coeff = current_coeffs[mode_idx] + direction * scale * np.exp(1j * np.random.uniform(0, 2 * np.pi))
                new_eps_fft[idx] = new_coeff
                
                # Ensure Hermitian symmetry
                idx_conj = ((-nx) % size, (-ny) % size)
                new_eps_fft[idx_conj] = np.conj(new_coeff)
                
                # Convert back to spatial domain and threshold
                new_eps = np.real(np.fft.ifft2(new_eps_fft * size * size))
                new_cell = (new_eps > (eps_high + eps_low) / 2).astype(int)
                
                # Ensure symmetry
                new_cell = np.maximum(new_cell, new_cell.T)
                new_cell = np.maximum(new_cell, new_cell[::-1, :])
                new_cell = np.maximum(new_cell, new_cell[:, ::-1])
                
                score = evaluate(new_cell)
                
                if score > best_score:
                    best_score = score
                    best_cell = new_cell.copy()
                    # Update Fourier coefficients
                    eps_initial = np.where(new_cell == 1, eps_high, eps_low)
                    eps_fft = np.fft.fft2(eps_initial) / (size * size)
                    current_coeffs = []
                    for (mnx, mny) in fourier_modes:
                        m_idx = (mnx % size, mny % size)
                        current_coeffs.append(eps_fft[m_idx].copy())
                    no_improve_count = 0
                    found_improvement = True
                    break
            if found_improvement:
                break
        
        if not found_improvement:
            no_improve_count += 1
        
        # Restart if stuck
        if no_improve_count > max_no_improve:
            cell = best_cell.copy()
            eps_initial = np.where(cell == 1, eps_high, eps_low)
            eps_fft = np.fft.fft2(eps_initial) / (size * size)
            current_coeffs = []
            for (nx, ny) in fourier_modes:
                idx = (nx % size, ny % size)
                current_coeffs.append(eps_fft[idx].copy())
            no_improve_count = 0
            step_size *= 0.7  # Reduce step size
    
    return best_cell.tolist()
