import numpy as np
import time

# EVOLVE-BLOCK-START
import math
import time

def _get_g_vectors(max_g=8):
    """Generate G vectors in a square grid [-max_g, max_g]"""
    g_vectors = []
    for h in range(-max_g, max_g + 1):
        for k in range(-max_g, max_g + 1):
            g_vectors.append((h, k))
    return np.array(g_vectors)

def _get_k_points(n=30):
    """Generate k points along X-M-Gamma"""
    k_points = []
    # Segment X-M: (0.5, 0) to (0.5, 0.5)
    n1 = n // 2
    for i in range(n1):
        t = i / (n1 - 1) if n1 > 1 else 0
        kx = 0.5
        ky = t * 0.5
        k_points.append((kx, ky))
    # Segment M-Gamma: (0.5, 0.5) to (0, 0)
    n2 = n - n1
    for i in range(n2):
        t = i / (n2 - 1) if n2 > 1 else 0
        kx = (1 - t) * 0.5
        ky = (1 - t) * 0.5
        k_points.append((kx, ky))
    return np.array(k_points)

def _compute_eps_fourier(pixel_cell, g_vectors, size=32):
    """
    Compute Fourier coefficients of epsilon using FFT.
    """
    fft_vals = np.fft.fft2(pixel_cell)
    fft_vals /= (size * size)
    
    eps_fourier = np.zeros(len(g_vectors), dtype=complex)
    for idx, (h, k) in enumerate(g_vectors):
        ih = h % size
        ik = k % size
        eps_fourier[idx] = fft_vals[ih, ik]
        
    return eps_fourier

def _solve_band_structure_tm(eps_fourier, g_vectors, k_points, max_g=8):
    """
    Solve TM band structure.
    """
    min_omega_5 = float('inf')
    max_omega_4 = 0.0
    
    G = g_vectors
    N_g = len(G)
    grid_size = 2 * max_g + 1
    
    # Precompute diff indices
    # diff_h = G[i, 0] - G[j, 0]
    # idx = (diff_h + max_g) * grid_size + (diff_k + max_g)
    # If diff_h or diff_k is outside [-max_g, max_g], idx is invalid.
    
    diff_idx = np.full((N_g, N_g), -1, dtype=int)
    for i in range(N_g):
        for j in range(N_g):
            dh = G[i, 0] - G[j, 0]
            dk = G[i, 1] - G[j, 1]
            if -max_g <= dh <= max_g and -max_g <= dk <= max_g:
                idx = (dh + max_g) * grid_size + (dk + max_g)
                # Check if this index corresponds to a valid G vector in our list
                # Our list is ordered h: -max_g..max_g, k: -max_g..max_g
                # So idx is the correct index in the flattened array.
                # But we need to ensure (dh, dk) is actually in g_vectors.
                # Since g_vectors contains all (h,k) in the range, it is.
                diff_idx[i, j] = idx
                
    # Precompute inv_eps_diff
    eps_diff_vals = eps_fourier[diff_idx]
    valid_mask = (diff_idx != -1) & (np.abs(eps_diff_vals) > 1e-10)
    inv_eps_diff = np.zeros((N_g, N_g), dtype=complex)
    inv_eps_diff[valid_mask] = 1.0 / eps_diff_vals[valid_mask]
    
    for k in k_points:
        kx, ky = k
        kg = np.array([kx, ky]) + G
        dot_prod = kg @ kg.T
        H = dot_prod * inv_eps_diff
        
        eigvals = np.linalg.eigvalsh(H)
        eigvals = np.real(eigvals)
        
        if len(eigvals) >= 5:
            lambda_4 = eigvals[3]
            lambda_5 = eigvals[4]
            
            if lambda_4 > 0 and lambda_5 > 0:
                omega_4 = np.sqrt(lambda_4)
                omega_5 = np.sqrt(lambda_5)
                
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
                    
    if max_omega_4 > 0 and min_omega_5 > 0:
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap > 0:
            return gap / midgap
    return -1.0

def _generate_cell_from_fourier(params, size=32):
    x = (np.arange(size) + 0.5) / size - 0.5
    y = (np.arange(size) + 0.5) / size - 0.5
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    eps = np.ones((size, size)) * 6.5
    
    # A10
    if params[0] != 0:
        eps += params[0] * np.cos(2 * np.pi * xx)
        eps += params[0] * np.cos(2 * np.pi * yy)
    # A11
    if params[1] != 0:
        eps += params[1] * np.cos(2 * np.pi * (xx + yy))
    # A20
    if params[2] != 0:
        eps += params[2] * np.cos(4 * np.pi * xx)
        eps += params[2] * np.cos(4 * np.pi * yy)
    # A21
    if params[3] != 0:
        eps += params[3] * np.cos(2 * np.pi * (2 * xx + yy))
        eps += params[3] * np.cos(2 * np.pi * (xx + 2 * yy))
    # A22
    if params[4] != 0:
        eps += params[4] * np.cos(2 * np.pi * (2 * xx + 2 * yy))
    # A30
    if params[5] != 0:
        eps += params[5] * np.cos(6 * np.pi * xx)
        eps += params[5] * np.cos(6 * np.pi * yy)
    # A31
    if params[6] != 0:
        eps += params[6] * np.cos(2 * np.pi * (3 * xx + yy))
        eps += params[6] * np.cos(2 * np.pi * (xx + 3 * yy))
    # A32
    if params[7] != 0:
        eps += params[7] * np.cos(2 * np.pi * (3 * xx + 2 * yy))
        eps += params[7] * np.cos(2 * np.pi * (2 * xx + 3 * yy))
    # A33
    if params[8] != 0:
        eps += params[8] * np.cos(2 * np.pi * (3 * xx + 3 * yy))
        
    cell = np.where(eps > 6.5, 12.0, 1.0)
    
    # Enforce symmetry
    cell = (cell + cell[::-1, :]) / 2.0
    cell = (cell + cell[:, ::-1]) / 2.0
    cell = (cell + cell.T) / 2.0
    
    cell = np.where(cell > 6.5, 1, 0).astype(int)
    return cell

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0
    
    size = problem.get("pixels", 32)
    
    g_vectors = _get_g_vectors(max_g=8)
    k_points_search = _get_k_points(n=10)
    
    best_params = np.zeros(9)
    best_score = -1.0
    
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
            
        params = np.random.uniform(-5, 5, 9)
        step_size = 1.0
        
        for step in range(20):
            if time.time() > deadline - 10:
                break
                
            cell = _generate_cell_from_fourier(params, size)
            eps_fourier = _compute_eps_fourier(cell, g_vectors, size)
            score = _solve_band_structure_tm(eps_fourier, g_vectors, k_points_search)
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                
            new_params = params + np.random.uniform(-step_size, step_size, 9)
            cell_new = _generate_cell_from_fourier(new_params, size)
            eps_fourier_new = _compute_eps_fourier(cell_new, g_vectors, size)
            score_new = _solve_band_structure_tm(eps_fourier_new, g_vectors, k_points_search)
            
            if score_new > score:
                params = new_params
            else:
                step_size *= 0.8
                
    best_cell = _generate_cell_from_fourier(best_params, size)
    return best_cell.tolist()

# EVOLVE-BLOCK-END
