import numpy as np
import time

def _get_symmetric_cell(mask):
    """
    mask is 16x16 boolean array.
    Returns 32x32 cell.
    """
    size = 32
    cell = np.zeros((size, size), dtype=np.int8)
    cell[:16, :16] = mask
    cell[:16, 16:] = mask[:, ::-1]
    cell[16:, :16] = mask[::-1, :]
    cell[16:, 16:] = mask[::-1, ::-1]
    return cell

def _fourier_coeffs(cell, eps_low=1, eps_high=12):
    size = cell.shape[0]
    eps_map = np.where(cell == 1, eps_high, eps_low)
    inv_eps_map = 1.0 / eps_map
    
    eps_fft = np.fft.fft2(eps_map) / (size * size)
    inv_eps_fft = np.fft.fft2(inv_eps_map) / (size * size)
    
    eps_fft = np.fft.fftshift(eps_fft)
    inv_eps_fft = np.fft.fftshift(inv_eps_fft)
    
    return eps_fft, inv_eps_fft

def _build_matrix(k, inv_eps_coeffs, G_list):
    N = len(G_list)
    M = np.zeros((N, N), dtype=np.complex128)
    
    kG = []
    for G in G_list:
        kG.append((k[0] + G[0], k[1] + G[1]))
        
    for i in range(N):
        for j in range(N):
            dm = G_list[i][0] - G_list[j][0]
            dn = G_list[i][1] - G_list[j][1]
            
            idx_i = int(dm) + 16
            idx_j = int(dn) + 16
            
            if 0 <= idx_i < 32 and 0 <= idx_j < 32:
                inv_eps_val = inv_eps_coeffs[idx_i, idx_j]
            else:
                inv_eps_val = 0.0
                
            dot_prod = kG[i][0] * kG[j][0] + kG[i][1] * kG[j][1]
            M[i, j] = inv_eps_val * dot_prod
            
    return M

def _solve_bands(k, inv_eps_coeffs, G_list, n_bands=6):
    M = _build_matrix(k, inv_eps_coeffs, G_list)
    eigvals = np.linalg.eigvalsh(M)
    eigvals = eigvals[eigvals > 1e-10]
    if len(eigvals) < n_bands:
        return None
    return np.sqrt(eigvals[:n_bands])

def _score_cell(cell, eps_low=1, eps_high=12, n_pw=10, n_k_points=10):
    eps_fft, inv_eps_fft = _fourier_coeffs(cell, eps_low, eps_high)
    
    G_list = []
    for m in range(-n_pw, n_pw + 1):
        for n in range(-n_pw, n_pw + 1):
            if m*m + n*n <= n_pw*n_pw:
                G_list.append((2*np.pi*m, 2*np.pi*n))
                
    k_points = [
        (0, 0),
        (np.pi/2, np.pi/2),
        (np.pi/2, 0),
        (np.pi/4, np.pi/4),
        (np.pi/2, np.pi/4),
        (np.pi/3, np.pi/6),
        (np.pi/2.5, np.pi/5),
        (np.pi/3.5, np.pi/7),
        (np.pi/4.5, np.pi/9),
        (np.pi/5, np.pi/10)
    ]
    
    min_omega_5 = float('inf')
    max_omega_4 = 0.0
    
    for k in k_points:
        bands = _solve_bands(k, inv_eps_fft, G_list, n_bands=6)
        if bands is None:
            continue
        omega_4 = bands[3]
        omega_5 = bands[4]
        
        if omega_5 < min_omega_5:
            min_omega_5 = omega_5
        if omega_4 > max_omega_4:
            max_omega_4 = omega_4
            
    if min_omega_5 == float('inf'):
        return -1.0
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    mask_rod = np.zeros((16, 16), dtype=np.int8)
    for i in range(16):
        for j in range(16):
            x = (i - 7.5) / 16.0
            y = (j - 7.5) / 16.0
            if x*x + y*y < (0.2 * 2)**2:
                mask_rod[i, j] = 1
                
    mask_cross = np.zeros((16, 16), dtype=np.int8)
    mask_cross[:, 7:9] = 1
    mask_cross[7:9, :] = 1
    
    mask_x = np.zeros((16, 16), dtype=np.int8)
    for i in range(16):
        mask_x[i, i] = 1
        mask_x[i, 15-i] = 1
    for i in range(16):
        if i < 15: mask_x[i, i+1] = 1
        if i > 0: mask_x[i, i-1] = 1
        if i < 15: mask_x[i, 15-i+1] = 1
        if i > 0: mask_x[i, 15-i-1] = 1

    candidates = [mask_rod, mask_cross, mask_x]
    
    best_score = -1.0
    best_mask = mask_rod
    
    for mask in candidates:
        cell = _get_symmetric_cell(mask)
        score = _score_cell(cell, eps_low, eps_high, n_pw=10, n_k_points=10)
        if score > best_score:
            best_score = score
            best_mask = mask.copy()
            
    current_mask = best_mask.copy()
    current_score = best_score
    
    search_pw = 10
    
    while time.time() < deadline:
        improved = False
        for i in range(16):
            for j in range(16):
                if time.time() > deadline:
                    break
                old_val = current_mask[i, j]
                current_mask[i, j] = 1 - old_val
                
                cell = _get_symmetric_cell(current_mask)
                score = _score_cell(cell, eps_low, eps_high, n_pw=search_pw, n_k_points=8)
                
                if score > current_score:
                    current_score = score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask.copy()
                else:
                    current_mask[i, j] = old_val
                    
            if time.time() > deadline:
                break
                
        if not improved:
            escaped = False
            for _ in range(10):
                if time.time() > deadline: break
                ri = np.random.randint(16)
                rj = np.random.randint(16)
                current_mask[ri, rj] = 1 - current_mask[ri, rj]
                cell = _get_symmetric_cell(current_mask)
                score = _score_cell(cell, eps_low, eps_high, n_pw=search_pw, n_k_points=8)
                if score > current_score:
                    current_score = score
                    escaped = True
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask.copy()
                else:
                    current_mask[ri, rj] = 1 - current_mask[ri, rj]
            if not escaped:
                break

    best_cell = _get_symmetric_cell(best_mask)
    return [[int(x) for x in row] for row in best_cell.tolist()]
