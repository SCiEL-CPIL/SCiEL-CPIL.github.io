import numpy as np
import scipy.optimize

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell using a superposition of 
    low-order Fourier harmonics (2,0), (0,2), and (2,2) modes.
    Optimizes amplitudes and phases to maximize the gap between TM bands 4 and 5.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Coordinates for the 32x32 grid, centered at 0
    idx = np.arange(size)
    x = (idx + 0.5) / size - 0.5
    y = (idx + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # C4v Symmetric Basis Functions
    b1 = np.cos(4 * np.pi * X) + np.cos(4 * np.pi * Y)
    b2 = np.cos(4 * np.pi * (X + Y)) + np.cos(4 * np.pi * (X - Y))
    
    # Precompute Plane Wave Indices for Surrogate
    M_PWE = 5 # 11x11 = 121 PWs
    G_indices = np.arange(-M_PWE, M_PWE + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    N_PW = len(Gx_flat)
    
    # Precompute G differences for Epsilon Matrix construction
    delta_gx = Gx_flat[:, None] - Gx_flat[None, :]
    delta_gy = Gy_flat[:, None] - Gy_flat[None, :]
    idx_x = (delta_gx % size).astype(int)
    idx_y = (delta_gy % size).astype(int)
    
    # Key k-points in the irreducible BZ
    k_points = [
        (0.0, 0.0), (0.5, 0.0), (0.5, 0.5),
        (0.25, 0.0), (0.25, 0.25), (0.35, 0.15),
        (0.4, 0.2), (0.3, 0.3), (0.1, 0.1),
        (0.45, 0.05), (0.45, 0.45), (0.2, 0.05)
    ]
    
    def calculate_surrogate_gap(cell):
        fft_vals = np.fft.fft2(cell) / (size * size)
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            kGx = kx + Gx_flat
            kGy = ky + Gy_flat
            D = kGx**2 + kGy**2
            
            E = fft_vals[idx_x, idx_y]
            
            try:
                sqrt_D_inv = 1.0 / np.sqrt(D)
                M = E * sqrt_D_inv[:, None] * sqrt_D_inv[None, :]
                M = (M + M.conj().T) / 2.0
                M = np.real(M)
                
                eigvals_mu = np.linalg.eigvalsh(M)
                eigvals_mu = np.sort(eigvals_mu)[::-1]
                
                if len(eigvals_mu) >= 5:
                    mu_4 = eigvals_mu[3]
                    mu_5 = eigvals_mu[4]
                    
                    if mu_4 > 0 and mu_5 > 0:
                        w4 = 1.0 / np.sqrt(mu_4)
                        w5 = 1.0 / np.sqrt(mu_5)
                        
                        max_omega_4 = max(max_omega_4, w4)
                        min_omega_5 = min(min_omega_5, w5)
            except (np.linalg.LinAlgError, ZeroDivisionError):
                return -np.inf
        
        if min_omega_5 == np.inf or max_omega_4 == 0.0:
            return -np.inf
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        score = gap / midgap
        return score

    def generate_cell_from_params(params):
        A1, A2, offset = params
        # Clip to valid range to ensure stability
        A1 = np.clip(A1, -1.5, 1.5)
        A2 = np.clip(A2, -1.5, 1.5)
        offset = np.clip(offset, 0.2, 0.8)
        
        field = offset + A1 * b1 + A2 * b2
        cell_raw = (field > 0.5).astype(int)
        
        sym_cells = [
            cell_raw, cell_raw[:, ::-1], cell_raw[::-1, :], cell_raw.T,
            cell_raw[::-1, ::-1], cell_raw.T[::-1, :], cell_raw[:, ::-1].T, cell_raw[::-1, :].T
        ]
        avg_cell = np.mean(sym_cells, axis=0)
        cell_sym = (avg_cell >= 0.5).astype(int)
        
        return cell_sym

    def surrogate_score_func(params):
        cell = generate_cell_from_params(params)
        fill = np.sum(cell) / (size * size)
        if fill < 0.15 or fill > 0.85:
            return 10.0 # Penalize by returning high "cost" (we minimize -score, so high cost is bad)
        score = calculate_surrogate_gap(cell)
        if np.isnan(score) or np.isinf(score):
            return 10.0
        return -score

    # Initial guess
    x0 = [0.5, 0.5, 0.5]
    bounds = [(-1.5, 1.5), (-1.5, 1.5), (0.2, 0.8)]
    
    # Optimize using Powell
    result = scipy.optimize.minimize(surrogate_score_func, x0, method='Powell', 
                                     bounds=bounds, 
                                     options={'maxiter': 100, 'xtol': 1e-4, 'ftol': 1e-4})
    
    best_params = result.x
    best_cell = generate_cell_from_params(best_params)
    
    candidates = [best_params]
    candidates.append([0.8, 0.4, 0.5])
    candidates.append([0.4, 0.8, 0.5])
    candidates.append([0.6, 0.6, 0.5])
    candidates.append([0.9, 0.2, 0.45])
    candidates.append([0.2, 0.9, 0.55])
    
    best_score = -np.inf
    final_cell = None
    
    for p in candidates:
        cell = generate_cell_from_params(p)
        fill = np.sum(cell) / (size * size)
        if fill < 0.15 or fill > 0.85:
            continue
        score = calculate_surrogate_gap(cell)
        if score > best_score:
            best_score = score
            final_cell = cell

    if final_cell is None:
        final_cell = np.zeros((size, size), dtype=int)
        center = size // 2
        for i in range(size):
            for j in range(size):
                if (i - center)**2 + (j - center)**2 < (size*0.2)**2:
                    final_cell[i,j] = 1
                    
    return final_cell.tolist()
