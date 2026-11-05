import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell to maximize the TM band gap
    between the 4th and 5th bands.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    size = problem.get("pixels", 32)
    if size != 32:
        # Fallback to seed if size is not 32, though task says 32
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell

    half = size // 2  # 16
    
    # Initialize with the seed design
    full_seed = np.zeros((size, size), dtype=int)
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.2 ** 2:
                full_seed[i, j] = 1
    
    domain = full_seed[:half, :half].copy()
    
    # Precompute G vectors for the solver
    N_pw = 7  # 7x7 = 49 plane waves
    g_vals = np.arange(-N_pw // 2, N_pw // 2 + 1)
    G_vectors = np.array([(gx, gy) for gy in g_vals for gx in g_vals], dtype=float)
    num_pw = len(G_vectors)
    
    def get_chi(grid_full):
        """
        Compute Fourier coefficients of 1/epsilon for the given full grid.
        """
        inv_eps = np.where(grid_full == 1, 1.0/12.0, 1.0)
        f = np.fft.fft2(inv_eps)
        f = f / (size * size)
        
        chi = {}
        max_g = N_pw - 1
        for gy in range(-max_g, max_g + 1):
            for gx in range(-max_g, max_g + 1):
                ix = gx % size
                iy = gy % size
                chi[(gx, gy)] = f[iy, ix]
        return chi

    def solve_bands(domain_grid):
        """
        Solve for the 4th and 5th TM band frequencies at Gamma, M, X.
        Returns the max gap ratio.
        """
        # Construct full grid
        full_grid = np.zeros((size, size), dtype=int)
        full_grid[:half, :half] = domain_grid
        full_grid[:half, half:] = domain_grid[:, ::-1]
        full_grid[half:, :half] = domain_grid[::-1, :]
        full_grid[half:, half:] = domain_grid[::-1, ::-1]
        
        chi = get_chi(full_grid)
        
        best_gap = -np.inf
        
        # k points in units of 2*pi/a
        k_points = [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([1.0, 1.0])
        ]
        
        for k in k_points:
            M = np.zeros((num_pw, num_pw), dtype=complex)
            for i in range(num_pw):
                ki = k + G_vectors[i]
                for j in range(num_pw):
                    kj = k + G_vectors[j]
                    gx_diff = int(G_vectors[i, 0] - G_vectors[j, 0])
                    gy_diff = int(G_vectors[i, 1] - G_vectors[j, 1])
                    chi_val = chi.get((gx_diff, gy_diff), 0.0)
                    dot_prod = np.dot(ki, kj)
                    M[i, j] = dot_prod * chi_val
            
            M_real = np.real(M)
            M_sym = (M_real + M_real.T) / 2.0
            eigvals = np.linalg.eigvalsh(M_sym)
            eigvals = np.sort(eigvals)
            
            if len(eigvals) >= 5:
                w4_sq = eigvals[3]
                w5_sq = eigvals[4]
                w4 = np.sqrt(max(0, w4_sq))
                w5 = np.sqrt(max(0, w5_sq))
                
                if w5 > w4:
                    gap_ratio = 2 * (w5 - w4) / (w5 + w4)
                    if gap_ratio > best_gap:
                        best_gap = gap_ratio
                        
        return best_gap

    # Optimization Loop
    current_domain = domain.copy()
    current_score = solve_bands(current_domain)
    best_domain = current_domain.copy()
    best_score = current_score
    
    pixels = [(i, j) for i in range(half) for j in range(half)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        np.random.shuffle(pixels)
        
        for i, j in pixels:
            old_val = current_domain[i, j]
            current_domain[i, j] = 1 - old_val
            
            new_score = solve_bands(current_domain)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_domain = current_domain.copy()
                improved = True
            else:
                current_domain[i, j] = old_val
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    # Return the best domain as a full 32x32 list
    final_domain = best_domain
    final_full = np.zeros((size, size), dtype=int)
    final_full[:half, :half] = final_domain
    final_full[:half, half:] = final_domain[:, ::-1]
    final_full[half:, :half] = final_domain[::-1, :]
    final_full[half:, half:] = final_domain[::-1, ::-1]
    
    return final_full.tolist()
