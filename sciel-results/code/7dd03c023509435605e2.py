import numpy as np
import time

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # PWE Setup
    # 17x17 G vectors
    g_range = np.arange(-8, 9)
    G_x, G_y = np.meshgrid(g_range, g_range, indexing='ij')
    G_vectors = np.column_stack((G_x.ravel(), G_y.ravel()))
    N_pw = len(G_vectors)
    
    # Precompute G vector indices for fast lookup
    # We will use a 2D array for coefficients
    # coeff[m_idx, n_idx] where m_idx = m + 8, n_idx = n + 8
    
    def get_epsilon_grid(cell):
        eps_grid = np.array(cell, dtype=float) * (eps_high - eps_low) + eps_low
        return eps_grid

    def compute_inv_eps_coeffs(eps_grid):
        inv_eps = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)
        return fft_inv_eps

    def build_tm_matrix(k_x, k_y, inv_eps_coeffs):
        # k_x, k_y are in radians (units of 1/a)
        # G vectors are in units of 2*pi/a
        # So k+G is in units of 2*pi/a?
        # The eigenvalue is (omega a / c)^2.
        # Let's work in units where a=1.
        # k is in [0, pi]. G is integer * 2*pi.
        # This mixing of units is tricky.
        # Standard PWE: k is in BZ (0 to pi/a). G is 2*pi*n/a.
        # |k+G|^2 has units 1/a^2.
        # Eigenvalue lambda = (omega a / c)^2.
        
        M = np.zeros((N_pw, N_pw), dtype=complex)
        
        # Precompute |k+G|^2 for all G
        # G_vectors are in units of 2*pi.
        # k is in units of 1 (where BZ edge is pi).
        # This is inconsistent.
        # Let's scale k by 2*pi? No.
        # Let's assume k is in radians per unit length, and G is 2*pi * integer.
        # Then k+G is in radians per unit length.
        
        k_plus_G_x = k_x + G_vectors[:, 0] * 2 * np.pi
        k_plus_G_y = k_y + G_vectors[:, 1] * 2 * np.pi
        k_plus_G_sq = k_plus_G_x**2 + k_plus_G_y**2
        
        for i in range(N_pw):
            mi, ni = G_vectors[i]
            for j in range(N_pw):
                mj, nj = G_vectors[j]
                dm = int(mi - mj)
                dn = int(ni - nj)
                # Check if dm, dn are in range [-8, 8]
                if -8 <= dm <= 8 and -8 <= dn <= 8:
                    idx_m = dm + 8
                    idx_n = dn + 8
                    coeff = inv_eps_coeffs[idx_m, idx_n]
                    M[i, j] = coeff * k_plus_G_sq[j]
                else:
                    M[i, j] = 0.0
                    
        return M

    def solve_bands_for_k(k_x, k_y, inv_eps_coeffs):
        M = build_tm_matrix(k_x, k_y, inv_eps_coeffs)
        try:
            eigvals = np.linalg.eigvalsh(M)
            return np.sqrt(eigvals)
        except np.linalg.LinAlgError:
            return None

    def evaluate_cell(cell):
        eps_grid = get_epsilon_grid(cell)
        inv_eps_coeffs = compute_inv_eps_coeffs(eps_grid)
        
        # Key k-points in irreducible BZ
        # Gamma (0,0), M (pi/2, pi/2), X (pi/2, 0)
        k_points = [
            (0.0, 0.0),
            (np.pi/2, 0.0),
            (np.pi/2, np.pi/2)
        ]
        
        max_omega_4 = 0
        min_omega_5 = np.inf
        
        for kx, ky in k_points:
            bands = solve_bands_for_k(kx, ky, inv_eps_coeffs)
            if bands is None:
                continue
            if len(bands) >= 5:
                w4 = bands[3]
                w5 = bands[4]
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == np.inf:
            return -1.0
            
        gap = 2 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return gap

    def make_cell(shape_type, param1, param2=0.0):
        cell = np.zeros((size, size), dtype=int)
        center = (size - 1) / 2.0
        
        for i in range(size):
            for j in range(size):
                x = (i - center) / size
                y = (j - center) / size
                val = 0
                
                if shape_type == 'square_rod':
                    if abs(x) < param1 and abs(y) < param1:
                        val = 1
                elif shape_type == 'rounded_square_rod':
                    w = param1
                    r = param2
                    dx = max(abs(x) - (w - r), 0)
                    dy = max(abs(y) - (w - r), 0)
                    if dx*dx + dy*dy <= r*r or (abs(x) < w - r) or (abs(y) < w - r):
                         val = 1
                elif shape_type == 'square_hole':
                    if abs(x) < param1 and abs(y) < param1:
                        val = 0
                    else:
                        val = 1
                elif shape_type == 'rounded_square_hole':
                    w = param1
                    r = param2
                    dx = max(abs(x) - (w - r), 0)
                    dy = max(abs(y) - (w - r), 0)
                    if dx*dx + dy*dy <= r*r or (abs(x) < w - r) or (abs(y) < w - r):
                        val = 0
                    else:
                        val = 1
                elif shape_type == 'circle_rod':
                    if x*x + y*y < param1*param1:
                        val = 1
                elif shape_type == 'circle_hole':
                    if x*x + y*y < param1*param1:
                        val = 0
                    else:
                        val = 1
                else:
                    val = 0
                    
                cell[i, j] = val
                
        return cell.tolist()

    best_score = -np.inf
    best_cell = None
    
    shapes = [
        'square_rod',
        'rounded_square_rod',
        'square_hole',
        'rounded_square_hole',
        'circle_rod',
        'circle_hole'
    ]
    
    param_range = np.linspace(0.15, 0.45, 8)
    radius_range = np.linspace(0.0, 0.1, 4)
    
    for shape_name in shapes:
        if time.time() > deadline:
            break
            
        for p1 in param_range:
            if shape_name in ['rounded_square_rod', 'rounded_square_hole']:
                for p2 in radius_range:
                    cell = make_cell(shape_name, p1, p2)
                    score = evaluate_cell(cell)
                    if score > best_score:
                        best_score = score
                        best_cell = cell
            else:
                cell = make_cell(shape_name, p1)
                score = evaluate_cell(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell

    if best_cell is None:
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell

    return best_cell
