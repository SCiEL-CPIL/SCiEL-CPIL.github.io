import numpy as np
import time
import math

def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Use a high-fidelity 17x17 plane-wave evaluator to match the task's 289 PW method.
    2. Optimize a low-dimensional parameter space (geometric shapes) rather than raw pixels.
    3. Generate C4v symmetric patterns from these parameters.
    4. Use Nelder-Mead optimization on the parameters.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 190.0

    # --- 1. High-Fidelity Evaluator (17x17 PW) ---
    pw_half = 8
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    Gx_arr = np.array([g[0] for g in G_vectors])
    Gy_arr = np.array([g[1] for g in G_vectors])
    
    # Precompute G differences indices for FFT
    # We need coeffs of 1/epsilon at G_i - G_j
    # Since 1/eps is discrete on 32x32, we can use FFT to get all coeffs up to size/2
    # But we only need specific G differences.
    # Let's precompute the index mapping for the required G differences.
    G_diff_h = []
    G_diff_k = []
    for i in range(N_pw):
        for j in range(N_pw):
            dh = G_vectors[i][0] - G_vectors[j][0]
            dk = G_vectors[i][1] - G_vectors[j][1]
            G_diff_h.append(dh)
            G_diff_k.append(dk)
    
    G_diff_h = np.array(G_diff_h)
    G_diff_k = np.array(G_diff_k)
    
    # Map to FFT indices (0..size-1)
    # FFT output index for frequency n is n % size
    G_diff_idx_h = G_diff_h % size
    G_diff_idx_k = G_diff_k % size
    
    # K-points in irreducible BZ
    # Gamma (0,0), M (0.5, 0), X (0.5, 0.5) and edges
    k_points = []
    # Gamma to M (0, 0) to (0.5, 0)
    for t in np.linspace(0, 0.5, 6):
        k_points.append((0.0, t))
    # M to X (0.5, 0) to (0.5, 0.5)
    for t in np.linspace(0, 0.5, 6):
        k_points.append((0.5 - t, 0.5))
    # X to Gamma (0.5, 0.5) to (0, 0) - usually covered by symmetry or separate line
    # The irreducible BZ for square lattice is triangle (0,0)-(0.5,0)-(0.5,0.5)
    # So we need points inside this triangle.
    # Let's add a few interior points
    k_points.append((0.25, 0.1))
    k_points.append((0.25, 0.2))
    k_points.append((0.3, 0.1))
    
    def evaluate_gap(cell_grid):
        """
        Computes the gap using 17x17 PW expansion.
        """
        eps_grid = np.where(cell_grid == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        
        # FFT of 1/eps
        fft_1eps = np.fft.fft2(inv_eps)
        
        # Extract coefficients for all G_i - G_j pairs
        # fft_1eps is indexed [ky, kx] in numpy convention?
        # np.fft.fft2 returns array where index [n, m] corresponds to frequency (m, n) in (x, y)?
        # Actually, numpy fft2: output[i, j] is coefficient for (j, i) if we assume standard math freqs?
        # Let's be careful.
        # Standard: F(u, v) = sum f(x, y) exp(-i 2pi (ux + vy) / N)
        # numpy.fft.fft2(a)[u, v] corresponds to frequency (v, u) in terms of x,y cycles?
        # Let's check documentation or just test.
        # Usually, if we want coeff for G=(hx, hy), we look at fft[hx, hy]?
        # No, numpy axes are (row, col) = (y, x).
        # So fft[hy, hx] is the coefficient for frequency (hx, hy).
        # Let's assume G_diff_h is hx and G_diff_k is hy.
        # Then we need fft_1eps[G_diff_idx_k, G_diff_idx_h].
        
        coeffs = fft_1eps[G_diff_idx_k, G_diff_idx_h] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            # H matrix
            # H_ij = (K_i dot K_j) * c_{G_i - G_j}
            # K_i dot K_j = Kx_i Kx_j + Ky_i Ky_j
            # We can form H as:
            # H = (Kx[:, None] * Kx[None, :] + Ky[:, None] * Ky[None, :]) * coeffs_matrix
            # coeffs_matrix needs to be reshaped to (N_pw, N_pw)
            
            c_mat = coeffs.reshape(N_pw, N_pw)
            dot_mat = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dot_mat * c_mat
            
            # H is Hermitian (real symmetric in this case because c is real for symmetric eps? 
            # Wait, 1/eps is real, so coeffs are real?
            # If eps is real, 1/eps is real. FFT of real signal is conjugate symmetric.
            # c_{-G} = c_G^*. Since G_i - G_j can be negative, we handle it via modulo.
            # Is H symmetric?
            # H_ij = (K_i . K_j) c_{G_i - G_j}
            # H_ji = (K_j . K_i) c_{G_j - G_i} = (K_i . K_j) c_{-(G_i - G_j)}
            # c_{-G} = c_G^* (conjugate). Since c is real (eps is real), c_{-G} = c_G.
            # So H is real symmetric.
            
            # Just in case, take real part
            H_real = H.real
            
            eigenvalues = np.linalg.eigvalsh(H_real)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                w4 = np.sqrt(max(0, w4_sq))
                w5 = np.sqrt(max(0, w5_sq))
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2 * gap / midgap
        return ratio

    # --- 2. Parameterized Pattern Generator ---
    def generate_cell(params):
        """
        params: [r1, r2, r3, x1, y1]
        r1, r2, r3: Radii of 3 concentric square-like shapes?
        Or maybe 3 distinct disks?
        Let's try 3 disks at (8,8), (24,8), (8,24), (24,24) is 4 disks.
        Let's stick to 4 disks at quarter points.
        params: [r1, r2, r3, r4] for the 4 disks.
        Or better: 1 radius for all 4, and 1 shape factor?
        Let's try:
        params[0]: radius of disk at (8,8)
        params[1]: radius of disk at (8,24)
        params[2]: radius of disk at (24,8)
        params[3]: radius of disk at (24,24)
        Due to C4v, these should be equal.
        Let's enforce C4v by using one radius R and one offset O?
        
        Let's try a more flexible shape:
        A "superlattice" of 4 disks.
        params: [R, cx, cy] where cx, cy are offsets from center (15.5, 15.5)?
        No, let's keep it simple: 4 disks at (8,8), (8,24), (24,8), (24,24).
        Vary their radii.
        params: [r1, r2, r3, r4]
        
        Wait, C4v symmetry requires r1=r2=r3=r4 if centers are symmetric.
        So we only need 1 parameter: R.
        This is too low dimensional.
        
        Let's try 2 parameters:
        1. Radius of central disk (if any)
        2. Radius of corner disks
        
        Let's try:
        params[0]: Radius of 4 corner disks
        params[1]: Radius of central disk (centered at 15.5, 15.5)
        
        """
        cell = np.zeros((size, size), dtype=int)
        
        R_corner = params[0]
        R_center = params[1]
        
        # Corner disks
        centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
        for cy, cx in centers:
            for i in range(size):
                for j in range(size):
                    dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                    if dist < R_corner:
                        cell[i, j] = 1
                        
        # Central disk
        cy, cx = 15.5, 15.5
        for i in range(size):
            for j in range(size):
                dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                if dist < R_center:
                    cell[i, j] = 1
            
        return cell

    # --- 3. Optimization ---
    # Initial guess
    x0 = np.array([4.0, 0.0])
    
    # Bounds for Nelder-Mead (not directly supported, so we'll use a penalty or just hope)
    # Let's use scipy.optimize.minimize
    
    def objective(params):
        cell = generate_cell(params)
        score = evaluate_gap(cell)
        return -score # Minimize negative score
        
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Fallback to simple hill climbing if scipy not available
        best_params = x0.copy()
        best_score = -objective(x0)
        step_size = 0.5
        for _ in range(1000):
            if time.time() > deadline:
                break
            for dim in range(len(x0)):
                for direction in [-1, 1]:
                    new_params = best_params.copy()
                    new_params[dim] += direction * step_size
                    new_score = -objective(new_params)
                    if new_score > best_score:
                        best_params = new_params
                        best_score = new_score
        x_opt = best_params
    else:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 0.1, 'fatol': 0.01})
        x_opt = result.x
        
    # Finalize
    final_cell = generate_cell(x_opt)
    
    # Ensure strict C4v symmetry (just in case floating point errors in generator broke it, though it shouldn't)
    def enforce_symmetry(c):
        sym = np.zeros_like(c)
        for i in range(size):
            for j in range(size):
                points = [
                    (i, j),
                    (size - 1 - i, j),
                    (i, size - 1 - j),
                    (size - 1 - i, size - 1 - j),
                    (j, i),
                    (size - 1 - j, j),
                    (j, size - 1 - i),
                    (size - 1 - j, size - 1 - i)
                ]
                s = sum(c[pi, pj] for pi, pj in points)
                sym[i, j] = 1 if s >= 4 else 0
        return sym
        
    final_cell = enforce_symmetry(final_cell)
    return final_cell.tolist()
