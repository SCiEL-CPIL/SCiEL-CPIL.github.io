import numpy as np
from scipy.optimize import minimize

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell using a superposition of 
    low-order Fourier harmonics (2,0) and (2,2) modes.
    Optimizes amplitudes and offset to maximize the TM band gap between 
    bands 4 and 5 using a fast PWE surrogate and Nelder-Mead optimization.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Coordinates for the 32x32 grid, centered at 0
    idx = np.arange(size)
    x = (idx + 0.5) / size - 0.5
    y = (idx + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Precompute basis functions for C4v-symmetric combinations
    # Mode 1: (2,0) + (0,2)
    basis_20 = np.cos(4 * np.pi * X) + np.cos(4 * np.pi * Y)
    # Mode 2: (2,2) + (2,-2)
    basis_22 = np.cos(4 * np.pi * (X + Y)) + np.cos(4 * np.pi * (X - Y))
    
    # Fast PWE Surrogate Parameters
    M_PWE = 4  # 9x9 PWs
    G_indices = np.arange(-M_PWE, M_PWE + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    N_PW = len(Gx_flat)
    
    # Key k-points in the irreducible BZ
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.35, 0.15),
        (0.4, 0.2),
        (0.3, 0.3)
    ]
    
    # Precompute E matrix structure for surrogate (depends only on G differences)
    # E[i,j] = eps(G_i - G_j)
    # We need the FFT of the cell to get eps(G)
    
    def calculate_surrogate_gap(cell):
        """
        Calculates an approximate gap using a small PWE expansion.
        Returns the score (gap-midgap ratio).
        """
        # Compute Fourier coefficients of the pixel map
        # Note: The evaluator uses exact coefficients. 
        # fft2 gives c_k = 1/N sum x_n e^{-2pi i k n / N}.
        # The PWE matrix uses <epsilon>_{G-G'} = c_{G-G'}.
        fft_vals = np.fft.fft2(cell) / (size * size)
        
        # Helper to get eps(Gx, Gy) with wrap-around
        # Gx, Gy are integers.
        def get_eps(gx, gy):
            ix = int(gx) % size
            iy = int(gy) % size
            return fft_vals[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            kGx = kx + Gx_flat
            kGy = ky + Gy_flat
            
            D = kGx**2 + kGy**2
            
            # Vectorized computation of epsilon matrix
            # We need E[i,j] = get_eps(Gx_flat[i] - Gx_flat[j], Gy_flat[i] - Gy_flat[j])
            # Since Gx_flat, Gy_flat are integers, differences are integers.
            # We can vectorize this lookup.
            dx = Gx_flat[:, None] - Gx_flat[None, :]
            dy = Gy_flat[:, None] - Gy_flat[None, :]
            
            # Map differences to indices in fft_vals
            # fft_vals is size x size.
            # Index = diff % size
            ix = dx % size
            iy = dy % size
            
            E = fft_vals[ix, iy]
            
            # A = D * E ? 
            # For TM mode:
            # (k+G)^2 * c_{G-G'} * a_{G'} 
            # The matrix element is (k+G_i)^2 * epsilon(G_i - G_j).
            # Wait, the standard PWE equation for TM is:
            # sum_j (k+G_i)^2 epsilon(G_i-G_j) a_j = omega^2 a_i ?
            # No, it's usually:
            # sum_j epsilon(G_i-G_j) (k+G_j)^2 a_j = omega^2 a_i ?
            # Let's check the standard form.
            # Maxwell: curl curl E = k^2 eps E.
            # In Fourier: (k+G)^2 a_{k+G} - sum ... 
            # For TM (E_z), the equation is:
            # (k+G)^2 a_{G} = omega^2 sum_{G'} epsilon(G-G') a_{G'}
            # So A_{i,j} = (k+G_i)^2 * epsilon(G_i - G_j) ?? No.
            # The operator is (k+G)^2 * eps^{-1}? No, for TM, it's direct.
            # Actually, the eigenvalue problem is:
            # sum_j epsilon(G_i - G_j) (k+G_j)^2 a_j = omega^2 a_i
            # This is NOT a symmetric matrix.
            # However, we can transform it.
            # Let b_j = sqrt(epsilon(G_j)) a_j? No, epsilon is not diagonal.
            
            # Let's look at the current code's logic:
            # A = D * E
            # A = (A + A.conj().T) / 2.0
            # This assumes A is Hermitian.
            # If A_{i,j} = D_i * E_{i,j}, then A_{j,i} = D_j * E_{j,i}.
            # E_{j,i} = epsilon(G_j - G_i) = epsilon(-(G_i - G_j)) = epsilon(G_i - G_j)^* (since real).
            # So A_{j,i} = D_j E_{i,j}.
            # A is not symmetric unless D_i = D_j.
            # The current code symmetrizes it. This is an approximation.
            # A better surrogate for TM is to solve the generalized eigenvalue problem or use the symmetric form:
            # H_{i,j} = sqrt(D_i D_j) * E_{i,j}
            # Then H is symmetric (if E is symmetric).
            # eigenvalues of H are sqrt(omega^2).
            
            # Let's use the symmetric form H = sqrt(D_i D_j) * E_{i,j}
            sqrt_D = np.sqrt(D)
            H = np.outer(sqrt_D, sqrt_D) * E
            H = (H + H.T) / 2.0 # Ensure symmetry
            H = np.real(H)
            
            try:
                eigvals = np.linalg.eigvalsh(H)
            except np.linalg.LinAlgError:
                return -np.inf
                
            eigvals = np.sort(eigvals)
            
            if len(eigvals) >= 5:
                # omega^2 are eigenvalues. omega = sqrt(eigval)
                # We need bands 4 and 5.
                # Indices 3 and 4.
                w4_sq = eigvals[3]
                w5_sq = eigvals[4]
                
                if w4_sq > 0 and w5_sq > 0:
                    w4 = np.sqrt(w4_sq)
                    w5 = np.sqrt(w5_sq)
                    max_omega_4 = max(max_omega_4, w4)
                    min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == np.inf or max_omega_4 == 0.0:
            return -np.inf
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        score = gap / midgap
        return score

    def generate_cell(A20, A22, offset):
        """Generate a binary cell from Fourier parameters."""
        field = offset + A20 * basis_20 + A22 * basis_22
        
        cell_raw = (field > 0.5).astype(int)
        
        # Enforce C4v symmetry by averaging over all 8 symmetry operations
        sym_cells = [
            cell_raw,
            cell_raw[:, ::-1],
            cell_raw[::-1, :],
            cell_raw.T,
            cell_raw[::-1, ::-1],
            cell_raw.T[::-1, :],
            cell_raw[:, ::-1].T,
            cell_raw[::-1, :].T
        ]
        
        avg_cell = np.mean(sym_cells, axis=0)
        cell_sym = (avg_cell >= 0.5).astype(int)
        
        return cell_sym

    def objective(params):
        """Objective function for optimization: negative surrogate gap."""
        A20, A22, offset = params
        cell = generate_cell(A20, A22, offset)
        
        fill = np.sum(cell) / (size * size)
        if fill < 0.10 or fill > 0.90:
            return 100.0 # Penalty for bad fill fraction
            
        score = calculate_surrogate_gap(cell)
        
        if np.isnan(score) or np.isinf(score):
            return 100.0
            
        return -score

    best_score = -np.inf
    best_cell = None
    best_params = None
    
    # Stage 1: Coarse Grid Search
    # A20, A22 in [-1.5, 1.5], offset in [0.3, 0.7]
    A20_coarse = np.linspace(-1.5, 1.5, 11)
    A22_coarse = np.linspace(-1.5, 1.5, 11)
    offset_coarse = np.linspace(0.3, 0.7, 9)
    
    coarse_results = []
    
    for A20 in A20_coarse:
        for A22 in A22_coarse:
            for offset in offset_coarse:
                obj_val = objective([A20, A22, offset])
                # objective is -score, so lower is better
                if obj_val < 50.0: # Filter out penalties
                    coarse_results.append((obj_val, A20, A22, offset))
    
    if not coarse_results:
        # Fallback
        return [[0]*size for _ in range(size)]

    # Sort by objective value (best first)
    coarse_results.sort(key=lambda x: x[0])
    
    # Take top 5 candidates for refinement
    top_candidates = coarse_results[:5]
    
    # Stage 2: Nelder-Mead Refinement
    for _, A20_init, A22_init, offset_init in top_candidates:
        # Define bounds for Nelder-Mead (it doesn't support bounds directly, but we can use penalty or just let it run and clip)
        # We'll use a standard simplex
        x0 = np.array([A20_init, A22_init, offset_init])
        
        # Use minimize with method='Nelder-Mead'
        # maxiter to keep it fast
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.001})
        
        if res.success or res.fun < 50.0:
            A20_opt, A22_opt, offset_opt = res.x
            # Clip to reasonable range
            A20_opt = np.clip(A20_opt, -2.0, 2.0)
            A22_opt = np.clip(A22_opt, -2.0, 2.0)
            offset_opt = np.clip(offset_opt, 0.2, 0.8)
            
            cell = generate_cell(A20_opt, A22_opt, offset_opt)
            fill = np.sum(cell) / (size * size)
            
            if 0.10 <= fill <= 0.90:
                score = calculate_surrogate_gap(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    best_params = (A20_opt, A22_opt, offset_opt)
    
    # Stage 3: Final Check and Fallback
    if best_cell is None:
        # If optimization failed, use the best coarse candidate
        _, A20_best, A22_best, offset_best = coarse_results[0]
        best_cell = generate_cell(A20_best, A22_best, offset_best).tolist()
        best_score = calculate_surrogate_gap(np.array(best_cell))

    # Final symmetry enforcement just in case
    best_cell_np = np.array(best_cell)
    sym_cells = [
        best_cell_np,
        best_cell_np[:, ::-1],
        best_cell_np[::-1, :],
        best_cell_np.T,
        best_cell_np[::-1, ::-1],
        best_cell_np.T[::-1, :],
        best_cell_np[:, ::-1].T,
        best_cell_np[::-1, :].T
    ]
    avg_cell = np.mean(sym_cells, axis=0)
    final_cell = (avg_cell >= 0.5).astype(int)
    
    return final_cell.tolist()
