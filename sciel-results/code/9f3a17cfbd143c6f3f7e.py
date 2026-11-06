import numpy as np

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell using a superposition of 
    low-order Fourier harmonics that directly couple to the 4th and 5th TM bands. 
    Optimizes amplitudes and phases to maximize the spectral separation between bands 4 and 5.
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
    # (1,0) + (0,1): cos(2πX) + cos(2πY)
    basis_10 = np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y)
    
    # (2,0) + (0,2): cos(4πX) + cos(4πY)
    basis_20 = np.cos(4 * np.pi * X) + np.cos(4 * np.pi * Y)
    
    # (3,0) + (0,3): cos(6πX) + cos(6πY)
    basis_30 = np.cos(6 * np.pi * X) + np.cos(6 * np.pi * Y)
    
    # (2,2) + (2,-2): cos(4π(X+Y)) + cos(4π(X-Y))
    basis_22 = np.cos(4 * np.pi * (X + Y)) + np.cos(4 * np.pi * (X - Y))
    
    # (1,1) + (1,-1): cos(2π(X+Y)) + cos(2π(X-Y))
    basis_11 = np.cos(2 * np.pi * (X + Y)) + np.cos(2 * np.pi * (X - Y))
    
    # (3,1) + permutations for C4v
    basis_31 = (np.cos(6*np.pi*X + 2*np.pi*Y) + np.cos(2*np.pi*X + 6*np.pi*Y) +
                np.cos(6*np.pi*X - 2*np.pi*Y) + np.cos(2*np.pi*X - 6*np.pi*Y))
    
    # Fast PWE Surrogate Parameters
    M_PWE = 5  # Indices -5 to 5, so 11x11 = 121 PWs
    G_indices = np.arange(-M_PWE, M_PWE + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    N_PW = len(Gx_flat)
    
    # Key k-points in the irreducible BZ (fractional coordinates)
    # More comprehensive set covering the BZ boundary and interior
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.35, 0.15),
        (0.4, 0.2),
        (0.3, 0.3),
        (0.45, 0.05),
        (0.4, 0.1),
        (0.35, 0.25),
        (0.45, 0.2),
        (0.3, 0.1),
        (0.2, 0.2),
        (0.4, 0.4),
        (0.45, 0.45),
    ]
    
    def calculate_surrogate_gap(cell):
        """
        Calculates an approximate gap using a small PWE expansion.
        Returns the score (gap-midgap ratio).
        """
        # Compute Fourier coefficients of the pixel map
        fft_vals = np.fft.fft2(cell)
        fft_vals /= (size * size)
        
        def get_eps(Gx_val, Gy_val):
            ix = int(Gx_val) % size
            iy = int(Gy_val) % size
            return fft_vals[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            kGx = kx + Gx_flat
            kGy = ky + Gy_flat
            
            D = np.outer(kGx, kGx) + np.outer(kGy, kGy)
            
            # Vectorized computation of epsilon matrix
            dx_matrix = Gx_flat[:, None] - Gx_flat[None, :]
            dy_matrix = Gy_flat[:, None] - Gy_flat[None, :]
            
            E = np.zeros((N_PW, N_PW), dtype=complex)
            for i in range(N_PW):
                for j in range(N_PW):
                    E[i, j] = get_eps(Gx_flat[i] - Gx_flat[j], Gy_flat[i] - Gy_flat[j])
            
            A = D * E
            A = (A + A.conj().T) / 2.0
            A = np.real(A)
            
            try:
                eigvals = np.linalg.eigvalsh(A)
            except np.linalg.LinAlgError:
                return -np.inf
                
            eigvals = np.sort(eigvals)
            
            if len(eigvals) >= 5:
                w4 = eigvals[3]
                w5 = eigvals[4]
                
                if w4 > 0 and w5 > 0:
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

    def generate_cell(A10, A20, A30, A22, A11, A31, offset, 
                      phase10=0.0, phase20=0.0, phase30=0.0, 
                      phase22=0.0, phase11=0.0, phase31=0.0):
        """Generate a binary cell from Fourier parameters."""
        field = offset
        if A10 > 0:
            field += A10 * (np.cos(2 * np.pi * X + phase10) + np.cos(2 * np.pi * Y + phase10))
        if A20 > 0:
            field += A20 * (np.cos(4 * np.pi * X + phase20) + np.cos(4 * np.pi * Y + phase20))
        if A30 > 0:
            field += A30 * (np.cos(6 * np.pi * X + phase30) + np.cos(6 * np.pi * Y + phase30))
        if A22 > 0:
            field += A22 * (np.cos(4 * np.pi * (X + Y) + phase22) + np.cos(4 * np.pi * (X - Y) + phase22))
        if A11 > 0:
            field += A11 * (np.cos(2 * np.pi * (X + Y) + phase11) + np.cos(2 * np.pi * (X - Y) + phase11))
        if A31 > 0:
            field += A31 * (np.cos(6*np.pi*X + 2*np.pi*Y + phase31) + np.cos(2*np.pi*X + 6*np.pi*Y + phase31) +
                            np.cos(6*np.pi*X - 2*np.pi*Y + phase31) + np.cos(2*np.pi*X - 6*np.pi*Y + phase31))
        
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

    def evaluate_params(params):
        """Evaluate a parameter set and return (score, cell)."""
        A10, A20, A30, A22, A11, A31, offset = params[:7]
        phase10, phase20, phase30, phase22, phase11, phase31 = params[7:13]
        
        cell = generate_cell(A10, A20, A30, A22, A11, A31, offset,
                            phase10, phase20, phase30, phase22, phase11, phase31)
        
        fill = np.sum(cell) / (size * size)
        if fill < 0.15 or fill > 0.85:
            return -np.inf, cell
        
        score = calculate_surrogate_gap(cell)
        return score, cell

    best_score = -np.inf
    best_cell = None
    best_params = None
    
    # Stage 1: Coarse grid search over key amplitudes and offset
    # Focus on the most impactful parameters first
    A20_coarse = np.linspace(0.2, 1.0, 6)
    A22_coarse = np.linspace(0.0, 0.8, 5)
    offset_coarse = np.linspace(0.35, 0.65, 7)
    
    for A20 in A20_coarse:
        for A22 in A22_coarse:
            for offset in offset_coarse:
                params = (0.0, A20, 0.0, A22, 0.0, 0.0, offset, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                score, cell = evaluate_params(params)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    best_params = params
    
    # Stage 2: Add (1,0) and (3,0) modes around best
    if best_cell is not None:
        A20_best = best_params[1]
        A22_best = best_params[3]
        offset_best = best_params[6]
        
        A10_fine = np.linspace(-0.3, 0.3, 5)
        A30_fine = np.linspace(0.0, 0.4, 4)
        
        for A10 in A10_fine:
            for A30 in A30_fine:
                params = (A10, A20_best, A30, A22_best, 0.0, 0.0, offset_best, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                score, cell = evaluate_params(params)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    best_params = params
    
    # Stage 3: Add (1,1) and (3,1) modes
    if best_cell is not None:
        A10_best = best_params[0]
        A20_best = best_params[1]
        A30_best = best_params[2]
        A22_best = best_params[3]
        offset_best = best_params[6]
        
        A11_fine = np.linspace(-0.3, 0.3, 4)
        A31_fine = np.linspace(0.0, 0.3, 3)
        
        for A11 in A11_fine:
            for A31 in A31_fine:
                params = (A10_best, A20_best, A30_best, A22_best, A11, A31, offset_best, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                score, cell = evaluate_params(params)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    best_params = params
    
    # Stage 4: Fine-tune all amplitudes and phases
    if best_cell is not None:
        A10_best, A20_best, A30_best, A22_best, A11_best, A31_best, offset_best = best_params[:7]
        
        # Fine grid around best values
        A10_fine2 = np.linspace(A10_best - 0.1, A10_best + 0.1, 3)
        A20_fine2 = np.linspace(max(0.1, A20_best - 0.1), min(1.2, A20_best + 0.1), 3)
        A30_fine2 = np.linspace(max(0.0, A30_best - 0.1), min(0.5, A30_best + 0.1), 3)
        A22_fine2 = np.linspace(max(0.0, A22_best - 0.1), min(1.0, A22_best + 0.1), 3)
        A11_fine2 = np.linspace(A11_best - 0.1, A11_best + 0.1, 3)
        A31_fine2 = np.linspace(max(0.0, A31_best - 0.1), min(0.4, A31_best + 0.1), 3)
        offset_fine2 = np.linspace(max(0.3, offset_best - 0.05), min(0.7, offset_best + 0.05), 3)
        
        for A10 in A10_fine2:
            for A20 in A20_fine2:
                for A30 in A30_fine2:
                    for A22 in A22_fine2:
                        for A11 in A11_fine2:
                            for A31 in A31_fine2:
                                for offset in offset_fine2:
                                    params = (A10, A20, A30, A22, A11, A31, offset, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                                    score, cell = evaluate_params(params)
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_cell = cell.tolist()
                                        best_params = params
    
    # Stage 5: Phase optimization
    if best_cell is not None:
        A10_best, A20_best, A30_best, A22_best, A11_best, A31_best, offset_best = best_params[:7]
        
        phase10_fine = np.linspace(-0.4, 0.4, 3)
        phase20_fine = np.linspace(-0.4, 0.4, 3)
        phase22_fine = np.linspace(-0.4, 0.4, 3)
        
        for p10 in phase10_fine:
            for p20 in phase20_fine:
                for p22 in phase22_fine:
                    params = (A10_best, A20_best, A30_best, A22_best, A11_best, A31_best, offset_best,
                              p10, p20, 0.0, p22, 0.0, 0.0)
                    score, cell = evaluate_params(params)
                    
                    if score > best_score:
                        best_score = score
                        best_cell = cell.tolist()
                        best_params = params
    
    # Stage 6: Try known good configurations
    test_configs = [
        (0.0, 0.7, 0.0, 0.6, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.6, 0.2, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.1, 0.8, 0.0, 0.4, 0.0, 0.0, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.5, 0.3, 0.7, 0.0, 0.0, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.7, 0.0, 0.5, 0.2, 0.0, 0.5, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0),
        (0.0, 0.7, 0.0, 0.5, -0.2, 0.0, 0.5, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0),
        (0.0, 0.9, 0.0, 0.3, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.4, 0.0, 0.8, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.7, 0.1, 0.5, 0.0, 0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.6, 0.0, 0.6, 0.0, 0.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    
    for params in test_configs:
        score, cell = evaluate_params(params)
        if score > best_score:
            best_score = score
            best_cell = cell.tolist()
            best_params = params
    
    if best_cell is None:
        # Fallback: Standard rod
        best_cell = []
        for i in range(size):
            xi = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                yj = (j + 0.5) / size - 0.5
                if xi*xi + yj*yj < 0.2**2:
                    row.append(1)
                else:
                    row.append(0)
            best_cell.append(row)
            
    return best_cell
