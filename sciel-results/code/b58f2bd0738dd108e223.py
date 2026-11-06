import numpy as np

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell using a superposition of 
    low-order Fourier harmonics (2,0), (0,2), and (2,2) modes that directly 
    couple to the 4th and 5th TM bands. Optimizes amplitudes and phases 
    to maximize the spectral separation between bands 4 and 5.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Coordinates for the 32x32 grid, centered at 0
    idx = np.arange(size)
    x = (idx + 0.5) / size - 0.5
    y = (idx + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Fast PWE Surrogate Parameters - use larger basis for better accuracy
    M_PWE = 5  # Indices -5 to 5, so 11x11 = 121 PWs
    G_indices = np.arange(-M_PWE, M_PWE + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    N_PW = len(Gx_flat)
    
    # More comprehensive k-points in the irreducible BZ
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.35, 0.15),
        (0.4, 0.2),
        (0.3, 0.3),
        (0.1, 0.0),
        (0.45, 0.05),
        (0.2, 0.1),
        (0.4, 0.4),
    ]
    
    def calculate_surrogate_gap(cell):
        """
        Calculates an approximate gap using a small PWE expansion.
        Returns the score (gap-midgap ratio).
        """
        # Compute Fourier coefficients of the pixel map
        fft_vals = np.fft.fft2(cell)
        fft_vals /= (size * size)
        
        # Precompute epsilon matrix for all pairs
        # E[i,j] = eps(G_i - G_j)
        E = np.zeros((N_PW, N_PW), dtype=complex)
        for i in range(N_PW):
            for j in range(N_PW):
                dx = Gx_flat[i] - Gx_flat[j]
                dy = Gy_flat[i] - Gy_flat[j]
                ix = int(dx) % size
                iy = int(dy) % size
                E[i, j] = fft_vals[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            kGx = kx + Gx_flat
            kGy = ky + Gy_flat
            
            # D matrix: (k+G)^2 for each plane wave
            D = (kGx ** 2 + kGy ** 2).astype(float)
            
            # A = D * E (element-wise product), then symmetrize
            A = np.outer(D, np.ones(N_PW)) * E
            # Actually, for PWE: A[i,j] = (k+G_i)^2 * eps(G_i - G_j)
            # This is not symmetric in general, so we need to handle carefully
            # The eigenvalue problem is: sum_j eps(G_i - G_j) * (k+G_j)^2 * c_j = omega^2 * c_i
            # So A[i,j] = eps(G_i - G_j) * (k+G_j)^2
            A = np.zeros((N_PW, N_PW), dtype=complex)
            for j in range(N_PW):
                A[:, j] = E[:, j] * (kGx[j] ** 2 + kGy[j] ** 2)
            
            # Make it Hermitian for eigvalsh
            A = (A + A.conj().T) / 2.0
            A_real = np.real(A)
            
            try:
                eigvals = np.linalg.eigvalsh(A_real)
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

    def generate_cell(A20, A22, offset, phase20=0.0, phase22=0.0):
        """Generate a binary cell from Fourier parameters with C4v symmetry."""
        # Build field with C4v-symmetric harmonics
        field = np.zeros((size, size), dtype=float)
        field += offset
        
        if A20 != 0:
            # (2,0) and (0,2) modes with C4v symmetry
            field += A20 * (np.cos(4 * np.pi * X + phase20) + np.cos(4 * np.pi * Y + phase20))
        
        if A22 != 0:
            # (2,2) and (2,-2) modes with C4v symmetry
            field += A22 * (np.cos(4 * np.pi * (X + Y) + phase22) + np.cos(4 * np.pi * (X - Y) + phase22))
        
        # Threshold to binary
        cell_raw = (field > 0.5).astype(int)
        
        # Enforce C4v symmetry strictly by averaging over all 8 operations
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

    best_score = -np.inf
    best_cell = None
    best_params = None
    
    # Stage 1: Coarse grid search over amplitudes and offset
    A20_coarse = np.linspace(0.0, 1.5, 10)
    A22_coarse = np.linspace(0.0, 1.2, 8)
    offset_coarse = np.linspace(0.25, 0.75, 11)
    
    for A20 in A20_coarse:
        for A22 in A22_coarse:
            for offset in offset_coarse:
                cell = generate_cell(A20, A22, offset)
                
                fill = np.sum(cell) / (size * size)
                if fill < 0.15 or fill > 0.85:
                    continue
                
                score = calculate_surrogate_gap(cell)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    best_params = (A20, A22, offset, 0.0, 0.0)
    
    # Stage 2: Fine-tune around best candidate with phases
    if best_cell is not None and best_params is not None:
        A20_best, A22_best, offset_best, p20_best, p22_best = best_params
        
        # Fine grid around best values
        A20_fine = np.linspace(max(0.0, A20_best - 0.2), min(1.8, A20_best + 0.2), 7)
        A22_fine = np.linspace(max(0.0, A22_best - 0.2), min(1.5, A22_best + 0.2), 7)
        offset_fine = np.linspace(max(0.2, offset_best - 0.1), min(0.8, offset_best + 0.1), 7)
        phase20_fine = np.linspace(-0.4, 0.4, 5)
        phase22_fine = np.linspace(-0.4, 0.4, 5)
        
        for A20 in A20_fine:
            for A22 in A22_fine:
                for offset in offset_fine:
                    for p20 in phase20_fine:
                        for p22 in phase22_fine:
                            cell = generate_cell(A20, A22, offset, p20, p22)
                            
                            fill = np.sum(cell) / (size * size)
                            if fill < 0.15 or fill > 0.85:
                                continue
                            
                            score = calculate_surrogate_gap(cell)
                            
                            if score > best_score:
                                best_score = score
                                best_cell = cell.tolist()
                                best_params = (A20, A22, offset, p20, p22)
    
    # Stage 3: Try additional targeted configurations
    test_configs = [
        (0.8, 0.5, 0.5, 0.0, 0.0),
        (0.6, 0.7, 0.45, 0.0, 0.0),
        (0.9, 0.3, 0.55, 0.0, 0.0),
        (0.7, 0.6, 0.5, 0.2, 0.0),
        (0.7, 0.6, 0.5, -0.2, 0.0),
        (0.5, 0.8, 0.4, 0.0, 0.3),
        (0.5, 0.8, 0.4, 0.0, -0.3),
        (1.0, 0.4, 0.6, 0.0, 0.0),
        (0.4, 1.0, 0.4, 0.0, 0.0),
        (1.2, 0.6, 0.5, 0.1, 0.1),
        (1.1, 0.8, 0.45, -0.1, 0.2),
        (0.9, 0.9, 0.5, 0.0, 0.0),
        (1.3, 0.3, 0.55, 0.15, -0.1),
        (0.7, 1.1, 0.4, 0.0, 0.25),
    ]
    
    for A20, A22, offset, p20, p22 in test_configs:
        cell = generate_cell(A20, A22, offset, p20, p22)
        
        fill = np.sum(cell) / (size * size)
        if fill < 0.15 or fill > 0.85:
            continue
        
        score = calculate_surrogate_gap(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell.tolist()
            best_params = (A20, A22, offset, p20, p22)
    
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
