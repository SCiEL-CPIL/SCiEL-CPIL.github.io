import numpy as np

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell using a superposition of
    (2,0), (0,2), and (2,2) Fourier harmonics with optimized amplitudes
    and phases to maximize the gap between TM bands 4 and 5.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Coordinates for the 32x32 grid
    idx = np.arange(size)
    x = (idx + 0.5) / size - 0.5
    y = (idx + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Basis functions for the three retained harmonics
    # (2,0) mode: cos(4πX) and sin(4πX)
    cos_20_x = np.cos(4 * np.pi * X)
    sin_20_x = np.sin(4 * np.pi * X)
    
    # (0,2) mode: cos(4πY) and sin(4πY)
    cos_02_y = np.cos(4 * np.pi * Y)
    sin_02_y = np.sin(4 * np.pi * Y)
    
    # (2,2) mode: cos(4π(X+Y)), sin(4π(X+Y)), cos(4π(X-Y)), sin(4π(X-Y))
    cos_22_plus = np.cos(4 * np.pi * (X + Y))
    sin_22_plus = np.sin(4 * np.pi * (X + Y))
    cos_22_minus = np.cos(4 * np.pi * (X - Y))
    sin_22_minus = np.sin(4 * np.pi * (X - Y))
    
    # PWE surrogate setup
    M_PWE = 3  # -3 to 3, so 7x7 = 49 PWs
    G_indices = np.arange(-M_PWE, M_PWE + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    N_PW = len(Gx_flat)
    
    # Key k-points in the irreducible BZ (fractional coordinates)
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.4, 0.1),
        (0.3, 0.3),
        (0.1, 0.1)
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
            
            E = np.zeros((N_PW, N_PW), dtype=complex)
            for i in range(N_PW):
                for j in range(N_PW):
                    dx = Gx_flat[i] - Gx_flat[j]
                    dy = Gy_flat[i] - Gy_flat[j]
                    E[i, j] = get_eps(dx, dy)
            
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
    
    def generate_cell(A20x, ph20x, A20y, ph20y, A22p, ph22p, A22m, ph22m, offset):
        """
        Generate a binary cell from the parameterized Fourier expansion.
        A20x, ph20x: amplitude and phase for (2,0) x-mode
        A20y, ph20y: amplitude and phase for (0,2) y-mode
        A22p, ph22p: amplitude and phase for (2,2) (X+Y) mode
        A22m, ph22m: amplitude and phase for (2,2) (X-Y) mode
        offset: DC offset to control fill fraction
        """
        field = offset
        # (2,0) mode
        field += A20x * (np.cos(ph20x) * cos_20_x + np.sin(ph20x) * sin_20_x)
        # (0,2) mode
        field += A20y * (np.cos(ph20y) * cos_02_y + np.sin(ph20y) * sin_02_y)
        # (2,2) (X+Y) mode
        field += A22p * (np.cos(ph22p) * cos_22_plus + np.sin(ph22p) * sin_22_plus)
        # (2,2) (X-Y) mode
        field += A22m * (np.cos(ph22m) * cos_22_minus + np.sin(ph22m) * sin_22_minus)
        
        # Threshold to binary
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
    
    # Parameter ranges for optimization
    # Amplitudes
    A20x_vals = np.linspace(0.0, 1.5, 5)
    A20y_vals = np.linspace(0.0, 1.5, 5)
    A22p_vals = np.linspace(0.0, 1.0, 4)
    A22m_vals = np.linspace(0.0, 1.0, 4)
    
    # Phases (for C4v symmetry, we restrict phases)
    ph20x_vals = np.array([0.0, np.pi/2, np.pi, -np.pi/2])
    ph20y_vals = np.array([0.0, np.pi/2, np.pi, -np.pi/2])
    ph22p_vals = np.array([0.0, np.pi/2, np.pi, -np.pi/2])
    ph22m_vals = np.array([0.0, np.pi/2, np.pi, -np.pi/2])
    
    # Offset values to control fill fraction
    offset_vals = np.linspace(0.3, 0.7, 5)
    
    best_score = -np.inf
    best_cell = None
    best_params = None
    
    # Coarse grid search
    for A20x in A20x_vals:
        for A20y in A20y_vals:
            for A22p in A22p_vals:
                for A22m in A22m_vals:
                    for ph20x in ph20x_vals:
                        for ph20y in ph20y_vals:
                            for ph22p in ph22p_vals:
                                for ph22m in ph22m_vals:
                                    for offset in offset_vals:
                                        cell = generate_cell(A20x, ph20x, A20y, ph20y, 
                                                              A22p, ph22p, A22m, ph22m, offset)
                                        
                                        fill = np.sum(cell) / (size * size)
                                        if fill < 0.1 or fill > 0.9:
                                            continue
                                        
                                        score = calculate_surrogate_gap(cell)
                                        
                                        if score > best_score:
                                            best_score = score
                                            best_cell = cell
                                            best_params = (A20x, ph20x, A20y, ph20y, 
                                                          A22p, ph22p, A22m, ph22m, offset)
    
    # If we found a good solution, do a local refinement
    if best_cell is not None and best_score > -np.inf:
        A20x0, ph20x0, A20y0, ph20y0, A22p0, ph22p0, A22m0, ph22m0, offset0 = best_params
        
        # Local search around best parameters
        local_A20x_vals = np.linspace(A20x0 - 0.2, A20x0 + 0.2, 3)
        local_A20y_vals = np.linspace(A20y0 - 0.2, A20y0 + 0.2, 3)
        local_A22p_vals = np.linspace(A22p0 - 0.2, A22p0 + 0.2, 3)
        local_A22m_vals = np.linspace(A22m0 - 0.2, A22m0 + 0.2, 3)
        local_offset_vals = np.linspace(offset0 - 0.1, offset0 + 0.1, 5)
        
        for A20x in local_A20x_vals:
            for A20y in local_A20y_vals:
                for A22p in local_A22p_vals:
                    for A22m in local_A22m_vals:
                        for offset in local_offset_vals:
                            cell = generate_cell(A20x, ph20x0, A20y, ph20y0, 
                                                A22p, ph22p0, A22m, ph22m0, offset)
                            
                            fill = np.sum(cell) / (size * size)
                            if fill < 0.1 or fill > 0.9:
                                continue
                            
                            score = calculate_surrogate_gap(cell)
                            
                            if score > best_score:
                                best_score = score
                                best_cell = cell
                                best_params = (A20x, ph20x0, A20y, ph20y0, 
                                              A22p, ph22p0, A22m, ph22m0, offset)
    
    if best_cell is None:
        # Fallback: Simple rod pattern
        best_cell = []
        for i in range(size):
            x_val = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                y_val = (j + 0.5) / size - 0.5
                if x_val*x_val + y_val*y_val < 0.2**2:
                    row.append(1)
                else:
                    row.append(0)
            best_cell.append(row)
            
    return best_cell.tolist()
