import numpy as np

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell using a superposition of 
    low-order Fourier harmonics that directly couple to the 4th and 5th TM bands.
    Optimizes amplitudes and phases to maximize the spectral separation.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Coordinates for the 32x32 grid, centered at 0
    idx = np.arange(size)
    x = (idx + 0.5) / size - 0.5
    y = (idx + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Fast PWE Surrogate Parameters
    M_PWE = 5  # Indices -5 to 5, so 11x11 = 121 PWs
    G_indices = np.arange(-M_PWE, M_PWE + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    N_PW = len(Gx_flat)
    
    # Precompute difference indices for epsilon lookup
    dx_all = Gx_flat[:, None] - Gx_flat[None, :]
    dy_all = Gy_flat[:, None] - Gy_flat[None, :]
    
    # Key k-points in the irreducible BZ (fractional coordinates)
    # The irreducible BZ is 0 <= ky <= kx, 0 <= kx <= 0.5
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.35, 0.15),
        (0.4, 0.2),
        (0.3, 0.3),
        (0.45, 0.1),
        (0.4, 0.0),
        (0.35, 0.35),
        (0.2, 0.1),
        (0.45, 0.25),
        (0.3, 0.15),
        (0.5, 0.2),
    ]
    
    def calculate_surrogate_gap(cell):
        """
        Calculates an approximate gap using a small PWE expansion.
        Returns the score (gap-midgap ratio).
        """
        # Compute Fourier coefficients of the pixel map
        fft_vals = np.fft.fft2(cell.astype(np.float64))
        fft_vals /= (size * size)
        
        # Precompute epsilon matrix lookup table for all possible differences
        # dx ranges from -2*M_PWE to 2*M_PWE, same for dy
        # Map (dx, dy) -> fft_vals[dx % size, dy % size]
        dx_lookup = np.mod(dx_all, size)
        dy_lookup = np.mod(dy_all, size)
        E = fft_vals[dx_lookup, dy_lookup].astype(complex)
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            kGx = kx + Gx_flat
            kGy = ky + Gy_flat
            
            D = kGx * kGx + kGy * kGy
            
            A = D[:, None] * E
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
                    if w4 > max_omega_4:
                        max_omega_4 = w4
                    if w5 < min_omega_5:
                        min_omega_5 = w5
        
        if min_omega_5 == np.inf or max_omega_4 == 0.0:
            return -np.inf
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        score = gap / midgap
        return score

    def generate_cell(A10, A20, A22, A30, offset, phase10=0.0, phase20=0.0, phase22=0.0, phase30=0.0):
        """Generate a binary cell from Fourier parameters with C4v symmetry."""
        field = np.full_like(X, offset, dtype=np.float64)
        
        # (1,0)+(0,1) mode: cos(2πX)+cos(2πY)
        if abs(A10) > 1e-10:
            field += A10 * (np.cos(2 * np.pi * X + phase10) + np.cos(2 * np.pi * Y + phase10))
        
        # (2,0)+(0,2) mode: cos(4πX)+cos(4πY)
        if abs(A20) > 1e-10:
            field += A20 * (np.cos(4 * np.pi * X + phase20) + np.cos(4 * np.pi * Y + phase20))
        
        # (2,2)+(2,-2) mode: cos(4π(X+Y))+cos(4π(X-Y))
        if abs(A22) > 1e-10:
            field += A22 * (np.cos(4 * np.pi * (X + Y) + phase22) + np.cos(4 * np.pi * (X - Y) + phase22))
        
        # (3,0)+(0,3) mode: cos(6πX)+cos(6πY)
        if abs(A30) > 1e-10:
            field += A30 * (np.cos(6 * np.pi * X + phase30) + np.cos(6 * np.pi * Y + phase30))
        
        cell_raw = (field > 0.5).astype(np.float64)
        
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

    best_score = -np.inf
    best_cell = None
    
    # Stage 1: Coarse grid search over amplitudes and offset
    # Parameters: A10, A20, A22, A30, offset
    A10_coarse = [0.0, 0.3, 0.6]
    A20_coarse = np.linspace(0.0, 1.2, 6)
    A22_coarse = np.linspace(0.0, 0.8, 5)
    A30_coarse = [0.0, 0.3, 0.6]
    offset_coarse = np.linspace(0.30, 0.70, 7)
    
    for A10 in A10_coarse:
        for A20 in A20_coarse:
            for A22 in A22_coarse:
                for A30 in A30_coarse:
                    for offset in offset_coarse:
                        cell = generate_cell(A10, A20, A22, A30, offset)
                        
                        fill = np.sum(cell) / (size * size)
                        if fill < 0.15 or fill > 0.85:
                            continue
                        
                        score = calculate_surrogate_gap(cell)
                        
                        if score > best_score:
                            best_score = score
                            best_cell = cell.tolist()
                            best_params = (A10, A20, A22, A30, offset, 0.0, 0.0, 0.0, 0.0)
    
    # Stage 2: Fine-tune around best candidate with coordinate descent
    if best_cell is not None:
        A10_best, A20_best, A22_best, A30_best, offset_best, p10_best, p20_best, p22_best, p30_best = best_params
        
        # Coordinate descent: optimize each parameter in turn
        for iteration in range(3):
            # Optimize offset
            for offset in np.linspace(max(0.25, offset_best - 0.1), min(0.75, offset_best + 0.1), 7):
                cell = generate_cell(A10_best, A20_best, A22_best, A30_best, offset, p10_best, p20_best, p22_best, p30_best)
                fill = np.sum(cell) / (size * size)
                if fill < 0.15 or fill > 0.85:
                    continue
                score = calculate_surrogate_gap(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    offset_best = offset
            
            # Optimize A10
            for A10 in np.linspace(max(0.0, A10_best - 0.2), min(1.0, A10_best + 0.2), 5):
                cell = generate_cell(A10, A20_best, A22_best, A30_best, offset_best, p10_best, p20_best, p22_best, p30_best)
                fill = np.sum(cell) / (size * size)
                if fill < 0.15 or fill > 0.85:
                    continue
                score = calculate_surrogate_gap(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    A10_best = A10
            
            # Optimize A20
            for A20 in np.linspace(max(0.0, A20_best - 0.2), min(1.5, A20_best + 0.2), 5):
                cell = generate_cell(A10_best, A20, A22_best, A30_best, offset_best, p10_best, p20_best, p22_best, p30_best)
                fill = np.sum(cell) / (size * size)
                if fill < 0.15 or fill > 0.85:
                    continue
                score = calculate_surrogate_gap(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    A20_best = A20
            
            # Optimize A22
            for A22 in np.linspace(max(0.0, A22_best - 0.2), min(1.2, A22_best + 0.2), 5):
                cell = generate_cell(A10_best, A20_best, A22, A30_best, offset_best, p10_best, p20_best, p22_best, p30_best)
                fill = np.sum(cell) / (size * size)
                if fill < 0.15 or fill > 0.85:
                    continue
                score = calculate_surrogate_gap(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    A22_best = A22
            
            # Optimize A30
            for A30 in np.linspace(max(0.0, A30_best - 0.2), min(1.0, A30_best + 0.2), 5):
                cell = generate_cell(A10_best, A20_best, A22_best, A30, offset_best, p10_best, p20_best, p22_best, p30_best)
                fill = np.sum(cell) / (size * size)
                if fill < 0.15 or fill > 0.85:
                    continue
                score = calculate_surrogate_gap(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    A30_best = A30
            
            # Optimize phases
            for p10 in np.linspace(max(-0.8, p10_best - 0.3), min(0.8, p10_best + 0.3), 3):
                cell = generate_cell(A10_best, A20_best, A22_best, A30_best, offset_best, p10, p20_best, p22_best, p30_best)
                fill = np.sum(cell) / (size * size)
                if fill < 0.15 or fill > 0.85:
                    continue
                score = calculate_surrogate_gap(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    p10_best = p10
            
            for p20 in np.linspace(max(-0.8, p20_best - 0.3), min(0.8, p20_best + 0.3), 3):
                cell = generate_cell(A10_best, A20_best, A22_best, A30_best, offset_best, p10_best, p20, p22_best, p30_best)
                fill = np.sum(cell) / (size * size)
                if fill < 0.15 or fill > 0.85:
                    continue
                score = calculate_surrogate_gap(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    p20_best = p20
            
            for p22 in np.linspace(max(-0.8, p22_best - 0.3), min(0.8, p22_best + 0.3), 3):
                cell = generate_cell(A10_best, A20_best, A22_best, A30_best, offset_best, p10_best, p20_best, p22, p30_best)
                fill = np.sum(cell) / (size * size)
                if fill < 0.15 or fill > 0.85:
                    continue
                score = calculate_surrogate_gap(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    p22_best = p22
            
            for p30 in np.linspace(max(-0.8, p30_best - 0.3), min(0.8, p30_best + 0.3), 3):
                cell = generate_cell(A10_best, A20_best, A22_best, A30_best, offset_best, p10_best, p20_best, p22_best, p30)
                fill = np.sum(cell) / (size * size)
                if fill < 0.15 or fill > 0.85:
                    continue
                score = calculate_surrogate_gap(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.tolist()
                    p30_best = p30
    
    # Stage 3: Try targeted configurations known to be good for TM gaps
    if best_cell is not None:
        test_configs = [
            (0.0, 0.8, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.6, 0.7, 0.0, 0.45, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.9, 0.3, 0.0, 0.55, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.7, 0.6, 0.0, 0.5, 0.0, 0.2, 0.0, 0.0),
            (0.0, 0.7, 0.6, 0.0, 0.5, 0.0, -0.2, 0.0, 0.0),
            (0.0, 0.5, 0.8, 0.0, 0.4, 0.0, 0.0, 0.3, 0.0),
            (0.0, 0.5, 0.8, 0.0, 0.4, 0.0, 0.0, -0.3, 0.0),
            (0.0, 1.0, 0.4, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.4, 1.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0),
            (0.3, 0.7, 0.5, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0),
            (0.5, 0.6, 0.6, 0.5, 0.45, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.8, 0.6, 0.0, 0.5, 0.0, 0.1, 0.1, 0.0),
            (0.0, 0.7, 0.7, 0.0, 0.48, 0.0, -0.1, -0.1, 0.0),
        ]
        
        for A10, A20, A22, A30, offset, p10, p20, p22, p30 in test_configs:
            cell = generate_cell(A10, A20, A22, A30, offset, p10, p20, p22, p30)
            
            fill = np.sum(cell) / (size * size)
            if fill < 0.15 or fill > 0.85:
                continue
            
            score = calculate_surrogate_gap(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell.tolist()
    
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
