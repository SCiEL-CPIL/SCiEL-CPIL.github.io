import numpy as np

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell by optimizing a parametric 
    dielectric distribution defined by a superposition of low-order Fourier 
    harmonics (2,0), (0,2), and (2,2) modes.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Coordinates for the 32x32 grid
    idx = np.arange(size)
    x = (idx + 0.5) / size - 0.5
    y = (idx + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Basis functions for C4v symmetry
    basis_20 = np.cos(4 * np.pi * X) + np.cos(4 * np.pi * Y)
    basis_22 = np.cos(4 * np.pi * (X + Y)) + np.cos(4 * np.pi * (X - Y))
    
    # Surrogate PWE Parameters
    M_PWE = 4 # 9x9 = 81 PWs
    G_indices = np.arange(-M_PWE, M_PWE + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    N_PW = len(Gx_flat)
    
    # Precompute differences for E matrix vectorization
    # dx[i,j] = Gx_flat[i] - Gx_flat[j]
    dx = Gx_flat[:, None] - Gx_flat[None, :]
    dy = Gy_flat[:, None] - Gy_flat[None, :]
    # Indices into FFT array
    ix = dx % size
    iy = dy % size
    
    # K-points in the irreducible Brillouin zone
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.25),
        (0.375, 0.125)
    ]

    def calculate_surrogate_gap(cell):
        """
        Calculates an approximate gap using a small PWE expansion.
        Returns the score (gap-midgap ratio).
        """
        # Compute Fourier coefficients of 1/epsilon
        eps_inv = np.where(cell == 0, 1.0, 1.0 / eps_high)
        fft_vals = np.fft.fft2(eps_inv)
        fft_vals /= (size * size)
        
        # E matrix is constant for all k-points if we just index it
        # E[i,j] = fft_vals[ix[i,j], iy[i,j]]
        E = fft_vals[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            kGx = kx + Gx_flat
            kGy = ky + Gy_flat
            
            D = np.outer(kGx, kGx) + np.outer(kGy, kGy)
            
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

    # Grid Search
    A20_vals = np.linspace(-0.5, 0.5, 15)
    A22_vals = np.linspace(-0.5, 0.5, 15)
    offset_vals = np.linspace(0.4, 0.6, 15)
    
    best_score = -np.inf
    best_cell = None
    
    for A20 in A20_vals:
        for A22 in A22_vals:
            for offset in offset_vals:
                field = offset + A20 * basis_20 + A22 * basis_22
                
                cell_raw = (field > 0.5).astype(int)
                
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
                
                fill = np.sum(cell_sym) / (size * size)
                if fill < 0.2 or fill > 0.8:
                    continue
                
                score = calculate_surrogate_gap(cell_sym)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell_sym.tolist()
    
    if best_cell is None:
        best_cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < 0.2**2:
                    row.append(1)
                else:
                    row.append(0)
            best_cell.append(row)
            
    return best_cell
