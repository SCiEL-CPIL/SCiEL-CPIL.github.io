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
    
    # Coordinates for the 32x32 grid, centered at 0
    idx = np.arange(size)
    x = (idx + 0.5) / size - 0.5
    y = (idx + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Precompute the basis functions for the harmonics
    # Mode (2,0) and (0,2) combined for C4v symmetry
    basis_20 = np.cos(4 * np.pi * X) + np.cos(4 * np.pi * Y)
    
    # Mode (2,2) and (2,-2) combined for C4v symmetry
    basis_22 = np.cos(4 * np.pi * (X + Y)) + np.cos(4 * np.pi * (X - Y))
    
    # Search parameters - refined grid for the specific modes requested
    # We optimize amplitudes A20, A22 and a DC offset (fill factor control)
    A20_vals = np.linspace(-0.5, 0.5, 11)
    A22_vals = np.linspace(-0.5, 0.5, 11)
    offset_vals = np.linspace(0.3, 0.7, 9)
    
    best_score = -np.inf
    best_cell = None
    
    # Fast PWE Surrogate Parameters
    # Use a larger number of plane waves for better accuracy
    M_PWE = 4  # Indices -4 to 4, so 9x9 = 81 PWs
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
        (0.3, 0.2)
    ]
    
    def calculate_surrogate_gap(cell):
        """
        Calculates an approximate gap using a small PWE expansion.
        Returns the score (gap-midgap ratio).
        """
        # 1. Compute Fourier coefficients of the pixel map
        # epsilon(G) = 1/N^2 sum epsilon(r) exp(-i G.r)
        # We assume epsilon is 0 or 1 (air/dielectric) in the pixel map, 
        # but the problem says eps_low=1, eps_high=12.
        # The evaluator uses exact Fourier coefficients of the pixel map.
        # The pixel map is 0/1. 
        # Usually, the PWE matrix for TM mode involves 1/epsilon.
        # Let's check the standard TM PWE formulation.
        # For TM: sum_j (k+Gi) x (k+Gj) * (1/eps(Gi-Gj)) * Hj = w^2 Hi
        # Wait, the current program uses eps(Gi-Gj) in the numerator?
        # Let's re-read the task: "exact-coefficient plane-wave method".
        # Standard TM PWE:
        # The equation is: sum_j (k+Gi) dot (k+Gj) * [1/eps(Gi-Gj)] * c_j = omega^2 c_i
        # Where eps(G) is the Fourier transform of 1/epsilon(r).
        # However, if epsilon(r) is piecewise constant (0 or 1 in the binary map, mapped to eps_low/eps_high),
        # then 1/epsilon(r) is also piecewise constant.
        
        # The current code computes `fft_vals` of the 0/1 cell.
        # If the cell is 1 for dielectric (eps=12) and 0 for air (eps=1):
        # eps(r) = 12 * cell(r) + 1 * (1 - cell(r)) = 1 + 11 * cell(r)
        # We need the Fourier coefficients of 1/eps(r).
        # Since eps(r) takes only two values, 1/eps(r) is:
        # 1/1 = 1 where cell=0
        # 1/12 where cell=1
        # Let inv_eps_cell be a 32x32 array of 1/eps(r).
        # Then we FFT inv_eps_cell to get 1/eps(G).
        
        # The current code FFTs the 0/1 cell directly. This is incorrect for TM if it assumes eps(G) is the transform of eps(r).
        # For TE, it's eps(G). For TM, it's 1/eps(G).
        # The task specifies TM.
        # Let's correct the surrogate to use 1/eps(G).
        
        # Construct inv_eps array
        # cell is 0 or 1.
        # eps_val = 12 if cell==1 else 1
        # inv_eps_val = 1/12 if cell==1 else 1
        inv_eps_arr = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        
        fft_inv_eps = np.fft.fft2(inv_eps_arr)
        fft_inv_eps /= (size * size)
        
        def get_inv_eps(Gx_val, Gy_val):
            ix = int(Gx_val) % size
            iy = int(Gy_val) % size
            return fft_inv_eps[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # Build PWE matrix
            # A_{ij} = (k+Gi) . (k+Gj) * (1/eps(Gi - Gj))
            
            # Precompute (k+G) vectors
            kGx = kx + Gx_flat
            kGy = ky + Gy_flat
            
            # Dot products
            D = np.outer(kGx, kGx) + np.outer(kGy, kGy)
            
            # Compute 1/eps matrix E_inv_{ij} = 1/eps(Gi - Gj)
            E_inv = np.zeros((N_PW, N_PW), dtype=complex)
            for i in range(N_PW):
                for j in range(N_PW):
                    dx = Gx_flat[i] - Gx_flat[j]
                    dy = Gy_flat[i] - Gy_flat[j]
                    E_inv[i, j] = get_inv_eps(dx, dy)
            
            # Matrix A = D * E_inv
            A = D * E_inv
            
            # A should be Hermitian
            A = (A + A.conj().T) / 2.0
            A = np.real(A)
            
            # Solve eigenvalues
            try:
                eigvals = np.linalg.eigvalsh(A)
            except np.linalg.LinAlgError:
                return -np.inf
                
            # Sort eigenvalues
            eigvals = np.sort(eigvals)
            
            # We need bands 4 and 5 (1-indexed), so indices 3 and 4
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
    for A20 in A20_vals:
        for A22 in A22_vals:
            for offset in offset_vals:
                # Generate continuous field
                field = offset + A20 * basis_20 + A22 * basis_22
                
                # Threshold to binary
                cell_raw = (field > 0.5).astype(int)
                
                # Enforce C4v symmetry
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
                
                # Average and re-threshold
                avg_cell = np.mean(sym_cells, axis=0)
                cell_sym = (avg_cell >= 0.5).astype(int)
                
                # Quick check: if cell is empty or full, skip
                fill = np.sum(cell_sym) / (size * size)
                if fill < 0.15 or fill > 0.85:
                    continue
                
                score = calculate_surrogate_gap(cell_sym)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell_sym.tolist()
    
    if best_cell is None:
        # Fallback: Standard rod
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
