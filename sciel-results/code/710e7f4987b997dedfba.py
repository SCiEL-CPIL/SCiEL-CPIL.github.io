import time
import numpy as np

def design(problem):
    """
    Optimizes a 32x32 photonic crystal unit cell to maximize the TM band gap 
    between the 4th and 5th bands using a coarse plane-wave surrogate model.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin for the evaluator
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # 1. Define the fundamental domain indices
    # The cell is 32x32. Symmetry: D4.
    # Fundamental domain: i in [0, 15], j in [i, 15]
    # We will store the values in a 32x32 array but only optimize the FD.
    
    def get_fd_indices():
        indices = []
        for i in range(16):
            for j in range(i, 16):
                indices.append((i, j))
        return indices
    
    fd_indices = get_fd_indices()
    
    def expand_cell(fd_values):
        """Expand the fundamental domain values to the full 32x32 symmetric cell."""
        cell = np.zeros((size, size), dtype=int)
        # Map fd_values to the upper-right triangle of the first quadrant
        # i is row, j is col
        for idx, (i, j) in enumerate(fd_indices):
            val = fd_values[idx]
            # Symmetry operations for D4 (Square)
            # 1. Identity: (i, j)
            cell[i, j] = val
            # 2. Transpose: (j, i)
            cell[j, i] = val
            # 3. 180 Rot: (size-1-i, size-1-j)
            cell[size-1-i, size-1-j] = val
            cell[size-1-j, size-1-i] = val
            # 4. Horizontal Flip: (size-1-i, j)
            cell[size-1-i, j] = val
            cell[size-1-i, i] = val
            # 5. Vertical Flip: (i, size-1-j)
            cell[i, size-1-j] = val
            cell[j, size-1-i] = val
            # 6. 90 Rot: (j, size-1-i)
            cell[j, size-1-i] = val
            cell[size-1-i, j] = val # Already covered by H flip? 
            # Let's just ensure all 8 symmetric positions are set.
            # Positions for (i,j) in Q1 (i<=j):
            # (i, j), (j, i), (size-1-i, size-1-j), (size-1-j, size-1-i),
            # (size-1-i, j), (j, size-1-i), (i, size-1-j), (size-1-j, i)
            
        return cell

    def compute_eps_inv_coeffs(cell):
        """
        Compute Fourier coefficients of epsilon^{-1} for a set of G vectors.
        G vectors are in units of 2*pi/a.
        """
        # Use a small basis, e.g., G in [-3, -2, -1, 0, 1, 2, 3]
        G_vals = np.arange(-3, 4)
        Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
        G_flat = np.column_stack([Gx.ravel(), Gy.ravel()])
        
        # epsilon(r) is piecewise constant on pixels
        # eps_inv(r) = 1/eps(r)
        eps_inv = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        
        # Compute FFT of eps_inv
        # The cell is 32x32 pixels.
        # G corresponds to frequency index k in FFT.
        # G = (nx, ny) where nx, ny are integers.
        # FFT index mapping:
        # Standard FFT gives frequencies 0..N-1.
        # We need negative frequencies too.
        
        fft_eps_inv = np.fft.fft2(eps_inv)
        
        # Normalize by N^2
        fft_eps_inv /= (size * size)
        
        # Extract coefficients for our G vectors
        coeffs = {}
        for gx, gy in G_flat:
            # Map integer G to FFT index
            # G=0 -> idx 0
            # G=1 -> idx 1
            # G=-1 -> idx size-1
            ix = int(gx) % size
            iy = int(gy) % size
            coeffs[(int(gx), int(gy))] = fft_eps_inv[ix, iy]
            
        return coeffs

    def solve_bands(cell, coeffs, kx, ky, num_bands=6):
        """
        Solve the eigenvalue problem for TM bands at wave vector (kx, ky).
        k is in units of 2*pi/a.
        """
        # Basis size: 7x7 = 49 vectors
        G_vals = np.arange(-3, 4)
        Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
        G_flat = np.column_stack([Gx.ravel(), Gy.ravel()])
        
        N = len(G_flat)
        A = np.zeros((N, N), dtype=complex)
        
        # Precompute (k+G) dot (k+G') terms
        k_vec = np.array([kx, ky])
        
        for i in range(N):
            Gi = G_flat[i]
            kpi = k_vec + Gi
            for j in range(N):
                Gj = G_flat[j]
                kpj = k_vec + Gj
                G_diff = Gi - Gj
                coeff = coeffs.get(tuple(G_diff), 0.0)
                if coeff != 0:
                    A[i, j] = (kpi[0]*kpj[0] + kpi[1]*kpj[1]) / coeff
        
        # Hermitian matrix
        A = (A + A.conj().T) / 2
        
        # Eigenvalues
        eigvals = np.linalg.eigvalsh(A)
        # Sort
        eigvals = np.sort(eigvals)
        return eigvals[:num_bands]

    # K-points in the irreducible Brillouin Zone
    # Gamma: (0,0)
    # X: (0.5, 0)
    # M: (0.5, 0.5)
    # Midpoints for better sampling
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.5, 0.25),
        (0.125, 0.125),
        (0.375, 0.125)
    ]

    def evaluate_score(cell):
        """
        Calculate the gap-midgap ratio for bands 4 and 5.
        """
        coeffs = compute_eps_inv_coeffs(cell)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            try:
                eigvals = solve_bands(cell, coeffs, kx, ky, num_bands=6)
                if len(eigvals) < 5:
                    continue
                w4 = eigvals[3] # 4th band (index 3)
                w5 = eigvals[4] # 5th band (index 4)
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
            except Exception:
                continue
        
        if min_omega_5 == float('inf') or max_omega_4 == 0.0:
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # Initialize with a Cross shape (often better than circle for higher bands)
    # Cross: Dielectric in the center, extending along axes.
    # Let's start with a "plus" sign dielectric.
    init_cell = np.zeros((size, size), dtype=int)
    center = size // 2
    arm_width = 6 # Half-width of the arm
    arm_length = 10 # Half-length of the arm
    
    for i in range(size):
        for j in range(size):
            # Distance from center
            di = i - center + 0.5
            dj = j - center + 0.5
            # Cross shape: |x| < w and |y| < L OR |y| < w and |x| < L
            # Adjusted for pixel centers
            if (abs(di) < arm_width and abs(dj) < arm_length) or (abs(dj) < arm_width and abs(di) < arm_length):
                init_cell[i, j] = 1
                
    # Extract initial FD values
    init_fd = []
    for i, j in fd_indices:
        init_fd.append(init_cell[i, j])
    init_fd = np.array(init_fd)
    
    best_score = evaluate_score(expand_cell(init_fd))
    best_fd = init_fd.copy()
    
    # Local Search
    # Randomly flip pixels in the FD
    np.random.seed(42)
    
    current_fd = best_fd.copy()
    current_score = best_score
    
    # Simulated Annealing or Hill Climbing
    # Given time constraints, simple Hill Climbing with random restarts
    max_iters = 2000
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iters:
        iter_count += 1
        
        # Pick a random pixel in FD to flip
        idx = np.random.randint(len(fd_indices))
        new_fd = current_fd.copy()
        new_fd[idx] = 1 - new_fd[idx]
        
        new_cell = expand_cell(new_fd)
        new_score = evaluate_score(new_cell)
        
        if new_score > current_score:
            current_fd = new_fd
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_fd = current_fd.copy()
        else:
            # With some probability, accept worse moves to escape local minima
            # Simple: always reject for now, or random restart
            pass
            
        # Check time
        if time.time() > deadline - 5.0:
            break

    # Finalize best cell
    final_cell = expand_cell(best_fd)
    
    # Convert to nested list
    return final_cell.tolist()
