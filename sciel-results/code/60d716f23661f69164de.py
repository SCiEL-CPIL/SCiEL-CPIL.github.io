import numpy as np
import time
import sys

def design(problem):
    """
    Optimizes a symmetric super-ellipse unit cell using a fast surrogate plane-wave solver.
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # Surrogate solver parameters
    # Use a smaller plane-wave grid for speed
    N_pw = 11  # 11x11 = 121 plane waves, fast enough for many evaluations
    
    def get_epsilon_map(R, n_exp):
        """Generate a symmetric 32x32 epsilon map for a super-ellipse."""
        # Super-ellipse equation: |x|^n + |y|^n = R^n
        # Centered at (0,0)
        x = np.linspace(-0.5, 0.5, size, endpoint=False) + 0.5/size
        y = np.linspace(-0.5, 0.5, size, endpoint=False) + 0.5/size
        X, Y = np.meshgrid(x, y)
        
        # Super-ellipse condition
        # To handle n_exp, we use the form: (|X|/R)^(2*n) + (|Y|/R)^(2*n) <= 1
        # This makes n=0.5 a square, n=1 a circle, n=2 a squarer shape
        # Let's use exponent m = 2 * n_exp
        m = 2.0 * n_exp
        dist = np.abs(X) ** m + np.abs(Y) ** m
        mask = dist <= (R ** m)
        
        # Ensure symmetry: The grid is centered, so it's symmetric by construction.
        # We need to return a list of lists of 0/1
        cell = mask.astype(int).tolist()
        return cell

    def compute_surrogate_gap(cell):
        """
        Fast TM band gap calculation using plane waves.
        Returns the gap-midgap ratio for bands 4 and 5.
        """
        eps = np.array(cell, dtype=float)
        # Inverse epsilon for TM mode
        # eps_inv = 1 / eps
        # But we need the average and Fourier coefficients
        # For a binary medium:
        # <1/eps> = f_air/1 + f_dielectric/eps_high
        # (1/eps)_G = integral (1/eps(x) exp(-iG.x)) d^2x
        
        f_dielectric = np.sum(eps) / (size * size)
        f_air = 1.0 - f_dielectric
        
        # Average inverse epsilon
        eps_inv_avg = f_air / eps_low + f_dielectric / eps_high
        
        # Fourier coefficients of 1/eps
        # G vectors: (Gx, Gy) where Gx, Gy are integers * 2*pi
        # We need coefficients for G in the grid
        G_vals = np.arange(-N_pw//2, N_pw//2 + 1)
        Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
        
        # Compute (1/eps)_G
        # (1/eps)_G = (1/N^2) sum_{i,j} (1/eps(i,j)) exp(-i 2pi (Gx*i/N + Gy*j/N))
        # This is just the 2D FFT of the 1/eps map, normalized
        inv_eps_map = np.zeros((size, size), dtype=float)
        inv_eps_map[eps == 1] = 1.0 / eps_high
        inv_eps_map[eps == 0] = 1.0 / eps_low
        
        # FFT
        fft_inv_eps = np.fft.fft2(inv_eps_map)
        # Normalize
        fft_inv_eps /= (size * size)
        
        # The DC component (G=0) should match eps_inv_avg
        # fft_inv_eps[center, center] is the DC term
        center = size // 2
        # Note: FFT frequency ordering is different from our G_vals
        # We need to map our G_vals to the FFT indices
        # FFT index k corresponds to frequency k (if k <= N/2) or k-N (if k > N/2)
        
        # Build the matrix for the eigenvalue problem
        # H = [ G^2/2 * delta_G,G' + (1/eps)_{G-G'} ]
        # Actually, the eigenvalue equation for TM is:
        # sum_{G'} [ (G+G')^2/2 delta_{G,G'} + (1/eps)_{G-G'} ] A_{G'} = omega^2 A_G
        # Let's construct the matrix M where M_{G,G'} = (G+G')^2/2 delta_{G,G'} + (1/eps)_{G-G'}
        # Wait, the standard form is:
        # sum_{G'} [ (k+G')^2 delta_{G,G'} + (1/eps)_{G-G'} (k+G) (k+G') ] ? No.
        
        # Correct TM eigenvalue equation:
        # sum_{G'} [ (k+G')^2 delta_{G,G'} + (1/eps)_{G-G'} ] A_{G'} = omega^2/c^2 A_G ?
        # No, it's:
        # sum_{G'} (1/eps)_{G-G'} (k+G')^2 A_{G'} = omega^2 A_G
        # Where (1/eps)_{G} are the Fourier coefficients of the inverse permittivity.
        
        # Let's build the matrix A such that A v = lambda v
        # A_{G,G'} = (1/eps)_{G-G'} * (k+G')^2
        # We need to find the 4th and 5th smallest eigenvalues for a set of k-points.
        
        # K-points: Irreducible BZ of square lattice
        # k = (kx, ky)
        # We'll sample a few points to estimate the min/max of bands 4 and 5
        # k-points: Gamma (0,0), M (0.5, 0), X (0.5, 0.5)
        k_points = [
            (0.0, 0.0),
            (0.5, 0.0),
            (0.5, 0.5)
        ]
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            k = np.array([kx, ky]) * 2 * np.pi
            
            # Construct matrix
            # Indices for G
            G_list = []
            for gx in G_vals:
                for gy in G_vals:
                    G_list.append(np.array([gx, gy]) * 2 * np.pi)
            N_pw_total = len(G_list)
            
            M = np.zeros((N_pw_total, N_pw_total), dtype=float)
            
            for i, Gi in enumerate(G_list):
                for j, Gj in enumerate(G_list):
                    # (1/eps)_{G_i - G_j}
                    dG = Gi - Gj
                    # dG is a vector of integer multiples of 2pi
                    # We need the Fourier coefficient (1/eps)_dG
                    # dG / (2*pi) gives the integer grid coordinates (dx, dy)
                    dx = int(round(dG[0] / (2 * np.pi)))
                    dy = int(round(dG[1] / (2 * np.pi)))
                    
                    # Map (dx, dy) to FFT index
                    # FFT index for frequency f is:
                    # if f >= 0: idx = f
                    # if f < 0: idx = N + f
                    # But our FFT is on a grid of size `size`.
                    # The frequencies in FFT are 0, 1, ..., size-1.
                    # Frequency `dx` corresponds to index `dx % size`.
                    idx_dx = dx % size
                    idx_dy = dy % size
                    
                    coeff = fft_inv_eps[idx_dx, idx_dy]
                    
                    # Term: (k + G_j)^2
                    k_plus_Gj = k + Gj
                    k_sq = np.dot(k_plus_Gj, k_plus_Gj)
                    
                    M[i, j] = coeff * k_sq
            
            # Solve eigenvalue problem
            # We only need the 4th and 5th smallest eigenvalues
            # Use sparse or full eig depending on size. 121x121 is small.
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals)
            
            # Bands are indexed 1-based in problem? "4th and 5th TM bands"
            # Usually band 1 is the lowest.
            # So we want eigvals[3] and eigvals[4]
            omega_4 = np.sqrt(eigvals[3])
            omega_5 = np.sqrt(eigvals[4])
            
            max_omega_4 = max(max_omega_4, omega_4)
            min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 <= max_omega_4:
            return -1.0
        
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        gap = min_omega_5 - max_omega_4
        ratio = 2.0 * gap / (min_omega_5 + max_omega_4)
        return ratio

    # Optimization Loop
    best_score = -1.0
    best_R = 0.3
    best_n = 1.0
    
    # Grid search over R and n_exp
    # R in [0.2, 0.45]
    # n_exp in [0.5, 2.0] (0.5 is square-ish, 1.0 is circle, 2.0 is very square)
    R_range = np.linspace(0.25, 0.45, 8)
    n_range = np.linspace(0.5, 2.0, 8)
    
    for R in R_range:
        for n_exp in n_range:
            if time.time() > deadline:
                break
            cell = get_epsilon_map(R, n_exp)
            score = compute_surrogate_gap(cell)
            if score > best_score:
                best_score = score
                best_R = R
                best_n = n_exp
        
        if time.time() > deadline:
            break
    
    # Refine with local search if time permits
    if time.time() < deadline - 20:
        # Simple Nelder-Mead or grid refinement
        R_min = max(0.2, best_R - 0.05)
        R_max = min(0.5, best_R + 0.05)
        n_min = max(0.5, best_n - 0.5)
        n_max = min(2.5, best_n + 0.5)
        
        R_range_fine = np.linspace(R_min, R_max, 5)
        n_range_fine = np.linspace(n_min, n_max, 5)
        
        for R in R_range_fine:
            for n_exp in n_range_fine:
                if time.time() > deadline:
                    break
                cell = get_epsilon_map(R, n_exp)
                score = compute_surrogate_gap(cell)
                if score > best_score:
                    best_score = score
                    best_R = R
                    best_n = n_exp
            if time.time() > deadline:
                break
    
    # Generate final cell
    final_cell = get_epsilon_map(best_R, best_n)
    
    # Ensure strict symmetry as per prompt:
    # cell == cell[::-1] row-wise and column-wise and cell == its transpose
    # My generation is symmetric by construction (centered grid, symmetric function).
    # But let's enforce it just in case of floating point errors in thresholding
    # Actually, the grid is discrete and centered, so it should be exact.
    
    return final_cell
