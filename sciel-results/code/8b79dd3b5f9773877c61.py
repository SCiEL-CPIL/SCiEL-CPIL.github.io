import numpy as np

def design(problem):
    size = problem["pixels"]
    
    # Precompute pixel coordinates
    coords = np.arange(size) / size - 0.5
    x, y = np.meshgrid(coords, coords, indexing='ij')
    
    # G vectors for the surrogate solver
    # We use a 5x5 grid of G vectors: -2, -1, 0, 1, 2
    g_vals = np.arange(-2, 3)
    Gx, Gy = np.meshgrid(g_vals, g_vals, indexing='ij')
    G_flat = np.stack([Gx.ravel(), Gy.ravel()], axis=-1)
    N_pw = len(G_flat)
    
    # K-points to sample in the irreducible BZ
    # Gamma (0,0), X (0.5, 0), M (0.5, 0.5)
    k_points = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [0.5, 0.5]
    ])
    
    def get_epsilon_coeffs(cell):
        # cell is 32x32 array of 0/1
        # Compute FFT
        # The Fourier coefficient is (1/N^2) sum epsilon_ij exp(-i G . r_ij)
        # numpy.fft.fft2 gives sum epsilon_ij exp(-i 2pi (m x_i + n y_j) / N)
        # Our G is integer n, m. r = i/N, j/N.
        # exp(-i 2pi (n i/N + m j/N)) matches G=(n,m).
        # So we can just use fft2.
        fft = np.fft.fft2(cell)
        # Normalize
        fft = fft / (size * size)
        # Reorder to match G_flat order
        # G_flat has indices -2 to 2.
        # fft indices are 0 to 31.
        # Map g to fft_index: if g >= 0, idx = g, else idx = size + g
        eps_coeffs = np.zeros(N_pw, dtype=complex)
        for idx, (gx, gy) in enumerate(G_flat):
            ix = gx % size
            iy = gy % size
            eps_coeffs[idx] = fft[iy, ix] # Note: fft2 output is [y, x] order? 
            # numpy.fft.fft2(a): a is [row, col]. row corresponds to y, col to x.
            # Our meshgrid: x is col, y is row.
            # So fft[iy, ix] is correct for G=(ix, iy) if we treat Gx as x-component and Gy as y-component.
            # Wait, G_flat has (Gx, Gy).
            # ix corresponds to Gx. iy corresponds to Gy.
            # So fft[iy, ix] is correct.
        return eps_coeffs

    def solve_bands(eps_coeffs):
        min_omega_5 = float('inf')
        max_omega_4 = -float('inf')
        
        for k in k_points:
            kx, ky = k
            # K_G = k + G
            Kx = kx + G_flat[:, 0]
            Ky = ky + G_flat[:, 1]
            
            # Build Matrix M
            # M_{ab} = eps_{G_a - G_b} * (K_a dot K_b)
            # We need eps for all differences G_a - G_b.
            # G_a, G_b are in [-2, 2]. Differences are in [-4, 4].
            # Our eps_coeffs only has [-2, 2].
            # We need to expand the epsilon coefficients or use the pixel data directly?
            # Using pixel data directly for the surrogate is more accurate but slower.
            # However, we only have 25 plane waves. The difference G_a - G_b can be up to 4.
            # We can compute the full FFT of the cell once, and index into it.
            
            # Let's restructure: compute full FFT once per candidate.
            pass
            
        return min_omega_5, max_omega_4

    # Refined approach: 
    # 1. Generate candidate cell.
    # 2. Compute full FFT (32x32).
    # 3. For each k, build 25x25 matrix using the full FFT for coefficients.
    
    def evaluate_candidate(A, B, T):
        # Construct continuous potential
        # f(x,y) = A(cos(4pi x) + cos(4pi y)) + 2B cos(4pi x)cos(4pi y)
        # cos(4pi x) = cos(2pi * 2 * x)
        c_x = np.cos(4 * np.pi * x)
        c_y = np.cos(4 * np.pi * y)
        f = A * (c_x + c_y) + 2 * B * c_x * c_y
        cell = (f > T).astype(int)
        
        # Compute FFT of this cell
        fft = np.fft.fft2(cell) / (size * size)
        
        min_omega_5 = float('inf')
        max_omega_4 = -float('inf')
        
        for k in k_points:
            kx, ky = k
            Kx = kx + G_flat[:, 0]
            Ky = ky + G_flat[:, 1]
            
            # Dot products K_a . K_b
            # K is N_pw x 2
            K = np.stack([Kx, Ky], axis=-1)
            dot_K = K @ K.T # N_pw x N_pw
            
            # Epsilon differences
            # We need eps_{G_a - G_b}
            # G_a = G_flat[a], G_b = G_flat[b]
            # Diff = G_a - G_b
            # Diff components are in [-4, 4]
            # We need to get fft[diff_y, diff_x]
            
            # Vectorized construction of matrix
            # Create indices for differences
            Gx_a = G_flat[:, 0:1] # N x 1
            Gy_a = G_flat[:, 0:1] # Correction: G_flat[:, 0] is x, G_flat[:, 1] is y
            Gx_a = G_flat[:, 0][:, np.newaxis]
            Gy_a = G_flat[:, 1][:, np.newaxis]
            
            Gx_b = G_flat[:, 0][np.newaxis, :]
            Gy_b = G_flat[:, 1][np.newaxis, :]
            
            Diff_x = Gx_a - Gx_b
            Diff_y = Gy_a - Gy_b
            
            # Map diff to fft indices
            # fft index = diff % size
            idx_x = Diff_x % size
            idx_y = Diff_y % size
            
            # Gather epsilon coefficients
            # fft is 2D array [y, x]
            eps_mat = fft[np.ix_(idx_y, idx_x)] # This creates a N x N matrix of coeffs
            
            # Construct Hamiltonian
            M = eps_mat * dot_K
            
            # Solve eigenvalues
            # M should be Hermitian. eps is real (symmetric cell), dot_K is real symmetric.
            # But eps_{-G} = eps_G^*. For real cell, eps is real.
            # So M is symmetric.
            evals = np.linalg.eigvalsh(M)
            evals = np.sort(evals)
            
            # Get 4th and 5th bands (indices 3 and 4)
            if len(evals) >= 5:
                w4 = evals[3]
                w5 = evals[4]
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
                
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap <= 0:
            return -1
        return gap / midgap

    # Grid search
    best_score = -np.inf
    best_params = (0, 0, 0)
    
    # Coarse search
    for A in np.linspace(-0.5, 0.5, 5):
        for B in np.linspace(-0.5, 0.5, 5):
            for T in np.linspace(-0.2, 0.2, 5):
                score = evaluate_candidate(A, B, T)
                if score > best_score:
                    best_score = score
                    best_params = (A, B, T)
                    
    # Fine search around best
    A0, B0, T0 = best_params
    for A in np.linspace(A0 - 0.1, A0 + 0.1, 5):
        for B in np.linspace(B0 - 0.1, B0 + 0.1, 5):
            for T in np.linspace(T0 - 0.05, T0 + 0.05, 5):
                score = evaluate_candidate(A, B, T)
                if score > best_score:
                    best_score = score
                    best_params = (A, B, T)

    A, B, T = best_params
    c_x = np.cos(4 * np.pi * x)
    c_y = np.cos(4 * np.pi * y)
    f = A * (c_x + c_y) + 2 * B * c_x * c_y
    cell = (f > T).astype(int)
    
    return cell.tolist()
