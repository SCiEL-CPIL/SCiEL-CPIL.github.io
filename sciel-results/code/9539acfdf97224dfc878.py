def _compute_gap(cell, problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Ensure cell is a 2D numpy array
    cell_arr = np.array(cell)
    if cell_arr.shape != (size, size):
        # Handle potential shape mismatches, though unlikely
        return 0.0
        
    eps_map = np.where(cell_arr == 1, eps_high, eps_low)
    inv_eps_map = 1.0 / eps_map
    
    # Use 5x5 plane waves for speed
    n_pw = 5
    G_vals = np.arange(-n_pw//2, n_pw//2 + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_list = list(zip(Gx.flatten(), Gy.flatten()))
    n_pw_total = len(G_list)
    
    # Precompute pixel coordinates
    pixel_coords = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(pixel_coords, pixel_coords, indexing='ij')
    
    # Compute Fourier coefficients of 1/eps
    # inv_eps(G) = sum_{i,j} (1/eps(i,j)) * exp(-i * 2pi * (Gx * X + Gy * Y)) / (size^2)
    # We need inv_eps(G1 - G2) for the matrix.
    # Due to convolution theorem, the matrix element is related to the autocorrelation?
    # No, the standard form is:
    # H_{G, G'} = (k+G) . (k+G') * inv_eps(G - G')
    
    # Let's compute the 2D FFT of inv_eps_map to get the Fourier coefficients efficiently.
    # The DFT is: F(G) = sum_{x,y} f(x,y) exp(-i 2pi (Gx x + Gy y))
    # numpy.fft.fft2 computes this.
    # Note: numpy.fft.fft2 uses unnormalized forward transform.
    # So F(G) = sum f(x,y) exp(-i 2pi ...).
    # We need inv_eps(G) = F(G) / (size * size).
    
    # However, the G indices in FFT are 0..N-1.
    # We need G in -N//2..N//2.
    # We can use fftshift.
    
    # The size of the FFT must be at least the number of plane waves we want to support?
    # Actually, the pixel map is 32x32. The FFT of size 32x32 gives coefficients for G = 0..31.
    # We want G = -2..2 (for n_pw=5).
    # These correspond to indices 0, 1, 2, 30, 31 in the FFT output (if we shift).
    # fftshift moves 0 to the center.
    # Indices -2, -1, 0, 1, 2 correspond to positions in the shifted array.
    
    fft_inv_eps = np.fft.fft2(inv_eps_map)
    fft_inv_eps_shifted = np.fft.fftshift(fft_inv_eps)
    
    # Now fft_inv_eps_shifted[i, j] corresponds to G = (i - 16, j - 16) for size 32.
    # We need to extract the coefficients for G in G_list.
    
    inv_eps_coeffs = np.zeros((n_pw_total, n_pw_total), dtype=complex)
    # Map G_list indices to FFT indices
    # G_list contains tuples (gx, gy) in range -2..2.
    # FFT index for gx is (gx + 16) % 32? No, after shift, index 16 is G=0.
    # So index for G is 16 + G.
    
    # Precompute the indices
    fft_indices = []
    for gx, gy in G_list:
        idx_x = 16 + gx
        idx_y = 16 + gy
        fft_indices.append((idx_x, idx_y))
        
    # Extract coefficients
    coeff_values = np.array([fft_inv_eps_shifted[ix, iy] for ix, iy in fft_indices]) / (size * size)
    
    # Now inv_eps_coeffs[i, j] should be inv_eps(G_i - G_j).
    # G_i - G_j is a vector. We need the Fourier coefficient at that vector.
    # The set of differences G_i - G_j will range from -4..4.
    # Our FFT only computed coefficients for G in -16..15 (effectively).
    # We need coefficients for G_diff in -4..4.
    # We can extract these from the shifted FFT array as well.
    
    # Let's build a dictionary or map for the coefficients we need.
    # We need inv_eps(G_diff) for all G_diff = G_i - G_j.
    # Max diff is 4. Min diff is -4.
    # So we need coefficients for G in -4..4.
    
    # Let's just compute the coefficients for all G in -4..4 from the FFT.
    diff_G_vals = np.arange(-4, 5)
    dGx, dGy = np.meshgrid(diff_G_vals, diff_G_vals, indexing='ij')
    diff_G_list = list(zip(dGx.flatten(), dGy.flatten()))
    n_diff = len(diff_G_list)
    
    diff_fft_indices = []
    for gx, gy in diff_G_list:
        idx_x = 16 + gx
        idx_y = 16 + gy
        diff_fft_indices.append((idx_x, idx_y))
        
    diff_coeff_values = np.array([fft_inv_eps_shifted[ix, iy] for ix, iy in diff_fft_indices]) / (size * size)
    
    # Map (gx, gy) to index in diff_coeff_values
    # diff_G_list is ordered by dGx then dGy?
    # meshgrid with indexing='ij' iterates i (dGx) then j (dGy).
    # So index = (gx - (-4)) * 9 + (gy - (-4)) = (gx+4)*9 + (gy+4).
    
    def get_diff_coeff_idx(gx, gy):
        return (gx + 4) * 9 + (gy + 4)
        
    # Build inv_eps_coeffs
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            dgx = G_list[i][0] - G_list[j][0]
            dgy = G_list[i][1] - G_list[j][1]
            idx = get_diff_coeff_idx(dgx, dgy)
            inv_eps_coeffs[i, j] = diff_coeff_values[idx]
            
    # Define k points
    k_points = []
    # Sample 30 points in the irreducible BZ triangle (0,0)-(0.5,0)-(0.5,0.5)
    # Use a grid
    for i in range(6):
        for j in range(6):
            kx = 0.5 * i / 5
            ky = 0.5 * j / 5
            if kx + ky <= 0.5 + 1e-9:
                k_points.append((kx, ky))
    while len(k_points) < 30:
        kx = random.random() * 0.5
        ky = random.random() * (0.5 - kx)
        k_points.append((kx, ky))
        
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in k_points:
        # Build H matrix
        # H[i, j] = (k+G_i) . (k+G_j) * inv_eps(G_i - G_j)
        H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
        for i in range(n_pw_total):
            for j in range(n_pw_total):
                kpi = (kx + G_list[i][0], ky + G_list[i][1])
                kpj = (kx + G_list[j][0], ky + G_list[j][1])
                dot = kpi[0] * kpj[0] + kpi[1] * kpj[1]
                H[i, j] = dot * inv_eps_coeffs[i, j]
                
        try:
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            if len(eigvals) >= 5:
                omega4 = math.sqrt(max(0, eigvals[3]))
                omega5 = math.sqrt(max(0, eigvals[4]))
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            pass
            
    if min_omega5 == float('inf'):
        return 0.0
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2
    if midgap == 0:
        return 0.0
    return gap / midgap
