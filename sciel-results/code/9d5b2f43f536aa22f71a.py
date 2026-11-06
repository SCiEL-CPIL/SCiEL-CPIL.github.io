    def compute_gap(cell, N_pw, k_points):
        """
        Compute the gap between band 4 and 5 for the given cell.
        """
        N_half = N_pw // 2
        G_indices_arr = np.arange(-N_half, N_half + 1)
        
        # Precompute pixel coordinates
        x = (np.arange(size) + 0.5) / size
        y = (np.arange(size) + 0.5) / size
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Compute inv_eps_fourier
        inv_eps_pixels = 1.0 / (eps_low + (eps_high - eps_low) * cell)
        inv_eps_fourier = np.zeros((N_pw, N_pw), dtype=complex)
        
        for idx_gx, gx in enumerate(G_indices_arr):
            for idx_gy, gy in enumerate(G_indices_arr):
                phase = np.exp(-1j * 2 * np.pi * (gx * X + gy * Y))
                inv_eps_fourier[idx_gx, idx_gy] = np.sum(inv_eps_pixels * phase) / (size * size)
                
        # Flatten inv_eps_fourier for fast lookup
        inv_eps_flat = inv_eps_fourier.flatten()
        
        # Precompute index differences for the matrix A
        # G_i is (G_indices_arr[i//N_pw], G_indices_arr[i%N_pw])
        # Diff = G_i - G_j
        # Index in inv_eps_flat = (idx_diff_x + N_half) * N_pw + (idx_diff_y + N_half)
        
        Gx_flat = np.repeat(G_indices_arr, N_pw)
        Gy_flat = np.tile(G_indices_arr, N_pw)
        
        # We need a matrix of indices of shape (N_pw^2, N_pw^2)
        # This might be too big for memory if N_pw is large, but 289x289 is fine.
        # idx_diff_x = Gx_flat[:, None] - Gx_flat[None, :]
        # idx_diff_y = Gy_flat[:, None] - Gy_flat[None, :]
        # flat_idx = (idx_diff_x + N_half) * N_pw + (idx_diff_y + N_half)
        
        # However, we only need to construct A for each k-point.
        # The structure of A is: A[i, j] = kG2[j] * inv_eps_flat[flat_idx[i, j]]
        
        # Precompute flat_idx
        idx_diff_x = Gx_flat[:, None] - Gx_flat[None, :]
        idx_diff_y = Gy_flat[:, None] - Gy_flat[None, :]
        flat_idx = (idx_diff_x + N_half) * N_pw + (idx_diff_y + N_half)
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for kx, ky in k_points:
            kG2 = (kx + Gx_flat)**2 + (ky + Gy_flat)**2
            
            # A[i, j] = kG2[j] * inv_eps_flat[flat_idx[i, j]]
            # kG2 is (N,). We want to multiply each column j by kG2[j].
            # So A = inv_eps_flat[flat_idx] * kG2[None, :]
            
            A = inv_eps_flat[flat_idx] * kG2[None, :]
            
            eigenvalues = np.linalg.eigvals(A)
            eigenvalues = np.sort(np.real(eigenvalues))
            
            if len(eigenvalues) >= 5:
                w4 = np.sqrt(eigenvalues[3])
                w5 = np.sqrt(eigenvalues[4])
                if w4 > max_omega4:
                    max_omega4 = w4
                if w5 < min_omega5:
                    min_omega5 = w5
                    
        if min_omega5 == np.inf or max_omega4 == 0:
            return -1.0
            
        gap = 2 * (min_omega5 - max_omega4)
        midgap = min_omega5 + max_omega4
        if midgap == 0:
            return -1.0
        return gap / midgap
