    def get_bands(cell, kx, ky):
        mu = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        mu_tilde = compute_fourier_coeffs(mu)
        
        # G_vectors is (N_PW, 2)
        # k_vec = [kx, ky]
        
        # We need M[i, j] = mu_tilde(G_i - G_j) * dot(G_i + k, G_j + k)
        
        # G_i + k: (N_PW, 2)
        Gk = G_vectors + np.array([kx, ky])
        
        # Dot product matrix: (N_PW, N_PW)
        # dot[i, j] = Gk[i] . Gk[j]
        dot_matrix = Gk @ Gk.T
        
        # mu_tilde matrix: (N_PW, N_PW)
        # We need mu_tilde(G_i - G_j)
        # G_i is (N_PW, 2). G_j is (N_PW, 2).
        # Diff = G_i[:, np.newaxis, :] - G_j[np.newaxis, :, :]
        # This is (N_PW, N_PW, 2).
        
        # We need to index mu_tilde with these differences.
        # mu_tilde is (33, 33).
        # Diff is integers.
        
        # This creates a large temporary array (289, 289, 2). ~200k ints. OK.
        
        diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
        # diff[i, j] is (Gx_i - Gx_j, Gy_i - Gy_j)
        
        # Index into mu_tilde
        # mu_tilde_idx[i, j] = mu_tilde[diff[i, j, 0] + 16, diff[i, j, 1] + 16]
        
        # We can do this with advanced indexing
        idx_x = diff[:, :, 0] + 16
        idx_y = diff[:, :, 1] + 16
        
        mu_matrix = mu_tilde[idx_x, idx_y]
        
        M = mu_matrix * dot_matrix
        
        eigvals = np.linalg.eigvalsh(M)
        eigvals = np.sort(eigvals)
        omega = np.sqrt(eigvals[:5])
        return omega
