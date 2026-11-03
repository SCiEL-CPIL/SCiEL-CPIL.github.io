    def get_svd_init(rank):
        # Unfold T to matrices
        # T shape: (n*m, m*p, p*n)
        T1 = T.reshape(n*m, -1) # (n*m, m*p*p*n)
        T2 = T.transpose(1, 0, 2).reshape(m*p, -1) # (m*p, n*m*p*n)
        T3 = T.transpose(2, 0, 1).reshape(p*n, -1) # (p*n, n*m*m*p)
        
        U_svd, s, Vt = np.linalg.svd(T1, full_matrices=False)
        U_init = U_svd[:, :rank] * np.sqrt(s[:rank])
        
        V_svd, s, Vt = np.linalg.svd(T2, full_matrices=False)
        V_init = V_svd[:, :rank] * np.sqrt(s[:rank])
        
        W_svd, s, Vt = np.linalg.svd(T3, full_matrices=False)
        W_init = W_svd[:, :rank] * np.sqrt(s[:rank])
        
        return U_init, V_init, W_init
