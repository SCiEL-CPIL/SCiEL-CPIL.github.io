    def compute_metrics(P):
        # H_{-1}: x - y
        # indices: x - y + 9
        idx_minus1 = np.arange(n)[:, None] - np.arange(n)[None, :] + n - 1
        p_minus1 = np.zeros(2*n - 1)
        np.add.at(p_minus1, idx_minus1, P)
        H_minus1 = shannon_entropy(p_minus1)
        
        # H_0: x
        p_x = np.sum(P, axis=0)
        H_0 = shannon_entropy(p_x)
        
        # H_1: x + y
        idx_xy = np.arange(n)[:, None] + np.arange(n)[None, :]
        p_xy = np.zeros(2*n - 1)
        np.add.at(p_xy, idx_xy, P)
        H_1 = shannon_entropy(p_xy)
        
        # H_2: x + 2y
        idx_x2y = np.arange(n)[:, None] + 2 * np.arange(n)[None, :]
        p_x2y = np.zeros(3*n - 1)
        np.add.at(p_x2y, idx_x2y, P)
        H_2 = shannon_entropy(p_x2y)
        
        M = max(H_0, H_1, H_2)
        return H_minus1, M
