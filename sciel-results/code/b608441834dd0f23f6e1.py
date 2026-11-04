    def _embed(U_sub, V_sub, W_sub, i_off, j_off, k_off, n, m, p, n_sub, m_sub, p_sub):
        # ...
        for col in range(U_sub.shape[1]):
            for i_local in range(n_sub):
                for j_local in range(m_sub):
                    val = U_sub[i_local * m_sub + j_local, col]
                    if val != 0:
                        i_global = i_off + i_local
                        j_global = j_off + j_local
                        U_global[i_global * m + j_global, col] = val
    