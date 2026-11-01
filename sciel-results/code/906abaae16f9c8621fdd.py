    def _tensor_residual_vec(U, V, W, n, m, p):
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # R[i,j,k] = sum_s U_r[i,j,s] * V_r[j,k,s] * W_r[k,i,s]
        R = np.zeros((n, m, p), dtype=float)
        for s in range(rank):
            # U_s: (n, m)
            # V_s: (m, p)
            # W_s: (p, n)
            # We need to sum over s of (U_s[i,j] * V_s[j,k] * W_s[k,i])
            # This is an element-wise product of three tensors?
            # U_s[:, :, None] * V_s[None, :, :] * W_s[None, None, :] -> (n, m, p)
            # W_s is (p, n). We need W_s[k, i].
            # So we need W_s transposed? No, W_s[k,i] is just accessing W_s.
            # W_s shape (p, n).
            # We want a tensor T_s[i, j, k] = U_s[i, j] * V_s[j, k] * W_s[k, i].
            
            # U_s: (n, m) -> (n, m, 1)
            # V_s: (m, p) -> (1, m, p)
            # W_s: (p, n) -> (n, 1, p) ?
            # W_s[k, i]. We want index i for dim 0, k for dim 2.
            # W_s is (p, n).
            # W_s.T is (n, p).
            # W_s.T[i, k] = W_s[k, i].
            # So we use W_s.T.
            # W_s.T shape (n, p) -> (n, 1, p).
            
            U_exp = U_r[:, :, s][:, :, np.newaxis]
            V_exp = V_r[:, :, s][np.newaxis, :, :]
            W_exp = W_r[:, :, s].T[np.newaxis, np.newaxis, :] # W_r[k,i,s].T is (n,p) [i,k]
            
            # Wait, W_r is (p, n, rank).
            # W_r[:, :, s] is (p, n).
            # We want W_s[k, i].
            # W_exp should be (n, 1, p) such that W_exp[i, 0, k] = W_s[k, i].
            # W_r[:, :, s] is (p, n).
            # Let's just use W_s = W_r[:, :, s].
            # We want to access W_s[k, i].
            # If we create W_exp of shape (n, 1, p):
            # W_exp[i, 0, k] should be W_s[k, i].
            # W_s is (p, n).
            # W_s.T is (n, p).
            # W_s.T[i, k] = W_s[k, i].
            # So W_exp = W_s.T[np.newaxis, :, np.newaxis]? No.
            # W_s.T is (n, p).
            # We want (n, 1, p).
            # W_s.T[:, np.newaxis, :] is (n, 1, p).
            # Element (i, 0, k) is W_s.T[i, k] = W_s[k, i]. Correct.
            
            W_exp = W_r[:, :, s].T[:, np.newaxis, :]
            
            R += U_exp * V_exp * W_exp
            
        return R
    