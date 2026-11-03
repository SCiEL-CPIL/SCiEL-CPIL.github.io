    def _evaluate(U, V, W, n, m, p):
        Ua = np.asarray(U, dtype=np.float64)
        Va = np.asarray(V, dtype=np.float64)
        Wa = np.asarray(W, dtype=np.float64)
        
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U_3d = Ua.reshape(n, m, -1)
        V_3d = Va.reshape(m, p, -1)
        W_3d = Wa.reshape(p, n, -1)
        
        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We want result[i,j,k]
        
        # Use einsum: 'ijr,jkr,kin->ijk'
        # Wait, W is indexed by k, i. So W[k, i, r].
        # V is indexed by j, k. So V[j, k, r].
        # U is indexed by i, j. So U[i, j, r].
        
        result = np.einsum('ijr,jkr,kin->ijk', U_3d, V_3d, W_3d)
        
        # The target is 1.0 for all i, j, k.
        # Error is max |result - 1|
        max_err = float(np.max(np.abs(result - 1.0)))
        return max_err
    