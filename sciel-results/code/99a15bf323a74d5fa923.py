<<<<<<< SEARCH
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify_fast(Ua, Va, Wa):
        # Vectorized verification: check all (i,j,k) at once
        # Ua shape: (n*m, rank), Va shape: (m*p, rank), Wa shape: (p*n, rank)
        # For each (i,j,k): sum over r of Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
        # Reshape to 3D and use einsum
        U3 = Ua.reshape(n, m, -1)  # (n, m, rank)
        V3 = Va.reshape(m, p, -1)  # (m, p, rank)
        W3 = Wa.reshape(p, n, -1)  # (p, n, rank)
        # Result[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(result, 1.0, atol=1e-9)
    
    if not verify_fast(U_arr, V_arr, W_arr):
=======
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if not verify_fast(U_arr, V_arr, W_arr):
>>>>>>> REPLACE

