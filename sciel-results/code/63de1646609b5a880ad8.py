def _verify(U, V, W, n, m, p):
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
    # Note: The tensor for matrix multiplication is 1 for all i,j,k?
    # Yes, the standard multiplication tensor has all 1s.
    # Because C_ik = sum_j A_ij B_jk.
    # The term A_ij B_jk contributes to C_ik.
    # So T_ij_k = 1.
    import numpy as np
    R = U.shape[1]
    # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
    U_3d = np.array(U).reshape(n, m, R)
    V_3d = np.array(V).reshape(m, p, R)
    W_3d = np.array(W).reshape(p, n, R)
    
    # Compute T_ij_k = sum_r U_ij_r V_jk_r W_ki_r
    # T[i, j, k]
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # Check if T is all 1s
    if not np.allclose(T, 1.0):
        return False
    return True
