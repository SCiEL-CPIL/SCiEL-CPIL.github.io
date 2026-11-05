def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    
    U_3d = U_np.reshape(n, m, -1)
    V_3d = V_np.reshape(m, p, -1)
    W_3d = W_np.reshape(p, n, -1)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    return np.all(T == 1)
