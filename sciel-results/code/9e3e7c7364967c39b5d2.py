<<<<<<< SEARCH
def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that sum_r U[:,r] V[:,r] W[:,r] == T_target
    T_target[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    """
    nm, mp, pn = n*m, m*p, p*n
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    T_target = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
                
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    return np.allclose(T_recon, T_target, atol=1e-9)
=======
def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that sum_r U[:,r] V[:,r] W[:,r] == T_target
    T_target[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    """
    nm, mp, pn = n*m, m*p, p*n
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_target = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    return np.allclose(T_recon, T_target, atol=1e-9)
>>>>>>> REPLACE
