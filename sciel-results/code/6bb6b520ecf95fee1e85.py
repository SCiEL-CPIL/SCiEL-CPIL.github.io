def snap_and_verify(U, V, W):
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    U_int = (U_snap * 2).astype(int)
    V_int = (V_snap * 2).astype(int)
    W_int = (W_snap * 2).astype(int)
    
    for a in range(n*m):
        for b in range(m*p):
            temp = U_int[a, :] * V_int[b, :]
            vals = temp @ W_int
            for c in range(p*n):
                target = 8 if T[a, b, c] == 1.0 else 0
                if vals[c] != target:
                    return None
    return U_snap, V_snap, W_snap
