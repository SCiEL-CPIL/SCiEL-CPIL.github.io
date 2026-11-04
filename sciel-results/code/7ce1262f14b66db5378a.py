import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Construct the target tensor T_mult as a 3D array
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    # Try to find a rank < nm*mp/pn? No, rank < n*m*p
    # Schoolbook rank is n*m*p = 100
    max_rank = n * m * p
    
    # Try ranks from max_rank - 1 down to max_rank - 10
    for R in range(max_rank - 1, max_rank - 10, -1):
        # Initialize U, V, W
        # Use random initialization or truncated schoolbook
        U = np.random.randn(nm, R)
        V = np.random.randn(mp, R)
        W = np.random.randn(pn, R)
        
        # ALS iterations
        for _ in range(50):
            # Solve for U
            # M_(1) = T.reshape(nm, mp*pn)
            # V_odot_W = np.kron(V, W) ? No, Khatri-Rao
            # V_odot_W[r, a] = V[a, r] * W[b, r] ?
            # Khatri-Rao product of V (mp x R) and W (pn x R) is (mp*pn) x R
            # The r-th column is V[:, r] * W[:, r] (element-wise? No, Kronecker? No, Khatri-Rao is column-wise Kronecker)
            # Actually, for CP decomposition, it's the Khatri-Rao product.
            # If V is (I x R) and W is (J x R), V ⊙ W is (I*J x R) where col r is v_r ⊗ w_r.
            VW = np.kron(V, W) # This is (mp*pn) x R
            # M_(1) is (nm, mp*pn)
            M1 = T.reshape(nm, mp * pn)
            # U = M1 @ VW @ inv(VW.T @ VW)
            try:
                U = M1 @ np.linalg.solve(VW.T @ VW, VW.T)
            except np.linalg.LinAlgError:
                break
                
            # Solve for V
            # M_(2) = T.transpose(1, 0, 2).reshape(mp, nm*pn)
            # U_odot_W = np.kron(U, W) ? (nm*pn) x R
            UW = np.kron(U, W)
            M2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            try:
                V = M2 @ np.linalg.solve(UW.T @ UW, UW.T)
            except np.linalg.LinAlgError:
                break
                
            # Solve for W
            # M_(3) = T.transpose(2, 0, 1).reshape(pn, nm*mp)
            # U_odot_V = np.kron(U, V) ? (nm*mp) x R
            UV = np.kron(U, V)
            M3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            try:
                W = M3 @ np.linalg.solve(UV.T @ UV, UV.T)
            except np.linalg.LinAlgError:
                break
        
        # Check if U, V, W are close to half-integers
        # Scale by 2 to check for integers
        U2 = 2 * U
        V2 = 2 * V
        W2 = 2 * W
        
        # Check if close to integers
        if np.allclose(U2, np.round(U2), atol=1e-5) and \
           np.allclose(V2, np.round(V2), atol=1e-5) and \
           np.allclose(W2, np.round(W2), atol=1e-5):
            
            U_int = np.round(U2).astype(int) / 2.0
            V_int = np.round(V2).astype(int) / 2.0
            W_int = np.round(W2).astype(int) / 2.0
            
            # Verify exact decomposition
            # Reconstruct T_rec
            # T_rec = sum_r u_r ⊗ v_r ⊗ w_r
            # Check if T_rec == T
            # We can check a few random slices or the whole thing
            # Full check is O(R * nm * mp * pn) which is small
            T_rec = np.zeros_like(T)
            for r in range(R):
                T_rec += np.einsum('i,j,k->ijk', U_int[:, r], V_int[:, r], W_int[:, r])
            
            if np.allclose(T_rec, T, atol=1e-9):
                return {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": W_int.tolist()
                }
                
    # Fallback to schoolbook
    rank = max_rank
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
