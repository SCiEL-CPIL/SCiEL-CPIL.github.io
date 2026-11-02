import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    schoolbook_rank = n * m * p  # 100
    
    start_time = time.time()
    deadline = start_time + 110  # safety margin for 120s limit
    
    # Build schoolbook decomposition
    U = [[0] * schoolbook_rank for _ in range(nm)]
    V = [[0] * schoolbook_rank for _ in range(mp)]
    W = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Convert to numpy for efficient verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def verify_decomposition(U_list, V_list, W_list, tol=1e-9):
        """Verify exact decomposition."""
        Ua = np.array(U_list, dtype=float)
        Va = np.array(V_list, dtype=float)
        Wa = np.array(W_list, dtype=float)
        r = Ua.shape[1]
        # Compute tensor: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        U_r = Ua.reshape(nm, r)
        V_r = Va.reshape(mp, r)
        W_r = Wa.reshape(pn, r)
        # For each (i,j,k), compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        tensor = np.zeros((nm, mp, pn), dtype=float)
        for r_idx in range(r):
            tensor += np.einsum('a,b,c->abc', U_r[:, r_idx], V_r[:, r_idx], W_r[:, r_idx])
        # Check against target: T[i*m+j, j*p+k, k*n+i] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    if abs(tensor[idx_u, idx_v, idx_w] - 1.0) > tol:
                        return False
        return True
    
    # Best known result
    best_U, best_V, best_W = U, V, W
    best_rank = schoolbook_rank
    
    # Strategy: Try to reduce rank by finding linear dependencies
    # Use tensor product of Strassen decomposition for 2x2x2 (rank 7)
    # with a padded version to get 4x5x5
    
    # Strassen decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M2 + M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M3 + M5 - M6
    
    # U entries: U[i*2+j, t]
    # i=0,j=0: A11 -> t=0(M1), t=2(M3), t=5(M6), t=6(M7) => [1,0,1,0,0,1,1]
    # i=0,j=1: A12 -> t=0(M1), t=4(M5), t=6(M7) => [1,0,0,0,1,0,-1]
    # i=1,j=0: A21 -> t=1(M2), t=5(M6) => [0,1,0,0,0,1,0]
    # i=1,j=1: A22 -> t=0(M1), t=1(M2), t=3(M4), t=6(M7) => [1,1,0,1,0,0,-1]
    
    strassen_U = np.array([
        [1, 0, 1, 0, 0, 1, 1],   # A11
        [1, 0, 0, 0, 1, 0, -1],  # A12
        [0, 1, 0, 0, 0, 1, 0],   # A21
        [1, 1, 0, 1, 0, 0, -1]   # A22
    ])
    
    # V entries: V[j*2+k, t]
    # j=0,k=0: B11 -> t=0(M1), t=1(M2), t=3(M4), t=5(M6) => [1,1,0,1,0,1,0]
    # j=0,k=1: B12 -> t=3(M4), t=5(M6) => [0,0,0,1,0,1,0]
    # j=1,k=0: B21 -> t=0(M1), t=2(M3), t=6(M7) => [1,0,1,0,0,0,1]
    # j=1,k=1: B22 -> t=0(M1), t=4(M5), t=6(M7) => [1,0,0,0,1,0,1]
    
    strassen_V = np.array([
        [1, 1, 0, 1, 0, 1, 0],   # B11
        [0, 0, 0, 1, 0, 1, 0],   # B12
        [1, 0, 1, 0, 0, 0, 1],   # B21
        [1, 0, 0, 0, 1, 0, 1]    # B22
    ])
    
    # W entries: W[k*2+i, t]
    # k=0,i=0: C11 -> t=0(M1), t=1(-M2), t=3(M4), t=6(M7) => [1,-1,0,1,0,0,1]
    # k=0,i=1: C12 -> t=2(M3), t=4(M5) => [0,0,1,0,1,0,0]
    # k=1,i=0: C21 -> t=1(M2), t=5(M6) => [0,1,0,0,0,1,0]
    # k=1,i=1: C22 -> t=0(M1), t=2(-M3), t=4(M5), t=5(-M6) => [1,0,-1,0,1,-1,0]
    
    strassen_W = np.array([
        [1, -1, 0, 1, 0, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 0, 0, 1, 0],   # C21
        [1, 0, -1, 0, 1, -1, 0]  # C22
    ])
    
    # Verify Strassen decomposition
    if not verify_decomposition(strassen_U.tolist(), strassen_V.tolist(), strassen_W.tolist()):
        # Fallback to schoolbook
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Now create tensor product to get 4x4x4 decomposition (rank 49)
    # Then restrict to 4x5x5
    
    # Tensor product of two 2x2x2 decompositions gives 4x4x4
    # U_prod: 16x49, V_prod: 16x49, W_prod: 16x49
    # Index mapping: U_prod[i1*2+i2, j1*2+j2, t1*7+t2]
    
    U_prod = np.zeros((16, 49))
    V_prod = np.zeros((16, 49))
    W_prod = np.zeros((16, 49))
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for t1 in range(7):
                        for t2 in range(7):
                            idx_u = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            idx_v = (j1 * 2 + j2) * 4 + (k1 := 0)  # placeholder
                            # Actually, let's be more careful with indexing
                            pass
    
    # Let me redo this more carefully
    # For 4x4x4: n=4, m=4, p=4
    # U: 16x49, V: 16x49, W: 16x49
    # U[i*4+j, t] where i,j in 0..3
    # V[j*4+k, t] where j,k in 0..3
    # W[k*4+i, t] where k,i in 0..3
    
    # Tensor product: t = t1*7 + t2
    # i = i1*2 + i2, j = j1*2 + j2, k = k1*2 + k2
    
    U_prod = np.zeros((16, 49))
    V_prod = np.zeros((16, 49))
    W_prod = np.zeros((16, 49))
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for t1 in range(7):
                        for t2 in range(7):
                            t = t1 * 7 + t2
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U_prod[i * 4 + j, t] = strassen_U[i1 * 2 + j1, t1] * strassen_U[i2 * 2 + j2, t2]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    for t1 in range(7):
                        for t2 in range(7):
                            t = t1 * 7 + t2
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V_prod[j * 4 + k, t] = strassen_V[j1 * 2 + k1, t1] * strassen_V[j2 * 2 + k2, t2]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    for t1 in range(7):
                        for t2 in range(7):
                            t = t1 * 7 + t2
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W_prod[k * 4 + i, t] = strassen_W[k1 * 2 + i1, t1] * strassen_W[k2 * 2 + i2, t2]
    
    # Verify 4x4x4 decomposition
    if not verify_decomposition(U_prod.tolist(), V_prod.tolist(), W_prod.tolist()):
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Now restrict to 4x5x5
    # We have 4x4x4, need 4x5x5
    # Pad the 4x4x4 to 4x5x5 by adding zero rows/cols
    # U: 20x49 (rows i*5+j for i=0..3, j=0..4)
    # V: 25x49 (rows j*5+k for j=0..4, k=0..4)
    # W: 20x49 (rows k*4+i for k=0..4, i=0..3)
    
    U_final = np.zeros((20, 49))
    V_final = np.zeros((25, 49))
    W_final = np.zeros((20, 49))
    
    # Copy from 4x4x4 to 4x5x5
    # U[i*5+j, t] = U_prod[i*4+j, t] for j=0..3, zero for j=4
    for i in range(4):
        for j in range(4):
            U_final[i * 5 + j, :] = U_prod[i * 4 + j, :]
        # j=4 is already zero
    
    # V[j*5+k, t] = V_prod[j*4+k, t] for j=0..3, k=0..3, zero otherwise
    for j in range(4):
        for k in range(4):
            V_final[j * 5 + k, :] = V_prod[j * 4 + k, :]
    
    # W[k*4+i, t] = W_prod[k*4+i, t] for k=0..3, zero for k=4
    for k in range(4):
        for i in range(4):
            W_final[k * 4 + i, :] = W_prod[k * 4 + i, :]
    
    # Verify 4x5x5 decomposition
    if verify_decomposition(U_final.tolist(), V_final.tolist(), W_final.tolist()):
        best_U = U_final.tolist()
        best_V = V_final.tolist()
        best_W = W_final.tolist()
        best_rank = 49
    
    # Return the best result
    return {"U": best_U, "V": best_V, "W": best_W}
