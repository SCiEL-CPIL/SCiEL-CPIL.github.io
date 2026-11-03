import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    
    n, m, p = 2, 3, 6
    
    # Build target tensor T for 2x3x6
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    rows_u = n * m  # 6
    rows_v = m * p  # 18
    rows_w = p * n  # 12

    best_result = None

    # Strategy 1: Try rank 18 via Kronecker product structure
    # T_{2x3x6} as Kronecker product of T_{2x1x1} (rank 1) and T_{1x3x6} (rank 18)
    # This gives total rank 1*18 = 18
    # 
    # Constructive attempt: 
    # The 1x3x6 tensor T1[j, 6j+k, k] = 1 has rank 18 (schoolbook)
    # For the 2x1x1 tensor T2[i, 0, i] = 1 (rank 1)
    # 
    # The Kronecker product structure suggests we can find a rank-18 decomposition
    # where the 18 terms are indexed by (j,k) pairs
    
    # Try structured initialization for rank 18
    R_target = 18
    
    # Structured initialization based on block structure
    # For each (j,k) pair, create a term that handles the interaction
    # U[i*3+j, r(j,k)]: needs to be nonzero for both i=0 and i=1
    # V[j*6+k, r(j,k)]: needs to be 1
    # W[k*2+i, r(j,k)]: needs to handle both i values
    
    # Let's try: U[i*3+j, r(j,k)] = 1 for all i, V[j*6+k, r(j,k)] = 1, 
    # W[k*2+0, r(j,k)] = a, W[k*2+1, r(j,k)] = b
    # Then sum over (j'',k'') of [j''=j][k''=k] * 1 * 1 * W[k2+i, r(j,k)]
    # = W[k2+i, r(j,k)] for the matching (j,k)
    # We need this to equal 1 for all i, so W[k2+0, r(j,k)] = 1 and W[k2+1, r(j,k)] = 1
    # But then the term contributes 1 to BOTH C[0,k] and C[1,k] from A[i,j]*B[j,k]
    # which is correct! C[i,k] = sum_j A[i,j] B[j,k]
    
    # Wait, let me verify: if U[i*3+j, r(j,k)] = 1 for all i, V[j*6+k, r(j,k)] = 1,
    # W[k*2+i, r(j,k)] = 1 for all i, then:
    # sum_r U[i'3+j', r] V[j'6+k', r] W[k'2+i', r]
    # = sum over (j'',k'') of [j'=j''][k'=k''] * U[i'3+j', r(j'',k'')] * V[j'6+k', r(j'',k'')] * W[k'2+i', r(j'',k'')]
    # = [j'=j'][k'=k''] * 1 * 1 * 1  (only the matching term contributes)
    # = 1 if j'=j and k'=k, else 0
    # 
    # But we need T[i'3+j', j'6+k', k'2+i'] = 1 iff (i',j',k') are consistent, i.e., always 1
    # for the actual tensor entries. The tensor has T[i*3+j, j*6+k, k*2+i] = 1 for all
    # valid (i,j,k). So for any (i',j',k'), T[i'3+j', j'6+k', k'2+i'] should be 1.
    # 
    # But my construction gives 1 only when the term index matches, which it always does
    # for the specific (j',k') we're evaluating. So it should work!
    
    # Let me construct this explicitly:
    U_struct = np.zeros((rows_u, R_target))
    V_struct = np.zeros((rows_v, R_target))
    W_struct = np.zeros((rows_w, R_target))
    
    r_idx = 0
    for j in range(m):
        for k in range(p):
            # Term r_idx corresponds to (j,k)
            # U[i*3+j, r_idx] = 1 for i=0,1
            for i in range(n):
                U_struct[i*m+j, r_idx] = 1.0
            # V[j*6+k, r_idx] = 1
            V_struct[j*p+k, r_idx] = 1.0
            # W[k*2+i, r_idx] = 1 for i=0,1
            for i in range(n):
                W_struct[k*n+i, r_idx] = 1.0
            r_idx += 1
    
    # Verify this structured decomposition
    T_check = np.einsum('ir,jr,kr->ijk', U_struct, V_struct, W_struct)
    if np.allclose(T_check, T, atol=1e-9):
        return {
            "U": U_struct.tolist(),
            "V": V_struct.tolist(),
            "W": W_struct.tolist()
        }

    # Strategy 2: ALS for rank 18 with multiple restarts
    for restart in range(400):
        if time.time() > deadline - 15:
            break
        
        # Initialize with small random values
        U = np.random.randn(rows_u, R_target) * 0.1
        V = np.random.randn(rows_v, R_target) * 0.1
        W = np.random.randn(rows_w, R_target) * 0.1
        
        for it in range(100):
            if time.time() > deadline - 15:
                break
                
            # Solve for U
            Z_u = np.zeros((R_target, rows_v * rows_w))
            for r in range(R_target):
                Z_u[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            T_flat_u = T.transpose(0, 1, 2).flatten().reshape(rows_u, -1)
            try:
                U_new, _, _, _ = np.linalg.lstsq(Z_u.T, T_flat_u.T, rcond=None)
                U = U_new.T
            except:
                break

            # Solve for V
            Z_v = np.zeros((R_target, rows_u * rows_w))
            for r in range(R_target):
                Z_v[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            T_flat_v = T.transpose(1, 0, 2).flatten().reshape(rows_v, -1)
            try:
                V_new, _, _, _ = np.linalg.lstsq(Z_v.T, T_flat_v.T, rcond=None)
                V = V_new.T
            except:
                break

            # Solve for W
            Z_w = np.zeros((R_target, rows_u * rows_v))
            for r in range(R_target):
                Z_w[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            T_flat_w = T.transpose(2, 0, 1).flatten().reshape(rows_w, -1)
            try:
                W_new, _, _, _ = np.linalg.lstsq(Z_w.T, T_flat_w.T, rcond=None)
                W = W_new.T
            except:
                break
        
        # After ALS, round to nearest half-integer
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        T_check = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
        if np.allclose(T_check, T, atol=1e-9):
            best_result = (U_rounded, V_rounded, W_rounded)
            break
    
    if best_result:
        U236, V236, W236 = best_result
        return {
            "U": U236.tolist(),
            "V": V236.tolist(),
            "W": W236.tolist()
        }

    # Strategy 3: Try the original 2x3x3 block approach for rank 30
    n2, m2, p2 = 2, 3, 3
    T233 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T233[i*m2+j, j*p2+k, k*n2+i] = 1.0
        
    R233 = 15
    best_233 = None
    
    for restart in range(100):
        if time.time() > deadline - 5:
            break
        
        U = np.random.randn(n2*m2, R233) * 0.1
        V = np.random.randn(m2*p2, R233) * 0.1
        W = np.random.randn(p2*n2, R233) * 0.1
        
        for it in range(50):
            if time.time() > deadline - 5:
                break
                
            Z_u = np.zeros((R233, m2*p2 * p2*n2))
            for r in range(R233):
                Z_u[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            T_flat_u = T233.transpose(0, 1, 2).flatten().reshape(n2*m2, -1)
            try:
                U_new, _, _, _ = np.linalg.lstsq(Z_u.T, T_flat_u.T, rcond=None)
                U = U_new.T
            except:
                break

            Z_v = np.zeros((R233, n2*m2 * p2*n2))
            for r in range(R233):
                Z_v[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            T_flat_v = T233.transpose(1, 0, 2).flatten().reshape(m2*p2, -1)
            try:
                V_new, _, _, _ = np.linalg.lstsq(Z_v.T, T_flat_v.T, rcond=None)
                V = V_new.T
            except:
                break

            Z_w = np.zeros((R233, n2*m2 * m2*p2))
            for r in range(R233):
                Z_w[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            T_flat_w = T233.transpose(2, 0, 1).flatten().reshape(p2*n2, -1)
            try:
                W_new, _, _, _ = np.linalg.lstsq(Z_w.T, T_flat_w.T, rcond=None)
                W = W_new.T
            except:
                break
        
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        T_check = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
        if np.allclose(T_check, T233, atol=1e-9):
            best_233 = (U_rounded, V_rounded, W_rounded)
            break

    if best_233:
        U233, V233, W233 = best_233
        R236 = 2 * R233
        U236 = np.zeros((n*m, R236))
        V236 = np.zeros((m*p, R236))
        W236 = np.zeros((p*n, R236))
        
        U236[:, :R233] = U233
        V236[:m2*p2, :R233] = V233
        W236[:p2*n2, :R233] = W233
        
        U236[:, R233:] = U233
        V236[m2*p2:, R233:] = V233
        W236[p2*n2:, R233:] = W233
        
        T_final = np.einsum('ir,jr,kr->ijk', U236, V236, W236)
        if np.allclose(T_final, T, atol=1e-9):
            return {
                "U": U236.tolist(),
                "V": V236.tolist(),
                "W": W236.tolist()
            }

    # Fallback to schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
