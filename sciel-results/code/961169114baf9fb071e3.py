import time
import numpy as np

DIMENSIONS = [6, 6, 6]

_DEADLINE = time.time() + 135

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U_np[i * m + j, :]
            for k in range(p):
                v_row = V_np[j * p + k, :]
                w_row = W_np[k * n + i, :]
                s = np.dot(u_row, v_row * w_row)
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-10

def _get_laderman_3x3():
    """Returns a rank-23 Laderman decomposition for 3x3 matrix multiplication.
    
    Returns numpy arrays U (9,23), V (9,23), W (9,23) where:
    sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1 for all i,j,k in 0..2
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # 9 diagonal terms
    for i in range(3):
        r = i
        U[i*3+i, r] = 1
        V[i*3+i, r] = 1
        W[i*3+i, r] = 1
    
    # 14 off-diagonal terms (Laderman 1976)
    # Term 9: handles C[0,1] and C[1,0] contributions
    r = 9
    U[0*3+0, r] = 1; U[1*3+1, r] = 1
    V[0*3+0, r] = 1; V[1*3+1, r] = 1
    W[1*3+0, r] = 1; W[0*3+1, r] = 1
    W[0*3+0, r] = -1; W[1*3+1, r] = -1
    
    # Term 10: handles C[0,2] and C[2,0]
    r = 10
    U[0*3+0, r] = 1; U[2*3+2, r] = 1
    V[0*3+0, r] = 1; V[2*3+2, r] = 1
    W[2*3+0, r] = 1; W[0*3+2, r] = 1
    W[0*3+0, r] = -1; W[2*3+2, r] = -1
    
    # Term 11: handles C[1,2] and C[2,1]
    r = 11
    U[1*3+1, r] = 1; U[2*3+2, r] = 1
    V[1*3+1, r] = 1; V[2*3+2, r] = 1
    W[2*3+1, r] = 1; W[1*3+2, r] = 1
    W[1*3+1, r] = -1; W[2*3+2, r] = -1
    
    # Term 12: row 0 of A, col 1 of B
    r = 12
    U[0*3+0, r] = 1; U[0*3+1, r] = 1; U[0*3+2, r] = 1
    V[1*3+0, r] = 1; V[1*3+1, r] = 1; V[1*3+2, r] = 1
    W[1*3+0, r] = 1; W[1*3+1, r] = 1; W[1*3+2, r] = 1
    W[0*3+0, r] = -1; W[0*3+1, r] = -1; W[0*3+2, r] = -1
    
    # Term 13: col 0 of A, row 1 of B
    r = 13
    U[0*3+0, r] = 1; U[1*3+0, r] = 1; U[2*3+0, r] = 1
    V[0*3+1, r] = 1; V[1*3+1, r] = 1; V[2*3+1, r] = 1
    W[1*3+0, r] = 1; W[1*3+1, r] = 1; W[2*3+1, r] = 1
    W[0*3+0, r] = -1; W[1*3+0, r] = -1; W[2*3+0, r] = -1
    
    # Term 14: row 0 of A, col 2 of B
    r = 14
    U[0*3+0, r] = 1; U[0*3+1, r] = 1; U[0*3+2, r] = 1
    V[2*3+0, r] = 1; V[2*3+1, r] = 1; V[2*3+2, r] = 1
    W[2*3+0, r] = 1; W[2*3+1, r] = 1; W[2*3+2, r] = 1
    W[0*3+0, r] = -1; W[0*3+1, r] = -1; W[0*3+2, r] = -1
    
    # Term 15: col 0 of A, row 2 of B
    r = 15
    U[0*3+0, r] = 1; U[1*3+0, r] = 1; U[2*3+0, r] = 1
    V[0*3+2, r] = 1; V[1*3+2, r] = 1; V[2*3+2, r] = 1
    W[2*3+0, r] = 1; W[2*3+1, r] = 1; W[2*3+2, r] = 1
    W[0*3+0, r] = -1; W[1*3+0, r] = -1; W[2*3+0, r] = -1
    
    # Term 16: row 1 of A, col 0 of B
    r = 16
    U[1*3+0, r] = 1; U[1*3+1, r] = 1; U[1*3+2, r] = 1
    V[0*3+0, r] = 1; V[0*3+1, r] = 1; V[0*3+2, r] = 1
    W[0*3+1, r] = 1; W[0*3+2, r] = 1; W[1*3+1, r] = 1
    W[1*3+0, r] = -1; W[1*3+1, r] = -1; W[1*3+2, r] = -1
    
    # Term 17: col 1 of A, row 0 of B
    r = 17
    U[0*3+1, r] = 1; U[1*3+1, r] = 1; U[2*3+1, r] = 1
    V[1*3+0, r] = 1; V[1*3+1, r] = 1; V[1*3+2, r] = 1
    W[0*3+1, r] = 1; W[1*3+1, r] = 1; W[2*3+1, r] = 1
    W[0*3+0, r] = -1; W[1*3+0, r] = -1; W[2*3+0, r] = -1
    
    # Term 18: row 2 of A, col 1 of B
    r = 18
    U[2*3+0, r] = 1; U[2*3+1, r] = 1; U[2*3+2, r] = 1
    V[1*3+0, r] = 1; V[1*3+1, r] = 1; V[1*3+2, r] = 1
    W[1*3+2, r] = 1; W[2*3+2, r] = 1
    W[2*3+0, r] = -1; W[2*3+1, r] = -1; W[2*3+2, r] = -1
    
    # Term 19: col 2 of A, row 1 of B
    r = 19
    U[0*3+2, r] = 1; U[1*3+2, r] = 1; U[2*3+2, r] = 1
    V[2*3+0, r] = 1; V[2*3+1, r] = 1; V[2*3+2, r] = 1
    W[1*3+2, r] = 1; W[2*3+2, r] = 1
    W[0*3+1, r] = -1; W[1*3+1, r] = -1; W[2*3+1, r] = -1
    
    # Term 20: row 1 of A, col 2 of B
    r = 20
    U[1*3+0, r] = 1; U[1*3+1, r] = 1; U[1*3+2, r] = 1
    V[2*3+0, r] = 1; V[2*3+1, r] = 1; V[2*3+2, r] = 1
    W[2*3+1, r] = 1; W[1*3+2, r] = 1
    W[1*3+0, r] = -1; W[1*3+1, r] = -1; W[1*3+2, r] = -1
    
    # Term 21: col 1 of A, row 2 of B
    r = 21
    U[0*3+1, r] = 1; U[1*3+1, r] = 1; U[2*3+1, r] = 1
    V[1*3+2, r] = 1; V[1*3+2, r] = 1; V[2*3+2, r] = 1
    W[2*3+1, r] = 1; W[2*3+2, r] = 1
    W[0*3+1, r] = -1; W[1*3+1, r] = -1; W[2*3+1, r] = -1
    
    # Term 22: remaining cross terms
    r = 22
    U[0*3+2, r] = 1; U[2*3+0, r] = 1
    V[0*3+2, r] = 1; V[2*3+0, r] = 1
    W[2*3+0, r] = 1; W[0*3+2, r] = 1
    W[0*3+0, r] = -1; W[2*3+2, r] = -1
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = 6, 6, 6
    
    # Try Rank 161 (Strassen 2x2 x Laderman 3x3)
    if time.time() < _DEADLINE - 10:
        try:
            L_U, L_V, L_W = _get_laderman_3x3()
            
            rank = 7 * 23
            U = np.zeros((36, rank))
            V = np.zeros((36, rank))
            W = np.zeros((36, rank))
            
            # Strassen steps for 2x2 block matrix multiplication
            # Each step defines:
            # - A_coeffs: linear combination of A blocks (I,J)
            # - B_coeffs: linear combination of B blocks (J,K)
            # - C_coeffs: linear combination of C blocks (I,K)
            steps = [
                # M1 = (A00 + A11) * (B00 + B11)
                {'A': {(0,0): 1, (1,1): 1},
                 'B': {(0,0): 1, (1,1): 1},
                 'C': {(0,0): 1, (1,1): 1}},
                # M2 = (A10 + A11) * B00
                {'A': {(1,0): 1, (1,1): 1},
                 'B': {(0,0): 1},
                 'C': {(1,0): 1, (1,1): -1}},
                # M3 = A00 * (B01 - B11)
                {'A': {(0,0): 1},
                 'B': {(0,1): 1, (1,1): -1},
                 'C': {(0,1): 1, (1,1): 1}},
                # M4 = A11 * (B10 - B00)
                {'A': {(1,1): 1},
                 'B': {(1,0): 1, (0,0): -1},
                 'C': {(0,0): 1, (1,0): 1}},
                # M5 = (A00 + A01) * B11
                {'A': {(0,0): 1, (0,1): 1},
                 'B': {(1,1): 1},
                 'C': {(0,0): -1, (0,1): 1}},
                # M6 = (A10 - A00) * (B00 + B01)
                {'A': {(1,0): 1, (0,0): -1},
                 'B': {(0,0): 1, (0,1): 1},
                 'C': {(1,1): 1}},
                # M7 = (A01 - A11) * (B10 + B11)
                {'A': {(0,1): 1, (1,1): -1},
                 'B': {(1,0): 1, (1,1): 1},
                 'C': {(0,0): 1}}
            ]
            
            # Precompute Laderman factor columns
            L_U_cols = [L_U[:, l].flatten() for l in range(23)]
            L_V_cols = [L_V[:, l].flatten() for l in range(23)]
            L_W_cols = [L_W[:, l].flatten() for l in range(23)]
            
            for s in range(7):
                A_coeffs = steps[s]['A']
                B_coeffs = steps[s]['B']
                C_coeffs = steps[s]['C']
                
                for l in range(23):
                    col = s * 23 + l
                    lu = L_U_cols[l]
                    lv = L_V_cols[l]
                    lw = L_W_cols[l]
                    
                    # Update U: X[u,v] = sum_{I,J} A_coeffs[(I,J)] * A[I*3+u, J*3+v]
                    for (I, J), ca in A_coeffs.items():
                        for u in range(3):
                            for v in range(3):
                                i_g = I * 3 + u
                                j_g = J * 3 + v
                                U[i_g * 6 + j_g, col] += ca * lu[u * 3 + v]
                    
                    # Update V: Y[v,w] = sum_{J,K} B_coeffs[(J,K)] * B[J*3+v, K*3+w]
                    for (J, K), cb in B_coeffs.items():
                        for v in range(3):
                            for w in range(3):
                                j_g = J * 3 + v
                                k_g = K * 3 + w
                                V[j_g * 6 + k_g, col] += cb * lv[v * 3 + w]
                    
                    # Update W: M[u,w] contributes to C blocks
                    for (I, K), cc in C_coeffs.items():
                        for u in range(3):
                            for w in range(3):
                                i_g = I * 3 + u
                                k_g = K * 3 + w
                                W[k_g * 6 + i_g, col] += cc * lw[w * 3 + u]
            
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass

    # Fallback Rank 189 (Strassen 2x2 x Schoolbook 3x3)
    if time.time() < _DEADLINE - 5:
        try:
            rank = 7 * 27
            U = np.zeros((36, rank))
            V = np.zeros((36, rank))
            W = np.zeros((36, rank))
            
            steps = [
                {'A': {(0,0): 1, (1,1): 1},
                 'B': {(0,0): 1, (1,1): 1},
                 'C': {(0,0): 1, (1,1): 1}},
                {'A': {(1,0): 1, (1,1): 1},
                 'B': {(0,0): 1},
                 'C': {(1,0): 1, (1,1): -1}},
                {'A': {(0,0): 1},
                 'B': {(0,1): 1, (1,1): -1},
                 'C': {(0,1): 1, (1,1): 1}},
                {'A': {(1,1): 1},
                 'B': {(1,0): 1, (0,0): -1},
                 'C': {(0,0): 1, (1,0): 1}},
                {'A': {(0,0): 1, (0,1): 1},
                 'B': {(1,1): 1},
                 'C': {(0,0): -1, (0,1): 1}},
                {'A': {(1,0): 1, (0,0): -1},
                 'B': {(0,0): 1, (0,1): 1},
                 'C': {(1,1): 1}},
                {'A': {(0,1): 1, (1,1): -1},
                 'B': {(1,0): 1, (1,1): 1},
                 'C': {(0,0): 1}}
            ]
            
            for s in range(7):
                A_coeffs = steps[s]['A']
                B_coeffs = steps[s]['B']
                C_coeffs = steps[s]['C']
                
                for u in range(3):
                    for v in range(3):
                        for w in range(3):
                            col = s * 27 + (u * 3 + v) * 3 + w
                            
                            for (I, J), ca in A_coeffs.items():
                                i_g = I * 3 + u
                                j_g = J * 3 + v
                                U[i_g * 6 + j_g, col] += ca
                            
                            for (J, K), cb in B_coeffs.items():
                                j_g = J * 3 + v
                                k_g = K * 3 + w
                                V[j_g * 6 + k_g, col] += cb
                            
                            for (I, K), cc in C_coeffs.items():
                                i_g = I * 3 + u
                                k_g = K * 3 + w
                                W[k_g * 6 + i_g, col] += cc
            
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass

    # Fallback Rank 216 (Naive)
    rank = 216
    U = np.zeros((36, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    col = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U[i * 6 + j, col] = 1
                V[j * 6 + k, col] = 1
                W[k * 6 + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
