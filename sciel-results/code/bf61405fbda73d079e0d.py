# EVOLVE-BLOCK-START
import numpy as np
import time

TIME_LIMIT = 145.0
START_TIME = time.time()

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def construct():
    n, m, p = 3, 4, 4
    
    # Strassen's algorithm for 2x2x2 matrix multiplication (rank 7)
    def strassen_2x2x2():
        """
        Returns U (4x7), V (4x7), W (4x7) for 2x2x2 matrix mult
        U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1   # a=A[0,0], d=A[1,1]
        V[0, 0] = 1; V[1, 0] = 1   # e=B[0,0], h=B[0,1]
        W[0, 0] = 1                 # C[0,0] += M1
        W[3, 0] = 1                 # C[1,1] += M1 (k=1,i=1 -> k*2+i=3)
        
        # M2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1   # c=A[1,0], d=A[1,1]
        V[0, 1] = 1                 # e=B[0,0]
        W[1, 1] = 1                 # C[1,0] += M2 (k=0,i=1 -> k*2+i=1)
        W[3, 1] = -1                # C[1,1] -= M2
        
        # M3 = a(f-h)
        U[0, 2] = 1                 # a=A[0,0]
        V[1, 2] = 1; V[3, 2] = -1  # f=B[0,1], -h=B[1,1]
        W[2, 2] = 1                 # C[0,1] += M3 (k=1,i=0 -> k*2+i=2)
        W[3, 2] = 1                 # C[1,1] += M3
        
        # M4 = d(g-e)
        U[3, 3] = 1                 # d=A[1,1]
        V[2, 3] = 1; V[0, 3] = -1  # g=B[1,0], -e=B[0,0]
        W[0, 3] = 1                 # C[0,0] += M4
        W[1, 3] = 1                 # C[1,0] += M4
        
        # M5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1   # a=A[0,0], b=A[0,1]
        V[3, 4] = 1                 # h=B[1,1]
        W[0, 4] = -1                # C[0,0] -= M5
        W[2, 4] = 1                 # C[0,1] += M5
        
        # M6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1  # c=A[1,0], -a=A[0,0]
        V[0, 5] = 1; V[1, 5] = 1   # e=B[0,0], f=B[0,1]
        W[3, 5] = 1                 # C[1,1] += M6
        
        # M7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1  # b=A[0,1], -d=A[1,1]
        V[2, 6] = 1; V[3, 6] = 1   # g=B[1,0], h=B[1,1]
        W[0, 6] = 1                 # C[0,0] += M7
        
        return U, V, W
    
    # Verify Strassen 2x2x2
    def verify_2x2x2():
        U, V, W = strassen_2x2x2()
        U_r = U.reshape(2, 2, 7)
        V_r = V.reshape(2, 2, 7)
        W_r = W.reshape(2, 2, 7)
        S = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(S, 1.0, atol=1e-9)
    
    # Build 3x4x4 decomposition with rank 44
    # Rows 0,1: 4 Strassen 2x2x2 blocks (rank 7 each = 28)
    # Row 2: trivial 1x4x4 (rank 16)
    
    total_rank = 44
    U_total = np.zeros((n * m, total_rank))
    V_total = np.zeros((m * p, total_rank))
    W_total = np.zeros((p * n, total_rank))
    
    U_str, V_str, W_str = strassen_2x2x2()
    
    offset = 0
    for js in range(2):
        for ks in range(2):
            for r in range(7):
                # U: A[i, j] where i in {0,1}, j in {js*2, js*2+1}
                for i in range(2):
                    for j_local in range(2):
                        j = js * 2 + j_local
                        U_total[i * m + j, offset + r] = U_str[i * 2 + j_local, r]
                
                # V: B[j, k] where j in {js*2, js*2+1}, k in {ks*2, ks*2+1}
                for j_local in range(2):
                    for k_local in range(2):
                        j = js * 2 + j_local
                        k = ks * 2 + k_local
                        V_total[j * p + k, offset + r] = V_str[j_local * 2 + k_local, r]
                
                # W: C[i, k] where i in {0,1}, k in {ks*2, ks*2+1}
                # W convention: W[k*n+i, r]
                for k_local in range(2):
                    for i in range(2):
                        k = ks * 2 + k_local
                        W_total[k * n + i, offset + r] = W_str[k_local * 2 + i, r]
            
            offset += 7
    
    # Row 2: trivial 1x4x4, rank 16
    for j in range(4):
        for k in range(4):
            U_total[2 * m + j, offset] = 1.0
            V_total[j * p + k, offset] = 1.0
            W_total[k * n + 2, offset] = 1.0
            offset += 1
    
    assert offset == 44, f"Expected 44, got {offset}"
    
    # Verify
    def verify(U, V, W, tol=1e-9):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        S = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(S, 1.0, atol=tol)
    
    if verify(U_total, V_total, W_total):
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    
    # Fallback: schoolbook
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=np.float64)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=np.float64)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, idx] = 1.0
                V_sb[j * p + k, idx] = 1.0
                W_sb[k * n + i, idx] = 1.0
                idx += 1
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
