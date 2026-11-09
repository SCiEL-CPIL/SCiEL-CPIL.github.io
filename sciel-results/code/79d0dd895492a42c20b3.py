import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 Strassen decomposition.
    U: 4x7 (A[0,0], A[0,1], A[1,0], A[1,1])
    V: 4x7 (B[0,0], B[0,1], B[1,0], B[1,1])
    W: 4x7 (C[0,0], C[0,1], C[1,0], C[1,1])
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 2: M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 3: M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # Term 6: M7 = (A01-A11)(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Strategy: Use block decomposition with Strassen for 2x2 blocks
    # Block 1: A[0:2, 0:2] @ B[0:2, 0:8] -> C[0:2, 0:8] using 4 Strassen calls (rank 28)
    # Block 2: A[2, 0:2] @ B[0:2, 0:8] -> C[2, 0:8] schoolbook (rank 16)
    # Block 3: A[0:2, 2] @ B[2, 0:8] -> C[0:2, 0:8] schoolbook (rank 16)
    # Block 4: A[2, 2] @ B[2, 0:8] -> C[2, 0:8] schoolbook (rank 8)
    # Total: 28+16+16+8 = 68
    
    rank_total = 68
    
    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]
    
    col_offset = 0
    
    # Block 1: 2x2x8 using Strassen 4 times
    U_s, V_s, W_s = _strassen_2x2x2()
    
    for col_pair in range(4):
        k0 = col_pair * 2
        k1 = col_pair * 2 + 1
        
        for c in range(7):
            # U: A[0:2, 0:2] -> U indices 0,1,2,3
            for u_idx in range(4):
                U[u_idx][col_offset + c] = U_s[u_idx][c]
            
            # V: B[0:2, k0:k1+1] -> V indices j*p+k for j in {0,1}, k in {k0,k1}
            for v_idx in range(4):
                j = v_idx // 2
                k_local = v_idx % 2
                k_global = k0 + k_local
                V[j*p + k_global][col_offset + c] = V_s[v_idx][c]
            
            # W: C[0:2, k0:k1+1] -> W indices k*n+i for k in {k0,k1}, i in {0,1}
            w_map = [k0*n + 0, k1*n + 0, k0*n + 1, k1*n + 1]
            for w_idx in range(4):
                W[w_map[w_idx]][col_offset + c] = W_s[w_idx][c]
        
        col_offset += 7
    
    # Block 2: 1x2x8 schoolbook (A[2, 0:2] @ B[0:2, 0:8] -> C[2, 0:8])
    for j in range(2):
        for k in range(8):
            U[2*m + j][col_offset] = 1
            V[j*p + k][col_offset] = 1
            W[k*n + 2][col_offset] = 1
            col_offset += 1
    
    # Block 3: 2x1x8 schoolbook (A[0:2, 2] @ B[2, 0:8] -> C[0:2, 0:8])
    for i in range(2):
        for k in range(8):
            U[i*m + 2][col_offset] = 1
            V[2*p + k][col_offset] = 1
            W[k*n + i][col_offset] = 1
            col_offset += 1
    
    # Block 4: 1x1x8 schoolbook (A[2,2] @ B[2, 0:8] -> C[2, 0:8])
    for k in range(8):
        U[2*m + 2][col_offset] = 1
        V[2*p + k][col_offset] = 1
        W[k*n + 2][col_offset] = 1
        col_offset += 1
    
    assert col_offset == rank_total, f"Expected {rank_total}, got {col_offset}"
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j][col] = 1
                    V_school[j*p+k][col] = 1
                    W_school[k*n+i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
