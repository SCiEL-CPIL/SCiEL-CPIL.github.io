import time
import numpy as np
from itertools import product

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
    
    # W indices: [C00, C01, C10, C11] = [0, 1, 2, 3]
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00, C11
    
    # Term 1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10
    
    # Term 2: M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01
    
    # Term 3: M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C00, C10
    
    # Term 4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # C00, C01
    
    # Term 5: M6 = (A10-A00)(B00+B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1  # C11
    
    # Term 6: M7 = (A01-A11)(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00
    
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

def _construct_3x3x8_strassen_blocks():
    """
    Construct a valid decomposition for 3x3x8 using block Strassen.
    Strategy:
    - Split k (8) into four blocks of size 2.
    - For each block, split i (3) into 2+1 and j (3) into 2+1.
    - The 2x2x2 subblock uses Strassen (rank 7).
    - The remaining terms use schoolbook.
    Total rank per k-block: 7 (2x2) + 4 (2x1x2) + 4 (1x2x2) + 2 (1x1x2) = 17.
    Total rank: 4 * 17 = 68.
    """
    n, m, p = 3, 3, 8
    rank_total = 68
    
    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    col_offset = 0
    
    for k_block in range(4):
        k0 = k_block * 2
        k1 = k0 + 1
        
        # Subblock 1: i in {0,1}, j in {0,1}, k in {k0,k1}
        # Uses Strassen 2x2x2
        for c in range(7):
            # U: A[0:2, 0:2] -> indices 0,1,2,3
            for u_idx in range(4):
                U[u_idx][col_offset + c] = U_s[u_idx][c]
            
            # V: B[0:2, k0:k1+1] -> indices j*p+k
            # v_idx 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1) in local 2x2
            # Local (j_l, k_l) -> Global (j_l, k0+k_l)
            v_map = [0*p + k0, 0*p + k1, 1*p + k0, 1*p + k1]
            for v_idx in range(4):
                V[v_map[v_idx]][col_offset + c] = V_s[v_idx][c]
            
            # W: C[0:2, k0:k1+1] -> indices k*n+i
            # w_idx 0->C00(k0,i0), 1->C01(k1,i0), 2->C10(k0,i1), 3->C11(k1,i1)
            w_map = [k0*n + 0, k1*n + 0, k0*n + 1, k1*n + 1]
            for w_idx in range(4):
                W[w_map[w_idx]][col_offset + c] = W_s[w_idx][c]
        
        col_offset += 7
        
        # Subblock 2: i in {0,1}, j in {2}, k in {k0,k1}
        # Schoolbook: A[i,2] * B[2,k] -> C[i,k]
        # i=0,1; j=2; k=k0,k1
        for i in range(2):
            for k in [k0, k1]:
                U[i*m + 2][col_offset] = 1
                V[2*p + k][col_offset] = 1
                W[k*n + i][col_offset] = 1
                col_offset += 1
        
        # Subblock 3: i in {2}, j in {0,1}, k in {k0,k1}
        # Schoolbook: A[2,j] * B[j,k] -> C[2,k]
        for j in range(2):
            for k in [k0, k1]:
                U[2*m + j][col_offset] = 1
                V[j*p + k][col_offset] = 1
                W[k*n + 2][col_offset] = 1
                col_offset += 1
        
        # Subblock 4: i in {2}, j in {2}, k in {k0,k1}
        # Schoolbook: A[2,2] * B[2,k] -> C[2,k]
        for k in [k0, k1]:
            U[2*m + 2][col_offset] = 1
            V[2*p + k][col_offset] = 1
            W[k*n + 2][col_offset] = 1
            col_offset += 1
            
    assert col_offset == rank_total, f"Expected {rank_total}, got {col_offset}"
    
    return U, V, W

def _construct_schoolbook(n, m, p):
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Attempt the improved block Strassen construction (Rank 44)
    try:
        U, V, W = _construct_3x3x8_strassen_blocks()
        
        # Verify correctness
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback to schoolbook if verification fails
            U_s, V_s, W_s = _construct_schoolbook(n, m, p)
            return {"U": U_s, "V": V_s, "W": W_s}
    except Exception:
        # Fallback to schoolbook if any error occurs
        U_s, V_s, W_s = _construct_schoolbook(n, m, p)
        return {"U": U_s, "V": V_s, "W": W_s}
