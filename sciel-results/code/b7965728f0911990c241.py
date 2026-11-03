import time
import sys

# EVOLVE-BLOCK-START

def _strassen_2x2_2x2_UVW():
    """
    Return (U, V, W) for the 2x2x2 Strassen decomposition.
    U: 4x7 (A has 4 elements, indexed by (i,j) -> i*2+j)
    V: 4x7 (B has 4 elements, indexed by (j,k) -> j*2+k)
    W: 4x7 (C has 4 elements, indexed by (k,i) -> k*2+i per trace convention)
    
    Trace convention: W[k*n+i, r] where n=2, so W[k*2+i, r]
    A is 2x2: A[i,j], index = i*2+j
    B is 2x2: B[j,k], index = j*2+k
    C is 2x2: C[i,k], but stored as W[k*2+i]
    
    Strassen:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1+M4-M5+M7  -> C[0,0], W index = 0*2+0 = 0
    C12 = M3+M5         -> C[0,1], W index = 1*2+0 = 2
    C21 = M2+M4         -> C[1,0], W index = 0*2+1 = 1
    C22 = M1-M2+M3+M6  -> C[1,1], W index = 1*2+1 = 3
    """
    U = [[0]*7 for _ in range(4)]  # 4 A-elements
    V = [[0]*7 for _ in range(4)]  # 4 B-elements
    W = [[0]*7 for _ in range(4)]  # 4 C-elements (trace convention: k*2+i)
    
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C indices (trace): C[0,0]->W[0], C[1,0]->W[1], C[0,1]->W[2], C[1,1]->W[3]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1  # A11
    U[3][0] = 1  # A22
    V[0][0] = 1  # B11
    V[3][0] = 1  # B22
    W[0][0] = 1  # C11
    W[3][0] = 1  # C22
    
    # M2 = (A21+A22)B11
    U[2][1] = 1  # A21
    U[3][1] = 1  # A22
    V[0][1] = 1  # B11
    W[1][1] = 1  # C21
    W[3][1] = -1 # C22
    
    # M3 = A11(B12-B22)
    U[0][2] = 1  # A11
    V[1][2] = 1  # B12
    V[3][2] = -1 # B22
    W[2][2] = 1  # C12
    W[3][2] = 1  # C22
    
    # M4 = A22(B21-B11)
    U[3][3] = 1  # A22
    V[2][3] = 1  # B21
    V[0][3] = -1 # B11
    W[0][3] = 1  # C11
    W[1][3] = 1  # C21
    
    # M5 = (A11+A12)B22
    U[0][4] = 1  # A11
    U[1][4] = 1  # A12
    V[3][4] = 1  # B22
    W[0][4] = -1 # C11
    W[2][4] = 1  # C12
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1  # A21
    U[0][5] = -1 # A11
    V[0][5] = 1  # B11
    V[1][5] = 1  # B12
    W[3][5] = 1  # C22
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1  # A12
    U[3][6] = -1 # A22
    V[2][6] = 1  # B21
    V[3][6] = 1  # B22
    W[0][6] = 1  # C11
    W[2][6] = 1  # C12
    
    return U, V, W

def _trivial_2x2x1_UVW():
    """
    Return (U, V, W) for trivial 2x2x1 decomposition.
    A is 2x2 (4 elements), B is 2x1 (2 elements), C is 2x1 (2 elements).
    C[i,0] = sum_j A[i,j] * B[j,0]
    Trace convention for C: k*n+i where k=0, n=2, so C[i,0] -> W[i*1+0]... 
    Wait, for p=1, W has p*n = 1*2 = 2 entries. W[k*n+i] = W[0*2+i] = W[i].
    So C[i,0] maps to W[i].
    
    U: 4x4 (A elements)
    V: 2x4 (B elements: B[0,0]=0, B[1,0]=1)
    W: 2x4 (C elements: C[0,0]->W[0], C[1,0]->W[1])
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    for i in range(2):
        for j in range(2):
            t = i*2 + j
            U[i*2+j][t] = 1  # A[i,j]
            V[j][t] = 1       # B[j,0]
            W[i][t] = 1       # C[i,0]
    
    return U, V, W

def _combine(U_list, V_list, W_list):
    """
    Combine multiple (U,V,W) triples by concatenating along the rank dimension.
    Each U has shape (n*m, r_i), V has shape (m*p, r_i), W has shape (p*n, r_i).
    Result: U has shape (n*m, sum(r_i)), etc.
    """
    if not U_list:
        return [], [], []
    
    total_rank = sum(U.shape for U in U_list) if hasattr(U_list[0], 'shape') else sum(len(U[0]) for U in U_list)
    
    # Determine dimensions from first element
    num_u_rows = len(U_list[0])
    num_v_rows = len(V_list[0])
    num_w_rows = len(W_list[0])
    
    U_out = [[0]*total_rank for _ in range(num_u_rows)]
    V_out = [[0]*total_rank for _ in range(num_v_rows)]
    W_out = [[0]*total_rank for _ in range(num_w_rows)]
    
    offset = 0
    for U, V, W in zip(U_list, V_list, W_list):
        r = len(U[0])
        for i in range(num_u_rows):
            for j in range(r):
                U_out[i][offset+j] = U[i][j]
        for i in range(num_v_rows):
            for j in range(r):
                V_out[i][offset+j] = V[i][j]
        for i in range(num_w_rows):
            for j in range(r):
                W_out[i][offset+j] = W[i][j]
        offset += r
    
    return U_out, V_out, W_out

def _block_2x2x7_UVW():
    """
    Build 2x2x7 decomposition by splitting the 7 columns into 2+2+2+1.
    Each 2x2x2 uses Strassen (rank 7), the 2x2x1 uses trivial (rank 4).
    Total rank: 3*7 + 4 = 25.
    
    U: 4x25 (A is 2x2)
    V: 14x25 (B is 2x7)
    W: 14x25 (C is 2x7, trace convention: k*n+i = k*2+i, so 7*2=14 entries)
    """
    U_parts = []
    V_parts = []
    W_parts = []
    
    # For each column block of B, we have a 2x2x2 or 2x2x1 sub-problem
    # B is 2x7: columns 0-1, 2-3, 4-5, 6
    
    # Block 1: columns 0-1 (2x2x2, Strassen)
    U1, V1, W1 = _strassen_2x2_2x2_UVW()
    # V1 is 4x7 for B[0:2, 0:2], we need to place it in the right position of 2x7
    # B index: j*7+k where j=0,1 and k=0,...,6
    # For columns 0-1: B[j, 0] and B[j, 1] -> indices j*7+0, j*7+1
    V1_full = [[0]*7 for _ in range(14)]
    for j in range(2):
        for k in range(2):
            v_idx = j*7 + k
            V1_full[v_idx] = V1[j*2+k]
    # W1 is for C[0:2, 0:2], trace convention: k*2+i
    # C[j, k] where j=0,1 (row of C, which is i in original), k=0,1 (column of C, which is k)
    # W index: k*n+i = k*2+i
    W1_full = [[0]*7 for _ in range(14)]
    for k in range(2):
        for i in range(2):
            w_idx = k*2 + i
            W1_full[w_idx] = W1[k*2+i]
    
    U_parts.append(U1)
    V_parts.append(V1_full)
    W_parts.append(W1_full)
    
    # Block 2: columns 2-3 (2x2x2, Strassen)
    U2, V2, W2 = _strassen_2x2_2x2_UVW()
    V2_full = [[0]*7 for _ in range(14)]
    for j in range(2):
        for k in range(2):
            v_idx = j*7 + (2+k)
            V2_full[v_idx] = V2[j*2+k]
    W2_full = [[0]*7 for _ in range(14)]
    for k in range(2):
        for i in range(2):
            w_idx = (2+k)*2 + i
            W2_full[w_idx] = W2[k*2+i]
    
    U_parts.append(U2)
    V_parts.append(V2_full)
    W_parts.append(W2_full)
    
    # Block 3: columns 4-5 (2x2x2, Strassen)
    U3, V3, W3 = _strassen_2x2_2x2_UVW()
    V3_full = [[0]*7 for _ in range(14)]
    for j in range(2):
        for k in range(2):
            v_idx = j*7 + (4+k)
            V3_full[v_idx] = V3[j*2+k]
    W3_full = [[0]*7 for _ in range(14)]
    for k in range(2):
        for i in range(2):
            w_idx = (4+k)*2 + i
            W3_full[w_idx] = W3[k*2+i]
    
    U_parts.append(U3)
    V_parts.append(V3_full)
    W_parts.append(W3_full)
    
    # Block 4: column 6 (2x2x1, trivial)
    U4, V4, W4 = _trivial_2x2x1_UVW()
    V4_full = [[0]*4 for _ in range(14)]
    for j in range(2):
        v_idx = j*7 + 6
        V4_full[v_idx] = V4[j]
    W4_full = [[0]*4 for _ in range(14)]
    for k in range(1):
        for i in range(2):
            w_idx = (6+k)*2 + i
            W4_full[w_idx] = W4[i]
    
    U_parts.append(U4)
    V_parts.append(V4_full)
    W_parts.append(W4_full)
    
    return _combine(U_parts, V_parts, W_parts)

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = 4, 4, 7
    
    # Build 4x4x7 by splitting A's rows into 2+2
    # For each half of A (2 rows), we have a 2x4x7 sub-problem
    # 2x4x7: split A's columns into 2+2, giving two 2x2x7 sub-problems
    
    # Get the 2x2x7 decomposition
    U_2x2x7, V_2x2x7, W_2x2x7 = _block_2x2x7_UVW()
    # U_2x2x7: 4x25 (A is 2x2)
    # V_2x2x7: 14x25 (B is 2x7)
    # W_2x2x7: 14x25 (C is 2x7)
    
    # Now build 4x4x7:
    # A is 4x4, split into top 2 rows (A[0:2,:]) and bottom 2 rows (A[2:4,:])
    # For top half: A_top is 2x4, B is 4x7
    #   Split B into top 2 rows (B[0:2,:]) and bottom 2 rows (B[2:4,:])
    #   C_top = A_top[:, 0:2] * B[0:2,:] + A_top[:, 2:4] * B[2:4,:]
    #   Each term is 2x2x7
    # For bottom half: similar
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Top half of A (rows 0-1), left half of B (cols 0-1 of B's row dimension, i.e., B[0:2,:])
    # A_top_left = A[0:2, 0:2], B_top = B[0:2, :]
    # This is a 2x2x7 problem
    U1 = [[0]*25 for _ in range(16)]  # A is 4x4, 16 elements
    V1 = [[0]*25 for _ in range(28)]  # B is 4x7, 28 elements
    W1 = [[0]*25 for _ in range(28)]  # C is 4x7, 28 elements (trace: k*4+i)
    
    for i in range(2):
        for j in range(2):
            u_idx = i*4 + j  # A[i,j]
            U1[u_idx] = U_2x2x7[i*2+j]
    
    for j in range(2):
        for k in range(7):
            v_idx = j*7 + k  # B[j,k]
            V1[v_idx] = V_2x2x7[j*7+k]
    
    for k in range(7):
        for i in range(2):
            w_idx = k*4 + i  # C[i,k] -> W[k*n+i] = W[k*4+i]
            W1[w_idx] = W_2x2x7[k*2+i]
    
    U_parts.append(U1)
    V_parts.append(V1)
    W_parts.append(W1)
    
    # Top half of A (rows 0-1), right half of B (B[2:4,:])
    U2 = [[0]*25 for _ in range(16)]
    V2 = [[0]*25 for _ in range(28)]
    W2 = [[0]*25 for _ in range(28)]
    
    for i in range(2):
        for j in range(2, 4):
            u_idx = i*4 + j  # A[i,j]
            U2[u_idx] = U_2x2x7[i*2+(j-2)]
    
    for j in range(2, 4):
        for k in range(7):
            v_idx = j*7 + k  # B[j,k]
            V2[v_idx] = V_2x2x7[(j-2)*7+k]
    
    for k in range(7):
        for i in range(2):
            w_idx = k*4 + i
            W2[w_idx] = W_2x2x7[k*2+i]
    
    U_parts.append(U2)
    V_parts.append(V2)
    W_parts.append(W2)
    
    # Bottom half of A (rows 2-3), left half of B (B[0:2,:])
    U3 = [[0]*25 for _ in range(16)]
    V3 = [[0]*25 for _ in range(28)]
    W3 = [[0]*25 for _ in range(28)]
    
    for i in range(2, 4):
        for j in range(2):
            u_idx = i*4 + j
            U3[u_idx] = U_2x2x7[(i-2)*2+j]
    
    for j in range(2):
        for k in range(7):
            v_idx = j*7 + k
            V3[v_idx] = V_2x2x7[j*7+k]
    
    for k in range(7):
        for i in range(2, 4):
            w_idx = k*4 + i
            W3[w_idx] = W_2x2x7[k*2+(i-2)]
    
    U_parts.append(U3)
    V_parts.append(V3)
    W_parts.append(W3)
    
    # Bottom half of A (rows 2-3), right half of B (B[2:4,:])
    U4 = [[0]*25 for _ in range(16)]
    V4 = [[0]*25 for _ in range(28)]
    W4 = [[0]*25 for _ in range(28)]
    
    for i in range(2, 4):
        for j in range(2, 4):
            u_idx = i*4 + j
            U4[u_idx] = U_2x2x7[(i-2)*2+(j-2)]
    
    for j in range(2, 4):
        for k in range(7):
            v_idx = j*7 + k
            V4[v_idx] = V_2x2x7[(j-2)*7+k]
    
    for k in range(7):
        for i in range(2, 4):
            w_idx = k*4 + i
            W4[w_idx] = W_2x2x7[k*2+(i-2)]
    
    U_parts.append(U4)
    V_parts.append(V4)
    W_parts.append(W4)
    
    U, V, W = _combine(U_parts, V_parts, W_parts)
    
    # Verify correctness with random test
    try:
        import numpy as np
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m))
        B_test = np.random.randint(-5, 5, (m, p))
        C_test = A_test @ B_test
        
        # Compute using decomposition
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a sum over all (i,j,k,r)
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for r in range(len(U_arr[0])):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_test, atol=1e-10):
            # Fallback to trivial if verification fails
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            t = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U[u_idx][t] = 1
                        V[v_idx][t] = 1
                        W[w_idx][t] = 1
                        t += 1
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
