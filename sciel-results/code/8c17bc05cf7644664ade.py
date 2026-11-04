import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def _build_strassen_222():
    """
    Returns U, V, W for 2x2 by 2x2 Strassen decomposition.
    U: (4, 7)
    V: (4, 7)
    W: (4, 7)
    Indices:
    U: A[0,0], A[0,1], A[1,0], A[1,1] -> 0,1,2,3
    V: B[0,0], B[0,1], B[1,0], B[1,1] -> 0,1,2,3
    W: C[0,0], C[0,1], C[1,0], C[1,1] -> 0,1,2,3 (k*n+i mapping? 
       Wait, W index is k*n+i.
       For 2x2, n=2, p=2.
       C[0,0] -> k=0, i=0 -> 0*2+0=0.
       C[0,1] -> k=1, i=0 -> 1*2+0=2.
       C[1,0] -> k=0, i=1 -> 0*2+1=1.
       C[1,1] -> k=1, i=1 -> 1*2+1=3.
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C00 += M1, C11 += M1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C10 += M2, C11 -= M2
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    # C01 += M3, C11 += M3
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    # C00 += M4, C10 += M4
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C00 -= M5, C01 += M5
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C11 += M6
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C00 += M7, C11 += M7
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_naive_221():
    """
    2x2 by 2x1. Rank 4.
    U: (4, 4)
    V: (2, 4)
    W: (2, 4)
    C[0,0] = A00 B00 + A01 B10
    C[1,0] = A10 B00 + A11 B10
    """
    U = np.zeros((4, 4), dtype=float)
    V = np.zeros((2, 4), dtype=float)
    W = np.zeros((2, 4), dtype=float)
    
    # Term 0: A00 B00 -> C00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 1: A01 B10 -> C00
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    # Term 2: A10 B00 -> C10
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    # Term 3: A11 B10 -> C10
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def _build_naive_217():
    """
    2x1 by 1x7. Rank 14.
    U: (2, 14)
    V: (7, 14)
    W: (14, 14)
    """
    U = np.zeros((2, 14), dtype=float)
    V = np.zeros((7, 14), dtype=float)
    W = np.zeros((14, 14), dtype=float)
    
    idx = 0
    for i in range(2):
        for k in range(7):
            # A[i,0] B[0,k] -> C[i,k]
            U[i, idx] = 1
            V[k, idx] = 1
            # W index for C[i,k] is k*n+i = k*2+i
            W[k*2+i, idx] = 1
            idx += 1
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # We construct a rank 39 decomposition.
    # Split A into A1 (2x2) and A2 (2x1).
    # Split B into B1 (2x7) and B2 (1x7).
    # B1 is split into three 2x2 blocks and one 2x1 block.
    
    # Total Rank:
    # A1 B1: 3 * 7 (Strassen 2x2x2) + 4 (Naive 2x2x1) = 25
    # A2 B2: 14 (Naive 2x1x7)
    # Total = 39
    
    rank = 39
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col_offset = 0
    
    # Block 1: A1 (rows 0,1 of A; cols 0,1) * B1_part1 (rows 0,1 of B; cols 0,1)
    U1, V1, W1 = _build_strassen_222()
    # Map indices
    # U1 indices 0..3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
    # In global U: A[i,j] is at i*m+j.
    # A[0,0] -> 0. A[0,1] -> 1. A[1,0] -> 3. A[1,1] -> 4.
    # Wait, m=3.
    # A[0,0] -> 0*3+0 = 0.
    # A[0,1] -> 0*3+1 = 1.
    # A[1,0] -> 1*3+0 = 3.
    # A[1,1] -> 1*3+1 = 4.
    
    # V1 indices 0..3 correspond to B[0,0], B[0,1], B[1,0], B[1,1]
    # In global V: B[j,k] is at j*p+k.
    # B[0,0] -> 0. B[0,1] -> 1. B[1,0] -> 7. B[1,1] -> 8.
    
    # W1 indices 0..3 correspond to C[0,0], C[0,1], C[1,0], C[1,1]
    # In global W: C[i,k] is at k*n+i.
    # C[0,0] -> 0. C[0,1] -> 1*2+0=2. C[1,0] -> 0*2+1=1. C[1,1] -> 1*2+1=3.
    
    u_map = [0, 1, 3, 4]
    v_map = [0, 1, 7, 8]
    w_map = [0, 2, 1, 3]
    
    for r in range(7):
        for u_idx in range(4):
            U[u_map[u_idx], col_offset + r] += U1[u_idx, r]
        for v_idx in range(4):
            V[v_map[v_idx], col_offset + r] += V1[v_idx, r]
        for w_idx in range(4):
            W[w_map[w_idx], col_offset + r] += W1[w_idx, r]
    col_offset += 7
    
    # Block 2: A1 * B1_part2 (rows 0,1 of B; cols 2,3)
    # B[0,2] -> 2. B[0,3] -> 3. B[1,2] -> 9. B[1,3] -> 10.
    # C[0,2] -> 2*2+0=4. C[0,3] -> 3*2+0=6. C[1,2] -> 2*2+1=5. C[1,3] -> 3*2+1=7.
    u_map2 = [0, 1, 3, 4]
    v_map2 = [2, 3, 9, 10]
    w_map2 = [4, 6, 5, 7]
    
    for r in range(7):
        for u_idx in range(4):
            U[u_map2[u_idx], col_offset + r] += U1[u_idx, r]
        for v_idx in range(4):
            V[v_map2[v_idx], col_offset + r] += V1[v_idx, r]
        for w_idx in range(4):
            W[w_map2[w_idx], col_offset + r] += W1[w_idx, r]
    col_offset += 7
    
    # Block 3: A1 * B1_part3 (rows 0,1 of B; cols 4,5)
    # B[0,4] -> 4. B[0,5] -> 5. B[1,4] -> 11. B[1,5] -> 12.
    # C[0,4] -> 4*2+0=8. C[0,5] -> 5*2+0=10. C[1,4] -> 4*2+1=9. C[1,5] -> 5*2+1=11.
    u_map3 = [0, 1, 3, 4]
    v_map3 = [4, 5, 11, 12]
    w_map3 = [8, 10, 9, 11]
    
    for r in range(7):
        for u_idx in range(4):
            U[u_map3[u_idx], col_offset + r] += U1[u_idx, r]
        for v_idx in range(4):
            V[v_map3[v_idx], col_offset + r] += V1[v_idx, r]
        for w_idx in range(4):
            W[w_map3[w_idx], col_offset + r] += W1[w_idx, r]
    col_offset += 7
    
    # Block 4: A1 * B1_part4 (rows 0,1 of B; col 6)
    # 2x2 by 2x1.
    U2, V2, W2 = _build_naive_221()
    # U2 indices 0..3: A[0,0], A[0,1], A[1,0], A[1,1]
    # V2 indices 0..1: B[0,6], B[1,6]
    # W2 indices 0..1: C[0,6], C[1,6]
    
    u_map4 = [0, 1, 3, 4]
    v_map4 = [6, 13] # B[0,6]=6, B[1,6]=13
    w_map4 = [12, 13] # C[0,6]=6*2+0=12, C[1,6]=6*2+1=13
    
    for r in range(4):
        for u_idx in range(4):
            U[u_map4[u_idx], col_offset + r] += U2[u_idx, r]
        for v_idx in range(2):
            V[v_map4[v_idx], col_offset + r] += V2[v_idx, r]
        for w_idx in range(2):
            W[w_map4[w_idx], col_offset + r] += W2[w_idx, r]
    col_offset += 4
    
    # Block 5: A2 * B2
    # A2 is A[0,2], A[1,2]. Indices 2, 5.
    # B2 is B[2,0]...B[2,6]. Indices 14...20.
    # C is C[0,0]...C[1,6].
    
    U3, V3, W3 = _build_naive_217()
    # U3 indices 0..1: A[0,2], A[1,2]
    # V3 indices 0..6: B[2,0]...B[2,6]
    # W3 indices 0..13: C[0,0]...C[1,6]
    
    u_map5 = [2, 5]
    v_map5 = [14, 15, 16, 17, 18, 19, 20]
    w_map5 = list(range(14))
    
    for r in range(14):
        for u_idx in range(2):
            U[u_map5[u_idx], col_offset + r] += U3[u_idx, r]
        for v_idx in range(7):
            V[v_map5[v_idx], col_offset + r] += V3[v_idx, r]
        for w_idx in range(14):
            W[w_map5[w_idx], col_offset + r] += W3[w_idx, r]
    col_offset += 14
    
    # Convert to list of lists of floats (or ints if possible)
    # The problem allows integer/half-integer. My construction uses integers.
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
