"""Rewritten program for matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 110  # Leave 40s safety margin
    
    # Trivial decomposition of rank n*m*p
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {'U': U, 'V': V, 'W': W}
    best_rank = rank_trivial
    
    # Build target tensor for verification
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
    
    def verify(U_mat, V_mat, W_mat):
        """Verify exact decomposition using integer arithmetic."""
        U_arr = np.array(U_mat, dtype=np.int64)
        V_arr = np.array(V_mat, dtype=np.int64)
        W_arr = np.array(W_mat, dtype=np.int64)
        T_comp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.array_equal(T_comp, T_target)
    
    # Verify trivial decomposition
    assert verify(U, V, W), "Trivial decomposition failed verification!"
    
    # Attempt to reduce rank by replacing a 2x2x2 sub-block with Strassen decomposition
    # Sub-block: i in {0,1}, j in {0,1}, k in {0,1}
    # Indices:
    # a: i*5 + j -> 0,1, 5,6
    # b: j*7 + k -> 0,1, 7,8
    # c: k*3 + i -> 0,1, 3,4
    
    # The 8 trivial terms for this sub-block are at columns:
    # (0,0,0) -> col 0
    # (0,0,1) -> col 1
    # (0,1,0) -> col 5
    # (0,1,1) -> col 6
    # (1,0,0) -> col 14
    # (1,0,1) -> col 15
    # (1,1,0) -> col 19
    # (1,1,1) -> col 20
    
    # We will replace these 8 columns with 7 new columns (Strassen)
    # New rank = 105 - 8 + 7 = 104
    
    U_new = []
    V_new = []
    W_new = []
    
    # Keep all columns except the 8 identified
    # Identifying the 8 columns:
    trivial_cols = [0, 1, 5, 6, 14, 15, 19, 20]
    
    for col_idx in range(rank_trivial):
        if col_idx not in trivial_cols:
            U_new.append(U[col_idx])
            V_new.append(V[col_idx])
            W_new.append(W[col_idx])
            
    # Now add the 7 Strassen terms
    # Strassen terms for 2x2:
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Map A entries to U indices (a = i*5+j)
    # A00: a=0, A01: a=1, A10: a=5, A11: a=6
    # Map B entries to V indices (b = j*7+k)
    # B00: b=0, B01: b=1, B10: b=7, B11: b=8
    # Map C entries to W indices (c = k*3+i)
    # C00: c=0, C01: c=1, C10: c=3, C11: c=4
    
    # Strassen vectors:
    # M1: U: {0:1, 6:1}, V: {0:1, 8:1}, W: {0:1, 1:1, 3:1, 4:1} ? No, W is for output.
    # The tensor is sum U[a,r] V[b,r] W[c,r].
    # For M1, the term is (A00+A11)(B00+B11).
    # This contributes to C00, C01, C10, C11.
    # C00 += M1, C01 += M1, C10 += M1, C11 += M1.
    # So W[0, r]=1, W[1, r]=1, W[3, r]=1, W[4, r]=1.
    # U[0, r]=1, U[6, r]=1.
    # V[0, r]=1, V[8, r]=1.
    
    # M2: (A10+A11)B00
    # Contributes to C10, C11.
    # C10 += M2, C11 += M2.
    # U[5, r]=1, U[6, r]=1.
    # V[0, r]=1.
    # W[3, r]=1, W[4, r]=1.
    
    # M3: A00(B10-B11)
    # Contributes to C01, C11.
    # C01 += M3, C11 -= M3.
    # U[0, r]=1.
    # V[7, r]=1, V[8, r]=-1.
    # W[1, r]=1, W[4, r]=-1.
    
    # M4: A11(B01-B00)
    # Contributes to C00, C10.
    # C00 += M4, C10 += M4.
    # U[6, r]=1.
    # V[1, r]=1, V[0, r]=-1.
    # W[0, r]=1, W[3, r]=1.
    
    # M5: (A00+A01)B11
    # Contributes to C01, C11.
    # C01 += M5, C11 -= M5.
    # U[0, r]=1, U[1, r]=1.
    # V[8, r]=1.
    # W[1, r]=1, W[4, r]=-1.
    
    # M6: (A10-A00)(B00+B01)
    # Contributes to C10, C11.
    # C10 += M6, C11 += M6.
    # U[5, r]=1, U[0, r]=-1.
    # V[0, r]=1, V[1, r]=1.
    # W[3, r]=1, W[4, r]=1.
    
    # M7: (A01-A11)(B10+B11)
    # Contributes to C00, C11.
    # C00 += M7, C11 += M7.
    # U[1, r]=1, U[6, r]=-1.
    # V[7, r]=1, V[8, r]=1.
    # W[0, r]=1, W[4, r]=1.
    
    # Let's add these 7 terms
    def add_term(u_vec, v_vec, w_vec):
        U_new.append(u_vec)
        V_new.append(v_vec)
        W_new.append(w_vec)
        
    # Initialize vectors
    u1 = [0]*15; v1 = [0]*35; w1 = [0]*21
    u1[0]=1; u1[6]=1
    v1[0]=1; v1[8]=1
    w1[0]=1; w1[1]=1; w1[3]=1; w1[4]=1
    add_term(u1, v1, w1)
    
    u2 = [0]*15; v2 = [0]*35; w2 = [0]*21
    u2[5]=1; u2[6]=1
    v2[0]=1
    w2[3]=1; w2[4]=1
    add_term(u2, v2, w2)
    
    u3 = [0]*15; v3 = [0]*35; w3 = [0]*21
    u3[0]=1
    v3[7]=1; v3[8]=-1
    w3[1]=1; w3[4]=-1
    add_term(u3, v3, w3)
    
    u4 = [0]*15; v4 = [0]*35; w4 = [0]*21
    u4[6]=1
    v4[1]=1; v4[0]=-1
    w4[0]=1; w4[3]=1
    add_term(u4, v4, w4)
    
    u5 = [0]*15; v5 = [0]*35; w5 = [0]*21
    u5[0]=1; u5[1]=1
    v5[8]=1
    w5[1]=1; w5[4]=-1
    add_term(u5, v5, w5)
    
    u6 = [0]*15; v6 = [0]*35; w6 = [0]*21
    u6[5]=1; u6[0]=-1
    v6[0]=1; v6[1]=1
    w6[3]=1; w6[4]=1
    add_term(u6, v6, w6)
    
    u7 = [0]*15; v7 = [0]*35; w7 = [0]*21
    u7[1]=1; u7[6]=-1
    v7[7]=1; v7[8]=1
    w7[0]=1; w7[4]=1
    add_term(u7, v7, w7)
    
    # Verify new decomposition
    if verify(U_new, V_new, W_new):
        best = {'U': U_new, 'V': V_new, 'W': W_new}
        best_rank = len(U_new)
        
    return best
# EVOLVE-BLOCK-END
