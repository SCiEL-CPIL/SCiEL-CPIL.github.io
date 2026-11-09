import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Construct rank-minimal decomposition for 2x3x3 (Rank 17)
    # Then replicate 3 times for 2x3x9 (Rank 51)
    
    n_sub, m_sub, p_sub = 2, 3, 3
    rank_sub = 17
    
    U_sub = np.zeros((n_sub * m_sub, rank_sub))
    V_sub = np.zeros((m_sub * p_sub, rank_sub))
    W_sub = np.zeros((p_sub * n_sub, rank_sub))
    
    col = 0
    
    # --- Part 1: M_{2,2,2} (Strassen) ---
    # A_1 (2x2) * B_11 (2x2) -> C_11 (2x2)
    # A_1 indices: i in 0..1, k in 0..1 (U idx: 3i+k)
    # B_11 indices: k in 0..1, j in 0..1 (V idx: 3k+j)
    # C_11 indices: j in 0..1, i in 0..1 (W idx: 2j+i)
    
    # Strassen factors for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # We define U, V, W for each of the 7 products.
    # U row: i*m+k. V row: k*p+j. W row: j*n+i.
    
    # M1: (a+d)(e+h)
    # a=A[0,0], d=A[1,1]. e=B[0,0], h=B[1,1].
    # U contributions: A[0,0] and A[1,1]
    # V contributions: B[0,0] and B[1,1]
    # W contribution: C[0,0] and C[1,1]? No, W is the output coefficient.
    # The tensor is sum_r U_r V_r W_r.
    # For M1, U_r has non-zeros at (0,0) and (1,1).
    # V_r has non-zeros at (0,0) and (1,1).
    # W_r has non-zeros at (0,0) and (1,1)?
    # C[0,0] gets M1. C[1,1] gets M1.
    # So W_r[0,0]=1, W_r[1,1]=1?
    # Wait, W index is j*n+i.
    # C[0,0] is j=0, i=0 -> idx 0.
    # C[1,1] is j=1, i=1 -> idx 3.
    # C[0,1] is j=0, i=1 -> idx 1.
    # C[1,0] is j=1, i=0 -> idx 2.
    
    # Let's define the 7 rank-1 tensors.
    
    # M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U1: U1[0,0]=1, U1[1,1]=1 (A[0,0], A[1,1])
    # V1: V1[0,0]=1, V1[1,1]=1 (B[0,0], B[1,1])
    # W1: W1[0,0]=1, W1[1,1]=1 (C[0,0], C[1,1])
    # Note: C[0,0] is W idx 0. C[1,1] is W idx 3.
    
    # M2: (A[1,0]+A[1,1]) * B[0,0]
    # U2: U2[1,0]=1, U2[1,1]=1
    # V2: V2[0,0]=1
    # W2: W2[1,0]=1 (C[1,0] is j=1, i=0 -> idx 2)
    
    # M3: A[0,0] * (B[0,1]-B[1,1])
    # U3: U3[0,0]=1
    # V3: V3[0,1]=1, V3[1,1]=-1
    # W3: W3[0,1]=1 (C[0,1] is j=0, i=1 -> idx 1)
    
    # M4: A[1,1] * (B[1,0]-B[0,0])
    # U4: U4[1,1]=1
    # V4: V4[1,0]=1, V4[0,0]=-1
    # W4: W4[1,0]=1 (C[1,0] is j=1, i=0 -> idx 2)
    # Wait, C[1,0] gets M4?
    # C[1,0] = M2 + M4. Yes.
    
    # M5: (A[0,0]+A[0,1]) * B[1,1]
    # U5: U5[0,0]=1, U5[0,1]=1
    # V5: V5[1,1]=1
    # W5: W5[0,1]=1 (C[0,1] is j=0, i=1 -> idx 1)
    # Wait, C[0,1] gets M5?
    # C[0,1] = M3 + M5. Yes.
    
    # M6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # U6: U6[1,0]=1, U6[0,0]=-1
    # V6: V6[0,0]=1, V6[0,1]=1
    # W6: W6[1,1]=1 (C[1,1] is j=1, i=1 -> idx 3)
    # Wait, C[1,1] gets M6?
    # C[1,1] = M1 - M2 + M3 + M6. Yes.
    
    # M7: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # U7: U7[0,1]=1, U7[1,1]=-1
    # V7: V7[1,0]=1, V7[1,1]=1
    # W7: W7[0,0]=1 (C[0,0] is j=0, i=0 -> idx 0)
    # Wait, C[0,0] gets M7?
    # C[0,0] = M1 + M4 - M5 + M7. Yes.

    # Fill U_sub, V_sub, W_sub for columns 0-6
    
    # M1
    U_sub[0, col] = 1; U_sub[3, col] = 1
    V_sub[0, col] = 1; V_sub[4, col] = 1
    W_sub[0, col] = 1; W_sub[3, col] = 1
    col += 1
    
    # M2
    U_sub[3, col] = 1; U_sub[4, col] = 1
    V_sub[0, col] = 1
    W_sub[2, col] = 1
    col += 1
    
    # M3
    U_sub[0, col] = 1
    V_sub[1, col] = 1; V_sub[4, col] = -1
    W_sub[1, col] = 1
    col += 1
    
    # M4
    U_sub[4, col] = 1
    V_sub[3, col] = 1; V_sub[0, col] = -1
    W_sub[2, col] = 1
    col += 1
    
    # M5
    U_sub[0, col] = 1; U_sub[1, col] = 1
    V_sub[4, col] = 1
    W_sub[1, col] = 1
    col += 1
    
    # M6
    U_sub[3, col] = 1; U_sub[0, col] = -1
    V_sub[0, col] = 1; V_sub[1, col] = 1
    W_sub[3, col] = 1
    col += 1
    
    # M7
    U_sub[1, col] = 1; U_sub[4, col] = -1
    V_sub[3, col] = 1; V_sub[4, col] = 1
    W_sub[0, col] = 1
    col += 1

    # --- Part 2: M_{2,2,1} (Trivial) ---
    # A_1 (2x2) * B_12 (2x1) -> C_12 (2x1)
    # A_1 indices: i in 0..1, k in 0..1
    # B_12 indices: k in 0..1, j=2
    # C_12 indices: j=2, i in 0..1
    
    # Terms: (i,k) for i in 0..1, k in 0..1
    # (0,0): U[0,0]=1, V[0,2]=1, W[2,0]=1 (W idx: j=2, i=0 -> 2*2+0=4)
    # (0,1): U[0,1]=1, V[1,2]=1, W[2,0]=1
    # (1,0): U[1,0]=1, V[0,2]=1, W[2,1]=1 (W idx: j=2, i=1 -> 2*2+1=5)
    # (1,1): U[1,1]=1, V[1,2]=1, W[2,1]=1
    
    # U idx: 3i+k.
    # V idx: 3k+2.
    # W idx: 2*2+i = 4+i.
    
    # (0,0)
    U_sub[0, col] = 1
    V_sub[2, col] = 1
    W_sub[4, col] = 1
    col += 1
    
    # (0,1)
    U_sub[1, col] = 1
    V_sub[5, col] = 1
    W_sub[4, col] = 1
    col += 1
    
    # (1,0)
    U_sub[3, col] = 1
    V_sub[2, col] = 1
    W_sub[5, col] = 1
    col += 1
    
    # (1,1)
    U_sub[4, col] = 1
    V_sub[5, col] = 1
    W_sub[5, col] = 1
    col += 1

    # --- Part 3: M_{2,1,3} (Trivial) ---
    # A_2 (2x1) * B_2 (1x3) -> C_2 (2x3)
    # A_2 indices: i in 0..1, k=2
    # B_2 indices: k=2, j in 0..2
    # C_2 indices: j in 0..2, i in 0..1
    
    # Terms: (i,j) for i in 0..1, j in 0..2
    # U idx: 3i+2.
    # V idx: 3*2+j = 6+j.
    # W idx: 2j+i.
    
    for i in range(2):
        for j in range(3):
            U_sub[3*i + 2, col] = 1
            V_sub[6 + j, col] = 1
            W_sub[2*j + i, col] = 1
            col += 1
            
    # Verify rank is 17
    assert col == 17
    
    # Construct 2x3x9
    total_rank = rank_sub * 3
    U_final = np.zeros((6, total_rank))
    V_final = np.zeros((27, total_rank))
    W_final = np.zeros((18, total_rank))
    
    for q in range(3):
        offset = q * rank_sub
        U_final[:, offset:offset+rank_sub] = U_sub
        
        # V: map (k, j_sub) to (k, 3q+j_sub)
        # V_sub idx: 3k + j_sub
        # V_final idx: 9k + (3q + j_sub)
        for k in range(3):
            for j_sub in range(3):
                v_sub_idx = 3*k + j_sub
                v_full_idx = 9*k + (3*q + j_sub)
                V_final[v_full_idx, offset:offset+rank_sub] = V_sub[v_sub_idx, :]
                
        # W: map (j_sub, i) to (3q+j_sub, i)
        # W_sub idx: 2*j_sub + i
        # W_final idx: 2*(3q+j_sub) + i
        for j_sub in range(3):
            for i in range(2):
                w_sub_idx = 2*j_sub + i
                w_full_idx = 2*(3*q + j_sub) + i
                W_final[w_full_idx, offset:offset+rank_sub] = W_sub[w_sub_idx, :]

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
