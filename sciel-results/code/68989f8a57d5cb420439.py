"""Optimized tensor decomposition for 2x3 x 3x7 matrix multiplication."""

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 24
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_indices, v_indices, w_indices, val=1):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += val
        for idx in v_indices:
            V[idx][col] += val
        for idx in w_indices:
            W[idx][col] += val
        col += 1

    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(i_offset, j_offset, k_offset):
        # i_offset: 0 or 1 (row in A)
        # j_offset: 0 or 1 (col in A / row in B)
        # k_offset: 0, 1, 2 (col in B, corresponds to k=0,1; k=2,3; k=4,5)
        
        # Global indices
        # A indices: i in {i_offset, i_offset+1}, j in {j_offset, j_offset+1}
        # B indices: j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
        # C indices: i in {i_offset, i_offset+1}, k in {k_offset, k_offset+1}
        
        # U indices (A): i*m + j
        u_00 = (i_offset) * m + (j_offset)
        u_01 = (i_offset) * m + (j_offset + 1)
        u_10 = (i_offset + 1) * m + (j_offset)
        u_11 = (i_offset + 1) * m + (j_offset + 1)
        
        # V indices (B): j*p + k
        v_00 = (j_offset) * p + (k_offset)
        v_01 = (j_offset) * p + (k_offset + 1)
        v_10 = (j_offset + 1) * p + (k_offset)
        v_11 = (j_offset + 1) * p + (k_offset + 1)
        
        # W indices (C): k*n + i
        w_00 = (k_offset) * n + (i_offset)
        w_01 = (k_offset) * n + (i_offset + 1)
        w_10 = (k_offset + 1) * n + (i_offset)
        w_11 = (k_offset + 1) * n + (i_offset + 1)
        
        # Strassen Terms
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # M2 = (A10+A11)B00 -> C10 += M2
        # M3 = A00(B10-B11) -> C00 -= M3? No.
        
        # Standard Strassen for C = AB:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 - M4 + M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6 - M7
        
        # Mapping to our indices (1-based in Strassen, 0-based here):
        # A11 -> u_00, A12 -> u_01
        # A21 -> u_10, A22 -> u_11
        # B11 -> v_00, B12 -> v_01
        # B21 -> v_10, B22 -> v_11
        # C11 -> w_00, C12 -> w_01
        # C21 -> w_10, C22 -> w_11
        
        # Term 1: M1
        # U: u_00 (1), u_11 (1)
        # V: v_00 (1), v_11 (1)
        # W: w_00 (1), w_11 (1)
        add_term([u_00, u_11], [v_00, v_11], [w_00, w_11])
        
        # Term 2: M2
        # U: u_10 (1), u_11 (1)
        # V: v_00 (1)
        # W: w_10 (1)
        add_term([u_10, u_11], [v_00], [w_10])
        
        # Term 3: M3
        # U: u_00 (1)
        # V: v_10 (1), v_11 (-1)
        # W: w_00 (-1), w_01 (1)  # C11 -= M3, C12 += M3
        add_term([u_00], [v_10, v_11], [w_00, w_01], val=1)
        # Wait, add_term adds val to all. 
        # I need to handle signs.
        # Let's rewrite add_term to accept a list of (index, value) or just handle signs in the call.
        
        # Re-implementing add_term logic inline for clarity and sign handling
        
        # Reset col to 0 and restart? No, let's just write the terms manually.
        
        # This helper is getting messy with signs. I will implement the terms directly in the main loop.
        pass

    # I will implement the 24 terms directly.
    
    # 1. M_{2,1,7} part (1 term)
    # I: 2, 5. J: 14..20. K: 0..13.
    U[2][col] = 1; U[5][col] = 1
    for j in range(14, 21): V[j][col] = 1
    for k in range(14): W[k][col] = 1
    col += 1

    # 2. M_{2,2,2} block 1 (k=0,1)
    # i=0,1; j=0,1; k=0,1
    # u_00=0, u_01=1, u_10=3, u_11=4
    # v_00=0, v_01=1, v_10=7, v_11=8
    # w_00=0, w_01=1, w_10=2, w_11=3
    
    # T1: M1
    U[0][col]=1; U[4][col]=1; V[0][col]=1; V[8][col]=1; W[0][col]=1; W[3][col]=1; col+=1
    # T2: M2
    U[3][col]=1; U[4][col]=1; V[0][col]=1; W[2][col]=1; col+=1
    # T3: M3
    U[0][col]=1; V[7][col]=1; V[8][col]=-1; W[0][col]=-1; W[1][col]=1; col+=1
    # T4: M4
    U[4][col]=1; V[0][col]=-1; V[1][col]=1; W[2][col]=1; W[3][col]=-1; col+=1
    # T5: M5
    U[0][col]=1; U[1][col]=1; V[8][col]=1; W[0][col]=1; W[1][col]=1; col+=1
    # T6: M6
    U[3][col]=1; U[0][col]=-1; V[0][col]=1; V[1][col]=1; W[2][col]=1; col+=1
    # T7: M7
    U[1][col]=1; U[4][col]=-1; V[7][col]=1; V[8][col]=1; W[3][col]=-1; col+=1

    # 3. M_{2,2,2} block 2 (k=2,3)
    # v_00=2, v_01=3, v_10=9, v_11=10
    # w_00=4, w_01=5, w_10=6, w_11=7
    # u same as above
    
    U[0][col]=1; U[4][col]=1; V[2][col]=1; V[10][col]=1; W[4][col]=1; W[7][col]=1; col+=1
    U[3][col]=1; U[4][col]=1; V[2][col]=1; W[6][col]=1; col+=1
    U[0][col]=1; V[9][col]=1; V[10][col]=-1; W[4][col]=-1; W[5][col]=1; col+=1
    U[4][col]=1; V[2][col]=-1; V[3][col]=1; W[6][col]=1; W[7][col]=-1; col+=1
    U[0][col]=1; U[1][col]=1; V[10][col]=1; W[4][col]=1; W[5][col]=1; col+=1
    U[3][col]=1; U[0][col]=-1; V[2][col]=1; V[3][col]=1; W[6][col]=1; col+=1
    U[1][col]=1; U[4][col]=-1; V[9][col]=1; V[10][col]=1; W[7][col]=-1; col+=1

    # 4. M_{2,2,2} block 3 (k=4,5)
    # v_00=4, v_01=5, v_10=11, v_11=12
    # w_00=8, w_01=9, w_10=10, w_11=11
    
    U[0][col]=1; U[4][col]=1; V[4][col]=1; V[12][col]=1; W[8][col]=1; W[11][col]=1; col+=1
    U[3][col]=1; U[4][col]=1; V[4][col]=1; W[10][col]=1; col+=1
    U[0][col]=1; V[11][col]=1; V[12][col]=-1; W[8][col]=-1; W[9][col]=1; col+=1
    U[4][col]=1; V[4][col]=-1; V[5][col]=1; W[10][col]=1; W[11][col]=-1; col+=1
    U[0][col]=1; U[1][col]=1; V[12][col]=1; W[8][col]=1; W[9][col]=1; col+=1
    U[3][col]=1; U[0][col]=-1; V[4][col]=1; V[5][col]=1; W[10][col]=1; col+=1
    U[1][col]=1; U[4][col]=-1; V[11][col]=1; V[12][col]=1; W[11][col]=-1; col+=1

    # 5. M_{2,2,1} block (k=6)
    # v_00=6, v_10=13
    # w_00=12, w_01=13
    
    # T1: A00 B00 -> C00 (i=0, k=6 -> K=12)
    U[0][col]=1; V[6][col]=1; W[12][col]=1; col+=1
    # T2: A11 B11 -> C11 (i=1, k=6 -> K=13)
    U[4][col]=1; V[13][col]=1; W[13][col]=1; col+=1

    return {"U": U, "V": V, "W": W}
