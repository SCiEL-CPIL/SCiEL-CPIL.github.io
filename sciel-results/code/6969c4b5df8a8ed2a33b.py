"""Program for 2x4x4 matrix multiplication tensor decomposition."""

def construct():
    # We decompose the 2x4x4 multiplication tensor into 4 independent 2x2x2 sub-tensors.
    # A is 2x4 (rows i=0..1, cols j=0..3)
    # B is 4x4 (rows j=0..3, cols k=0..3)
    # C is 2x4 (rows i=0..1, cols k=0..3)
    #
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # We partition B's rows into two groups: {0,1} and {2,3}.
    # We partition C's columns into two groups: {0,1} and {2,3}.
    #
    # This gives 4 sub-problems, each of which is a 2x2x2 multiplication:
    #   C[i, k] for i in {0,1}, k in {0,1} uses A[i, j] for j in {0,1} and B[j, k] for j in {0,1}, k in {0,1}
    #   C[i, k] for i in {0,1}, k in {2,3} uses A[i, j] for j in {0,1} and B[j, k] for j in {0,1}, k in {2,3}
    #   C[i, k] for i in {0,1}, k in {0,1} uses A[i, j] for j in {2,3} and B[j, k] for j in {2,3}, k in {0,1}
    #   C[i, k] for i in {0,1}, k in {2,3} uses A[i, j] for j in {2,3} and B[j, k] for j in {2,3}, k in {2,3}
    #
    # Each 2x2x2 sub-tensor can be decomposed using Strassen's algorithm with rank 7.
    # Total rank = 4 * 7 = 28.
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen decomposition for a 2x2x2 tensor:
    # Input: A (2x2), B (2x2), Output C (2x2)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Strassen's 7 multiplications:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # We apply this to each of the 4 sub-tensors.
    # For each sub-tensor, we assign 7 consecutive columns in U, V, W.
    
    def strassen_block(u_rows, v_rows, w_rows, col_offset):
        """
        Fill in the Strassen decomposition for one 2x2x2 sub-tensor.
        
        u_rows: list of 4 indices into U for A00, A01, A10, A11
        v_rows: list of 4 indices into V for B00, B01, B10, B11
        w_rows: list of 4 indices into W for C00, C01, C10, C11
        col_offset: starting column index
        """
        # Define the 7 Strassen multiplications
        # Each is (a_indices_with_signs, b_indices_with_signs, c_indices_with_signs)
        # where a_indices_with_signs is a list of (u_row_index, sign)
        
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        strassen = [
            # M1
            ([(0, 1), (2, 1)], [(0, 1), (3, 1)], [(0, 1), (3, -1)]),
            # M2
            ([(1, 1), (2, 1)], [(0, 1)], [(1, 1), (2, 1)]),
            # M3
            ([(0, 1)], [(1, 1), (3, -1)], [(0, 1), (1, 1)]),
            # M4
            ([(2, 1)], [(2, 1), (0, -1)], [(0, 1), (1, -1)]),
            # M5
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6
            ([(1, 1), (0, -1)], [(0, 1), (1, 1)], [(2, 1), (3, 1)]),
            # M7
            ([(1, 1), (2, -1)], [(2, 1), (3, 1)], [(0, 1), (2, 1)]),
        ]
        
        for m_idx, (a_entries, b_entries, c_entries) in enumerate(strassen):
            col = col_offset + m_idx
            for u_rel, sign in a_entries:
                U[u_rows[u_rel]][col] += sign
            for v_rel, sign in b_entries:
                V[v_rows[v_rel]][col] += sign
            for w_rel, sign in c_entries:
                W[w_rows[w_rel]][col] += sign
    
    # Sub-tensor 1: i in {0,1}, j in {0,1}, k in {0,1}
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5 -> u_rows = [0, 1, 4, 5]
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5 -> v_rows = [0, 1, 4, 5]
    # C indices: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3 -> w_rows = [0, 2, 1, 3]
    strassen_block([0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3], 0)
    
    # Sub-tensor 2: i in {0,1}, j in {0,1}, k in {2,3}
    # A indices: same -> u_rows = [0, 1, 4, 5]
    # B indices: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7 -> v_rows = [2, 3, 6, 7]
    # C indices: C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7 -> w_rows = [4, 6, 5, 7]
    strassen_block([0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7], 7)
    
    # Sub-tensor 3: i in {0,1}, j in {2,3}, k in {0,1}
    # A indices: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7 -> u_rows = [2, 3, 6, 7]
    # B indices: B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13 -> v_rows = [8, 9, 12, 13]
    # C indices: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3 -> w_rows = [0, 2, 1, 3]
    strassen_block([2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3], 14)
    
    # Sub-tensor 4: i in {0,1}, j in {2,3}, k in {2,3}
    # A indices: same -> u_rows = [2, 3, 6, 7]
    # B indices: B[2,2]=10, B[2,3]=11, B[3,2]=14, B[3,3]=15 -> v_rows = [10, 11, 14, 15]
    # C indices: C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7 -> w_rows = [4, 6, 5, 7]
    strassen_block([2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7], 21)
    
    return {"U": U, "V": V, "W": W}
