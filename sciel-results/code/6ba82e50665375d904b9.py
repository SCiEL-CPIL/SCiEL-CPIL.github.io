"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = make_schoolbook()
    best_rank = n * m * p

    # Precompute index arrays for efficient verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        u_vals = Ua[u_idx, :]  # (n*m*p, r)
        v_vals = Va[v_idx, :]  # (n*m*p, r)
        w_vals = Wa[w_idx, :]  # (n*m*p, r)
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))

    def combine_blocks(block_decomps, n, m, p):
        """
        Combine multiple block decompositions into one for the full problem.
        block_decomps: list of (U_block, V_block, W_block, rank_block)
        where each block handles a sub-problem.
        """
        total_rank = sum(r for _, _, _, r in block_decomps)
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col_offset = 0
        for U_b, V_b, W_b, r_b in block_decomps:
            for i in range(len(U_b)):
                for c in range(r_b):
                    if U_b[i][c] != 0:
                        U[i][col_offset + c] = U_b[i][c]
            for j in range(len(V_b)):
                for c in range(r_b):
                    if V_b[j][c] != 0:
                        V[j][col_offset + c] = V_b[j][c]
            for k in range(len(W_b)):
                for c in range(r_b):
                    if W_b[k][c] != 0:
                        W[k][col_offset + c] = W_b[k][c]
            col_offset += r_b
        
        return {"U": U, "V": V, "W": W}, total_rank

    def strassen_2x2x2():
        """
        Strassen's algorithm for 2x2 * 2x2 = 2x2.
        Returns U (4x7), V (4x7), W (4x7) with rank 7.
        """
        # Strassen's 7 products
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
        V = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
        W = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
        
        # A indices: A11=0, A12=1, A21=2, A22=3
        # B indices: B11=0, B12=1, B21=2, B22=3
        # C indices: C11=0, C12=1, C21=2, C22=3
        
        # M1 = (A11+A22)*(B11+B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1
        
        # M2 = (A21+A22)*B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3 = A11*(B12-B22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A22*(B21-B11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A11+A12)*B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A21-A11)*(B11+B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A12-A22)*(B21+B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W, 7

    def schoolbook_2x2x1():
        """
        Schoolbook for 2x2 * 2x1 = 2x1.
        Returns U (4x4), V (2x4), W (2x4) with rank 4.
        """
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        
        # C1 = A11*B1 + A12*B2
        # C2 = A21*B1 + A22*B2
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        
        return U, V, W, 4

    def schoolbook_2x1x2():
        """
        Schoolbook for 2x1 * 1x2 = 2x2.
        Returns U (2x4), V (2x4), W (4x4) with rank 4.
        """
        U = [[0]*4 for _ in range(2)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(4)]
        
        # C11 = A1*B1, C12 = A1*B2
        # C21 = A2*B1, C22 = A2*B2
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
        U[1][2] = 1; V[0][2] = 1; W[2][2] = 1
        U[1][3] = 1; V[1][3] = 1; W[3][3] = 1
        
        return U, V, W, 4

    def schoolbook_1x2x2():
        """
        Schoolbook for 1x2 * 2x2 = 1x2.
        Returns U (2x4), V (4x4), W (2x4) with rank 4.
        """
        U = [[0]*4 for _ in range(2)]
        V = [[0]*4 for _ in range(4)]
        W = [[0]*4 for _ in range(2)]
        
        # C1 = A1*B11 + A2*B21
        # C2 = A1*B12 + A2*B22
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
        U[1][2] = 1; V[2][2] = 1; W[0][2] = 1
        U[1][3] = 1; V[3][3] = 1; W[1][3] = 1
        
        return U, V, W, 4

    # Strategy: Split 4x4 * 4x7 into blocks
    # We can split the 4x4 matrix into 2x2 blocks and the 4x7 into appropriate blocks
    
    # Option 1: Split into four 2x2 * 2x2 and 2x2 * 2x1 blocks
    # But this gets complex. Let's try a simpler approach.
    
    # Actually, let's use the known result: for 4x4 * 4x7, we can achieve rank 93
    # by using Strassen for the 4x4 * 4x4 part (rank 49) and schoolbook for 4x4 * 4x3 (rank 48)
    # Total: 49 + 48 = 97. But we can do better.
    
    # Better: Use 4 separate 2x2 * 2x7 decompositions
    # Each 2x2 * 2x7 can be done with rank 11 (Strassen 2x2x2 + schoolbook 2x2x1 for each of 3 remaining cols)
    # Actually, 2x2 * 2x7: split into 2x2*2x2 (rank 7) + 2x2*2x1 (rank 4) + 2x2*2x1 (rank 4) + 2x2*2x1 (rank 4) = 19? No.
    
    # Let me think differently. The 4x4 * 4x7 tensor can be decomposed by:
    # Splitting the 7 columns into groups: 2+2+2+1
    # For each group of 2 columns: 4x4 * 4x2
    # For the last group of 1 column: 4x4 * 4x1
    
    # 4x4 * 4x2: This is two 4x4 * 4x1 problems
    # 4x4 * 4x1: Can be done with rank 16 (schoolbook) or less
    
    # Actually, let's use the approach from inspiration 2:
    # Strassen 4x4x4 (rank 49) + schoolbook 4x4x3 (rank 48) = 97
    
    # But we can do better by using smaller blocks.
    
    # Let's try: Split 4x4 into four 2x2 blocks
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B1, B2], [B3, B4]] where B1, B2, B3, B4 are 2x7
    
    # C11 = A11*B1 + A12*B3
    # C12 = A11*B2 + A12*B4
    # C21 = A21*B1 + A22*B3
    # C22 = A21*B2 + A22*B4
    
    # Each Ci is 2x7
    # We need to compute 8 products of form 2x2 * 2x7
    
    # For each 2x2 * 2x7, we can split into 2x2*2x2 + 2x2*2x2 + 2x2*2x2 + 2x2*2x1
    # That's 7+7+7+4 = 25 per block, total 8*25 = 200. Too high.
    
    # Better: Use Strassen for the 2x2*2x2 parts
    # Actually, 2x2 * 2x7 can be done with rank 11:
    # Split 2x7 into 2x2 + 2x2 + 2x2 + 2x1
    # Use Strassen for each 2x2*2x2 (rank 7 each) and schoolbook for 2x2*2x1 (rank 4)
    # But we can share some computations.
    
    # Let me just implement the 97-rank solution first and see if we can improve.
    
    # Implementation of 4x4 * 4x4 using Strassen (rank 49)
    # This is complex. Let's use a known construction.
    
    # Actually, let's try a different structural approach:
    # Use the fact that we can decompose 4x4 * 4x7 as a sum of tensor products
    
    # For now, let's return the schoolbook solution and add a simple block decomposition
    
    # Block decomposition: Split into 2x2 * 2x7 blocks
    # We'll have 4 such blocks (for the 2x2 blocks of the 4x4 matrix)
    # Each 2x2 * 2x7 needs rank 11 (using Strassen + schoolbook)
    
    def make_2x2x7():
        """
        Decompose 2x2 * 2x7 with rank 11.
        Split 2x7 into 2x2 + 2x2 + 2x2 + 2x1
        Use Strassen for 2x2*2x2 (rank 7) and schoolbook for 2x2*2x1 (rank 4)
        Total: 7 + 4 + 4 + 4 = 19? No, we can do better.
        
        Actually, 2x2 * 2x7: 
        A = [[a, b], [c, d]], B = [b1, b2, b3, b4, b5, b6, b7] (2x7)
        C = [a*b1 + b*b3, a*b2 + b*b4, a*b5 + b*b6, a*b7 + b*b7, ...]
        
        Let me just use schoolbook for 2x2 * 2x7: rank 14
        """
        U = [[0]*14 for _ in range(4)]
        V = [[0]*14 for _ in range(14)]
        W = [[0]*14 for _ in range(14)]
        
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(7):
                    U[i*2+j][col] = 1
                    V[j*7+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        
        return U, V, W, 14

    # This gives 4 * 14 = 56 for the 4x4 * 4x7? No, that's not right.
    # The 4x4 matrix has 4 blocks of 2x2, and each multiplies the full 2x7.
    # So we need 4 separate 2x2 * 2x7 decompositions, each with rank 14.
    # Total rank: 4 * 14 = 56. But this is not correct because the blocks don't overlap.
    
    # Actually, I'm overcomplicating this. Let me just return the schoolbook solution
    # and add a simple improvement: try to find a lower rank using random search with
    # a better initialization.
    
    # For now, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
