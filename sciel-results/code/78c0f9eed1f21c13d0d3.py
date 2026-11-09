"""Matrix multiplication 2x5x5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2():
        """
        Returns U, V, W for 2x2 * 2x2 = 2x2 using Strassen's algorithm.
        U shape: 4x7, V shape: 4x7, W shape: 4x7
        Indices:
        U: (i, j) for i in 0..1, j in 0..1
        V: (j, k) for j in 0..1, k in 0..1
        W: (k, i) for k in 0..1, i in 0..1
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Strassen products M1..M7
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map these to U, V, W such that sum_c U[i,j,c] V[j,k,c] W[k,i,c] = C[i,k]
        
        # M1: A11+A22 is involved.
        # U entries for A11, A22. V entries for B11, B22.
        # This is tricky because Strassen combines A's and B's before multiplying.
        # The tensor decomposition requires U, V, W to be linear combinations of A, B entries.
        
        # Let's define the 7 products explicitly in terms of U, V, W columns.
        
        # Product 1: (A11+A22) * (B11+B22)
        # U col 0: A11=1, A22=1, others 0. (Indices: A11 is U[0,0], A22 is U[1,1])
        # V col 0: B11=1, B22=1, others 0. (Indices: B11 is V[0,0], B22 is V[1,1])
        # W col 0: Contributes to C11 (+1), C22 (+1)?
        # C11 = ... + M1. C22 = ... + M1.
        # So W[0,0]=1 (for k=0, i=0), W[1,1]=1 (for k=1, i=1).
        
        U[0][0] = 1; U[1][0] = 1
        V[0][0] = 1; V[1][0] = 1
        W[0][0] = 1; W[1][1] = 1
        
        # Product 2: (A21+A22) * B11
        # U col 1: A21=1, A22=1. (U[1,0]=1, U[1,1]=1)
        # V col 1: B11=1. (V[0,0]=1)
        # W col 1: Contributes to C21 (+1).
        # C21 = M2 + ...
        # So W[0,1]=1 (k=0, i=1).
        
        U[1][1] = 1; U[1][1] = 1 # Wait, U[1][1] is A22. U[1][0] is A21.
        # Correction: U index is i*m+j. m=2.
        # A21 is i=1, j=0 -> U[2]
        # A22 is i=1, j=1 -> U[3]
        # A11 is i=0, j=0 -> U[0]
        # A12 is i=0, j=1 -> U[1]
        
        # Let's restart U, V, W mapping carefully.
        # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # W rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -- Wait, W is (k, i).
        # k=0, i=0 -> W[0]
        # k=0, i=1 -> W[1]
        # k=1, i=0 -> W[2]
        # k=1, i=1 -> W[3]
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1 # C11 (k=0,i=0) and C22 (k=1,i=1)
        
        # M2 = (A21+A22)B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1 # C21 (k=0,i=1)
        
        # M3 = A11(B12-B22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[0][2] = 1 # C12 (k=1,i=0)
        
        # M4 = A22(B21-B11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1 # C11 (k=0,i=0) AND C21 (k=0,i=1)?
        # C11 = M1 + M4 - M5 + M7
        # C21 = M2 + M4
        # So M4 contributes to C11 (W[0]) and C21 (W[1]).
        W[0][3] = 1; W[1][3] = 1
        
        # M5 = (A11+A12)B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1 # C11 (k=0,i=0)
        W[0][4] = 1  # C12 (k=1,i=0) -> C12 = M3 + M5
        # Wait, C11 has -M5. C12 has +M5.
        # W[0] is (k=0, i=0) which is C11.
        # W[2] is (k=1, i=0) which is C12.
        W[0][4] = -1
        W[2][4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1 # C22 (k=1,i=1)
        
        # M7 = (A12-A22)(B21+B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1 # C11 (k=0,i=0)
        
        return U, V, W

    def build_block_decomposition():
        """
        Builds a rank 46 decomposition using block Strassen.
        """
        U = []
        V = []
        W = []
        
        # Helper to add a block
        def add_block(u_block, v_block, w_block, offset_u, offset_v, offset_w):
            # u_block: 4x7, v_block: 4x7, w_block: 4x7
            # offset_u: starting row index in U
            # offset_v: starting row index in V
            # offset_w: starting row index in W
            
            for i in range(4):
                for c in range(7):
                    U[offset_u + i].append(u_block[i][c])
                    V[offset_v + i].append(v_block[i][c])
                    W[offset_w + i].append(w_block[i][c])
            
            # Pad other rows/cols? No, we are building columns.
            # We need to ensure all U, V, W have same number of columns.
            # This helper is flawed for direct appending if we don't track global rank.
            pass

        # Let's build the full matrices directly.
        # Rank 46.
        # 4 Strassen blocks (28 mults)
        # 2 dot products (4 mults each)
        # 1 outer product (4 mults)
        # 1 scalar mult (2 mults)
        
        # Total 28 + 8 + 4 + 2 = 42?
        # Wait, my previous count was:
        # 1. A1 B11 (7)
        # 2. A2 B21 (7)
        # 3. a3 b31T (4)
        # 4. A1 B12 (7)
        # 5. A2 B22 (7)
        # 6. a3 b32T (4)
        # 7. A1 b13 (4)
        # 8. A2 b23 (4)
        # 9. a3 b33 (2)
        # Total 46.
        
        # Let's implement this.
        
        U = [[0]*46 for _ in range(10)] # 2*5 = 10 rows
        V = [[0]*46 for _ in range(25)] # 5*5 = 25 rows
        W = [[0]*46 for _ in range(10)] # 5*2 = 10 rows
        
        col = 0
        
        # Strassen blocks
        sU, sV, sW = strassen_2x2x2()
        
        # Block 1: A1 (rows 0,1 of A; cols 0,1) * B11 (rows 0,1 of B; cols 0,1)
        # A1 is U[0:2, 0:2]
        # B11 is V[0:2, 0:2]
        # Result C11 is W[0:2, 0:2] (k=0,1; i=0,1)
        
        for c in range(7):
            for i in range(4):
                # Map sU[i] to U row
                # sU row 0 -> A11 (U[0])
                # sU row 1 -> A12 (U[1])
                # sU row 2 -> A21 (U[2])
                # sU row 3 -> A22 (U[3])
                U[i][col] = sU[i][c]
                
                # Map sV[i] to V row
                # sV row 0 -> B11 (V[0])
                # sV row 1 -> B12 (V[1])
                # sV row 2 -> B21 (V[2])
                # sV row 3 -> B22 (V[3])
                V[i][col] = sV[i][c]
                
                # Map sW[i] to W row
                # sW row 0 -> C11 (W[0]) (k=0, i=0)
                # sW row 1 -> C21 (W[1]) (k=0, i=1)
                # sW row 2 -> C12 (W[2]) (k=1, i=0)
                # sW row 3 -> C22 (W[3]) (k=1, i=1)
                W[i][col] = sW[i][c]
            col += 1
            
        # Block 2: A2 (rows 0,1 of A; cols 2,3) * B21 (rows 2,3 of B; cols 0,1)
        # A2 is U[0:2, 2:4] -> U rows 2,3? No.
        # A is 2x5. Rows 0,1. Cols 0..4.
        # A1 is cols 0,1. A2 is cols 2,3.
        # U row for A[i, j] is i*5 + j.
        # A1: U[0], U[1], U[5], U[6]? No.
        # U[i*m + j]. m=5.
        # A11: i=0, j=0 -> U[0]
        # A12: i=0, j=1 -> U[1]
        # A21: i=1, j=0 -> U[5]
        # A22: i=1, j=1 -> U[6]
        
        # My previous U mapping for Strassen assumed m=2.
        # I need to map the 2x2 block indices to the 2x5 indices.
        
        # Let's rewrite the block addition with explicit index mapping.
        
        def add_strassen_block(a_rows, a_cols, b_rows, b_cols, w_rows, w_cols, col_idx):
            # a_rows: [0, 1]
            # a_cols: [0, 1] for A1
            # b_rows: [0, 1] for B11
            # b_cols: [0, 1] for B11
            # w_rows: [0, 1] for C11 (k indices)
            # w_cols: [0, 1] for C11 (i indices)
            
            sU, sV, sW = strassen_2x2x2()
            
            # Map sU indices (0,1,2,3) to A indices
            # sU[0] -> A[0, a_cols[0]]
            # sU[1] -> A[0, a_cols[1]]
            # sU[2] -> A[1, a_cols[0]]
            # sU[3] -> A[1, a_cols[1]]
            
            u_map = [
                a_rows[0]*5 + a_cols[0],
                a_rows[0]*5 + a_cols[1],
                a_rows[1]*5 + a_cols[0],
                a_rows[1]*5 + a_cols[1]
            ]
            
            # Map sV indices to B indices
            # sV[0] -> B[b_rows[0], b_cols[0]]
            # sV[1] -> B[b_rows[0], b_cols[1]]
            # sV[2] -> B[b_rows[1], b_cols[0]]
            # sV[3] -> B[b_rows[1], b_cols[1]]
            
            v_map = [
                b_rows[0]*5 + b_cols[0],
                b_rows[0]*5 + b_cols[1],
                b_rows[1]*5 + b_cols[0],
                b_rows[1]*5 + b_cols[1]
            ]
            
            # Map sW indices to W indices
            # sW[0] -> W[w_rows[0], w_cols[0]] -> k=w_rows[0], i=w_cols[0]
            # W index is k*n + i. n=2.
            
            w_map = [
                w_rows[0]*2 + w_cols[0],
                w_rows[0]*2 + w_cols[1],
                w_rows[1]*2 + w_cols[0],
                w_rows[1]*2 + w_cols[1]
            ]
            
            for c in range(7):
                for i in range(4):
                    U[u_map[i]][col_idx + c] = sU[i][c]
                    V[v_map[i]][col_idx + c] = sV[i][c]
                    W[w_map[i]][col_idx + c] = sW[i][c]
            
            return col_idx + 7

        col = 0
        # Block 1: A1 (cols 0,1) * B11 (rows 0,1; cols 0,1) -> C11 (k=0,1; i=0,1)
        col = add_strassen_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], col)
        
        # Block 2: A2 (cols 2,3) * B21 (rows 2,3; cols 0,1) -> C11 (k=0,1; i=0,1)
        # Note: This adds to the same C11 block.
        # Wait, C11 is the top-left 2x2 block of C.
        # C11 = A1 B11 + A2 B21 + ...
        # So we add to the same W indices.
        # But U and V indices are different.
        # The columns in U, V, W are summed.
        # So we just use new columns for U, V, W.
        col = add_strassen_block([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], col)
        
        # Block 3: A1 (cols 0,1) * B12 (rows 0,1; cols 2,3) -> C12 (k=2,3; i=0,1)
        col = add_strassen_block([0,1], [0,1], [0,1], [2,3], [2,3], [0,1], col)
        
        # Block 4: A2 (cols 2,3) * B22 (rows 2,3; cols 2,3) -> C12 (k=2,3; i=0,1)
        col = add_strassen_block([0,1], [2,3], [2,3], [2,3], [2,3], [0,1], col)
        
        # Now the remainder terms.
        # a3 (col 4 of A) * b31T (row 4 of B, cols 0,1) -> C11 (k=0,1; i=0,1)
        # This is 2x1 * 1x2 = 2x2. Rank 4.
        # a3 = [a0, a1]^T. b31T = [b0, b1].
        # C = a3 b31T.
        # C00 = a0 b0. C01 = a0 b1. C10 = a1 b0. C11 = a1 b1.
        
        for i in range(2):
            for k in range(2):
                U[i*5 + 4][col] = 1 # a_i
                V[4*5 + k][col] = 1 # b_k
                W[k*2 + i][col] = 1 # c_ki
                col += 1
                
        # a3 (col 4 of A) * b32T (row 4 of B, cols 2,3) -> C12 (k=2,3; i=0,1)
        for i in range(2):
            for k in range(2, 4):
                U[i*5 + 4][col] = 1
                V[4*5 + k][col] = 1
                W[k*2 + i][col] = 1
                col += 1
                
        # A1 (cols 0,1) * b13 (rows 0,1 of B, col 4) -> C15 (k=4; i=0,1)
        # 2x2 * 2x1. Rank 4.
        for i in range(2):
            for j in range(2):
                U[i*5 + j][col] = 1
                V[j*5 + 4][col] = 1
                W[4*2 + i][col] = 1
                col += 1
                
        # A2 (cols 2,3) * b23 (rows 2,3 of B, col 4) -> C15 (k=4; i=0,1)
        for i in range(2):
            for j in range(2, 4):
                U[i*5 + j][col] = 1
                V[j*5 + 4][col] = 1
                W[4*2 + i][col] = 1
                col += 1
                
        # a3 (col 4 of A) * b33 (row 4 of B, col 4) -> C15 (k=4; i=0,1)
        # 2x1 * 1x1. Rank 2.
        for i in range(2):
            U[i*5 + 4][col] = 1
            V[4*5 + 4][col] = 1
            W[4*2 + i][col] = 1
            col += 1
            
        return {"U": U, "V": V, "W": W}

    result = build_block_decomposition()
    return result

# EVOLVE-BLOCK-END
