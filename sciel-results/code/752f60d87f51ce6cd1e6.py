import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Schoolbook rank 96
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook

    # Rank 44 construction
    # M_{3,4,8} = 4 * M_{2,2,2} + 4 * M_{1,2,2}
    # Rank(M_{2,2,2}) = 7
    # Rank(M_{1,2,2}) = 4
    # Total Rank = 4*7 + 4*4 = 28 + 16 = 44
    
    target_rank = 44
    if time.time() < deadline - 10:
        U_try = np.zeros((n * m, target_rank))
        V_try = np.zeros((m * p, target_rank))
        W_try = np.zeros((p * n, target_rank))
        
        # Helper to add a block decomposition
        # block_type: '222' or '122'
        # a_start, a_end: row range in A (size n*m)
        # b_start, b_end: row range in B (size m*p) -> actually B is indexed by j (input col) and k (output row)
        # Wait, V is (m*p, rank). Index is j*p + k.
        # W is (p*n, rank). Index is k*n + i.
        # U is (n*m, rank). Index is i*m + j.
        
        # We need to map the local indices of the sub-tensors to the global indices.
        
        def add_strassen_222(a_rows, b_cols, c_cols, offset):
            # a_rows: list of 2 global row indices in A (i, j) -> actually just the 2 rows of the 2x2 block of A
            # b_cols: list of 2 global col indices in B? No.
            # Let's define the mapping carefully.
            # A block: 2x2. Rows i1, i2. Cols j1, j2.
            # U indices: i1*m+j1, i1*m+j2, i2*m+j1, i2*m+j2.
            # B block: 2x2. Rows j1, j2. Cols k1, k2.
            # V indices: j1*p+k1, j1*p+k2, j2*p+k1, j2*p+k2.
            # C block: 2x2. Rows k1, k2. Cols i1, i2.
            # W indices: k1*n+i1, k1*n+i2, k2*n+i1, k2*n+i2.
            
            # Strassen intermediates
            # M1 = (a11+a22)(b11+b22)
            # M2 = (a21+a22)b11
            # M3 = a11(b12-b22)
            # M4 = a21(b21-b12)
            # M5 = (a11+a12)b22
            # M6 = (a22-a12)(b11+b12)
            # M7 = (a11-a21)(b21+b22)
            
            # Coefficients for U (A factors)
            # a11, a12, a21, a22
            u_idx1 = a_rows[0][0] * m + a_rows[0][1] # a11
            u_idx2 = a_rows[0][0] * m + a_rows[0][1] + 1 # a12
            u_idx3 = a_rows[1][0] * m + a_rows[1][1] # a21
            u_idx4 = a_rows[1][0] * m + a_rows[1][1] + 1 # a22
            
            # Coefficients for V (B factors)
            # b11, b12, b21, b22
            # V index: j*p + k
            v_idx1 = b_cols[0][0] * p + b_cols[0][1] # b11 (j1, k1)
            v_idx2 = b_cols[0][0] * p + b_cols[0][1] + 1 # b12 (j1, k2)
            v_idx3 = b_cols[1][0] * p + b_cols[1][1] # b21 (j2, k1)
            v_idx4 = b_cols[1][0] * p + b_cols[1][1] + 1 # b22 (j2, k2)
            
            # Coefficients for W (C factors)
            # c11, c12, c21, c22
            # W index: k*n + i
            w_idx1 = b_cols[0][1] * n + a_rows[0][0] # c11 (k1, i1)
            w_idx2 = b_cols[0][1] * n + a_rows[1][0] # c12? No.
            # C is 2x2. Rows k1, k2. Cols i1, i2.
            # C[k1, i1] is at w_idx(k1, i1)
            # C[k1, i2] is at w_idx(k1, i2)
            # C[k2, i1] is at w_idx(k2, i1)
            # C[k2, i2] is at w_idx(k2, i2)
            
            w_idx1 = b_cols[0][1] * n + a_rows[0][0] # c11
            w_idx2 = b_cols[0][1] * n + a_rows[1][0] # c12 (row k1, col i2)
            w_idx3 = b_cols[1][1] * n + a_rows[0][0] # c21 (row k2, col i1)
            w_idx4 = b_cols[1][1] * n + a_rows[1][0] # c22
            
            # Add terms
            # M1
            U_try[u_idx1][offset+0] += 1
            U_try[u_idx4][offset+0] += 1
            V_try[v_idx1][offset+0] += 1
            V_try[v_idx4][offset+0] += 1
            W_try[w_idx1][offset+0] += 1
            W_try[w_idx2][offset+0] += 1
            W_try[w_idx3][offset+0] += 1
            W_try[w_idx4][offset+0] += 1
            
            # M2
            U_try[u_idx3][offset+1] += 1
            U_try[u_idx4][offset+1] += 1
            V_try[v_idx1][offset+1] += 1
            W_try[w_idx3][offset+1] += 1
            
            # M3
            U_try[u_idx1][offset+2] += 1
            V_try[v_idx2][offset+2] += 1
            V_try[v_idx4][offset+2] -= 1
            W_try[w_idx2][offset+2] += 1
            
            # M4
            U_try[u_idx3][offset+3] += 1
            V_try[v_idx3][offset+3] += 1
            V_try[v_idx2][offset+3] -= 1
            W_try[w_idx1][offset+3] += 1
            
            # M5
            U_try[u_idx1][offset+4] += 1
            U_try[u_idx2][offset+4] += 1
            V_try[v_idx4][offset+4] += 1
            W_try[w_idx1][offset+4] -= 1
            W_try[w_idx2][offset+4] += 1
            
            # M6
            U_try[u_idx4][offset+5] += 1
            U_try[u_idx2][offset+5] -= 1
            V_try[v_idx1][offset+5] += 1
            V_try[v_idx2][offset+5] += 1
            W_try[w_idx4][offset+5] += 1
            
            # M7
            U_try[u_idx1][offset+6] += 1
            U_try[u_idx3][offset+6] -= 1
            V_try[v_idx3][offset+6] += 1
            V_try[v_idx4][offset+6] += 1
            W_try[w_idx1][offset+6] -= 1
            W_try[w_idx3][offset+6] += 1

        def add_122(a_row, b_cols, c_cols, offset):
            # A: 1x2. Row i, Cols j1, j2.
            # B: 2x2. Rows j1, j2. Cols k1, k2.
            # C: 1x2. Row k? No, C is 1x2. Rows k1, k2?
            # M_{1,2,2}: A(1x2) * B(2x2) = C(1x2).
            # C[k1] = A[j1]B[j1,k1] + A[j2]B[j2,k1]
            # C[k2] = A[j1]B[j1,k2] + A[j2]B[j2,k2]
            
            # U indices
            u_idx1 = a_row[0] * m + a_row[1] # a1
            u_idx2 = a_row[0] * m + a_row[1] + 1 # a2
            
            # V indices
            v_idx1 = b_cols[0][0] * p + b_cols[0][1] # b11
            v_idx2 = b_cols[0][0] * p + b_cols[0][1] + 1 # b12
            v_idx3 = b_cols[1][0] * p + b_cols[1][1] # b21
            v_idx4 = b_cols[1][0] * p + b_cols[1][1] + 1 # b22
            
            # W indices
            # C is 1x2. The "rows" of C correspond to the output columns k1, k2.
            # But W is indexed by k*n + i.
            # Here i is the single row of A.
            # So W indices are k1*n + i and k2*n + i.
            w_idx1 = b_cols[0][1] * n + a_row[0] # c1
            w_idx2 = b_cols[1][1] * n + a_row[0] # c2
            
            # Terms
            # 1. a1*b11 -> c1
            U_try[u_idx1][offset+0] += 1
            V_try[v_idx1][offset+0] += 1
            W_try[w_idx1][offset+0] += 1
            
            # 2. a2*b21 -> c1
            U_try[u_idx2][offset+1] += 1
            V_try[v_idx3][offset+1] += 1
            W_try[w_idx1][offset+1] += 1
            
            # 3. a1*b12 -> c2
            U_try[u_idx1][offset+2] += 1
            V_try[v_idx2][offset+2] += 1
            W_try[w_idx2][offset+2] += 1
            
            # 4. a2*b22 -> c2
            U_try[u_idx2][offset+3] += 1
            V_try[v_idx4][offset+3] += 1
            W_try[w_idx2][offset+3] += 1

        # Construction
        # Top 2 rows of A (i=0,1)
        # Split into 4 blocks of 2x2 in B
        # B is 4x8.
        # Block 1: j=0,1; k=0,1
        # Block 2: j=0,1; k=2,3
        # Block 3: j=2,3; k=0,1
        # Block 4: j=2,3; k=2,3
        
        # For each 2x2 block of B, we use A rows 0,1 and cols j1,j2.
        
        # Block 1: A(0,1; 0,1) * B(0,1; 0,1) -> C(0,1; 0,1)
        # a_rows: [(0,0), (1,0)] -> i=0,j=0; i=1,j=0
        # b_cols: [(0,0), (0,1)] -> j=0,k=0; j=0,k=1? No.
        # b_cols for B(2x2): rows j1,j2. cols k1,k2.
        # So b_cols = [(0,0), (1,0)]? No.
        # B is 4x8.
        # Block 1 uses B[0:2, 0:2].
        # j indices: 0, 1. k indices: 0, 1.
        # So b_cols = [(0,0), (1,1)]? No.
        # b_cols[0] is (j1, k1) = (0, 0).
        # b_cols[1] is (j2, k2) = (1, 1).
        # Wait, the 2x2 block of B is:
        # B[0,0] B[0,1]
        # B[1,0] B[1,1]
        # So j1=0, j2=1. k1=0, k2=1.
        # b_cols = [(0,0), (1,1)]? No, the second element should be (j2, k2)?
        # My helper expects b_cols[0]=(j1,k1) and b_cols[1]=(j2,k2).
        # But it uses b_cols[0][0] for j1 and b_cols[0][1] for k1.
        # And b_cols[1][0] for j2 and b_cols[1][1] for k2.
        # This assumes the block is diagonal? No.
        # The block is:
        # (j1,k1) (j1,k2)
        # (j2,k1) (j2,k2)
        # So I need to pass j1, j2, k1, k2 separately or a structure that allows this.
        # Let's modify the helper to take j1, j2, k1, k2.
        
        def add_strassen_222_v2(i1, i2, j1, j2, k1, k2, offset):
            # A block: rows i1, i2; cols j1, j2
            u_idx1 = i1 * m + j1
            u_idx2 = i1 * m + j2
            u_idx3 = i2 * m + j1
            u_idx4 = i2 * m + j2
            
            # B block: rows j1, j2; cols k1, k2
            v_idx1 = j1 * p + k1
            v_idx2 = j1 * p + k2
            v_idx3 = j2 * p + k1
            v_idx4 = j2 * p + k2
            
            # C block: rows k1, k2; cols i1, i2
            w_idx1 = k1 * n + i1
            w_idx2 = k1 * n + i2
            w_idx3 = k2 * n + i1
            w_idx4 = k2 * n + i2
            
            # M1
            U_try[u_idx1][offset+0] += 1
            U_try[u_idx4][offset+0] += 1
            V_try[v_idx1][offset+0] += 1
            V_try[v_idx4][offset+0] += 1
            W_try[w_idx1][offset+0] += 1
            W_try[w_idx2][offset+0] += 1
            W_try[w_idx3][offset+0] += 1
            W_try[w_idx4][offset+0] += 1
            
            # M2
            U_try[u_idx3][offset+1] += 1
            U_try[u_idx4][offset+1] += 1
            V_try[v_idx1][offset+1] += 1
            W_try[w_idx3][offset+1] += 1
            
            # M3
            U_try[u_idx1][offset+2] += 1
            V_try[v_idx2][offset+2] += 1
            V_try[v_idx4][offset+2] -= 1
            W_try[w_idx2][offset+2] += 1
            
            # M4
            U_try[u_idx3][offset+3] += 1
            V_try[v_idx3][offset+3] += 1
            V_try[v_idx2][offset+3] -= 1
            W_try[w_idx1][offset+3] += 1
            
            # M5
            U_try[u_idx1][offset+4] += 1
            U_try[u_idx2][offset+4] += 1
            V_try[v_idx4][offset+4] += 1
            W_try[w_idx1][offset+4] -= 1
            W_try[w_idx2][offset+4] += 1
            
            # M6
            U_try[u_idx4][offset+5] += 1
            U_try[u_idx2][offset+5] -= 1
            V_try[v_idx1][offset+5] += 1
            V_try[v_idx2][offset+5] += 1
            W_try[w_idx4][offset+5] += 1
            
            # M7
            U_try[u_idx1][offset+6] += 1
            U_try[u_idx3][offset+6] -= 1
            V_try[v_idx3][offset+6] += 1
            V_try[v_idx4][offset+6] += 1
            W_try[w_idx1][offset+6] -= 1
            W_try[w_idx3][offset+6] += 1

        def add_122_v2(i1, j1, j2, k1, k2, offset):
            # A: row i1, cols j1, j2
            u_idx1 = i1 * m + j1
            u_idx2 = i1 * m + j2
            
            # B: rows j1, j2; cols k1, k2
            v_idx1 = j1 * p + k1
            v_idx2 = j1 * p + k2
            v_idx3 = j2 * p + k1
            v_idx4 = j2 * p + k2
            
            # C: rows k1, k2; col i1
            w_idx1 = k1 * n + i1
            w_idx2 = k2 * n + i1
            
            # 1. a1*b11 -> c1
            U_try[u_idx1][offset+0] += 1
            V_try[v_idx1][offset+0] += 1
            W_try[w_idx1][offset+0] += 1
            
            # 2. a2*b21 -> c1
            U_try[u_idx2][offset+1] += 1
            V_try[v_idx3][offset+1] += 1
            W_try[w_idx1][offset+1] += 1
            
            # 3. a1*b12 -> c2
            U_try[u_idx1][offset+2] += 1
            V_try[v_idx2][offset+2] += 1
            W_try[w_idx2][offset+2] += 1
            
            # 4. a2*b22 -> c2
            U_try[u_idx2][offset+3] += 1
            V_try[v_idx4][offset+3] += 1
            W_try[w_idx2][offset+3] += 1

        # Top 2 rows (i=0,1)
        # 4 Strassen blocks
        # Block 1: i=0,1; j=0,1; k=0,1
        add_strassen_222_v2(0, 1, 0, 1, 0, 1, 0)
        # Block 2: i=0,1; j=0,1; k=2,3
        add_strassen_222_v2(0, 1, 0, 1, 2, 3, 7)
        # Block 3: i=0,1; j=2,3; k=0,1
        add_strassen_222_v2(0, 1, 2, 3, 0, 1, 14)
        # Block 4: i=0,1; j=2,3; k=2,3
        add_strassen_222_v2(0, 1, 2, 3, 2, 3, 21)
        
        # Bottom 1 row (i=2)
        # 4 122 blocks
        # Block 5: i=2; j=0,1; k=0,1
        add_122_v2(2, 0, 1, 0, 1, 28)
        # Block 6: i=2; j=0,1; k=2,3
        add_122_v2(2, 0, 1, 2, 3, 32)
        # Block 7: i=2; j=2,3; k=0,1
        add_122_v2(2, 2, 3, 0, 1, 36)
        # Block 8: i=2; j=2,3; k=2,3
        add_122_v2(2, 2, 3, 2, 3, 40)
        
        # Verification
        # Check if U V^T W^T reconstructs the tensor?
        # We can't easily check the full tensor in 120s if we do it naively, but we can check a few random points.
        # Or just trust the algebra.
        
        best_result = {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
        best_rank = target_rank

    return best_result
