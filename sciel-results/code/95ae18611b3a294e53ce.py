import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    # We split the multiplication into:
    # 1. 4x4x4 block (A is 4x4, B is 4x4, C is 4x4) -> Rank 49 (Strassen)
    # 2. 4x4x1 block (A is 4x4, B is 4x1, C is 4x1) -> Rank 16 (Schoolbook)
    # Total Rank R = 49 + 16 = 65
    
    R = 65
    U = [[0.0] * R for _ in range(n * m)] # 16 x 65
    V = [[0.0] * R for _ in range(m * p)] # 20 x 65
    W = [[0.0] * R for _ in range(p * n)] # 20 x 65
    
    # --- Helper: Generate Strassen Tensor for 2x2x2 ---
    def strassen_2x2():
        U_s = [[0.0]*7 for _ in range(4)]
        V_s = [[0.0]*7 for _ in range(4)]
        W_s = [[0.0]*7 for _ in range(4)]
        
        # Indices for 2x2: A[i][j] -> i*2+j, B[i][j] -> i*2+j, C[i][j] -> i*2+j
        # M1 = (A11+A22)(B11+B22) -> C11, C22
        U_s[0][0] = 1; U_s[3][0] = 1
        V_s[0][0] = 1; V_s[3][0] = 1
        W_s[0][0] = 1; W_s[3][0] = 1
        
        # M2 = (A21+A22)B11 -> C21
        U_s[2][1] = 1; U_s[3][1] = 1
        V_s[0][1] = 1
        W_s[2][1] = 1
        
        # M3 = A11(B12-B22) -> C12
        U_s[0][2] = 1
        V_s[1][2] = 1; V_s[3][2] = -1
        W_s[1][2] = 1
        
        # M4 = A22(B21-B11) -> C11, C21
        U_s[3][3] = 1
        V_s[2][3] = 1; V_s[0][3] = -1
        W_s[0][3] = 1; W_s[2][3] = 1
        
        # M5 = (A11+A12)B22 -> C11, C12
        U_s[0][4] = 1; U_s[1][4] = 1
        V_s[3][4] = 1
        W_s[0][4] = -1; W_s[1][4] = 1
        
        # M6 = (A21-A11)(B11+B12) -> C22
        U_s[2][5] = 1; U_s[0][5] = -1
        V_s[0][5] = 1; V_s[1][5] = 1
        W_s[3][5] = 1
        
        # M7 = (A12-A22)(B21+B22) -> C11, C22
        U_s[1][6] = 1; U_s[3][6] = -1
        V_s[2][6] = 1; V_s[3][6] = 1
        W_s[0][6] = 1; W_s[3][6] = 1
        
        return U_s, V_s, W_s

    # --- Helper: Generate Strassen Tensor for 4x4x4 ---
    def strassen_4x4():
        U_4 = [[0.0]*49 for _ in range(16)]
        V_4 = [[0.0]*49 for _ in range(16)]
        W_4 = [[0.0]*49 for _ in range(16)]
        
        U_s, V_s, W_s = strassen_2x2()
        
        # Strassen recursion: 7 multiplications of 2x2 blocks.
        # Block indices for 4x4:
        # A11: 0-3, A12: 4-7, A21: 8-11, A22: 12-15
        # B11: 0-3, B12: 4-7, B21: 8-11, B22: 12-15
        # C11: 0-3, C12: 4-7, C21: 8-11, C22: 12-15
        
        # We define the 7 terms M1..M7.
        # Each term is a 2x2x2 multiplication.
        # We map the local 2x2 indices (0..3) to global 4x4 indices.
        
        # Mapping functions for A, B, C blocks
        # A_block(r, c, i, j) where r,c in {0,1} are block row/col, i,j in {0,1} are local
        # Global A index: (block_row*2 + i) * 4 + (block_col*2 + j)
        # Same for B and C.
        
        def map_idx(block_r, block_c, local_i, local_j):
            return (block_r * 2 + local_i) * 4 + (block_c * 2 + local_j)

        # Term 1: M1 = (A11+A22)(B11+B22)
        # A parts: A11 (0,0), A22 (1,1)
        # B parts: B11 (0,0), B22 (1,1)
        # C parts: C11 (0,0), C22 (1,1)
        # Local 2x2 indices:
        # A11 local (i,j) -> Global (0,0,i,j)
        # A22 local (i,j) -> Global (1,1,i,j)
        # B11 local (i,j) -> Global (0,0,i,j)
        # B22 local (i,j) -> Global (1,1,i,j)
        # C11 local (i,j) -> Global (0,0,i,j)
        # C22 local (i,j) -> Global (1,1,i,j)
        
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    # U (A)
                    # A11
                    U_4[map_idx(0, 0, i, j)][c] += U_s[i*2+j][c]
                    # A22
                    U_4[map_idx(1, 1, i, j)][c] += U_s[i*2+j][c]
                    
                    # V (B)
                    # B11
                    V_4[map_idx(0, 0, i, j)][c] += V_s[i*2+j][c]
                    # B22
                    V_4[map_idx(1, 1, i, j)][c] += V_s[i*2+j][c]
                    
                    # W (C)
                    # C11
                    W_4[map_idx(0, 0, i, j)][c] += W_s[i*2+j][c]
                    # C22
                    W_4[map_idx(1, 1, i, j)][c] += W_s[i*2+j][c]

        # Term 2: M2 = (A21+A22)B11
        # A parts: A21 (1,0), A22 (1,1)
        # B parts: B11 (0,0)
        # C parts: C21 (1,0)
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    # U
                    U_4[map_idx(1, 0, i, j)][c+7] += U_s[i*2+j][c]
                    U_4[map_idx(1, 1, i, j)][c+7] += U_s[i*2+j][c]
                    # V
                    V_4[map_idx(0, 0, i, j)][c+7] += V_s[i*2+j][c]
                    # W
                    W_4[map_idx(1, 0, i, j)][c+7] += W_s[i*2+j][c]

        # Term 3: M3 = A11(B12-B22)
        # A parts: A11 (0,0)
        # B parts: B12 (0,1), B22 (1,1)
        # C parts: C12 (0,1)
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    # U
                    U_4[map_idx(0, 0, i, j)][c+14] += U_s[i*2+j][c]
                    # V
                    V_4[map_idx(0, 1, i, j)][c+14] += V_s[i*2+j][c]
                    V_4[map_idx(1, 1, i, j)][c+14] -= V_s[i*2+j][c]
                    # W
                    W_4[map_idx(0, 1, i, j)][c+14] += W_s[i*2+j][c]

        # Term 4: M4 = A22(B21-B11)
        # A parts: A22 (1,1)
        # B parts: B21 (1,0), B11 (0,0)
        # C parts: C11 (0,0), C21 (1,0) -- Wait, Strassen M4 contributes to C11 and C21?
        # Let's check standard Strassen:
        # M4 = A22 * (B21 - B11)
        # C11 = M1 - M4 + M5 - M7
        # C21 = M2 + M4
        # Yes, M4 contributes to C11 and C21.
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    # U
                    U_4[map_idx(1, 1, i, j)][c+21] += U_s[i*2+j][c]
                    # V
                    V_4[map_idx(1, 0, i, j)][c+21] += V_s[i*2+j][c]
                    V_4[map_idx(0, 0, i, j)][c+21] -= V_s[i*2+j][c]
                    # W
                    # C11
                    W_4[map_idx(0, 0, i, j)][c+21] += W_s[i*2+j][c]
                    # C21
                    W_4[map_idx(1, 0, i, j)][c+21] += W_s[i*2+j][c]

        # Term 5: M5 = (A11+A12)B22
        # A parts: A11 (0,0), A12 (0,1)
        # B parts: B22 (1,1)
        # C parts: C11 (0,0), C12 (0,1)
        # C11 = ... + M5
        # C12 = ... - M5 + M3
        # Wait, standard Strassen:
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 - M5 + M7
        # C21 = M2 + M4
        # C22 = M1 - M2 + M6 + M7
        # So M5 contributes to C11 (+) and C12 (-).
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    # U
                    U_4[map_idx(0, 0, i, j)][c+28] += U_s[i*2+j][c]
                    U_4[map_idx(0, 1, i, j)][c+28] += U_s[i*2+j][c]
                    # V
                    V_4[map_idx(1, 1, i, j)][c+28] += V_s[i*2+j][c]
                    # W
                    # C11
                    W_4[map_idx(0, 0, i, j)][c+28] += W_s[i*2+j][c]
                    # C12
                    W_4[map_idx(0, 1, i, j)][c+28] -= W_s[i*2+j][c]

        # Term 6: M6 = (A21-A11)(B11+B12)
        # A parts: A21 (1,0), A11 (0,0)
        # B parts: B11 (0,0), B12 (0,1)
        # C parts: C22 (1,1)
        # C22 = M1 - M2 + M6 + M7
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    # U
                    U_4[map_idx(1, 0, i, j)][c+35] += U_s[i*2+j][c]
                    U_4[map_idx(0, 0, i, j)][c+35] -= U_s[i*2+j][c]
                    # V
                    V_4[map_idx(0, 0, i, j)][c+35] += V_s[i*2+j][c]
                    V_4[map_idx(0, 1, i, j)][c+35] += V_s[i*2+j][c]
                    # W
                    # C22
                    W_4[map_idx(1, 1, i, j)][c+35] += W_s[i*2+j][c]

        # Term 7: M7 = (A12-A22)(B21+B22)
        # A parts: A12 (0,1), A22 (1,1)
        # B parts: B21 (1,0), B22 (1,1)
        # C parts: C11 (0,0), C12 (0,1), C22 (1,1)
        # C11 = ... - M7
        # C12 = ... + M7
        # C22 = ... + M7
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    # U
                    U_4[map_idx(0, 1, i, j)][c+42] += U_s[i*2+j][c]
                    U_4[map_idx(1, 1, i, j)][c+42] -= U_s[i*2+j][c]
                    # V
                    V_4[map_idx(1, 0, i, j)][c+42] += V_s[i*2+j][c]
                    V_4[map_idx(1, 1, i, j)][c+42] += V_s[i*2+j][c]
                    # W
                    # C11
                    W_4[map_idx(0, 0, i, j)][c+42] -= W_s[i*2+j][c]
                    # C12
                    W_4[map_idx(0, 1, i, j)][c+42] += W_s[i*2+j][c]
                    # C22
                    W_4[map_idx(1, 1, i, j)][c+42] += W_s[i*2+j][c]

        return U_4, V_4, W_4

    U1, V1, W1 = strassen_4x4()
    
    # Map U1, V1, W1 to U, V, W
    # U1: 16x49. U: 16x65.
    for i in range(16):
        for c in range(49):
            U[i][c] = U1[i][c]
            
    # V1: 16x49. V: 20x65.
    # V1 indices j*4+k. V indices j*5+k.
    # V1 covers B[0:4, 0:4]. Global V covers B[0:4, 0:5].
    # So V1[j*4+k] maps to V[j*5+k]
    for j in range(4):
        for k in range(4):
            v_idx = j*4+k
            global_v_idx = j*5+k
            for c in range(49):
                V[global_v_idx][c] = V1[v_idx][c]
                
    # W1: 16x49. W: 20x65.
    # W1 indices k*4+i. W indices k*5+i?
    # Wait, W is p*n = 5*4 = 20.
    # The output C is 4x5.
    # The Strassen part computes C[0:4, 0:4].
    # C[k, i] where k in 0..3, i in 0..3.
    # Global W index for C[k, i] is k*5 + i?
    # The problem statement says: "output factor follows the official trace convention W[k*n+i,r]"
    # Here n=4. So W index is k*4 + i?
    # Let's re-read: "W[k*n+i,r]".
    # If n=4, then index is k*4 + i.
    # The output matrix is C (4x5).
    # Usually C is p x n? Or n x p?
    # "multiplying a 4 by 4 matrix by a 4 by 5 matrix".
    # A is 4x4. B is 4x5.
    # C = A * B. C is 4x5.
    # Rows of C: 4 (from A rows). Cols of C: 5 (from B cols).
    # So C[k, i] where k is row (0..3), i is col (0..4).
    # The convention W[k*n+i] with n=4 means index = k*4 + i.
    # This matches the shape of C (4x5) flattened row-major?
    # Row k has 5 elements. Index k*5 + i would be row-major for 4x5.
    # But the prompt says W[k*n+i]. If n=4, it's k*4+i.
    # This implies the matrix is stored column-major? Or n is the number of columns?
    # In matrix mult A(m x n) * B(n x p) -> C(m x p).
    # Here A is 4x4 (m=4, n=4). B is 4x5 (n=4, p=5).
    # C is 4x5 (m=4, p=5).
    # The prompt says "W[k*n+i]".
    # If this is standard row-major for C(m x p), it should be k*p + i.
    # If it is k*n + i, and n=4, p=5, then k*4+i.
    # This looks like column-major storage of C (4x5) or row-major of C^T.
    # Let's assume the prompt is literal: Index = k * n + i.
    # For C[k, i], k in 0..3, i in 0..4.
    # Max index: 3*4 + 4 = 16. But W has size p*n = 20.
    # If index is k*4+i, max is 16. Indices 17,18,19 are unused?
    # Or maybe i goes up to n-1?
    # If the convention is W[k*n+i], and it must cover all p*n elements, then i must go up to p-1?
    # If i goes up to p-1 (4), and n=4, then k*4+i covers 0..19 if k goes up to 4? No k is 0..3.
    # 3*4+4 = 16.
    # There is a mismatch.
    # Let's check the seed program's W shape: (p*n, rank) = (20, rank).
    # If the index is k*n+i, and it must be unique for all 20 elements, then n must be p?
    # Or the formula is k*p + i?
    # "official trace convention W[k*n+i,r]"
    # In many contexts, n is the inner dimension.
    # If A is n x m and B is m x p, C is n x p.
    # Here "4 by 4 matrix by a 4 by 5 matrix".
    # A: 4x4. B: 4x5.
    # Usually "A by B" means A*B.
    # So A is 4x4, B is 4x5.
    # Inner dim m=4.
    # A is n x m? 4x4. So n=4, m=4.
    # B is m x p? 4x5. So m=4, p=5.
    # C is n x p? 4x5.
    # W index k*n + i.
    # If k is row (0..3) and i is col (0..4), k*4+i gives 0..16.
    # This doesn't fill 20.
    # Maybe the convention is W[i*n + k]?
    # If i is col (0..4) and k is row (0..3), i*4+k gives 0..19.
    # This fits perfectly.
    # So W[i*n + k] where i is col index of C, k is row index of C.
    # This is column-major storage of C.
    
    # Let's verify with the failed program's logic.
    # Failed program: W[global_w_idx][c] = W1[w_idx][c]
    # w_idx = k*4+i.
    # It assumed row-major.
    # If the evaluator uses W[k*n+i] literally, and n=4, and C is 4x5,
    # then the indices 0..19 must be generated.
    # If I use k*4+i, I only get 0..16.
    # If I use i*4+k (col-major), I get 0..19.
    # Given "W[k*n+i]" is the prompt, but n=4, p=5, it's ambiguous which is k and which is i.
    # However, "trace convention" often implies specific ordering.
    # Let's look at the dimensions in the prompt:
    # U: (n*m, rank) -> 4*4=16.
    # V: (m*p, rank) -> 4*5=20.
    # W: (p*n, rank) -> 5*4=20.
    # So W has 20 entries.
    # If the formula is k*n+i, and it must cover 20 unique indices, then one index must range 0..4 and the other 0..3.
    # If k ranges 0..3 (rows) and i ranges 0..4 (cols), k*4+i fails.
    # If i ranges 0..3 and k ranges 0..4, i*4+k works.
    # So likely: index = i * n + k, where i is the "outer" loop variable in the formula's first slot?
    # Or the prompt's k is the column and i is the row?
    # Let's assume the standard: C is p x n? No, C is n x p.
    # Let's assume the index is `col * n + row` (column-major).
    # So for C[row, col], W index is col * 4 + row.
    
    # In my Strassen part, I computed C[0:4, 0:4].
    # row k (0..3), col i (0..3).
    # W index should be i * 4 + k.
    
    # Let's re-map W1 to W.
    # W1 is 16x49. W1 index was k*4+i (row-major 4x4).
    # So W1[row*4+col] corresponds to C[row, col].
    # I need to place this at W[col*4 + row].
    
    for row in range(4):
        for col in range(4):
            w1_idx = row * 4 + col
            global_w_idx = col * 4 + row
            for c in range(49):
                W[global_w_idx][c] = W1[w1_idx][c]

    # --- Schoolbook 4x4x1 ---
    # Computes C[0:4, 4] (the 5th column).
    # C[row, 4] = sum_j A[row, j] B[j, 4].
    # row: 0..3. j: 0..4.
    # U index: row * 4 + j.
    # V index: j * 5 + 4.
    # W index: col * 4 + row = 4 * 4 + row = 16 + row.
    
    for row in range(4):
        for j in range(4):
            c = row * 4 + j
            U[c][49+c] = 1.0
            V[j * 5 + 4][49+c] = 1.0
            W[16 + row][49+c] = 1.0

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    print(construct())
