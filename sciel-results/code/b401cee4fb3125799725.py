import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by concatenating blocks.
    # Block 1: C[0:3, 0:4] from A[0:3, 0:4] * B[0:4, 0:4]
    # Block 2: C[0:3, 4:7] from A[0:3, 0:4] * B[0:4, 4:7]
    
    # Helper to add a schoolbook block
    def add_schoolbook_block(U, V, W, i_start, i_end, j_start, j_end, k_start, k_end, offset):
        rank_added = 0
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    col = offset + rank_added
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    rank_added += 1
        return rank_added

    # Helper to add a Strassen 2x2x2 block
    # Maps local (li, lj, lk) to global (gi, gj, gk)
    def add_strassen_block(U, V, W, i_base, j_base, k_base, offset):
        # i_base, j_base, k_base are the starting indices for the 2x2x2 block
        # Local indices 0,1 map to global i_base+0, i_base+1, etc.
        
        # Strassen formulas
        # M1: (A11+A22)(B11+B22)
        # M2: (A21+A22)B11
        # M3: A11(B12-B22)
        # M4: A22(B21-B11)
        # M5: (A11+A12)B22
        # M6: (A21-A11)(B11+B12)
        # M7: (A12-A22)(B21+B22)
        
        rank_added = 0
        
        # Define the 7 multiplications
        # Each mult c involves U factors, V factors, W factors
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        
        # M1
        # A parts: (1,0) and (1,1) -> li=0, lj=0 and li=1, lj=1
        # B parts: (0,0) and (1,1) -> lj=0, lk=0 and lj=1, lk=1
        # C parts: (0,0), (1,1), (0,1), (1,0)
        
        # We will create 7 columns.
        
        # M1
        col = offset + rank_added
        # U
        U[(i_base+0)*m + (j_base+0)][col] = 1
        U[(i_base+1)*m + (j_base+1)][col] = 1
        # V
        V[(j_base+0)*p + (k_base+0)][col] = 1
        V[(j_base+1)*p + (k_base+1)][col] = 1
        # W
        W[(k_base+0)*n + (i_base+0)][col] = 1
        W[(k_base+1)*n + (i_base+1)][col] = -1
        W[(k_base+1)*n + (i_base+0)][col] = 1
        W[(k_base+0)*n + (i_base+1)][col] = 1
        rank_added += 1
        
        # M2: (A21+A22)B11
        col = offset + rank_added
        # A: (1,0) and (1,1)
        U[(i_base+1)*m + (j_base+0)][col] = 1
        U[(i_base+1)*m + (j_base+1)][col] = 1
        # B: (0,0)
        V[(j_base+0)*p + (k_base+0)][col] = 1
        # C: (1,0)
        W[(k_base+0)*n + (i_base+1)][col] = 1
        rank_added += 1
        
        # M3: A11(B12-B22)
        col = offset + rank_added
        # A: (0,0)
        U[(i_base+0)*m + (j_base+0)][col] = 1
        # B: (0,1) and (1,1)
        V[(j_base+0)*p + (k_base+1)][col] = 1
        V[(j_base+1)*p + (k_base+1)][col] = -1
        # C: (0,1)
        W[(k_base+1)*n + (i_base+0)][col] = 1
        rank_added += 1
        
        # M4: A22(B21-B11)
        col = offset + rank_added
        # A: (1,1)
        U[(i_base+1)*m + (j_base+1)][col] = 1
        # B: (1,0) and (0,0)
        V[(j_base+1)*p + (k_base+0)][col] = 1
        V[(j_base+0)*p + (k_base+0)][col] = -1
        # C: (1,0)
        W[(k_base+0)*n + (i_base+1)][col] = 1
        rank_added += 1
        
        # M5: (A11+A12)B22
        col = offset + rank_added
        # A: (0,0) and (0,1)
        U[(i_base+0)*m + (j_base+0)][col] = 1
        U[(i_base+0)*m + (j_base+1)][col] = 1
        # B: (1,1)
        V[(j_base+1)*p + (k_base+1)][col] = 1
        # C: (0,1)
        W[(k_base+1)*n + (i_base+0)][col] = 1
        rank_added += 1
        
        # M6: (A21-A11)(B11+B12)
        col = offset + rank_added
        # A: (1,0) and (0,0)
        U[(i_base+1)*m + (j_base+0)][col] = 1
        U[(i_base+0)*m + (j_base+0)][col] = -1
        # B: (0,0) and (0,1)
        V[(j_base+0)*p + (k_base+0)][col] = 1
        V[(j_base+0)*p + (k_base+1)][col] = 1
        # C: (1,1)
        W[(k_base+1)*n + (i_base+1)][col] = 1
        rank_added += 1
        
        # M7: (A12-A22)(B21+B22)
        col = offset + rank_added
        # A: (0,1) and (1,1)
        U[(i_base+0)*m + (j_base+1)][col] = 1
        U[(i_base+1)*m + (j_base+1)][col] = -1
        # B: (1,0) and (1,1)
        V[(j_base+1)*p + (k_base+0)][col] = 1
        V[(j_base+1)*p + (k_base+1)][col] = 1
        # C: (1,1)
        W[(k_base+1)*n + (i_base+1)][col] = 1
        rank_added += 1
        
        return rank_added

    # Initialize matrices with large enough rank
    # Max rank estimate: 64
    max_rank = 100
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    current_offset = 0
    
    # Block 1: C[0:3, 0:4]
    # Sub-block 1a: C[0:2, 0:4] from A[0:2, 0:4] * B[0:4, 0:4]
    # Split A[0:2, 0:4] into A[0:2, 0:2] and A[0:2, 2:4]
    # Split B[0:4, 0:4] into B[0:2, 0:4] and B[2:4, 0:4]
    # C = A_L B_T + A_R B_B
    # A_L is 2x2, B_T is 2x4. R(2,2,4) = R(2,2,2)+R(2,2,2) = 7+7=14?
    # Wait, B_T is 2x4. Split B_T into B_T1 (2x2) and B_T2 (2x2).
    # A_L B_T1 is 2x2x2 (Strassen, 7)
    # A_L B_T2 is 2x2x2 (Strassen, 7)
    # A_R B_B1 is 2x2x2 (Strassen, 7)
    # A_R B_B2 is 2x2x2 (Strassen, 7)
    # Total 28.
    
    # My previous calc: R(2,4,4) = 14.
    # Let's re-verify.
    # A is 2x4. B is 4x4.
    # A = [A1 A2] (2x2 each)
    # B = [B1; B2] (2x4 each)
    # C = A1 B1 + A2 B2
    # A1 is 2x2. B1 is 2x4.
    # R(2,2,4). Split B1 into B11 (2x2) and B12 (2x2).
    # A1 B11 is 2x2x2 (Rank 7).
    # A1 B12 is 2x2x2 (Rank 7).
    # Total 14 for A1 B1.
    # Similarly 14 for A2 B2.
    # Total 28.
    
    # So R(2,4,4) is 28, not 14.
    # My previous error: I assumed R(2,2,4)=7 which is wrong.
    # So Block 1a rank is 28.
    
    # Sub-block 1b: C[2:3, 0:4] from A[2:3, 0:4] * B[0:4, 0:4]
    # A is 1x4. B is 4x4.
    # Rank 16.
    
    # Total Block 1 rank: 28 + 16 = 44.
    
    # Block 2: C[0:3, 4:7]
    # Sub-block 2a: C[0:2, 4:7] from A[0:2, 0:4] * B[0:4, 4:7]
    # A is 2x4. B is 4x3.
    # A = [A1 A2]. B = [B1; B2] (2x3 each).
    # C = A1 B1 + A2 B2.
    # R(2,2,3) = 11.
    # Total 22.
    
    # Sub-block 2b: C[2:3, 4:7] from A[2:3, 0:4] * B[0:4, 4:7]
    # A is 1x4. B is 4x3.
    # Rank 12.
    
    # Total Block 2 rank: 22 + 12 = 34.
    
    # Total Rank: 44 + 34 = 78.
    # This is worse than 84? No, 78 < 84.
    # But my previous estimate was 64. Where did I go wrong?
    # Ah, I thought R(2,4,4)=14. It is 28.
    # Is there a better way?
    
    # What if we use Strassen on 2x2x2 blocks directly?
    # We have 4 Strassen blocks for 2x4x4.
    # 4 * 7 = 28.
    
    # Let's stick with 78. It's an improvement.
    
    # Construct Block 1a:
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    current_offset += add_strassen_block(U, V, W, 0, 0, 0, current_offset)
    
    # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # B is 2x2. k_base=2.
    current_offset += add_strassen_block(U, V, W, 0, 0, 2, current_offset)
    
    # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A is 2x2 (j_base=2). B is 2x2 (k_base=0).
    current_offset += add_strassen_block(U, V, W, 0, 2, 0, current_offset)
    
    # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # A is 2x2 (j_base=2). B is 2x2 (k_base=2).
    current_offset += add_strassen_block(U, V, W, 0, 2, 2, current_offset)
    
    # Block 1b: Schoolbook for i=2, j=0..3, k=0..3
    current_offset += add_schoolbook_block(U, V, W, 2, 3, 0, 4, 0, 4, current_offset)
    
    # Block 2a:
    # A[0:2, 0:2] * B[0:2, 4:7] -> C[0:2, 4:7]
    # B is 2x3. Split B into 2x2 and 2x1?
    # Strassen only works for 2x2x2.
    # We can use Strassen for k=4,5 (2x2) and schoolbook for k=6 (2x1).
    # A[0:2, 0:2] * B[0:2, 4:6] (2x2x2) -> Strassen
    current_offset += add_strassen_block(U, V, W, 0, 0, 4, current_offset)
    
    # A[0:2, 0:2] * B[0:2, 6:7] (2x2x1) -> Schoolbook (Rank 4)
    current_offset += add_schoolbook_block(U, V, W, 0, 2, 0, 2, 6, 7, current_offset)
    
    # A[0:2, 2:4] * B[2:4, 4:6] (2x2x2) -> Strassen
    current_offset += add_strassen_block(U, V, W, 0, 2, 4, current_offset)
    
    # A[0:2, 2:4] * B[2:4, 6:7] (2x2x1) -> Schoolbook (Rank 4)
    current_offset += add_schoolbook_block(U, V, W, 0, 2, 2, 4, 6, 7, current_offset)
    
    # Block 2b: Schoolbook for i=2, j=0..3, k=4..7
    current_offset += add_schoolbook_block(U, V, W, 2, 3, 0, 4, 4, 7, current_offset)
    
    # Trim matrices to actual rank
    U = [row[:current_offset] for row in U]
    V = [row[:current_offset] for row in V]
    W = [row[:current_offset] for row in W]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
