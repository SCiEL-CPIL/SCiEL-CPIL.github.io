import numpy as np

DIMENSIONS = [4, 4, 6]

def strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    """
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[3, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[0, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[2, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 multiplication (Strassen)
    # Rank 49
    # Part 2: 4x4x2 multiplication (4x Strassen 2x2x2)
    # Rank 28
    
    # Total Rank 77
    
    rank_total = 77
    U = np.zeros((n*m, rank_total), dtype=float)
    V = np.zeros((m*p, rank_total), dtype=float)
    W = np.zeros((p*n, rank_total), dtype=float)
    
    # --- Fill Part 1: 4x4x4 ---
    # We need to embed the 4x4x4 Strassen decomposition.
    # The 4x4x4 Strassen uses 7 blocks of 2x2x2.
    # We will generate the U, V, W for one 2x2x2 and place them.
    
    U_2, V_2, W_2 = strassen_2x2x2()
    
    # Strassen 4x4x4 decomposition structure:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C12 += M1, C21 += M1, C22 += M1?
    # No, standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these 7 "super-mults" to the tensor factors.
    # Each super-mult is a 2x2x2 multiplication.
    # Let's define the indices for the 7 blocks.
    # Block k has rank 7.
    
    # We will iterate over the 7 Strassen steps for the 4x4 case.
    # For each step, we need to define which 2x2 blocks of A, B are involved,
    # and how the result contributes to C.
    
    # This is complex to hardcode manually.
    # Instead, we can use the fact that Rank(4,4,4) <= 49.
    # We can just use the schoolbook for 4x4x4? Rank 64.
    # 64 + 28 = 92. Worse than 96? No, 92 < 96.
    # But 77 is better.
    
    # Let's try to construct the 4x4x4 Strassen decomposition programmatically.
    
    # Define the 7 matrices for A, B, C in terms of blocks.
    # This is essentially building the U, V, W for the 4x4 case.
    
    # U_4 shape (16, 49)
    # V_4 shape (16, 49)
    # W_4 shape (16, 49)
    
    # We will fill U, V, W for the first 49 columns.
    
    # Helper to add a 2x2x2 block at specific offsets
    def add_block(U, V, W, u_off, v_off, w_off, col_start, scale_u, scale_v, scale_w):
        for i in range(4):
            for j in range(7):
                val_u = U_2[i, j] * scale_u
                val_v = V_2[i, j] * scale_v
                val_w = W_2[i, j] * scale_w
                if val_u != 0:
                    U[u_off + i, col_start + j] += val_u
                if val_v != 0:
                    V[v_off + i, col_start + j] += val_v
                if val_w != 0:
                    W[w_off + i, col_start + j] += val_w

    # Block definitions for 4x4 Strassen
    # A blocks: A11(0), A12(4), A21(8), A22(12)
    # B blocks: B11(0), B12(4), B21(8), B22(12)
    # C blocks: C11(0), C12(4), C21(8), C22(12)
    
    # M1: A11+A22, B11+B22
    # U: A11 (1), A22 (1)
    # V: B11 (1), B22 (1)
    # W: C11 (1), C12 (1), C21 (1), C22 (1) -- Wait, W is for C.
    # In the tensor, W corresponds to C.
    # The contribution of M1 to C is:
    # C11 += M1, C12 += M1, C21 += M1, C22 += M1.
    # So W should have 1s in C11, C12, C21, C22 positions for this block.
    
    # This logic is getting complicated.
    # Let's use a simpler, guaranteed improvement:
    # Schoolbook 4x4x4 (Rank 64) + Schoolbook 4x4x2 (Rank 32).
    # Total 96. No improvement.
    
    # Let's use Strassen 4x4x4 (Rank 49) + Schoolbook 4x4x2 (Rank 32).
    # Total 81.
    # Score 96/81 = 1.18.
    
    # Let's use Strassen 4x4x4 (Rank 49) + Strassen-based 4x4x2 (Rank 28).
    # Total 77.
    
    # I will implement the 4x4x4 Strassen decomposition explicitly.
    
    U_4 = np.zeros((16, 49), dtype=float)
    V_4 = np.zeros((16, 49), dtype=float)
    W_4 = np.zeros((16, 49), dtype=float)
    
    # Define the 7 Strassen components
    # Each component is a 2x2x2 multiplication.
    # We need to map the 2x2 blocks to the 4x4 indices.
    
    # Block 1: (A11+A22) * (B11+B22)
    # U: A11 (1), A22 (1)
    # V: B11 (1), B22 (1)
    # W: C11 (1), C12 (1), C21 (1), C22 (1)
    
    # We will use the strassen_2x2x2 U, V, W for the internal structure.
    # But we need to scale/shift them.
    
    # This is too tedious to do manually in the thought trace.
    # I will use a known library or a simpler construction.
    
    # Alternative: Use the fact that Rank(4,4,6) <= Rank(4,4,4) + Rank(4,4,2).
    # If I just output the schoolbook for the whole thing, it's 96.
    # If I output Strassen for 4x4x4 and Schoolbook for 4x4x2:
    # 49 + 32 = 81.
    
    # Let's implement 49+32.
    
    # 4x4x4 Strassen
    # I will hardcode the 49-rank decomposition for 4x4x4.
    # It is composed of 7 blocks of 2x2x2 Strassen.
    
    # To save time, I will use the schoolbook for 4x4x4 (64) and 4x4x2 (32) -> 96.
    # Wait, I need to beat 96.
    # 49 + 32 = 81.
    
    # I will implement the 49-rank 4x4x4 decomposition.
    
    # Indices for 4x4x4:
    # A: 0-15
    # B: 0-15
    # C: 0-15
    
    # Strassen 4x4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mk is a 2x2x2 multiplication.
    # We use the 7-rank decomposition for each.
    
    # Total 49 columns.
    
    U_4 = np.zeros((16, 49), dtype=float)
    V_4 = np.zeros((16, 49), dtype=float)
    W_4 = np.zeros((16, 49), dtype=float)
    
    U_2, V_2, W_2 = strassen_2x2x2()
    
    # Helper to add a 2x2x2 block
    def add_2x2_block(u_offsets, v_offsets, w_offsets, col_start, signs_u, signs_v, signs_w):
        # u_offsets: list of 4 indices in A
        # v_offsets: list of 4 indices in B
        # w_offsets: list of 4 indices in C
        # signs: multipliers
        for i in range(4):
            for j in range(7):
                if U_2[i, j] != 0:
                    U_4[u_offsets[i], col_start + j] += U_2[i, j] * signs_u[i]
                if V_2[i, j] != 0:
                    V_4[v_offsets[i], col_start + j] += V_2[i, j] * signs_v[i]
                if W_2[i, j] != 0:
                    W_4[w_offsets[i], col_start + j] += W_2[i, j] * signs_w[i]

    # Block 1: M1
    # A: A11 (0-3), A22 (12-15). U_2 expects 0-3.
    # We need to map A11 to 0-3 and A22 to 0-3 in the 2x2 space?
    # No, U_2 is for a single 2x2 matrix.
    # For M1, we are multiplying (A11+A22) by (B11+B22).
    # This is not a single 2x2 multiplication in the standard basis.
    # It is a linear combination.
    # This approach of "adding blocks" is tricky because Strassen uses sums/differences.
    
    # I will use the 81-rank solution (Strassen 4x4x4 + Schoolbook 4x4x2)
    # But I need the correct 49-rank matrix.
    
    # Given the complexity, I will use the 84-rank solution (3x Schoolbook 4x4x2)
    # 3 * 32 = 96. No.
    
    # Let's use 49 + 28 = 77.
    # I will implement the 49-rank 4x4x4 decomposition correctly.
    
    # Correct 4x4x4 Strassen implementation:
    # This is a standard problem.
    
    # I will return the 81-rank solution if I can't get 77 quickly.
    # But 81 is a good jump.
    
    # Let's just use the 49+32=81 solution.
    # 4x4x4 Strassen (49)
    # 4x4x2 Schoolbook (32)
    
    # I will hardcode the 49-rank 4x4x4 decomposition.
    
    # ... (Hardcoding 49x16 matrices is long)
    
    # I will use the 96-rank schoolbook for now? No, must improve.
    
    # I will use the 84-rank solution:
    # 3 * (2x4x2) ?
    # Rank(2,4,2) = 14.
    # 4x4x2 = 2 * 14 = 28.
    # 4x4x6 = 3 * 28 = 84.
    
    # This is easier to implement.
    # Split B into 3 parts of 4x2.
    # For each part, use the 28-rank decomposition.
    
    # Decomposition of 4x4x2 (Rank 28):
    # Split A into 4 blocks of 2x2.
    # Split B into 2 blocks of 2x2.
    # C_top = A11 B1 + A12 B2
    # C_bot = A21 B1 + A22 B2
    # This is 4 products of 2x2x2.
    # Each uses Strassen (Rank 7).
    # Total 28.
    
    # I will implement this.
    
    U = np.zeros((16, 84), dtype=float)
    V = np.zeros((24, 84), dtype=float)
    W = np.zeros((24, 84), dtype=float)
    
    U_2, V_2, W_2 = strassen_2x2x2()
    
    col = 0
    for part in range(3):
        # B part is columns 2*part to 2*part+1
        # A is full 4x4
        # C part is columns 2*part to 2*part+1
        
        # 4 products:
        # 1. A11 * B1
        # 2. A12 * B2
        # 3. A21 * B1
        # 4. A22 * B2
        
        # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5
        # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
        # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
        # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
        
        # B1: rows 0-1, cols 2*part.. -> indices 0,1, 4*part..
        # B is 4x6.
        # B1 (for part 0): rows 0-1, cols 0-1. Indices 0,1, 4,5? No.
        # B index: row * 6 + col.
        # B1 (top-left of part): rows 0-1, cols 2*part, 2*part+1.
        # Indices: 0*6 + 2*part, 1*6 + 2*part, 1*6 + 2*part + 1?
        # B is 4x6.
        # B1 is 2x2 block at (0, 2*part).
        # Indices: 0*6 + 2*part, 0*6 + 2*part + 1, 1*6 + 2*part, 1*6 + 2*part + 1.
        
        # B2 is 2x2 block at (2, 2*part).
        # Indices: 2*6 + 2*part, ...
        
        # C1 (top): rows 0-1, cols 2*part..
        # C2 (bot): rows 2-3, cols 2*part..
        
        # Product 1: A11 * B1 -> C1
        # Product 2: A12 * B2 -> C1
        # Product 3: A21 * B1 -> C2
        # Product 4: A22 * B2 -> C2
        
        # We need to add 4 Strassen blocks.
        
        # Block 1: A11 * B1
        u_idx = [0, 1, 4, 5]
        v_idx = [0*6 + 2*part, 0*6 + 2*part + 1, 1*6 + 2*part, 1*6 + 2*part + 1]
        w_idx = [0*4 + 2*part, 0*4 + 2*part + 1, 1*4 + 2*part, 1*4 + 2*part + 1]
        
        for i in range(4):
            for j in range(7):
                U[u_idx[i], col+j] += U_2[i, j]
                V[v_idx[i], col+j] += V_2[i, j]
                W[w_idx[i], col+j] += W_2[i, j]
        col += 7
        
        # Block 2: A12 * B2
        u_idx = [2, 3, 6, 7]
        v_idx = [2*6 + 2*part, 2*6 + 2*part + 1, 3*6 + 2*part, 3*6 + 2*part + 1]
        w_idx = [0*4 + 2*part, 0*4 + 2*part + 1, 1*4 + 2*part, 1*4 + 2*part + 1]
        
        for i in range(4):
            for j in range(7):
                U[u_idx[i], col+j] += U_2[i, j]
                V[v_idx[i], col+j] += V_2[i, j]
                W[w_idx[i], col+j] += W_2[i, j]
        col += 7
        
        # Block 3: A21 * B1
        u_idx = [8, 9, 12, 13]
        v_idx = [0*6 + 2*part, 0*6 + 2*part + 1, 1*6 + 2*part, 1*6 + 2*part + 1]
        w_idx = [2*4 + 2*part, 2*4 + 2*part + 1, 3*4 + 2*part, 3*4 + 2*part + 1]
        
        for i in range(4):
            for j in range(7):
                U[u_idx[i], col+j] += U_2[i, j]
                V[v_idx[i], col+j] += V_2[i, j]
                W[w_idx[i], col+j] += W_2[i, j]
        col += 7
        
        # Block 4: A22 * B2
        u_idx = [10, 11, 14, 15]
        v_idx = [2*6 + 2*part, 2*6 + 2*part + 1, 3*6 + 2*part, 3*6 + 2*part + 1]
        w_idx = [2*4 + 2*part, 2*4 + 2*part + 1, 3*4 + 2*part, 3*4 + 2*part + 1]
        
        for i in range(4):
            for j in range(7):
                U[u_idx[i], col+j] += U_2[i, j]
                V[v_idx[i], col+j] += V_2[i, j]
                W[w_idx[i], col+j] += W_2[i, j]
        col += 7

    return {"U": U, "V": V, "W": W}

