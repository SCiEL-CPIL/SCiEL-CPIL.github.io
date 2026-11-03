import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We'll build the decomposition using Strassen for the 4x4x4 part and naive for the 4x4x5 part
    # Total rank = 114 (7*16 for 4x4x4 via Strassen + 20 for 4x4x5 naive)
    # Wait, let me reconsider.
    
    # For 4x4x4: Strassen gives rank 16 (7 for outer blocks, each block is 2x2x2 with rank 7? No)
    # Actually, Strassen for 4x4x4: 
    #   Split 4x4 into 2x2 blocks (4 blocks)
    #   Use Strassen formulas: 7 products of 2x2 matrices
    #   Each 2x2x2 product can be done with rank 7 (Strassen) or rank 8 (naive)
    #   So total rank for 4x4x4 = 7 * 7 = 49 if using Strassen recursively
    #   Or 7 * 8 = 56 if using naive for the 2x2x2
    #   
    # Actually, the known optimal rank for 4x4x4 is 49 (using Strassen recursively).
    # But we can also just use the standard Strassen: 7 products of 2x2, each done naively (rank 8) = 56
    # Or use Strassen for the 2x2: 7*7 = 49
    #
    # For 4x4x5: We can split the 5 into 4+1, use 4x4x4 (rank 49) + 4x4x1 (rank 16) = 65
    # Total: 49 + 65 = 114
    #
    # Alternatively, we could try to be smarter, but let's start with this.
    #
    # Actually, let me just implement the naive approach first to get a valid solution,
    # then optimize.
    
    # Let's implement a rank-114 decomposition:
    # 4x4x4 part: Strassen with recursive 2x2x2 (rank 49)
    # 4x4x5 part: split 5 = 4+1, so 4x4x4 (rank 49) + 4x4x1 (rank 16) = 65
    # Total: 49 + 65 = 114
    
    rank = 114
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper: add a Strassen 2x2x2 decomposition at given offsets
    def add_strassen_2x2x2(U, V, W, i_off, j_off, k_off, col_start):
        """Add Strassen 2x2x2 decomposition for blocks starting at (i_off, j_off, k_off)"""
        # Strassen formulas for 2x2x2:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each M is a rank-1 tensor (product of 3 vectors)
        # For M1: U has (A00+A11), V has (B00+B11), W has identity for C00,C11
        # Actually, each M corresponds to a rank-1 term in the tensor
        
        # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
        # U indices: (i_off, j_off) and (i_off+1, j_off+1) with coeff 1
        # V indices: (j_off, k_off) and (j_off+1, k_off+1) with coeff 1
        # W indices: (k_off, i_off) with coeff 1 (for C00), (k_off+1, i_off+1) with coeff 1 (for C11)
        
        U[i_off * m + j_off, col_start] = 1.0
        U[(i_off+1) * m + (j_off+1), col_start] = 1.0
        V[j_off * p + k_off, col_start] = 1.0
        V[(j_off+1) * p + (k_off+1), col_start] = 1.0
        W[k_off * n + i_off, col_start] = 1.0
        W[(k_off+1) * n + (i_off+1), col_start] = 1.0
        col_start += 1
        
        # M2 = (A10+A11)B00 -> contributes to C10 and C11
        U[i_off * m + j_off, col_start] = 0.0  # A00 not in M2
        U[(i_off+1) * m + j_off, col_start] = 1.0  # A10
        U[(i_off+1) * m + (j_off+1), col_start] = 1.0  # A11
        V[j_off * p + k_off, col_start] = 1.0  # B00
        W[k_off * n + (i_off+1), col_start] = 1.0  # C10
        W[(k_off+1) * n + (i_off+1), col_start] = 1.0  # C11
        col_start += 1
        
        # M3 = A00(B01-B11) -> contributes to C01 and C11
        U[i_off * m + j_off, col_start] = 1.0  # A00
        V[j_off * p + (k_off+1), col_start] = 1.0  # B01
        V[(j_off+1) * p + (k_off+1), col_start] = -1.0  # -B11
        W[(k_off+1) * n + i_off, col_start] = 1.0  # C01
        W[(k_off+1) * n + (i_off+1), col_start] = 1.0  # C11
        col_start += 1
        
        # M4 = A11(B10-B00) -> contributes to C00 and C10
        U[(i_off+1) * m + (j_off+1), col_start] = 1.0  # A11
        V[(j_off+1) * p + k_off, col_start] = 1.0  # B10
        V[j_off * p + k_off, col_start] = -1.0  # -B00
        W[k_off * n + i_off, col_start] = 1.0  # C00
        W[k_off * n + (i_off+1), col_start] = 1.0  # C10
        col_start += 1
        
        # M5 = (A00+A01)B11 -> contributes to C00 and C01
        U[i_off * m + j_off, col_start] = 1.0  # A00
        U[i_off * m + (j_off+1), col_start] = 1.0  # A01
        V[(j_off+1) * p + (k_off+1), col_start] = 1.0  # B11
        W[k_off * n + i_off, col_start] = -1.0  # -C00 (since C00 = M1+M4-M5+M7)
        W[(k_off+1) * n + i_off, col_start] = 1.0  # C01
        col_start += 1
        
        # M6 = (A10-A00)(B00+B01) -> contributes to C11
        U[(i_off+1) * m + j_off, col_start] = 1.0  # A10
        U[i_off * m + j_off, col_start] = -1.0  # -A00
        V[j_off * p + k_off, col_start] = 1.0  # B00
        V[j_off * p + (k_off+1), col_start] = 1.0  # B01
        W[(k_off+1) * n + (i_off+1), col_start] = 1.0  # C11
        col_start += 1
        
        # M7 = (A01-A11)(B10+B11) -> contributes to C00
        U[i_off * m + (j_off+1), col_start] = 1.0  # A01
        U[(i_off+1) * m + (j_off+1), col_start] = -1.0  # -A11
        V[(j_off+1) * p + k_off, col_start] = 1.0  # B10
        V[(j_off+1) * p + (k_off+1), col_start] = 1.0  # B11
        W[k_off * n + i_off, col_start] = 1.0  # C00
        col_start += 1
        
        return col_start
    
    # Helper: add naive 2x2x2 decomposition
    def add_naive_2x2x2(U, V, W, i_off, j_off, k_off, col_start):
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    u_idx = (i_off + di) * m + (j_off + dj)
                    v_idx = (j_off + dj) * p + (k_off + dk)
                    w_idx = (k_off + dk) * n + (i_off + di)
                    U[u_idx, col_start] = 1.0
                    V[v_idx, col_start] = 1.0
                    W[w_idx, col_start] = 1.0
                    col_start += 1
        return col_start
    
    # Part 1: 4x4x4 (columns 0-3 of p)
    # Use Strassen on 2x2 blocks, each block uses Strassen 2x2x2
    # 4 outer blocks, 7 Strassen terms per 4x4, but we do it recursively
    
    # Actually, let's just do: 4x4x4 = 4 blocks of 2x2x2, each done with Strassen (rank 7)
    # Total: 4 * 7 = 28? No, that's not right.
    
    # Let me just use naive for everything first to get a valid solution, then optimize.
    # Naive rank = 4*4*9 = 144
    
    # For now, let's just return the naive solution to ensure validity.
    # We can optimize later.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
