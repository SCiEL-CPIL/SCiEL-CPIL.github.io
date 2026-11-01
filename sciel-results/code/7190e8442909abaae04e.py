"""Rewritten program: rank-32 decomposition for 2x2x9 matrix multiplication."""
DIMENSIONS = [2, 2, 9]

def construct():
    """
    Rank-32 exact tensor decomposition for multiplying a 2x2 matrix by a 2x9 matrix.
    
    Strategy:
    Split the 2x9 matrix B into four 2x2 blocks and one 2x1 block.
    Use Strassen's algorithm (rank 7) for each 2x2 block.
    Use naive multiplication (rank 4) for the 2x1 block.
    Total rank = 4 * 7 + 4 = 32.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 18 x 32
    W = [[0] * rank for _ in range(p * n)]  # 18 x 32
    
    col = 0
    
    # Strassen factors for A (2x2)
    # Indices in U: 0: A00, 1: A01, 2: A10, 3: A11
    # s1 = A00 + A11
    # s2 = A10 + A11
    # s3 = A00
    # s4 = A11
    # s5 = A00 + A01
    # s6 = A10 - A00
    # s7 = A01 - A11
    
    def add_strassen_block(b_start_col):
        """
        Adds 7 terms for a 2x2 block of B starting at column b_start_col.
        b_start_col is 0, 2, 4, 6.
        """
        nonlocal col
        # V indices for B block:
        # B0, B1 are at rows 0, 1 (for col 0) -> actually V is m x p = 2 x 9.
        # V[j * p + k] corresponds to B[j, k].
        # For block starting at k, B[0, k], B[1, k] are V[k], V[9+k].
        # B[0, k+1], B[1, k+1] are V[k+1], V[9+k+1].
        
        # t1 = B0k + B1,k+1
        U[0][col] = 1; U[3][col] = 1
        V[b_start_col][col] = 1; V[9 + b_start_col + 1][col] = 1
        # W: C0k = P1...
        # C0k = P1 - P3 - P4 + P7
        # C0,k+1 = P4 + P5 + P7
        # C1k = P2 + P3 - P6
        # C1,k+1 = P1 - P2 + P6
        
        # W indices: W[k*n + i] -> W[k*2 + i].
        # C0k is W[k*2 + 0]
        # C0,k+1 is W[(k+1)*2 + 0]
        # C1k is W[k*2 + 1]
        # C1,k+1 is W[(k+1)*2 + 1]
        
        W[b_start_col * 2 + 0][col] = 1
        W[(b_start_col + 1) * 2 + 0][col] = 0
        W[b_start_col * 2 + 1][col] = 0
        W[(b_start_col + 1) * 2 + 1][col] = 1
        # Wait, W is the output coefficient.
        # Term 1 (P1) contributes to C0k (+1), C1,k+1 (+1).
        # Let's map Strassen outputs to W.
        
        # P1: C0k += 1, C1,k+1 += 1
        # P2: C1k += 1, C1,k+1 -= 1
        # P3: C0k -= 1, C1k += 1
        # P4: C0k -= 1, C0,k+1 += 1
        # P5: C0,k+1 += 1
        # P6: C1k -= 1, C1,k+1 += 1
        # P7: C0k += 1, C0,k+1 += 1
        
        # Re-initialize W for this term
        W[b_start_col * 2 + 0][col] = 1
        W[(b_start_col + 1) * 2 + 1][col] = 1
        col += 1
        
        # P2
        U[2][col] = 1; U[3][col] = 1
        V[b_start_col][col] = 1
        W[b_start_col * 2 + 1][col] = 1
        W[(b_start_col + 1) * 2 + 1][col] = -1
        col += 1
        
        # P3
        U[0][col] = 1
        V[9 + b_start_col][col] = 1; V[9 + b_start_col + 1][col] = -1
        W[b_start_col * 2 + 0][col] = -1
        W[b_start_col * 2 + 1][col] = 1
        col += 1
        
        # P4
        U[3][col] = 1
        V[b_start_col + 1][col] = 1; V[b_start_col][col] = -1
        W[b_start_col * 2 + 0][col] = -1
        W[(b_start_col + 1) * 2 + 0][col] = 1
        col += 1
        
        # P5
        U[0][col] = 1; U[1][col] = 1
        V[9 + b_start_col + 1][col] = 1
        W[(b_start_col + 1) * 2 + 0][col] = 1
        col += 1
        
        # P6
        U[2][col] = 1; U[0][col] = -1
        V[b_start_col][col] = 1; V[b_start_col + 1][col] = 1
        W[b_start_col * 2 + 1][col] = -1
        W[(b_start_col + 1) * 2 + 1][col] = 1
        col += 1
        
        # P7
        U[1][col] = 1; U[3][col] = -1
        V[9 + b_start_col][col] = 1; V[9 + b_start_col + 1][col] = 1
        W[b_start_col * 2 + 0][col] = 1
        W[(b_start_col + 1) * 2 + 0][col] = 1
        col += 1

    # Add 4 Strassen blocks for columns 0-1, 2-3, 4-5, 6-7
    for b_start in [0, 2, 4, 6]:
        add_strassen_block(b_start)
        
    # Naive block for column 8
    # C08 = A00 B08 + A01 B18
    # C18 = A10 B08 + A11 B18
    
    # Term 1: A00 B08
    U[0][col] = 1
    V[8][col] = 1
    W[8 * 2 + 0][col] = 1
    col += 1
    
    # Term 2: A01 B18
    U[1][col] = 1
    V[9 + 8][col] = 1
    W[8 * 2 + 0][col] = 1
    col += 1
    
    # Term 3: A10 B08
    U[2][col] = 1
    V[8][col] = 1
    W[8 * 2 + 1][col] = 1
    col += 1
    
    # Term 4: A11 B18
    U[3][col] = 1
    V[9 + 8][col] = 1
    W[8 * 2 + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
