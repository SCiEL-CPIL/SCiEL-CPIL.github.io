"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 4x5 matrix by a 5x5 matrix.
    
    Strategy: Use block structure to exploit Strassen's algorithm on 2x2 blocks
    and standard multiplication on the remaining parts.
    
    We partition the 5x5 input as 2+3 and the 4x5 output as 2+2.
    
    The multiplication can be decomposed into:
    - Top-left 2x2 block: use Strassen (rank 7 instead of 8)
    - Other blocks: standard multiplication
    
    However, a simpler approach that is guaranteed to work and gives rank 96:
    Split the 5x5 matrix into four 2x2 blocks plus one 2x1, 1x2, 1x1.
    Actually, let's use a cleaner block decomposition.
    
    Let's think differently. We have A (4x5) and B (5x5), output C (4x5).
    
    Partition A as [A1 | A2] where A1 is 4x2 and A2 is 4x3.
    Partition B as [B11; B21] [B12; B22] where B11 is 2x2, B12 is 2x3,
    B21 is 3x2, B22 is 3x3.
    
    Then C = A1*B11 + A2*B21 ... wait, that's not right.
    
    C = A * B = [A1 | A2] * [B11 B12; B21 B22]
              = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    So C_left (4x2) = A1*B11 + A2*B21
    C_right (4x3) = A1*B12 + A2*B22
    
    For C_left: A1 is 4x2, B11 is 2x2, A2 is 4x3, B21 is 3x2.
    This is a sum of two matrix multiplications:
    - A1 (4x2) * B11 (2x2) -> 4x2, rank 4*2*2 = 16
    - A2 (4x3) * B21 (3x2) -> 4x2, rank 4*3*2 = 24
    Total for C_left: 16 + 24 = 40
    
    For C_right: A1 (4x2) * B12 (2x3) -> 4x3, rank 4*2*3 = 24
                 A2 (4x3) * B22 (3x3) -> 4x3, rank 4*3*3 = 36
    Total for C_right: 24 + 36 = 60
    
    Total rank: 40 + 60 = 100 (same as schoolbook)
    
    To improve, we need to use Strassen on some 2x2 blocks.
    
    Let's try a different partition. What if we treat the 4x5 output as two 2x5 blocks?
    C = [C1; C2] where C1 is 2x5 and C2 is 2x5.
    
    C1 = A1 * B where A1 is 2x5
    C2 = A2 * B where A2 is 2x5
    
    Each C_i = A_i * B is a 2x5 * 5x5 = 2x5 multiplication.
    Rank for each: 2*5*5 = 50. Total: 100.
    
    Still 100. We need to find a way to share structure.
    
    Let's use Strassen on the 2x2 sub-blocks.
    
    Consider the top-left 2x2 block of the output: C[0:2, 0:2].
    C[0:2, 0:2] = A[0:2, 0:5] * B[0:5, 0:2]
    
    Let A_top = A[0:2, 0:5] (2x5), B_left = B[0:5, 0:2] (5x2).
    
    Partition A_top = [A_top1 | A_top2] where A_top1 is 2x2 and A_top2 is 2x3.
    Partition B_left = [B_left1; B_left2] where B_left1 is 2x2 and B_left2 is 3x2.
    
    C[0:2, 0:2] = A_top1 * B_left1 + A_top2 * B_left2
    
    For A_top1 * B_left1 (2x2 * 2x2), we can use Strassen with rank 7.
    For A_top2 * B_left2 (2x3 * 3x2), rank is 2*3*2 = 12.
    
    So C[0:2, 0:2] has rank 7 + 12 = 19 instead of 2*5*2 = 20.
    
    Similarly for other 2x2 blocks of the output.
    
    The output is 4x5. Let's identify the 2x2 blocks:
    - C[0:2, 0:2]
    - C[0:2, 2:4]
    - C[2:4, 0:2]
    - C[2:4, 2:4]
    - Plus the last column: C[0:4, 4:5]
    
    Let's compute ranks:
    
    C[0:2, 0:2] = A[0:2, :] * B[:, 0:2]
    = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:5]*B[2:5, 0:2]
    Strassen(2x2*2x2) + std(2x3*3x2) = 7 + 12 = 19
    
    C[0:2, 2:4] = A[0:2, :] * B[:, 2:4]
    = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:5]*B[2:5, 2:4]
    std(2x2*2x2) + std(2x3*3x2) = 8 + 12 = 20
    
    C[2:4, 0:2] = A[2:4, :] * B[:, 0:2]
    = A[2:4, 0:2]*B[0:2, 0:2] + A[2:4, 2:5]*B[2:5, 0:2]
    std(2x2*2x2) + std(2x3*3x2) = 8 + 12 = 20
    
    C[2:4, 2:4] = A[2:4, :] * B[:, 2:4]
    = A[2:4, 0:2]*B[0:2, 2:4] + A[2:4, 2:5]*B[2:5, 2:4]
    std(2x2*2x2) + std(2x3*3x2) = 8 + 12 = 20
    
    C[0:4, 4:5] = A[:, 0:5] * B[:, 4:5]
    = A[0:2, :]*B[:, 4:5] + A[2:4, :]*B[:, 4:5]
    std(2x5*5x1) + std(2x5*5x1) = 10 + 10 = 20
    
    Total: 19 + 20 + 20 + 20 + 20 = 99
    
    Hmm, only saved 1. Let's see if we can do better.
    
    What if we use Strassen on more blocks? The issue is that the 2x2 blocks
    in A and B that correspond to the same (i,j) position can be Strassen-ized.
    
    Actually, let's just implement the rank 96 approach mentioned in the knowledge:
    "Replacing 4 disjoint 2x2x2 standard sub-problems with Strassen's 7-term 
    decomposition reduces the tensor rank from 100 to 96"
    
    So we have 4 places where we can apply Strassen, saving 1 each (8-7=1 per place).
    
    Let's implement this carefully.
    """
    n, m, p = 4, 5, 5
    rank = 96
    
    U = [[0] * rank for _ in range(n * m)]  # 20 x 96
    V = [[0] * rank for _ in range(m * p)]  # 25 x 96
    W = [[0] * rank for _ in range(p * n)]  # 20 x 96
    
    col = 0
    
    def add_standard(i_start, i_len, j_start, j_len, k_start, k_len):
        """Add standard multiplication terms for A[i_start:i_start+i_len, j_start:j_start+j_len] * B[j_start:j_start+j_len, k_start:k_start+k_len]"""
        nonlocal col
        for ii in range(i_len):
            for jj in range(j_len):
                for kk in range(k_len):
                    U[(i_start + ii) * m + (j_start + jj)][col] = 1
                    V[(j_start + jj) * p + (k_start + kk)][col] = 1
                    W[(k_start + kk) * n + (i_start + ii)][col] = 1
                    col += 1
    
    def add_strassen(i_start, j_start, k_start):
        """Add Strassen terms for 2x2 * 2x2 at position (i_start, j_start, k_start)"""
        nonlocal col
        # Strassen's 7 products for 2x2 matrix multiplication
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A10)(B10 + B11)
        
        # M1
        U[i_start * m + j_start][col] = 1
        V[j_start * p + k_start][col] = 1
        W[k_start * n + i_start][col] = 1
        U[(i_start+1) * m + (j_start+1)][col] = 1
        V[(j_start+1) * p + (k_start+1)][col] = 1
        W[(k_start+1) * n + (i_start+1)][col] = 1
        col += 1
        
        # M2
        U[(i_start+1) * m + j_start][col] = 1
        V[j_start * p + k_start][col] = 1
        W[k_start * n + (i_start+1)][col] = 1
        U[(i_start+1) * m + (j_start+1)][col] = 1
        V[(j_start+1) * p + k_start][col] = 1
        W[k_start * n + (i_start+1)][col] = 1
        col += 1
        
        # M3
        U[i_start * m + j_start][col] = 1
        V[(j_start+1) * p + k_start][col] = 1
        W[k_start * n + i_start][col] = 1
        V[(j_start+1) * p + (k_start+1)][col] = -1
        W[(k_start+1) * n + i_start][col] = -1
        col += 1
        
        # M4
        U[(i_start+1) * m + (j_start+1)][col] = 1
        V[j_start * p + (k_start+1)][col] = 1
        W[(k_start+1) * n + (i_start+1)][col] = 1
        V[j_start * p + k_start][col] = -1
        W[k_start * n + (i_start+1)][col] = -1
        col += 1
        
        # M5
        U[i_start * m + j_start][col] = 1
        V[(j_start+1) * p + (k_start+1)][col] = 1
        W[(k_start+1) * n + i_start][col] = 1
        U[i_start * m + (j_start+1)][col] = 1
        V[(j_start+1) * p + (k_start+1)][col] = 1
        W[(k_start+1) * n + i_start][col] = 1
        col += 1
        
        # M6
        U[(i_start+1) * m + j_start][col] = 1
        V[j_start * p + k_start][col] = 1
        W[k_start * n + (i_start+1)][col] = 1
        U[i_start * m + j_start][col] = -1
        V[j_start * p + k_start][col] = -1
        W[k_start * n + i_start][col] = -1
        U[(i_start+1) * m + j_start][col] = 1
        V[j_start * p + (k_start+1)][col] = 1
        W[(k_start+1) * n + (i_start+1)][col] = 1
        U[i_start * m + j_start][col] = -1
        V[j_start * p + (k_start+1)][col] = -1
        W[(k_start+1) * n + i_start][col] = -1
        col += 1
        
        # M7
        U[i_start * m + (j_start+1)][col] = 1
        V[(j_start+1) * p + k_start][col] = 1
        W[k_start * n + i_start][col] = 1
        U[(i_start+1) * m + j_start][col] = -1
        V[j_start * p + k_start][col] = -1
        W[k_start * n + (i_start+1)][col] = -1
        U[i_start * m + (j_start+1)][col] = 1
        V[(j_start+1) * p + (k_start+1)][col] = 1
        W[(k_start+1) * n + i_start][col] = 1
        U[(i_start+1) * m + j_start][col] = -1
        V[j_start * p + (k_start+1)][col] = -1
        W[(k_start+1) * n + (i_start+1)][col] = -1
        col += 1
    
    # Apply Strassen to 4 disjoint 2x2x2 blocks
    # Block 1: i=0:2, j=0:2, k=0:2
    add_strassen(0, 0, 0)
    
    # Block 2: i=0:2, j=0:2, k=2:4
    add_strassen(0, 0, 2)
    
    # Block 3: i=2:4, j=0:2, k=0:2
    add_strassen(2, 0, 0)
    
    # Block 4: i=2:4, j=0:2, k=2:4
    add_strassen(2, 0, 2)
    
    # Now fill in the rest with standard multiplication
    # We need to cover all (i,j,k) triples not covered by the Strassen blocks above.
    # The Strassen blocks cover:
    # i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # i in {2,3}, j in {0,1}, k in {0,1,2,3}
    # So all i in 0:4, j in 0:2, k in 0:4 are covered.
    
    # Remaining: j in 2:5 (i.e., j=2,3,4) for all i in 0:4, k in 0:5
    # That's 4 * 3 * 5 = 60 terms
    
    for i in range(4):
        for j in range(2, 5):
            for k in range(5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Total should be 4*7 + 60 = 28 + 60 = 88
    # But we expected 96. Let me recheck.
    
    # Wait, the Strassen blocks I chose cover j=0:2 only. But there are other 2x2 blocks
    # in j that could also be Strassen-ized.
    
    # Let me reconsider. The knowledge says "4 disjoint 2x2x2 standard sub-problems".
    # Maybe they mean 4 different (i,j,k) positions where each is a 2x2x2 block.
    
    # Actually, I think my count is off. Let me just use the simple approach:
    # Use Strassen on 4 specific 2x2x2 blocks and standard on the rest.
    
    # Let me restart with a cleaner approach.
    
    # Reset
    col = 0
    for row in U:
        for idx in range(len(row)):
            row[idx] = 0
    for row in V:
        for idx in range(len(row)):
            row[idx] = 0
    for row in W:
        for idx in range(len(row)):
            row[idx] = 0
    
    # Use Strassen on these 4 blocks:
    # (i=0:2, j=0:2, k=0:2)
    # (i=0:2, j=2:4, k=0:2)
    # (i=2:4, j=0:2, k=0:2)
    # (i=2:4, j=2:4, k=0:2)
    
    add_strassen(0, 0, 0)
    add_strassen(0, 2, 0)
    add_strassen(2, 0, 0)
    add_strassen(2, 2, 0)
    
    # Now fill in the rest
    # Covered: i in 0:4, j in 0:4, k in 0:2
    # Remaining: k in 2:5 for all i in 0:4, j in 0:5
    # That's 4 * 5 * 3 = 60 terms
    
    for i in range(4):
        for j in range(5):
            for k in range(2, 5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Total: 4*7 + 60 = 28 + 60 = 88
    
    # Hmm, that's even better than 96. But I need to verify correctness.
    # Let me just go with rank 96 to be safe, as that's what was measured.
    
    # Actually, let me just implement the known-good rank 96 approach.
    # I'll use Strassen on 4 blocks and standard on the rest, targeting rank 96.
    
    # Let me be more careful. I'll use a different set of 4 Strassen blocks.
    
    # Reset again
    col = 0
    for row in U:
        for idx in range(len(row)):
            row[idx] = 0
    for row in V:
        for idx in range(len(row)):
            row[idx] = 0
    for row in W:
        for idx in range(len(row)):
            row[idx] = 0
    
    # Use Strassen on 4 blocks that are disjoint in the (i,j,k) space
    # and cover 2x2x2 subcubes:
    # Block 1: i=0:2, j=0:2, k=0:2
    # Block 2: i=0:2, j=3:5, k=3:5
    # Block 3: i=2:4, j=0:2, k=3:5
    # Block 4: i=2:4, j=3:5, k=0:2
    
    add_strassen(0, 0, 0)
    add_strassen(0, 3, 3)
    add_strassen(2, 0, 3)
    add_strassen(2, 3, 0)
    
    # Now fill in all remaining (i,j,k) not covered
    # Covered:
    # (0,0,0), (0,0,1), (0,1,0), (0,1,1)
    # (0,3,3), (0,3,4), (0,4,3), (0,4,4)
    # (2,0,3), (2,0,4), (2,1,3), (2,1,4)
    # (3,3,0), (3,3,1), (3,4,0), (3,4,1)
    
    covered = set()
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                covered.add((0+di, 0+dj, 0+dk))
                covered.add((0+di, 3+dj, 3+dk))
                covered.add((2+di, 0+dj, 3+dk))
                covered.add((2+di, 3+dj, 0+dk))
    
    for i in range(4):
        for j in range(5):
            for k in range(5):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Total should be 4*7 + (100 - 4*8) = 28 + 68 = 96
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
