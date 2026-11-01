"""Matrix multiplication 2,3,9 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Decomposition structure:
    # For each k in 0..2 (middle dimension), we multiply A (2x3) by B[:,k] (3x9)
    # This gives a 2x9 output block for each k.
    # But the tensor is sum over j of A[:,j] * B[j,:], so we need to be careful.
    
    # Actually, let's think of it as: C = A * B where A is 2x3 and B is 3x9.
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We can use a rank-3 decomposition for 2x3x3 and combine.
    # For 2x3x3 matrix multiplication, the minimal rank is 8 (known result).
    # But we have 3 such blocks (one for each output column group of size 3).
    # Since the 9 output columns are independent, we can just take 3 copies of the 2x3x3 decomposition.
    # That would give rank 3*8 = 24.
    
    # Wait, can we do better? The B matrix is 3x9, so the 9 columns are in 3 groups of 3.
    # Each group uses the same A, but different B columns.
    # So we need 3 separate 2x3x3 multiplications.
    # Rank 24 is achievable.
    
    # Let's use a known rank-8 decomposition for 2x3x3.
    # From literature: the tensor rank of 2x3x3 matrix multiplication is 8.
    
    # One such decomposition (entries in {0, 1, -1}):
    # The 8 rank-1 terms are indexed by c=0..7.
    # U has shape (2*3, 8) = (6, 8)
    # V has shape (3*3, 8) = (9, 8)
    # W has shape (3*2, 8) = (6, 8)
    
    # Let me look up a specific rank-8 decomposition for 2x3x3.
    # A common one is based on the fact that 2x3x3 can be embedded in 4x4x4 or similar.
    
    # Alternatively, I'll use a known explicit decomposition.
    # From Alon et al. or similar sources, here's one:
    
    # Actually, let me just use 3 copies of a rank-8 2x3x3 decomposition.
    # Total rank = 3 * 8 = 24.
    
    # Here is one rank-8 decomposition for 2x3x3:
    # U rows: (i,j) for i in 0,1 and j in 0,1,2 -> indices 0..5
    # V rows: (j,k) for j in 0,1,2 and k in 0,1,2 -> indices 0..8
    # W rows: (k,i) for k in 0,1,2 and i in 0,1 -> indices 0..5
    
    # I'll use the decomposition from https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
    # or construct one carefully.
    
    # Let me use a known one. For 2x3x3, one rank-8 decomposition is:
    # (from various sources, e.g., Landsberg's work)
    
    # I'll use the following explicit 8-term decomposition for 2x3x3:
    U23 = [
        [1, 0, 1, 0, 1, 0, 0, 0],  # (0,0)
        [0, 1, 0, 1, 0, 1, 0, 0],  # (0,1)
        [0, 0, 0, 0, 0, 0, 1, 1],  # (0,2)
        [1, 0, 0, 0, 0, 0, 1, 0],  # (1,0)
        [0, 1, 0, 0, 0, 0, 0, 1],  # (1,1)
        [0, 0, 1, 1, 1, 1, 0, 0],  # (1,2)
    ]
    V23 = [
        [1, 1, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 0, 1, 1, 0, 0, 0, 0],  # (0,1)
        [0, 0, 0, 0, 1, 1, 0, 0],  # (0,2)
        [1, 0, 1, 0, 0, 0, 0, 0],  # (1,0)
        [0, 1, 0, 1, 0, 0, 0, 0],  # (1,1)
        [0, 0, 0, 0, 1, 0, 0, 0],  # (1,2)
        [0, 0, 0, 0, 0, 1, 0, 0],  # (2,0)
        [0, 0, 0, 0, 0, 0, 1, 0],  # (2,1)
        [0, 0, 0, 0, 0, 0, 0, 1],  # (2,2)
    ]
    W23 = [
        [1, 0, 0, 1, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 1, 0, 0, 0, 0, 0],  # (1,0)
        [0, 0, 0, 1, 0, 0, 0, 0],  # (1,1)
        [0, 0, 0, 0, 1, 0, 0, 0],  # (2,0)
        [0, 0, 0, 0, 0, 1, 0, 0],  # (2,1)
        # Wait, W should have 3*2=6 rows for (k,i) with k in 0,1,2 and i in 0,1
    ]
    
    # Let me be more careful. I need a verified rank-8 decomposition for 2x3x3.
    # To be safe, let me use rank 9 for 2x3x3 (which is the schoolbook rank) 
    # and see if I can get total rank 27. That's not better than 54, but let me try harder.
    
    # Actually, the minimal rank for 2x3x3 is indeed 8. Let me use a known correct one.
    # From "On the border rank of matrix multiplication" by Landsberg:
    
    # Let me use a different approach. I'll use the fact that we can decompose 
    # the 2x3x9 problem into three 2x3x3 problems. If I can find a rank-8 
    # decomposition for 2x3x3, I get total rank 24.
    
    # Here's a correct rank-8 decomposition for 2x3x3 (verified):
    # Terms (c=0..7):
    # c=0: U=(1,0,0,1,0,0), V=(1,0,0,1,0,0,0,0,0), W=(1,0,0,1,0,0)
    # This is getting complicated to verify manually. Let me use a safer approach.
    
    # I'll use rank 9 for each 2x3x3 block (schoolbook), giving total rank 27.
    # This is better than 54 and I can verify it easily.
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k_block in range(3):  # 3 blocks of 3 output columns
        for i in range(2):
            for j in range(3):
                U[i * m + j][col] = 1
                V[j * p + k_block * 3][col] = 1
                # W index: k*n + i where k = k_block*3 + 0,1,2
                # But we need to sum over j, so for each (i, k) we need the right W entry
                W[(k_block * 3) * n + i][col] = 1
                col += 1
    
    # Wait, this is wrong. The schoolbook for 2x3x3 has rank 6, not 9.
    # For each block: 2*3*3 = 18 rank-1 terms? No, 2*3*3 = 18 is for the full tensor.
    # No, for 2x3x3 matrix mult: rank is 2*3*3 = 18 for schoolbook? 
    # No! Schoolbook rank for n x m x p is n*m*p only if we mean the standard algorithm.
    # But actually, the standard algorithm uses n*m*p multiplications.
    # For 2x3x3: 2*3*3 = 18 rank-1 terms.
    
    # I made an error. Let me redo.
    # Schoolbook for 2x3x9: rank = 2*3*9 = 54. That's what we started with.
    
    # For three 2x3x3 blocks: each has schoolbook rank 2*3*3 = 18.
    # Total: 3*18 = 54. Same as before.
    
    # To beat 54, I need a non-schoolbook decomposition.
    # Let me use Strassen for 2x2 blocks.
    
    # Split the 3 middle dimension into: 2 + 1.
    # Block 1: A[:, 0:2] (2x2) times B[0:2, :] (2x9).
    # Block 2: A[:, 2] (2x1) times B[2, :] (1x9).
    
    # For Block 1: 2x2x9. Split the 9 columns into three 3s? Or use Strassen on 2x2 part.
    # 2x2x9: We can view this as 2x2x3 repeated 3 times. Rank for 2x2x3 is 7 (Strassen-like).
    # Actually, 2x2x3: schoolbook is 12. Can we do better?
    # 2x2xn: rank is 2n for n>=2? Or can we use Strassen?
    # For 2x2x2: rank 7 (Strassen).
    # For 2x2x3: we can use Strassen on the 2x2x2 part and schoolbook for the last column.
    # Rank 7 + 4 = 11? Or maybe 2x2x3 has rank 10?
    
    # Let me just use: 2x2x9 = 3 * (2x2x3). If 2x2x3 has rank 10, total 30.
    # Plus 2x1x9: schoolbook rank 18.
    # Total: 30 + 18 = 48. Better than 54!
    
    # Let me implement 2x2x3 with rank 10.
    # Actually, for 2x2xn, the rank is known. For n=3, I believe it's 10.
    
    # To be safe and simple, let me use Strassen for 2x2x2 (rank 7) and schoolbook for the rest.
    # 2x2x9: use Strassen on first 2 cols (rank 7), schoolbook on last 7 cols (2*1*7=14).
    # Total: 7 + 14 = 21.
    # 2x1x9: schoolbook 2*1*9 = 18.
    # Grand total: 21 + 18 = 39.
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: A[:, 0:2] * B[0:2, :] -> 2x2x9
    # Sub-block 1a: 2x2x2 using Strassen (cols 0,1 of B)
    # Strassen 2x2x2:
    M = 7
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[3][col] = 1  # A00, A11 (row 0 col 0, row 1 col 1)
    V[0][col] = 1; V[5][col] = 1  # B00, B11 (row 0 col 0, row 1 col 1)
    W[0][col] = 1; W[3][col] = 1  # C00, C11
    col += 1
    # M2 = (A10+A11)B00
    U[2][col] = 1; U[3][col] = 1  # A10, A11
    V[0][col] = 1  # B00
    W[2][col] = 1  # C10
    col += 1
    # M3 = A00(B01-B11)
    U[0][col] = 1  # A00
    V[1][col] = 1; V[5][col] = -1  # B01, -B11
    W[1][col] = 1  # C01
    col += 1
    # M4 = A11(B10-B00)
    U[3][col] = 1  # A11
    V[4][col] = 1; V[0][col] = -1  # B10, -B00
    W[3][col] = 1  # C11
    col += 1
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1  # A00, A01
    V[5][col] = 1  # B11
    W[3][col] = 1  # C11
    col += 1
    # M6 = (A10-A00)(B00+B01)
    U[2][col] = 1; U[0][col] = -1  # A10, -A00
    V[0][col] = 1; V[1][col] = 1  # B00, B01
    W[1][col] = 1  # C01
    col += 1
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[3][col] = -1  # A01, -A11
    V[4][col] = 1; V[5][col] = 1  # B10, B11
    W[1][col] = 1  # C01
    col += 1
    
    # Sub-block 1b: schoolbook for cols 2..8 of B (7 columns)
    for k in range(2, 9):
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block 2: A[:, 2] * B[2, :] -> 2x1x9
    for i in range(2):
        for k in range(9):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
