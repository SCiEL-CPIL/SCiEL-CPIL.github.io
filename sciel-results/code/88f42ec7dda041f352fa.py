"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_222():
    """Return U (4x7), V (4x7), W (4x7) for 2x2x2 Strassen decomposition.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]"""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # 0: A00*B00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # 1: A11*B11
    U[3, 1] = 1; V[3, 1] = 1; W[3, 1] = 1
    # 2: (A00+A11)*(B00+B11)
    U[0, 2] = 1; U[3, 2] = 1; V[0, 2] = 1; V[3, 2] = 1; W[0, 2] = -1; W[3, 2] = -1
    # 3: (A10+A11)*B01
    U[2, 3] = 1; U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    # 4: A00*(B10-B00)
    U[0, 4] = 1; V[2, 4] = 1; V[0, 4] = -1; W[2, 4] = 1
    # 5: (A01+A11)*B10
    U[1, 5] = 1; U[3, 5] = 1; V[2, 5] = 1; W[2, 5] = 1
    # 6: (A10-A00)*(B01+B11)
    U[2, 6] = 1; U[0, 6] = -1; V[1, 6] = 1; V[3, 6] = 1; W[1, 6] = 1
    return U, V, W

def _combine_222_to_444(U222, V222, W222):
    """Combine four 2x2x2 Strassen decompositions into one 4x4x4.
    U is (16, 28), V is (16, 28), W is (16, 28)
    Indices: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]"""
    U4 = np.zeros((16, 28))
    V4 = np.zeros((16, 28))
    W4 = np.zeros((16, 28))
    for block_i in range(2):
        for block_j in range(2):
            for block_k in range(2):
                # The (i,j) block of U gets contribution from c where the i2 block matches block_i
                # and j2 block matches block_j
                for c in range(7):
                    # Determine which 2x2 block this c belongs to
                    # In the 2x2x2, U has rows i2*2+j2, so block is (i2, j2)
                    i2, j2 = c // 2, c % 2  # This is wrong! c is a column index
                    pass
    # Let me think more carefully.
    # For 4x4x4, we split into 2x2 blocks.
    # A = [[A00, A01],[A10, A11]] where each is 2x2
    # B = [[B00, B01],[B10, B11]] where each is 2x2
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # C[i1*2+i2, k1*2+k2] = sum_{j1,j2} A[i1*2+i2, j1*2+j2]*B[j1*2+j2, k1*2+k2]
    # = sum_{j1,j2} A_{i1,j1}[i2,j2] * B_{j1,k1}[j2,k2]
    # For fixed (i1,k1), C_{i1,k1}[i2,k2] = sum_{j1} (sum_{j2} A_{i1,j1}[i2,j2] * B_{j1,k1}[j2,k2])
    # The inner sum over j2 is a 2x2x2 matrix multiplication: A_{i1,j1} * B_{j1,k1}
    # So for each (i1,k1) pair, we have 4 terms (j1=0,1), each a 2x2x2 product.
    # Each 2x2x2 product can be done with Strassen (rank 7).
    # Total: 4 * 2 * 7 = 56? No wait, that's not right either.
    # Actually, for each (i1,k1), we compute C_{i1,k1} = A_{i1,0}*B_{0,k1} + A_{i1,1}*B_{1,k1}
    # This is a sum of two 2x2x2 matrix products.
    # We can use Strassen for each 2x2x2 product (rank 7 each), total 14 per (i1,k1) pair.
    # 4 (i1,k1) pairs * 14 = 56. That's worse than just doing 16 multiplications!
    # 
    # Actually, the standard approach is to use Strassen recursively:
    # 4x4x4 = 8 * 2x2x2 (strassen on 4x4x4 gives rank 8*7=56)
    # But 16 schoolbook multiplications give rank 16.
    # So for 4x4x4, schoolbook is better!
    # 
    # Let me just use schoolbook for 4x4x4: rank 16.
    pass
    # Schoolbook 4x4x4:
    U4 = np.zeros((16, 16))
    V4 = np.zeros((16, 16))
    W4 = np.zeros((16, 16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                c = i*4 + j + k*16  # need 16 columns
                U4[i*4+j, c] = 1
                V4[j*4+k, c] = 1
                W4[k*4+i, c] = 1
    return U4, V4, W4

def _combine_to_447(U444, V444, W444):
    """Create 4x4x7 from 4x4x4 by adding 3 more columns of B.
    U: (16, 16), V: (16, 16), W: (16, 16) -> U: (16, 28), V: (28, 28), W: (28, 28)
    """
    # For 4x4x7: A is 4x4, B is 4x7, C is 4x7
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # U: (16, rank), V: (28, rank), W: (28, rank)
    # We can use the 4x4x4 decomposition for the first 4 columns of B,
    # and schoolbook for the remaining 3 columns.
    # 4x4x4: rank 16
    # 4x4x3: rank 12 (schoolbook: 4*4*3 = 48? No, schoolbook is 4*4*3=48 but we can do better)
    # Actually schoolbook for 4x4x3: for each i,j,k (k=0,1,2): one term. Total 4*4*3=48.
    # But we can share: for fixed i,j, we need A[i,j]*B[j,k] for k=0,1,2.
    # That's 3 terms per (i,j), total 16*3=48.
    # Hmm, that's still schoolbook.
    # Total: 16 + 48 = 64.
    # 
    # Can we do better? What if we use Strassen for the 4x4x4 part?
    # Strassen 4x4x4: rank 56 (8 * 7). But schoolbook is 16. So schoolbook is better.
    # 
    # Let me just do: first 4 columns with schoolbook 4x4x4 (rank 16), last 3 with schoolbook 4x4x3 (rank 48).
    # Total: 64.
    #
    # Actually wait. Let me reconsider. The known rank for 4x4x7... 
    # I'll just go with a reasonable construction.
    
    rank_total = 16 + 12  # I'll try to get 4x4x3 to rank 12 somehow
    # For 4x4x3: C[i,k] = sum_j A[i,j]*B[j,k], k=0,1,2
    # U: (16, 12), V: (28, 12) [rows 16-27 for k=4,5,6], W: (28, 12)
    # Schoolbook: for each (i,j,k) with k in {0,1,2} (the 3 new columns):
    #   U[i*4+j, c] = 1, V[(j)*3 + k_new, c] = 1, W[(k_new+4)*4 + i, c] = 1
    # Wait, I need to be careful with indexing.
    
    # Let me just do full schoolbook for 4x4x7: rank = 4*4*7 = 112.
    # That's the same as the seed. Not useful.
    
    # Let me try a different approach: use the 4x4x4 Strassen-like decomposition.
    # Actually, let me just implement a clean schoolbook but try to find any structure.
    
    # For now, let me just return schoolbook 4x4x7 with rank 112, same as seed.
    # This won't improve, but let me at least make it correct.
    
    # Actually, let me try: split B into two parts: B1 (4x4) and B2 (4x3).
    # C = A*B1 + A*B2
    # A*B1: use schoolbook 4x4x4, rank 16
    # A*B2: schoolbook 4x4x3, rank 48
    # Total: 64
    
    U = np.zeros((16, 64))
    V = np.zeros((28, 64))
    W = np.zeros((28, 64))
    
    c = 0
    # Part 1: A * B1 where B1 is 4x4 (columns 0-3 of B)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j, c] = 1
                V[j*4+k, c] = 1
                W[k*4+i, c] = 1
                c += 1
    
    # Part 2: A * B2 where B2 is 4x3 (columns 4-6 of B)
    for i in range(4):
        for j in range(4):
            for k in range(3):
                k_actual = k + 4  # column index in B
                U[i*4+j, c] = 1
                V[j*7 + k_actual, c] = 1
                W[k_actual*4 + i, c] = 1
                c += 1
    
    # Verify shapes
    assert U.shape == (16, 64)
    assert V.shape == (28, 64)
    assert W.shape == (28, 64)
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    start_time = time.time()
    U, V, W = _combine_to_447(None, None, None)
    elapsed = time.time() - start_time
    if elapsed > 140:
        # Return schoolbook as fallback
        n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
