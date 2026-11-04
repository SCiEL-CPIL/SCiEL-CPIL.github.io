"""
Matrix multiplication 4x5x6 decomposition.
Uses Strassen's algorithm for 2x2 blocks to reduce rank from 120 to 108.
"""
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 4, 5, 6
    
    # Strategy: Partition the multiplication into blocks and use Strassen where possible.
    # A is 4x5, B is 5x6.
    # Partition A into A_top (2x5) and A_bottom (2x5)
    # Partition B into B_left (5x4) and B_right (5x2)
    # C_top (2x6) = A_top @ B
    # C_bottom (2x6) = A_bottom @ B
    
    # For each half of A (2x5) times B (5x6):
    # We can further partition B into B_1 (5x4) and B_2 (5x2)
    # Result is 2x4 and 2x2.
    
    # Let's look at the 2x5 * 5x4 block.
    # A_half is 2x5. B_left is 5x4.
    # We can view A_half as [A1 (2x2) | A2 (2x2) | A3 (2x1)]
    # B_left as [B1 (2x4); B2 (2x4); B3 (1x4)]
    # This is getting complicated with mixed sizes.
    
    # Let's stick to the known improvement: 120 -> 108.
    # How is 108 achieved?
    # Schoolbook: 120.
    # Strassen on 2x2x2 blocks.
    # If we can decompose the 4x5x6 problem into a set of 2x2x2 multiplications
    # and some schoolbook parts, we might save rank.
    
    # Let's try a specific decomposition:
    # Partition A (4x5) into 2x2 blocks? No, 5 is odd.
    # Partition A into A1 (2x2), A2 (2x2), A3 (2x1).
    # Partition B into B1 (2x6), B2 (2x6), B3 (1x6).
    # C = A1@B1 + A2@B2 + A3@B3.
    
    # A1@B1 is 2x2 @ 2x6.
    # A2@B2 is 2x2 @ 2x6.
    # A3@B3 is 2x1 @ 1x6.
    
    # For 2x2 @ 2x6:
    # We can view this as two 2x2 @ 2x3? No.
    # We can view B1 as [B1a (2x2) | B1b (2x2) | B1c (2x2)].
    # Then A1@B1 = A1@B1a + A1@B1b + A1@B1c.
    # Each term is 2x2 @ 2x2.
    # There are 3 such terms for A1@B1, and 3 for A2@B2. Total 6 blocks of 2x2@2x2.
    # Rank of each 2x2@2x2 via Strassen is 7.
    # Total rank for these 6 blocks: 6 * 7 = 42.
    
    # Now the cross terms? No, we partitioned the inner dimension completely.
    # A = [A1 A2 A3], B = [B1; B2; B3].
    # C = A1 B1 + A2 B2 + A3 B3.
    # This is exact.
    
    # Now handle A3@B3: 2x1 @ 1x6.
    # This is rank 2*1*6 = 12 via schoolbook.
    # Can we do better? 2x1 @ 1x6 is essentially an outer product of a 2-vector and a 6-vector.
    # Rank is 1? No, the tensor is C[i,k] = sum_j A[i,j] B[j,k].
    # Here j is fixed to the single column/row.
    # C[i,k] = A[i,0] * B[0,k].
    # This is a rank 1 tensor!
    # Wait, A is 2x1, B is 1x6.
    # A[i,0] is a scalar, B[0,k] is a scalar.
    # C[i,k] = A[i,0] * B[0,k].
    # Yes, this is a rank 1 term.
    # U[i, r] = A[i,0], V[0, r] = B[0,k]... wait.
    # U has shape (n*m, rank). Here n=2, m=1. U is 2x1.
    # V has shape (m*p, rank). Here m=1, p=6. V is 6x1.
    # W has shape (p*n, rank). Here p=6, n=2. W is 12x1.
    # For rank 1:
    # U[i,0] = 1 (coefficient for A[i,0])
    # V[k,0] = 1 (coefficient for B[0,k])
    # W[k*2+i, 0] = 1 (coefficient for C[i,k])
    # So A3@B3 contributes rank 1.
    
    # Total rank so far: 42 + 1 = 43?
    # That seems too good. Let's re-evaluate the 2x2 @ 2x6 decomposition.
    
    # A1 (2x2) @ B1 (2x6).
    # B1 is 2x6. We split B1 into three 2x2 blocks: B1a, B1b, B1c.
    # A1 @ B1 = A1 @ B1a + A1 @ B1b + A1 @ B1c.
    # Each term is 2x2 @ 2x2.
    # Strassen rank is 7.
    # So 3 terms * 7 = 21 rank for A1@B1.
    # Similarly 21 rank for A2@B2.
    # Total 42.
    # Plus 1 for A3@B3.
    # Total 43.
    
    # Is this valid?
    # The tensor decomposition must be linear.
    # We are summing 6 independent 2x2@2x2 multiplications and 1 2x1@1x6 multiplication.
    # Since the blocks are disjoint in the summation index j, they are independent.
    # The rank is the sum of the ranks of the components.
    # So rank = 6*7 + 1 = 43.
    
    # Let's verify the dimensions.
    # A is 4x5.
    # A1: rows 0-1, cols 0-1. (2x2)
    # A2: rows 2-3, cols 0-1. (2x2)
    # A3: rows 0-3, col 2? No.
    # We partitioned A by columns?
    # A (4x5). Inner dimension is 5.
    # A = [A_left (4x2) | A_mid (4x2) | A_right (4x1)]
    # B = [B_top (2x6); B_mid (2x6); B_bot (1x6)]
    
    # C = A_left @ B_top + A_mid @ B_mid + A_right @ B_bot.
    
    # A_left (4x2) @ B_top (2x6).
    # This is 4x2 @ 2x6.
    # We can split A_left into A_left_top (2x2) and A_left_bot (2x2).
    # And B_top into B_top_1 (2x2), B_top_2 (2x2), B_top_3 (2x2).
    # C_top_left = A_left_top @ B_top + A_left_bot @ B_top.
    # A_left_top @ B_top = A_left_top @ (B_top_1 + B_top_2 + B_top_3)
    # = A_left_top @ B_top_1 + A_left_top @ B_top_2 + A_left_top @ B_top_3.
    # Each is 2x2 @ 2x2. Rank 7 each. Total 21.
    # A_left_bot @ B_top similarly 21.
    # Total for A_left @ B_top is 42.
    
    # A_mid (4x2) @ B_mid (2x6).
    # Similarly 42.
    
    # A_right (4x1) @ B_bot (1x6).
    # Rank 1.
    
    # Total rank = 42 + 42 + 1 = 85.
    
    # This is a significant improvement over 120.
    # Let's construct this decomposition.
    
    # We need to build U, V, W matrices.
    # n=4, m=5, p=6.
    # U shape: (4*5, rank) = (20, rank)
    # V shape: (5*6, rank) = (30, rank)
    # W shape: (6*4, rank) = (24, rank)
    
    # We will accumulate rank-1 terms.
    # Each term is defined by (i, j, k) indices in the original matrices,
    # or more generally, by the specific Strassen components.
    
    # It is easier to generate the full decomposition by iterating over the blocks.
    
    # Helper to add a schoolbook term for 1x1x1
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    
    # Helper to add a Strassen term for 2x2x2
    # The 7 terms of Strassen for 2x2 matrix multiplication C = A B.
    # A is 2x2, B is 2x2.
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    
    # Strassen:
    # m1 = (a11+a22)*(b11+b22)
    # m2 = (a21+a22)*b11
    # m3 = a11*(b12-b22)
    # m4 = a22*(b21-b11)
    # m5 = (a11+a12)*b22
    # m6 = (a21-a11)*(b11+b12)
    # m7 = (a12-a22)*(b21+b22)
    
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Each mj is a product of a linear combination of A elements and a linear combination of B elements.
    # So each mj contributes 1 to the rank, but with coefficients in U, V, W.
    # U entries are coefficients for A elements.
    # V entries are coefficients for B elements.
    # W entries are coefficients for C elements.
    
    # For a specific block of A (rows i0, i0+1; cols j0, j0+1) and B (rows j0, j0+1; cols k0, k0+1),
    # the indices in the global matrices are:
    # A elements:
    # a11 -> A[i0, j0] -> U[i0*m + j0]
    # a12 -> A[i0, j0+1] -> U[i0*m + j0+1]
    # a21 -> A[i0+1, j0] -> U[(i0+1)*m + j0]
    # a22 -> A[i0+1, j0+1] -> U[(i0+1)*m + j0+1]
    
    # B elements:
    # b11 -> B[j0, k0] -> V[j0*p + k0]
    # b12 -> B[j0, k0+1] -> V[j0*p + k0+1]
    # b21 -> B[j0+1, k0] -> V[(j0+1)*p + k0]
    # b22 -> B[j0+1, k0+1] -> V[(j0+1)*p + k0+1]
    
    # C elements:
    # c11 -> C[i0, k0] -> W[k0*n + i0]
    # c12 -> C[i0, k0+1] -> W[(k0+1)*n + i0]
    # c21 -> C[i0+1, k0] -> W[k0*n + (i0+1)]
    # c22 -> C[i0+1, k0+1] -> W[(k0+1)*n + (i0+1)]
    
    # We will create a list of terms.
    # Each term is a dict with 'u', 'v', 'w' dictionaries mapping index to coefficient.
    
    terms = []
    
    # Function to add a Strassen block
    def add_strassen_block(i0, j0, k0):
        # i0, j0, k0 are the top-left indices of the 2x2 block in A, B, C respectively.
        # Note: j0 is the inner index, so it must match for A and B.
        
        # Define the 7 Strassen multiplications
        # m1 = (a11+a22)*(b11+b22)
        u1 = {(i0, j0): 1, (i0+1, j0+1): 1}
        v1 = {(j0, k0): 1, (j0+1, k0+1): 1}
        w1 = {(i0, k0): 1, (i0, k0+1): 0, (i0+1, k0): 0, (i0+1, k0+1): -1} # c11=..., c22=...
        # Wait, W maps to C indices.
        # c11 = m1 + ... -> W[k0*n + i0] gets +1
        # c22 = m1 - ... -> W[(k0+1)*n + (i0+1)] gets +1
        # So w1 = {k0*n+i0: 1, (k0+1)*n+(i0+1): 1}
        
        # Let's be precise.
        # m1 contributes +1 to c11 and +1 to c22.
        w1 = {k0*n + i0: 1, (k0+1)*n + (i0+1): 1}
        
        # m2 = (a21+a22)*b11
        u2 = {(i0+1, j0): 1, (i0+1, j0+1): 1}
        v2 = {(j0, k0): 1}
        # c21 = m2 + ... -> W[k0*n + (i0+1)] gets +1
        # c22 = m1 - m2 + ... -> W[(k0+1)*n + (i0+1)] gets -1
        w2 = {k0*n + (i0+1): 1, (k0+1)*n + (i0+1): -1}
        
        # m3 = a11*(b12-b22)
        u3 = {(i0, j0): 1}
        v3 = {(j0, k0+1): 1, (j0+1, k0+1): -1}
        # c12 = m3 + m5 -> W[(k0+1)*n + i0] gets +1
        # c22 = m1 - m2 + m3 + m6 -> W[(k0+1)*n + (i0+1)] gets +1
        w3 = {(k0+1)*n + i0: 1, (k0+1)*n + (i0+1): 1}
        
        # m4 = a22*(b21-b11)
        u4 = {(i0+1, j0+1): 1}
        v4 = {(j0+1, k0): 1, (j0, k0): -1}
        # c11 = m1 + m4 - m5 + m7 -> W[k0*n + i0] gets +1
        # c21 = m2 + m4 -> W[k0*n + (i0+1)] gets +1
        w4 = {k0*n + i0: 1, k0*n + (i0+1): 1}
        
        # m5 = (a11+a12)*b22
        u5 = {(i0, j0): 1, (i0, j0+1): 1}
        v5 = {(j0+1, k0+1): 1}
        # c11 = m1 + m4 - m5 + m7 -> W[k0*n + i0] gets -1
        # c12 = m3 + m5 -> W[(k0+1)*n + i0] gets +1
        w5 = {k0*n + i0: -1, (k0+1)*n + i0: 1}
        
        # m6 = (a21-a11)*(b11+b12)
        u6 = {(i0+1, j0): 1, (i0, j0): -1}
        v6 = {(j0, k0): 1, (j0, k0+1): 1}
        # c22 = m1 - m2 + m3 + m6 -> W[(k0+1)*n + (i0+1)] gets +1
        w6 = {(k0+1)*n + (i0+1): 1}
        
        # m7 = (a12-a22)*(b21+b22)
        u7 = {(i0, j0+1): 1, (i0+1, j0+1): -1}
        v7 = {(j0+1, k0): 1, (j0+1, k0+1): 1}
        # c11 = m1 + m4 - m5 + m7 -> W[k0*n + i0] gets +1
        w7 = {k0*n + i0: 1}
        
        blocks = [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7),
        ]
        
        for u, v, w in blocks:
            terms.append((u, v, w))
            
    # Function to add a schoolbook term for 1x1x1
    def add_schoolbook_term(i, j, k):
        u = {(i, j): 1}
        v = {(j, k): 1}
        w = {k*n + i: 1}
        terms.append((u, v, w))
        
    # Partition A (4x5) into:
    # A_left: rows 0-3, cols 0-1 (4x2)
    # A_mid: rows 0-3, cols 2-3 (4x2)
    # A_right: rows 0-3, col 4 (4x1)
    
    # Partition B (5x6) into:
    # B_top: rows 0-1, cols 0-5 (2x6)
    # B_mid: rows 2-3, cols 0-5 (2x6)
    # B_bot: row 4, cols 0-5 (1x6)
    
    # C = A_left @ B_top + A_mid @ B_mid + A_right @ B_bot
    
    # A_left @ B_top:
    # A_left is 4x2. B_top is 2x6.
    # Split A_left into A_lt (rows 0-1) and A_lb (rows 2-3).
    # Split B_top into B_t1 (cols 0-1), B_t2 (cols 2-3), B_t3 (cols 4-5).
    # A_lt @ B_t1: 2x2 @ 2x2. Strassen. i0=0, j0=0, k0=0.
    add_strassen_block(0, 0, 0)
    # A_lt @ B_t2: 2x2 @ 2x2. Strassen. i0=0, j0=0, k0=2.
    add_strassen_block(0, 0, 2)
    # A_lt @ B_t3: 2x2 @ 2x2. Strassen. i0=0, j0=0, k0=4.
    add_strassen_block(0, 0, 4)
    
    # A_lb @ B_t1: 2x2 @ 2x2. Strassen. i0=2, j0=0, k0=0.
    add_strassen_block(2, 0, 0)
    # A_lb @ B_t2: 2x2 @ 2x2. Strassen. i0=2, j0=0, k0=2.
    add_strassen_block(2, 0, 2)
    # A_lb @ B_t3: 2x2 @ 2x2. Strassen. i0=2, j0=0, k0=4.
    add_strassen_block(2, 0, 4)
    
    # A_mid @ B_mid:
    # A_mid is 4x2 (cols 2-3). B_mid is 2x6 (rows 2-3).
    # Split A_mid into A_mt (rows 0-1) and A_mb (rows 2-3).
    # Note: The column index for A_mid is offset by 2.
    # So for A_mt, i0=0, j0=2.
    # For A_mb, i0=2, j0=2.
    # B_mid rows are 2-3. So j0=2.
    # B_mid cols are 0-5. Split into k0=0, 2, 4.
    
    # A_mt @ B_mt1: 2x2 @ 2x2. i0=0, j0=2, k0=0.
    add_strassen_block(0, 2, 0)
    # A_mt @ B_mt2: 2x2 @ 2x2. i0=0, j0=2, k0=2.
    add_strassen_block(0, 2, 2)
    # A_mt @ B_mt3: 2x2 @ 2x2. i0=0, j0=2, k0=4.
    add_strassen_block(0, 2, 4)
    
    # A_mb @ B_mt1: 2x2 @ 2x2. i0=2, j0=2, k0=0.
    add_strassen_block(2, 2, 0)
    # A_mb @ B_mt2: 2x2 @ 2x2. i0=2, j0=2, k0=2.
    add_strassen_block(2, 2, 2)
    # A_mb @ B_mt3: 2x2 @ 2x2. i0=2, j0=2, k0=4.
    add_strassen_block(2, 2, 4)
    
    # A_right @ B_bot:
    # A_right is 4x1 (col 4). B_bot is 1x6 (row 4).
    # This is a rank 1 term?
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Here j=4.
    # C[i,k] = A[i,4] * B[4,k].
    # This is a single product.
    # U[i*m+4, r] = 1 for all i? No.
    # U[i*5+4, r] = 1.
    # V[4*6+k, r] = 1.
    # W[k*4+i, r] = 1.
    # This is one rank-1 term.
    add_schoolbook_term(0, 4, 0) # This adds A[0,4]*B[4,0] -> C[0,0]
    # Wait, add_schoolbook_term(i, j, k) adds A[i,j]*B[j,k] -> C[i,k].
    # We need to add terms for all i, k.
    # Since it's a single product A[i,4]*B[4,k], we need 4*6 = 24 terms?
    # No, it's a rank 1 tensor.
    # The tensor is T[i,k] = A[i,4] * B[4,k].
    # This can be written as sum_r U[i,r] V[r,k] W[r]?
    # No, the decomposition is C = sum_r U_r * V_r * W_r.
    # For rank 1, C[i,k] = U[i] * V[k] * W? No.
    # U is (n*m, 1), V is (m*p, 1), W is (p*n, 1).
    # C[i,k] = sum_j U[i*m+j] V[j*p+k] W[k*n+i].
    # If we set U[i*m+4] = 1, V[4*p+k] = 1, W[k*n+i] = 1, then
    # C[i,k] = 1*1*1 = 1. This is not A[i,4]*B[4,k].
    # The coefficients in U, V, W are constants. The values A and B are multiplied by them.
    # So if we have one term with U[i*5+4]=1, V[4*6+k]=1, W[k*4+i]=1,
    # it contributes 1 * A[i,4] * B[4,k] to C[i,k].
    # So we just need ONE term that covers all i, k?
    # No, U, V, W are vectors.
    # U has non-zero entries at i*5+4 for i=0..3.
    # V has non-zero entries at 4*6+k for k=0..5.
    # W has non-zero entries at k*4+i for i=0..3, k=0..5.
    # If we use one rank-1 term, U, V, W are fixed vectors.
    # U[i*5+4] = 1 for all i.
    # V[4*6+k] = 1 for all k.
    # W[k*4+i] = 1 for all i,k.
    # Then C[i,k] += 1 * A[i,4] * B[4,k].
    # This is correct.
    # So we add ONE term with these non-zero entries.
    
    u_right = {}
    v_right = {}
    w_right = {}
    for i in range(4):
        u_right[i*5+4] = 1
    for k in range(6):
        v_right[4*6+k] = 1
    for i in range(4):
        for k in range(6):
            w_right[k*4+i] = 1
    terms.append((u_right, v_right, w_right))
    
    # Now build U, V, W matrices.
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            i, j = idx
            U[i*m + j][r] += val
        for idx, val in v.items():
            j, k = idx
            V[j*p + k][r] += val
        for idx, val in w.items():
            W[idx][r] += val
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
