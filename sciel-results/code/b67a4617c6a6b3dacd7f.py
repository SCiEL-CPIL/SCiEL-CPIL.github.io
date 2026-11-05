import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _verify_decomposition(U, V, W, n=6, m=6, p=6):
    """
    Verifies if the given U, V, W matrices form a valid tensor decomposition
    for n x m by m x p matrix multiplication.
    The condition is:
    sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] == 1 for all i, j, k.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = np.dot(U_np[u_idx, :], V_np[v_idx, :] * W_np[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct_laderman_2x2_blocks():
    """
    Laderman 3x3 matrix multiplication applied to 3x3 grid of 2x2 blocks.
    Rank: 15 * (2*2) = 60.
    """
    n, m, p = 6, 6, 6
    bs = 2  # block size
    rank = 60
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Laderman 15-term decomposition for 3x3 matrices.
    # Each term: (A_contrib, B_contrib, C_contrib)
    # where each contrib is a list of (block_row, block_col, coeff)
    # Block indices 0,1,2 for the 3x3 block grid.
    
    laderman_terms = [
        # Term 0: diagonal sums
        ([(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)]),
        # Term 1
        ([(0,1,1),(0,2,1)], [(1,0,1),(2,0,1)], [(0,1,1),(0,2,1)]),
        # Term 2
        ([(0,1,1),(1,2,1)], [(1,0,1),(2,1,1)], [(0,1,1),(1,2,1)]),
        # Term 3
        ([(1,0,1),(2,1,1)], [(0,1,1),(2,2,1)], [(1,0,1),(2,1,1)]),
        # Term 4
        ([(2,0,1),(0,2,1)], [(0,2,1),(1,1,1)], [(2,0,1),(0,2,1)]),
        # Term 5
        ([(0,2,1),(1,0,1)], [(0,1,1),(2,0,1)], [(0,2,1),(1,0,1)]),
        # Term 6
        ([(1,2,1),(2,0,1)], [(0,2,1),(1,0,1)], [(1,2,1),(2,0,1)]),
        # Term 7
        ([(0,1,1),(0,2,1),(1,0,-1)], [(1,0,1),(2,0,1),(0,1,-1)], [(0,1,1),(0,2,-1),(1,0,-1)]),
        # Term 8
        ([(0,1,1),(1,2,1),(2,0,-1)], [(1,0,1),(2,1,1),(0,2,-1)], [(0,1,1),(1,2,1),(2,0,-1)]),
        # Term 9
        ([(1,0,1),(2,1,1),(0,2,-1)], [(0,1,1),(2,2,1),(1,1,-1)], [(1,0,1),(2,1,1),(0,2,-1)]),
        # Term 10
        ([(2,0,1),(0,2,1),(1,2,-1)], [(0,2,1),(1,1,1),(2,1,-1)], [(2,0,1),(0,2,1),(1,2,-1)]),
        # Term 11
        ([(0,2,1),(1,0,1),(1,2,-1)], [(0,1,1),(2,0,1),(2,1,-1)], [(0,2,1),(1,0,1),(1,2,-1)]),
        # Term 12
        ([(1,2,1),(2,0,1),(2,1,-1)], [(0,2,1),(1,0,1),(0,1,-1)], [(1,2,1),(2,0,1),(2,1,-1)]),
        # Term 13
        ([(0,1,1),(0,2,1),(2,1,-1)], [(1,0,1),(2,0,1),(1,1,-1)], [(0,1,1),(0,2,1),(2,1,-1)]),
        # Term 14
        ([(1,0,1),(2,1,1),(0,2,-1)], [(0,1,1),(2,2,1),(1,1,-1)], [(1,0,1),(2,1,1),(0,2,-1)]),
    ]
    
    for t, (a_contrib, b_contrib, c_contrib) in enumerate(laderman_terms):
        for i in range(bs):
            for k in range(bs):
                for j in range(bs):
                    r = t * 4 + i * bs + k  # rank index: 4 per term (2*2)
                    # Actually, we need 4 rank indices per Laderman term (for the 2x2 inner product)
                    # So total rank = 15 * 4 = 60
                    pass
    
    # Let me redo this more carefully.
    # For each Laderman term t (0..14), we need 4 rank-1 terms (one for each (i,k,j) in 2x2x2x2... 
    # Actually, for each Laderman term, the "inner" 2x2 multiplication is done via 4 rank-1 terms
    # (schoolbook for 2x2: rank 4).
    # So total rank = 15 * 4 = 60.
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for t, (a_contrib, b_contrib, c_contrib) in enumerate(laderman_terms):
        for i in range(bs):
            for k in range(bs):
                for j in range(bs):
                    # This doesn't work - we need rank 4 per term, not 8
                    pass
    
    # The 2x2 schoolbook has 4 terms:
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    # But as rank-1 decompositions, 2x2 matrix mult has rank 7 (Strassen) or 4 (schoolbook).
    # Schoolbook rank 4: 4 independent rank-1 terms.
    
    # For each Laderman term t, we compute M_t = (sum a_contrib) * (sum b_contrib)
    # Then C += M_t * (sum c_contrib)
    # The multiplication of two 2x2 matrices (the "inner" multiplication) needs rank 4 (schoolbook).
    # So for each Laderman term, we have 4 rank-1 terms in the full decomposition.
    
    # 2x2 schoolbook rank-1 terms:
    # Term A: a00 * b00 -> contributes to c00
    # Term B: a01 * b10 -> contributes to c00
    # Term C: a00 * b01 -> contributes to c01
    # Term D: a01 * b11 -> contributes to c01
    # Term E: a10 * b00 -> contributes to c10
    # Term F: a11 * b10 -> contributes to c10
    # Term G: a10 * b01 -> contributes to c11
    # Term H: a11 * b11 -> contributes to c11
    # That's 8 terms, but they can be grouped into 4 rank-1 terms for the full 2x2 mult:
    # Actually, the 2x2 matrix multiplication tensor has rank 7 (Strassen) or 4 is not correct.
    # The rank of 2x2 matrix multiplication is 7.
    # Wait, no. The rank of the 2x2 matrix multiplication tensor is 7 (Strassen).
    # The "schoolbook" uses 4 multiplications but that's for the entries, not rank-1 terms.
    # Each entry product a_ij * b_jk is a rank-1 term, and there are 4*4 = 16 such terms,
    # but they share the same j index... 
    
    # I'm overcomplicating this. Let me think differently.
    
    # The 6x6 matrix multiplication tensor M_6 has some rank.
    # If we use Laderman on 3x3 blocks of 2x2, the structure is:
    # C_block[br][bc] = sum_t (A_combined_t[br][bc]) * (B_combined_t[br][bc]) * (C_combined_t[br][bc])
    # where each "block" is 2x2.
    # The "inner" 2x2 multiplication has rank 7 (Strassen) or we can use the 4 schoolbook multiplications
    # but each schoolbook multiplication a_ij*b_jk for fixed j is already a rank-1 term in the 2x2 mult.
    
    # Actually, for 2x2 matrix multiplication:
    # C = A * B where A, B, C are 2x2.
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # The tensor has rank 7 (Strassen) or 4 is the "naive" count of multiplications.
    # But the rank of the 2x2 matrix multiplication tensor is 7.
    
    # So Laderman (rank 15 for 3x3) composed with 2x2 schoolbook (rank 7) gives rank 15*7 = 105.
    # That's worse than 189? No, 105 < 189.
    
    # Wait, I think I'm confusing myself. Let me restart.
    
    # The 6x6 matrix multiplication can be viewed as:
    # Option 1: 3x3 grid of 2x2 blocks. Use Laderman (rank 15) for the 3x3 block mult,
    #           and for each "block multiplication" (which is a 2x2 * 2x2 -> 2x2), use rank 7 (Strassen).
    #           Total rank: 15 * 7 = 105.
    # Option 2: 2x2 grid of 3x3 blocks. Use Strassen (rank 7) for the 2x2 block mult,
    #           and for each "block multiplication" (3x3 * 3x3 -> 3x3), use rank 27 (schoolbook) or 15 (Laderman).
    #           Total rank: 7 * 27 = 189 (schoolbook) or 7 * 15 = 105 (Laderman).
    
    # Both give 105! Let me use Option 2: Strassen on 2x2 blocks of 3x3, with Laderman for the 3x3 block mult.
    # But that's complex. Let me just use Option 1: Laderman on 3x3 blocks of 2x2, with Strassen for 2x2 block mult.
    
    # Actually, the simplest improvement: use Strassen (rank 7) on 2x2 blocks of 3x3, 
    # but use Laderman (rank 15) instead of schoolbook (rank 27) for the 3x3 block multiplication.
    # Total rank: 7 * 15 = 105.
    
    # Let me implement this.
    
    U = np.zeros((n * m, 105))
    V = np.zeros((m * p, 105))
    W = np.zeros((p * n, 105))
    
    # Strassen 2x2 block decomposition
    # Block indices: 0,1 for top/bottom rows, 0,1 for left/right cols
    # Block (br, bc): row offset br*3, col offset bc*3
    strassen_blocks = [
        # (A_contrib, B_contrib, C_contrib)
        # A_contrib: list of (block_idx, coeff) where block_idx = br*2+bc
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    # Laderman 3x3 decomposition
    # This is the tricky part. I need the exact Laderman 15-term decomposition.
    # Let me use a known correct implementation.
    
    # For now, let me just return the 189 solution which I know works.
    # The improvement from 189 to 105 requires a correct Laderman implementation.
    
    return None  # Placeholder

def construct_strassen_3x3_blocks():
    """
    Strassen 2x2 blocks of 3x3. Rank: 7 * 27 = 189.
    This is the known-good solution.
    """
    n, m, p = 6, 6, 6
    
    U = np.zeros((n * m, 189))
    V = np.zeros((m * p, 189))
    W = np.zeros((p * n, 189))
    
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    col = 0
    for term_idx in range(7):
        a_comb, b_comb, c_comb = strassen_terms[term_idx]
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    for (blk_a, coeff_a) in a_comb:
                        row_a = (blk_a // 2) * 3
                        col_a = (blk_a % 2) * 3
                        u_idx = (row_a + i) * m + (col_a + k)
                        U[u_idx, col] += coeff_a
                    for (blk_b, coeff_b) in b_comb:
                        row_b = (blk_b // 2) * 3
                        col_b = (blk_b % 2) * 3
                        v_idx = (row_b + k) * p + (col_b + j)
                        V[v_idx, col] += coeff_b
                    for (blk_c, coeff_c) in c_comb:
                        row_c = (blk_c // 2) * 3
                        col_c = (blk_c % 2) * 3
                        w_idx = (col_c + j) * n + (row_c + i)
                        W[w_idx, col] += coeff_c
                    col += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Return the known-good 189 rank solution
    result = construct_strassen_3x3_blocks()
    
    # Verify
    if _verify_decomposition(result['U'], result['V'], result['W']):
        return result
    else:
        # Fallback: schoolbook rank 216
        n, m, p = 6, 6, 6
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    r = i * m * p + k * p + j
                    U[i * m + k, r] = 1
                    V[k * p + j, r] = 1
                    W[j * n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
