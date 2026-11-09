"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    deadline = time.time() + 120.0
    
    # Strategy: Use Strassen's 7-mult algorithm for 2x2x2 blocks.
    # We decompose 4x4 * 4x6 into 2x3 blocks of 2x2 * 2x2 multiplications.
    # Each 2x2*2x2 uses 7 mults. Total: 2*3*7 = 42 mults.
    # We need to verify this is correct and that the tensor decomposition is valid.
    
    # For each pair (I, K) where I in {0,1} (row block of A) and K in {0,1,2} (col block of C/B),
    # we compute C[I,K] = A[I,:] * B[:,K] where A[I,:] is 2x4 and B[:,K] is 4x2.
    # Split A[I,:] into A[I,0] and A[I,1] (each 2x2).
    # Split B[:,K] into B[0,K] and B[1,K] (each 2x2).
    # C[I,K] = A[I,0]*B[0,K] + A[I,1]*B[1,K]
    # Each 2x2*2x2 product uses Strassen's 7 mults.
    # So for each (I,K): 14 mults.
    # Total: 2 * 3 * 14 = 84 mults.
    
    # Wait, that's the same as before. Let me think about sharing.
    # Actually, we can't easily share between different (I,K) blocks because
    # they involve different elements of A and B.
    
    # Let me try a different approach: use the full Strassen recursion.
    # 4x4 * 4x6: 
    # Split A into 4 2x2 blocks (2x2 grid), B into 6 2x2 blocks (3x2 grid).
    # C is 4x6, split into 6 2x2 blocks (2x3 grid).
    # C[i,j] = sum_k A[i,k] * B[k,j] for i in 0..1, j in 0..2, k in 0..1
    # Each C[i,j] is 2x2 = sum of two 2x2*2x2 products.
    # Using Strassen for each 2x2*2x2: 7 mults per product, 14 per C block.
    # Total: 6 * 14 = 84.
    
    # Can we do better? What if we use a different decomposition?
    # Let me try: compute all 4x4*4x2 subproblems using a shared Strassen structure.
    # For fixed K (column block of B), we compute C[:,K] = A * B[:,K].
    # A is 4x4, B[:,K] is 4x2.
    # Split A into 4 2x2 blocks, B[:,K] into 2 2x2 blocks.
    # C[:,K] has 2 2x2 blocks (rows 0-1 and 2-3).
    # C[0:2,K] = A[0:2,0:2]*B[0:2,K] + A[0:2,2:4]*B[2:4,K]
    # C[2:4,K] = A[2:4,0:2]*B[0:2,K] + A[2:4,2:4]*B[2:4,K]
    
    # For each K, we have 4 2x2*2x2 products, each with 7 mults = 28 mults per K.
    # Total: 3 * 28 = 84. Same.
    
    # Let me try to use the fact that Strassen's algorithm has structure that can be shared.
    # In Strassen for 2x2*2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # If we have multiple products sharing the same A blocks, we can share some M's.
    # For example, for fixed I (row block of A), we have products with different K.
    # A[I,0] and A[I,1] are shared across K.
    # But B[0,K] and B[1,K] differ for each K.
    # So M1 = (A[I,0]00 + A[I,1]11)(B[0,K]00 + B[1,K]11) - the A part is shared but B part changes.
    # We can't directly share the multiplication unless we precompute A combinations.
    
    # Alternative: precompute all Strassen combinations of A blocks, then multiply with B.
    # For A (4x4), there are 4 2x2 blocks. The Strassen combinations involve sums/differences.
    # Let me enumerate:
    # A00, A01, A10, A11 (each 2x2, indexed by [row_block, col_block])
    # Strassen needs:
    # For product A[i,j]*B[j,k]:
    # M1 uses (A[i,j]00 + A[i,j]11) - wait, no. A[i,j] is a 2x2 matrix with elements A[i,j][0,0], A[i,j][0,1], A[i,j][1,0], A[i,j][1,1].
    # Let me re-index. A is 4x4. A[i,j] where i,j in {0,1} refers to the 2x2 block at position (2i:2i+2, 2j:2j+2).
    # Within that block, let's call the elements a00, a01, a10, a11.
    # Strassen for this block times B[j,k] (another 2x2 block):
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b10 - b11)
    # M4 = a11(b01 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b10)
    # M7 = (a01 - a11)(b01 + b11)
    
    # The A-side combinations are:
    # a00 + a11, a10 + a11, a00, a11, a00 + a01, a10 - a00, a01 - a11
    # That's 7 combinations per A block.
    # We have 4 A blocks, so 28 A-side precomputations.
    # Similarly, B-side combinations per B block:
    # b00 + b11, b00, b10 - b11, b01 - b00, b11, b00 + b10, b01 + b11
    # 7 combinations per B block. We have 6 B blocks (3 col blocks * 2 row blocks), so 42 B-side precomputations.
    
    # But this doesn't reduce the number of multiplications. Each product still needs 7 mults.
    # The sharing would be if the same A combination is used with multiple B combinations.
    # For A block (i,j), it's used in products with B block (j,k) for all k.
    # So A block (i,j) is used for k=0,1,2.
    # The 7 A combinations for block (i,j) are each multiplied by 7 B combinations for block (j,k).
    # But that's still 7*7=49 mults per (i,j,k) triple, which is worse than 7.
    
    # I think the issue is that Strassen's 7 mults are for a single 2x2*2x2 product.
    # There's no obvious sharing between different products unless they share both A and B blocks.
    
    # Let me just implement the 84-rank solution carefully and verify it.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    
    # Helper: apply Strassen to 2x2 blocks at given indices
    def strassen_block(a_base, b_base, c_base, col_start):
        """
        a_base: (row_offset, col_offset) in A (4x4)
        b_base: (row_offset, col_offset) in B (4x6)
        c_base: (row_offset, col_offset) in C (4x6), which maps to W index (col*n + row)
        Returns col_end
        """
        ai, aj = a_base
        bi, bk = b_base
        ci, ck = c_base
        
        # A block elements (2x2):
        a00 = ai * m + aj
        a01 = ai * m + aj + 1
        a10 = (ai+1) * m + aj
        a11 = (ai+1) * m + aj + 1
        
        # B block elements (2x2):
        b00 = bi * p + bk
        b01 = bi * p + bk + 1
        b10 = (bi+1) * p + bk
        b11 = (bi+1) * p + bk + 1
        
        # C block elements (2x2) -> W indices:
        # W index = col * n + row
        c00 = ck * n + ci
        c01 = (ck+1) * n + ci
        c10 = ck * n + (ci+1)
        c11 = (ck+1) * n + (ci+1)
        
        c = col_start
        
        # M1 = (a00 + a11)(b00 + b11)
        U[a00][c] = 1
        U[a11][c] = 1
        V[b00][c] = 1
        V[b11][c] = 1
        W[c00][c] = 1
        W[c11][c] = 1
        c += 1
        
        # M2 = (a10 + a11)b00
        U[a10][c] = 1
        U[a11][c] = 1
        V[b00][c] = 1
        W[c10][c] = 1
        W[c11][c] = -1
        c += 1
        
        # M3 = a00(b10 - b11)
        U[a00][c] = 1
        V[b10][c] = 1
        V[b11][c] = -1
        W[c01][c] = 1
        W[c11][c] = 1
        c += 1
        
        # M4 = a11(b01 - b00)
        U[a11][c] = 1
        V[b01][c] = 1
        V[b00][c] = -1
        W[c00][c] = 1
        W[c10][c] = 1
        c += 1
        
        # M5 = (a00 + a01)b11
        U[a00][c] = 1
        U[a01][c] = 1
        V[b11][c] = 1
        W[c00][c] = -1
        W[c01][c] = 1
        c += 1
        
        # M6 = (a10 - a00)(b00 + b10)
        U[a10][c] = 1
        U[a00][c] = -1
        V[b00][c] = 1
        V[b10][c] = 1
        W[c11][c] = 1
        c += 1
        
        # M7 = (a01 - a11)(b01 + b11)
        U[a01][c] = 1
        U[a11][c] = -1
        V[b01][c] = 1
        V[b11][c] = 1
        W[c00][c] = 1
        c += 1
        
        return c
    
    # Now, for each (i, j, k) where i in {0,1} (A row block), j in {0,1} (A col block / B row block), k in {0,1,2} (B col block):
    # C[i,k] += A[i,j] * B[j,k]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                col = strassen_block((i, j), (j, k), (i, k), col)
    
    # Verify
    if time.time() > deadline - 30:
        return {"U": U, "V": V, "W": W}
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
    
    T_decomp = np.zeros_like(T_expected)
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    if not np.allclose(T_decomp, T_expected, atol=1e-10):
        # Fall back to naive
        rank_naive = n * m * p
        U_naive = [[0] * rank_naive for _ in range(n * m)]
        V_naive = [[0] * rank_naive for _ in range(m * p)]
        W_naive = [[0] * rank_naive for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i*m+j][col] = 1
                    V_naive[j*p+k][col] = 1
                    W_naive[k*n+i][col] = 1
                    col += 1
        return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
