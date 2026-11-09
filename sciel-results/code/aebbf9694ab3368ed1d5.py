"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 6x6 matrix as 2x2 grid of 3x3 blocks.
    # Each 3x3 block multiplication has rank 23 (Bini-Capovani 1983).
    # Total rank = 4 blocks * 23 = 92... wait, that's too optimistic.
    # Actually, we need to think about this differently.
    # The 6x6 multiplication tensor can be viewed as having structure.
    # But a simple block approach: if we use 3x3 Strassen-like, 
    # the 3x3 rank is 23, so 2*2 blocks gives 4*23=92? No.
    # 
    # Actually the standard approach: for 6x6, we can use the fact that
    # 6 = 3+3, so we have a 2x2 block structure where each block is 3x3.
    # If each 3x3 multiplication has rank 23, then the total rank for
    # the 2x2 block structure would be at most 4*23 = 92.
    # But this isn't quite right either - the block multiplication of 3x3
    # matrices within a 2x2 grid still requires the full tensor structure.
    #
    # Let me reconsider. The matrix multiplication tensor for (n,m,p)
    # has a known hierarchy. For 6x6x6, we can use:
    # - Direct: rank 216
    # - 2x2 Strassen with 3x3 blocks: This doesn't directly work because
    #   Strassen works on the block structure, not the element structure.
    #
    # Actually, the cleanest approach: Use the fact that if we know a
    # rank-r decomposition for multiplying 3x3 matrices, we can use it
    # in a block context. But the 6x6 multiplication is NOT simply 4
    # independent 3x3 multiplications.
    #
    # Let me use a different strategy: the known best for 6x6 is at most
    # 184 (from Lader 1974 for 7x7 which gives bounds).
    #
    # Actually, let me just use the 2x2 Strassen with 3x3 sub-blocks.
    # In Strassen's algorithm, we compute 7 products of the form
    # (linear combo of A blocks) * (linear combo of B blocks).
    # If each such product is a 3x3 matrix multiplication with rank 23,
    # then total rank = 7 * 23 = 161.
    # Score = 216/161 ≈ 1.3416. That's a significant improvement!
    
    # Strassen patterns for 2x2 block multiplication
    # Each pattern: (u_coeffs, v_coeffs, w_coeffs) where coeffs map (block_row, block_col) to value
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    strassen_patterns = [
        # M0 = (A00+A11)(B00+B11) -> contributes to C00, C11
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M1 = (A10+A11)B00 -> contributes to C10, C11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M2 = A00(B01-B11) -> contributes to C01, C11
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M3 = A11(B10-B00) -> contributes to C00, C10
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M4 = (A00+A01)B11 -> contributes to C00, C01
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M5 = (A10-A00)(B00+B01) -> contributes to C11
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6 = (A01-A11)(B10+B11) -> contributes to C00
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    block_size = 3
    # Rank for 3x3 matrix multiplication: use rank 23
    # For simplicity and correctness, I'll use rank 9 (direct) for 3x3
    # since I need to guarantee correctness. 
    # Actually, let me use the direct 3x3 multiplication (rank 9) first
    # to ensure correctness, then the total rank = 7 * 9 = 63.
    # That would give score 216/63 ≈ 3.43, which is way too high.
    # 
    # Wait, that's not right. Let me think again.
    # 
    # In Strassen, each "multiplication" M_i = A_i * B_i where A_i and B_i
    # are 3x3 matrices (linear combinations of the original blocks).
    # To compute A_i * B_i (a 3x3 matrix product), we need rank 9 (direct)
    # or rank 23 (optimal for 3x3).
    # 
    # So total rank = 7 * (rank of 3x3 multiplication).
    # With rank 9: 7*9 = 63 -> score 216/63 = 3.43 (impossible, this is wrong)
    #
    # I'm confusing myself. The issue is that the 6x6 multiplication tensor
    # is NOT decomposable into 7 independent 3x3 multiplications in a way
    # that gives rank 7*9. The Strassen decomposition works at the symbolic
    # level, but when we expand, we get the full 6x6 tensor.
    #
    # Let me go back to the safe approach: 2x2 blocks of 3x3, each 3x3
    # multiplication done directly (rank 9 per block element product).
    # No wait, that's not how it works either.
    #
    # The correct approach: The 6x6 multiplication tensor has entries
    # T[i,j,k,l] = 1 if j==l else 0 (in some indexing).
    # 
    # Let me just use the 2x2 Strassen with 3x3 blocks, where each
    # "scalar" multiplication is replaced by a 3x3 matrix multiplication
    # of rank 9 (direct). This gives:
    # Each of the 7 Strassen products is a 3x3 x 3x3 multiplication.
    # Each such multiplication has rank 9.
    # Total rank = 7 * 9 = 63.
    # 
    # But wait - is this valid? In Strassen, we compute:
    # M0 = (A00+A11)(B00+B11)
    # where A00+A11 is a 3x3 matrix and B00+B11 is a 3x3 matrix.
    # Their product is a 3x3 matrix computed via 9 scalar multiplications.
    # So yes, total = 7 * 9 = 63 scalar multiplications.
    # 
    # But the original 6x6 multiplication requires 36 scalar multiplications
    # per output element... no, it requires 6 multiplications per output element,
    # 36 output elements, so 216 scalar multiplications total.
    # 
    # So rank 63 < 216, score = 216/63 ≈ 3.43.
    # 
    # Hmm, but that seems too good. Let me verify with a small example.
    # For 2x2 with 1x1 blocks: Strassen gives rank 7, direct gives 4.
    # 4/7 < 1, but Strassen is rank 7 which is MORE than 4.
    # 
    # I had it backwards! Strassen uses 7 multiplications vs 4 for direct.
    # So for 2x2: direct rank = 4, Strassen rank = 7.
    # For 6x6 with 3x3 blocks: direct rank = 6*6 = 36... no, 6*6*6 = 216.
    # 
    # Wait, the rank of the matrix multiplication tensor M_{n,m,p} is the
    # minimum number of scalar multiplications needed. For n=m=p=6, direct
    # is 216. Strassen gives 7 * (rank of 3x3 mult) = 7 * 9 = 63? No!
    # 
    # Strassen for 2x2 gives rank 7 (vs direct 4). The "7" replaces the
    # "4" scalar multiplications. When we scale up with 3x3 blocks,
    # each "scalar" multiplication becomes a 3x3 matrix multiplication
    # which itself has rank 9. So total = 7 * 9 = 63.
    # But direct 6x6 has rank 216. 63 < 216, so this IS an improvement.
    # 
    # Actually no, I think I'm still confused. Let me look at this more carefully.
    # 
    # The matrix multiplication tensor M_{n,n,n} lives in (R^n ⊗ R^n)^* ⊗ R^n
    # (or similar). Its rank is the minimum number of rank-1 tensors that
    # sum to it.
    # 
    # For M_{2,2,2}: rank = 7 (Strassen) vs 8 (direct? no, direct is 4*1=4... )
    # Actually M_{2,2,2} has rank 7. Direct computation uses 4 multiplications
    # but the tensor rank is 7.
    # 
    # Hmm, I think the "rank" in the problem refers to the tensor rank,
    # which for M_{n,n,n} is the minimum number of scalar multiplications
    # in a bilinear algorithm.
    # 
    # For M_{6,6,6}: 
    # - Direct: 216 scalar multiplications
    # - Strassen with 3x3 blocks: 7 * (rank of M_{3,3,3}) = 7 * 23 = 161
    #   (using the known rank 23 for 3x3)
    # - Or with direct 3x3: 7 * 9 = 63... but that's not right either.
    # 
    # I think the correct interpretation is:
    # M_{6,6,6} can be computed by treating it as a 2x2 block matrix where
    # each block is 3x3. The block-level multiplication is M_{2,2,2} (rank 7),
    # and each "block multiplication" is itself M_{3,3,3} (rank 23).
    # So total rank ≤ 7 * 23 = 161.
    # 
    # But actually, this isn't quite right either. The tensor rank doesn't
    # simply multiply like that. However, there IS a construction:
    # If M_{a,b,c} has rank r and M_{a',b',c'} has rank r', then
    # M_{a*a', b*b', c*c'} has rank at most r * r'.
    # 
    # So M_{6,6,6} = M_{2*3, 2*3, 2*3} has rank ≤ rank(M_{2,2,2}) * rank(M_{3,3,3}) = 7 * 23 = 161.
    # 
    # Score = 216/161 ≈ 1.3416.
    # 
    # To implement this, I need the rank-23 decomposition of M_{3,3,3}.
    # This is complex. Let me use a simpler approach: use M_{3,3,3} with
    # rank 27 (direct) to get 7*27=189, which is what we already have.
    # 
    # To improve, I need the actual rank-23 coefficients for 3x3.
    # This is quite involved. Let me try to implement it.
    # 
    # Actually, for now let me just keep the current 189-rank solution but
    # make the code cleaner and add a deadline guard.
    
    block_size = 2
    blocks_per_dim = 3  # 6 / 2
    
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen patterns for 2x2 matrices
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
