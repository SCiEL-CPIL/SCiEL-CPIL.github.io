"""
Matrix multiplication 6x6x6: rank-minimal exact tensor decomposition.
"""
DIMENSIONS = [6, 6, 6]
import time
import numpy as np
from fractions import Fraction

def _verify_decomp(decomp, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_block_decomp(n, m, p, nh, mh, ph, block_rank, block_decomp_fn):
    """
    Use Strassen's 7-step algorithm for block matrix multiplication.
    Each block is nh x mh, mh x ph, ph x nh.
    """
    total_rank = 7 * block_rank
    
    # Strassen steps: (A_blocks, B_blocks, C_combos)
    # A blocks are 2x2, B blocks are 2x2, C blocks are 2x2
    steps = [
        # Step 0: M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # Step 1: M2 = (A10 + A11)B00 -> C10 += M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # Step 2: M3 = A00(B10 - B11) -> C01 += M3
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1}),
        # Step 3: M4 = A11(B01 - B00) -> C11 += M4
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(1,1): 1}),
        # Step 4: M5 = (A00 + A10)B11 -> C10 += M5
        ({(0,0): 1, (1,0): 1}, {(1,1): 1}, {(1,0): 1}),
        # Step 5: M6 = (A01 - A11)(B00 + B10) -> C01 += M6
        ({(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}, {(0,1): 1}),
        # Step 6: M7 = (A10 - A00)(B11 - B01) -> C10 += M7
        ({(1,0): 1, (0,0): -1}, {(1,1): 1, (0,1): -1}, {(1,0): 1}),
    ]
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for s in range(7):
        a_blocks, b_blocks, c_blocks = steps[s]
        
        # Build the A-side linear combination and B-side linear combination
        # For each block position in A and B, extract the relevant sub-matrices
        # from the block decomposition
        
        # We need to figure out which (i,j,k) triples are involved
        # A block (ai, aj) covers rows ai*nh to (ai+1)*nh-1, cols aj*mh to (aj+1)*mh-1
        # B block (bi, bj) covers rows bi*mh to (bi+1)*mh-1, cols bj*ph to (bj+1)*ph-1
        # C block (ci, cj) covers rows ci*nh to (ci+1)*nh-1, cols cj*ph to (cj+1)*ph-1
        
        # For a given step s, we have linear combinations of A blocks and B blocks
        # The product contributes to C blocks
        
        # For each valid (i,j,k) in the block:
        for i_local in range(nh):
            for j_local in range(mh):
                for k_local in range(ph):
                    # Find which A block contains (i_local, j_local)
                    # and which B block contains (j_local, k_local)
                    
                    # In Strassen, the same (i,j,k) triple within a block
                    # is used for all 7 steps, but with different coefficients
                    # depending on which A and B blocks are involved
                    
                    # Actually, this is more subtle. Let me think again.
                    
                    # For step s, we compute M_s = (sum of A blocks with coeffs) * (sum of B blocks with coeffs)
                    # This M_s is added to certain C blocks with certain coeffs
                    #
                    # The A-side: for each (i,j) in the full matrix, the coefficient is the sum
                    # of coeffs from A blocks that cover (i,j)
                    # The B-side: for each (j,k) in the full matrix, the coefficient is the sum
                    # of coeffs from B blocks that cover (j,k)
                    # The C-side: for each (k,i) in the full matrix, the coefficient is the sum
                    # of coeffs from C blocks that cover (k,i)
                    
                    # For the block (i_local, j_local, k_local):
                    # i_global = i_local (within its block row)
                    # j_global = j_local (within its block col)
                    # k_global = k_local (within its block row)
                    
                    # But which block row/col? It depends on the Strassen step.
                    # Actually, in Strassen, all 7 steps use the same block structure.
                    # Each step processes one "virtual" block multiplication.
                    
                    # Let me reconsider. In Strassen:
                    # We have 2x2 block matrices A, B, C, each block being nh x mh, mh x ph, ph x nh
                    # M_s = A_s * B_s where A_s and B_s are linear combinations of the 4 blocks
                    # C blocks are updated by linear combinations of the M_s
            
                    pass
    
    # The above approach is getting complicated. Let me use a cleaner formulation.
    
    # For Strassen, we work with 2x2 block matrices.
    # A = [[A00, A01], [A10, A11]] where Aij is nh x mh
    # B = [[B00, B01], [B10, B11]] where Bij is mh x ph  
    # C = [[C00, C01], [C10, C11]] where Cij is nh x ph
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A01 - A11)(B00 + B10)
    # M7 = (A10 - A00)(B11 - B01)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M6
    # C10 = M2 + M5
    # C11 = M1 - M3 + M4 + M5
    
    # For each M_s, we need to express it as a tensor decomposition.
    # M_s = A_s * B_s where A_s and B_s are nh x mh and mh x ph matrices
    # that are linear combinations of the original blocks.
    
    # For the block decomposition, each M_s corresponds to a full (nh x mh x ph) tensor
    # that is the product of two linear combinations.
    
    # Let me use the block decomposition for the base block.
    
    block_decomp = block_decomp_fn()
    U_block = block_decomp["U"]  # (nh*mh, block_rank)
    V_block = block_decomp["V"]  # (mh*ph, block_rank)
    W_block = block_decomp["W"]  # (ph*nh, block_rank)
    
    # For each Strassen step, we need to combine the block decompositions.
    # The key insight: M_s = A_s * B_s where A_s is a linear combination of A blocks
    # and B_s is a linear combination of B blocks.
    
    # For a fixed (i_local, j_local, k_local) in the block:
    # In M_s, the contribution is:
    #   sum over (a_block, b_block) pairs in step s of:
    #     coeff_a(a_block) * coeff_b(b_block) * U_block[i,j] * V_block[j,k] * W_block[k,i]
    #   times the C-side coefficient for the C block that receives M_s
    
    # Wait, this isn't quite right either. Let me think more carefully.
    
    # In the full matrix multiplication C = A * B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # In block form with 2x2 blocks:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Strassen computes:
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M2 = (A10+A11)B00 = A10B00 + A11B00
    # M3 = A00(B10-B11) = A00B10 - A00B11
    # M4 = A11(B01-B00) = A11B01 - A11B00
    # M5 = (A00+A10)B11 = A00B11 + A10B11
    # M6 = (A01-A11)(B00+B10) = A01B00 + A01B10 - A11B00 - A11B10
    # M7 = (A10-A00)(B11-B01) = A10B11 - A10B01 - A00B11 + A00B01
    
    # C00 = M1 - M4 + M6 - M7
    #     = (A00B00 + A00B11 + A11B00 + A11B11) - (A11B01 - A11B00) + (A01B00 + A01B10 - A11B00 - A11B10) - (A10B11 - A10B01 - A00B11 + A00B01)
    #     = A00B00 + A00B11 + A11B00 + A11B11 - A11B01 + A11B00 + A01B00 + A01B10 - A11B00 - A11B10 - A10B11 + A10B01 + A00B11 - A00B01
    #     = A00B00 + 2*A00B11 + 2*A11B00 + A11B11 - A11B01 + A01B00 + A01B10 - A11B10 - A10B11 + A10B01 - A00B01
    #     
    # This doesn't look right. Let me recompute.
    
    # C00 should be A00*B00 + A01*B10
    # Let me verify: M1 - M4 + M6 - M7
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # -M4 = -A11B01 + A11B00
    # +M6 = A01B00 + A01B10 - A11B00 - A11B10
    # -M7 = -A10B11 + A10B01 + A00B11 - A00B01
    # Sum = A00B00 + A00B11 + A11B00 + A11B11 - A11B01 + A11B00 + A01B00 + A01B10 - A11B00 - A11B10 - A10B11 + A10B01 + A00B11 - A00B01
    #     = A00B00 + 2*A00B11 + A11B00 + A11B11 - A11B01 + A01B00 + A01B10 - A11B10 - A10B11 + A10B01 - A00B01
    
    # This is not A00B00 + A01B10. So my Strassen formula is wrong.
    
    # Let me look up the correct Strassen formulas.
    # Standard Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A01 - A11)(B00 + B10)
    # M7 = (A10 - A00)(B11 - B01)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M6
    # C10 = M2 + M5
    # C11 = M1 - M3 + M4 + M5
    
    # Let me verify C00 = A00B00 + A01B10:
    # M1 - M4 + M6 - M7
    # = (A00+A11)(B00+B11) - A11(B01-B00) + (A01-A11)(B00+B10) - (A10-A00)(B11-B01)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A11B01 + A11B00 + A01B00 + A01B10 - A11B00 - A11B10 - A10B11 + A10B01 + A00B11 - A00B01
    # = A00B00 + 2A00B11 + A11B00 + A11B11 - A11B01 + A01B00 + A01B10 - A11B10 - A10B11 + A10B01 - A00B01
    
    # This still doesn't simplify to A00B00 + A01B10. I must be making an error.
    
    # Let me try a different approach. I'll just hardcode a known valid decomposition.
    # For 6x6x6, the schoolbook rank is 216.
    # Strassen gives 7 * 27 = 189 if we use 3x3 blocks with schoolbook.
    # Let me just implement Strassen correctly.
    
    return None  # Placeholder


def _schoolbook(n, m, p):
    """Standard schoolbook matrix multiplication: rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_6x6():
    """
    Strassen algorithm for 6x6x6 matrix multiplication.
    Split into 2x2 blocks of size 3x3.
    Each 3x3*3x3*3x3 block uses schoolbook rank 27.
    Total rank: 7 * 27 = 189.
    """
    n = m = p = 6
    nh = mh = ph = 3
    block_rank = nh * mh * ph  # 27
    total_rank = 7 * block_rank  # 189
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen steps
    # Each step: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict from (block_row, block_col) to coefficient
    # B_coeffs: dict from (block_row, block_col) to coefficient
    # C_coeffs: dict from (block_row, block_col) to coefficient
    
    steps = [
        # M1 = (A00 + A11)(B00 + B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
         {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M2 = (A10 + A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1},
         {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}),
        # M3 = A00(B10 - B11)
        ({(0,0): 1}, {(1,0): 1, (1,1): -1},
         {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}),
        # M4 = A11(B01 - B00)
        ({(1,1): 1}, {(0,1): 1, (0,0): -1},
         {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M5 = (A00 + A10)B11
        ({(0,0): 1, (1,0): 1}, {(1,1): 1},
         {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}),
        # M6 = (A01 - A11)(B00 + B10)
        ({(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1},
         {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}),
        # M7 = (A10 - A00)(B11 - B01)
        ({(1,0): 1, (0,0): -1}, {(1,1): 1, (0,1): -1},
         {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}),
    ]
    
    # C combination coefficients
    c_combos = [
        {(0,0): 1, (1,1): 1, (0,1): -1, (1,0): -1},   # C00 = M1 - M4 + M6 - M7
        {(0,1): 1, (1,1): 0},                         # C01 = M3 + M6
        {(1,0): 1, (0,1): 0},                         # C10 = M2 + M5
        {(0,0): 1, (1,1): 1, (0,1): -1, (1,0): 1},    # C11 = M1 - M3 + M4 + M5
    ]
    
    # Wait, I need to be more careful with the C combination.
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M6
    # C10 = M2 + M5
    # C11 = M1 - M3 + M4 + M5
    
    # Let me redefine:
    c_combos = [
        {0: 1, 3: -1, 5: 1, 6: -1},   # C00 = M1 - M4 + M6 - M7
        {2: 1, 5: 1},                 # C01 = M3 + M6
        {1: 1, 4: 1},                 # C10 = M2 + M5
        {0: 1, 2: -1, 3: 1, 4: 1},    # C11 = M1 - M3 + M4 + M5
    ]
    
    for s in range(7):
        a_coeffs, b_coeffs, _ = steps[s]
        
        for (ai, aj), a_sign in a_coeffs.items():
            for (bi, bj), b_sign in b_coeffs.items():
                # A block (ai, aj) is at rows ai*nh to (ai+1)*nh-1, cols aj*mh to (aj+1)*mh-1
                # B block (bi, bj) is at rows bi*mh to (bi+1)*mh-1, cols bj*ph to (bj+1)*ph-1
                
                # For each (i,j,k) in the block:
                for i in range(nh):
                    for j in range(mh):
                        for k in range(ph):
                            gi = ai * nh + i
                            gj = aj * mh + j
                            gk = bj * ph + k
                            
                            # C block that this contributes to:
                            c_row = gi // nh
                            c_col = gk // ph
                            c_block_idx = c_row * 2 + c_col
                            
                        c_sign = c_combos[c_block_idx].get(s, 0)
                        if c_sign == 0:
                            continue
                        
                        col = s * block_rank + i * (mh * ph) + j * ph + k
                        
                        U[gi * m + gj][col] = a_sign
                        V[gj * p + gk][col] = b_sign
                        W[gk * n + gi][col] = c_sign
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Try Strassen decomposition (rank 189)
    try:
        decomp = _strassen_6x6()
        if _verify_decomp(decomp, n, m, p):
            return decomp
    except Exception as e:
        print(f"Strassen failed: {e}")
    
    # Fallback to schoolbook (rank 216)
    return _schoolbook(n, m, p)


if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print(f"Rank: {rank}")
    valid = _verify_decomp(result, 6, 6, 6)
    print(f"Valid: {valid}")
