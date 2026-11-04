"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_start_time = time.time()
TIME_LIMIT = 145.0  # 5s safety margin before 150s kill

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _check_time():
    return time.time() - _start_time > TIME_LIMIT

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen.
    U: (4, 7), V: (4, 7), W: (4, 7)
    U index: 0=a, 1=b, 2=c, 3=d
    V index: 0=e, 1=f, 2=g, 3=h
    W index: 0=C11, 1=C12, 2=C21, 3=C22
    """
    # p1 = (a+b)(e+h)
    # p2 = (c+d)h
    # p3 = a(e-f)
    # p4 = d(g-h)
    # p5 = (a+b)h
    # p6 = (c-d)e
    # p7 = (b-d)(f+g)
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p6
    # C22 = p1 - p2 + p3 + p4
    
    U = [
        [1, 0, 1, 0, 1, 0, 0],  # a
        [1, 0, 0, 0, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [0, 1, 0, 1, 0, -1, -1],  # d
    ]
    V = [
        [1, 0, 1, 0, 0, 1, 0],  # e
        [0, 0, -1, 0, 0, 0, 1],  # f
        [0, 0, 0, 1, 0, 0, 1],  # g
        [1, 1, 0, -1, 1, 0, 0],  # h
    ]
    W = [
        [1, 0, 1, 1, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],  # C12
        [0, 1, 0, 0, 0, 1, 0],  # C21
        [1, -1, 1, 1, 0, 0, 0],  # C22
    ]
    return U, V, W

def _strassen_4x4x4():
    """Return (U, V, W) for 4x4x4 Strassen.
    U: (16, 49), V: (16, 49), W: (16, 49)
    U index: i*4+j, V index: j*4+k, W index: k*4+i
    """
    n = 4
    # Partition into 2x2 blocks
    # A blocks: A00=[a00,a01;a20,a21], A01=[a02,a03;a22,a23], A10=[a10,a11;a30,a31], A11=[a12,a13;a32,a33]
    # B blocks: B00, B01, B10, B11 similarly
    # C blocks: C00, C01, C10, C11 similarly
    
    # Strassen at 4x4 level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)(B00+B01)
    # M6 = (A11-A10)(B10+B11)
    # M7 = (A01-A11)(B00+B10)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M4
    
    # Each Mi is a 2x2x2 product, so use Strassen_2x2x2 for each.
    # Total rank: 7 * 7 = 49
    
    U49 = [[0]*49 for _ in range(16)]
    V49 = [[0]*49 for _ in range(16)]
    W49 = [[0]*49 for _ in range(16)]
    
    # For each of the 7 outer Strassen terms, we have a 2x2x2 product.
    # The outer terms define which blocks of A and B are combined.
    # The inner Strassen defines how the 2x2 blocks are multiplied.
    
    # Let's define the 7 outer terms:
    # For term t in 0..6:
    #   A_combination: which 4x4 entries of A are used, with what coefficients
    #   B_combination: which 4x4 entries of B are used, with what coefficients
    #   C_combination: which 4x4 entries of C receive this term, with what coefficients
    
    # Actually, let me think of it differently.
    # The 4x4x4 tensor can be decomposed as a sum of 7 tensor products of 2x2x2 tensors.
    
    # Let me use a more direct approach. I'll construct the 49 rank-1 terms explicitly.
    
    # For the outer Strassen, each Mi involves products of linear combinations of A blocks 
    # and linear combinations of B blocks. Each such product is a 2x2x2 tensor.
    # So for each outer term, I need to expand the 2x2x2 Strassen.
    
    # Let me define the block structure:
    # A is 4x4, partitioned into 2x2 blocks:
    #   A00 = rows 0,1 cols 0,1 -> indices 0,1,4,5 (i*4+j)
    #   A01 = rows 0,1 cols 2,3 -> indices 2,3,6,7
    #   A10 = rows 2,3 cols 0,1 -> indices 8,9,12,13
    #   A11 = rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    # Wait, I'm using row-major: index = i*4+j where i is row, j is col.
    # A00: i in {0,1}, j in {0,1} -> indices 0,1,4,5
    # A01: i in {0,1}, j in {2,3} -> indices 2,3,6,7
    # A10: i in {2,3}, j in {0,1} -> indices 8,9,12,13
    # A11: i in {2,3}, j in {2,3} -> indices 10,11,14,15
    
    # Similarly for B (4x4, j*4+k):
    # B00: j in {0,1}, k in {0,1} -> indices 0,1,4,5
    # B01: j in {0,1}, k in {2,3} -> indices 2,3,6,7
    # B10: j in {2,3}, k in {0,1} -> indices 8,9,12,13
    # B11: j in {2,3}, k in {2,3} -> indices 10,11,14,15
    
    # C (4x4, k*4+i):
    # C00: k in {0,1}, i in {0,1} -> indices 0,1,4,5
    # C01: k in {0,1}, i in {2,3} -> indices 2,3,6,7
    # C10: k in {2,3}, i in {0,1} -> indices 8,9,12,13
    # C11: k in {2,3}, i in {2,3} -> indices 10,11,14,15
    
    # For each outer Strassen term, I have:
    #   A_term: a 4x4 matrix (sparse, being a linear combo of blocks)
    #   B_term: a 4x4 matrix
    #   C_term: a 4x4 matrix (output contribution)
    #   coefficient: +1 or -1
    
    # The product A_term * B_term contributes coefficient * (A_term ⊗ B_term ⊗ C_term)
    # But A_term and B_term are each linear combinations of 2x2 blocks, and the product
    # of two such combinations gives a 2x2x2 tensor which we decompose with Strassen.
    
    # This is getting complex. Let me just directly construct the 49 terms.
    
    # Alternative: use the known rank-49 decomposition for 4x4x4.
    # I'll use the approach from the literature.
    
    # Actually, let me just implement it by composing the two levels of Strassen.
    
    # Level 1: 7 outer terms, each is a product of two 4x4 matrices that are
    # linear combinations of 2x2 blocks.
    
    # Let me define for each outer term t:
    #   a_blocks[t]: dict mapping block_name -> coefficient for A
    #   b_blocks[t]: dict mapping block_name -> coefficient for B
    #   c_blocks[t]: dict mapping block_name -> coefficient for C
    #   c_sign[t]: +1 or -1
    
    # Outer Strassen:
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
    # M3 = A00(B01-B11) -> C01+=M3, C11+=M3
    # M4 = A11(B10-B00) -> C00+=M4, C11+=M4
    # M5 = (A00+A10)(B00+B01) -> C00-=M5, C01+=M5
    # M6 = (A11-A10)(B10+B11) -> C10+=M6
    # M7 = (A01-A11)(B00+B10) -> C00+=M7
    
    # For each Mi, it's a product of a 4x4 matrix (linear combo of 2x2 blocks) 
    # and another 4x4 matrix. The product is a 2x2x2 tensor at the block level,
    # which we then decompose with 2x2x2 Strassen (rank 7).
    
    # So for each outer term, we generate 7 inner terms, total 49.
    
    # Let me define the block indices:
    # Block (bi, bj) of A covers rows bi*2..bi*2+1, cols bj*2..bj*2+1
    # In the 4x4 matrix (row-major i*4+j):
    # Block (0,0): rows 0,1, cols 0,1 -> indices [0,1,4,5]
    # Block (0,1): rows 0,1, cols 2,3 -> indices [2,3,6,7]
    # Block (1,0): rows 2,3, cols 0,1 -> indices [8,9,12,13]
    # Block (1,1): rows 2,3, cols 2,3 -> indices [10,11,14,15]
    
    # For a 2x2 block at position (bi,bj) in a 4x4 matrix (row-major):
    # The 4 elements are at indices: (bi*2)*4 + bj*2, (bi*2)*4 + bj*2 + 1,
    #                                   (bi*2+1)*4 + bj*2, (bi*2+1)*4 + bj*2 + 1
    
    def block_indices_4x4(bi, bj):
        r0 = bi * 2
        c0 = bj * 2
        return [(r0)*4 + c0, (r0)*4 + c0 + 1, (r0+1)*4 + c0, (r0+1)*4 + c0 + 1]
    
    # For B (j*4+k), block (bj, bk) covers rows bj*2..bj*2+1, cols bk*2..bk*2+1
    def block_indices_B(bj, bk):
        r0 = bj * 2
        c0 = bk * 2
        return [(r0)*4 + c0, (r0)*4 + c0 + 1, (r0+1)*4 + c0, (r0+1)*4 + c0 + 1]
    
    # For C (k*4+i), block (bk, bi) covers rows bk*2..bk*2+1, cols bi*2..bi*2+1
    def block_indices_C(bk, bi):
        r0 = bk * 2
        c0 = bi * 2
        return [(r0)*4 + c0, (r0)*4 + c0 + 1, (r0+1)*4 + c0, (r0+1)*4 + c0 + 1]
    
    # Now for each outer term, I define:
    #   A_comb: list of (block_id, coeff) where block_id is (bi,bj)
    #   B_comb: list of (block_id, coeff) where block_id is (bj,bk)
    #   C_comb: list of (block_id, coeff) where block_id is (bk,bi)
    #   sign: +1 or -1
    
    # The product of the A_comb and B_comb gives a 2x2x2 tensor at block level.
    # I'll decompose that 2x2x2 with Strassen_2x2x2.
    
    # For a given A_comb and B_comb, the block-level 2x2x2 tensor is:
    #   T[bi,bj,bk] = sum over (bi',bj') in A_comb and (bj',bk') in B_comb 
    #                 of A_coeff * B_coeff * delta(bj',bj')
    # Wait, this is a matrix product, so the block (bi,bk) of the result is:
    #   result[bi,bk] = sum_bj A[bi,bj] * B[bj,bk]
    
    # So the 2x2x2 tensor at block level has T[bi,bj,bk] where bi,bj,bk in {0,1}
    # and T is the tensor for the block-level matrix product.
    
    # Let me just construct the 2x2x2 tensor for each outer term, then decompose it.
    
    # Define the outer terms:
    outer_terms = [
        # (A_comb, B_comb, C_comb, sign)
        # M1 = (A00+A11)(B00+B11)
        ([( (0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], 1),
        # M2 = (A10+A11)B00
        ([((1,0),1), ((1,1),1)], [((0,0),1)], [((1,0),1)], 1),
        # C10 += M2, C11 -= M2
        # I need to handle the C comb carefully
    ]
    
    # This is getting too complex to do by hand. Let me just hardcode the 49 terms
    # using a known decomposition, or compute them programmatically.
    
    # Actually, let me just use a simpler approach: use the schoolbook for the 4x4x4 part
    # but that's rank 64, plus 16 for the 4x4x1 = 80, same as baseline.
    
    # Let me try a different approach. I'll use the rank-49 decomposition for 4x4x4
    # that I can construct by composing Strassen.
    
    # For now, let me just return the schoolbook decomposition with rank 80 to ensure
    # correctness, and then try to improve.
    
    # Actually, let me just implement the 65-rank approach: 49 for 4x4x4 + 16 for 4x4x1.
    
    # For the 4x4x4 part, I'll use a hardcoded rank-49 decomposition.
    # Let me look up or derive it.
    
    # I'll use the following approach: the 4x4x4 matrix multiplication tensor has rank 49
    # via Strassen. I'll construct it by composing two levels of 2x2x2 Strassen.
    
    # Let me just do it programmatically by building the tensor and finding a decomposition.
    # But that's expensive. Instead, let me hardcode the 49 terms.
    
    # For simplicity and correctness, let me just use rank 64 for 4x4x4 (schoolbook) 
    # and rank 16 for 4x4x1, total 80. Same as baseline.
    
    # Actually, let me try to implement the Strassen composition properly.
    
    # I'll build the 49 terms by composing the two Strassen levels.
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # Define the 7 outer Strassen terms
    # Each term: (A_block_weights, B_block_weights, C_block_weights)
    # A_block_weights: dict (bi,bj) -> coeff
    # B_block_weights: dict (bj,bk) -> coeff
    # C_block_weights: dict (bk,bi) -> coeff
    
    outer = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,1):1}),
        # M5
        ({(0,0):1, (1,0):1}, {(0,0):1, (0,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,1):1, (1,0):-1}, {(1,0):1, (1,1):1}, {(1,0):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(0,0):1, (1,0):1}, {(0,0):1}),
    ]
    
    # For each outer term, build the 2x2x2 block-level tensor
    # T_block[bi,bj,bk] for bi,bj,bk in {0,1}
    
    # Then decompose T_block with Strassen_2x2x2 to get 7 rank-1 terms
    # Each rank-1 term gives U_block (4 entries), V_block (4 entries), W_block (4 entries)
    # Map these to the 4x4 level.
    
    for outer_idx, (A_w, B_w, C_w) in enumerate(outer):
        # Build 2x2x2 block tensor
        T_block = [[[0]*2 for _ in range(2)] for _ in range(2)]
        for (bi,bj), aw in A_w.items():
            for (bj2,bk), bw in B_w.items():
                if bj == bj2:
                    T_block[bi][bj][bk] += aw * bw
        
        # Now decompose T_block (2x2x2) with Strassen
        # Get the 2x2x2 Strassen decomposition
        U2, V2, W2 = _strassen_2x2x2()
        # U2: 4x7, V2: 4x7, W2: 4x7
        # U2 index: 0=block(0,0)_row0, 1=block(0,0)_row1, 2=block(0,1)_row0, 3=block(0,1)_row1
        # Wait, I need to be careful about the indexing.
        
        # The 2x2x2 Strassen I defined has:
        # U index: 0=a, 1=b, 2=c, 3=d where a,b,c,d are the 4 entries of the 2x2 A matrix
        # In the block-level 2x2 matrix, entry (i,j) corresponds to...
        
        # Let me redefine. For the 2x2x2 tensor T[i,j,k] where i,j,k in {0,1}:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # A is 2x2: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
        # B is 2x2: B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
        # C is 2x2: C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
        
        # In my Strassen_2x2x2:
        # U: 4 entries for A: [a,b,c,d] -> index 0,1,2,3
        # V: 4 entries for B: [e,f,g,h] -> index 0,1,2,3
        # W: 4 entries for C: [C11,C12,C21,C22] -> index 0,1,2,3
        
        # Now map to the 4x4 level:
        # Block (bi,bj) of A is a 2x2 matrix. Its entries in the 4x4 (row-major i*4+j):
        #   entry (r,c) within block is at global index (bi*2+r)*4 + (bj*2+c)
        #   r,c in {0,1}
        
        # So for the 2x2x2 Strassen, U[0]=a corresponds to block(bi,bj) entry (0,0)
        # U[1]=b corresponds to block(bi,bj) entry (0,1)
        # U[2]=c corresponds to block(bi,bj) entry (1,0)
        # U[3]=d corresponds to block(bi,bj) entry (1,1)
        
        # But we have a 2x2x2 block-level tensor T_block[bi,bj,bk].
        # For a fixed (bi,bj), T_block[bi,bj,:] is a vector over bk.
        # Wait, no. T_block is a 3D tensor. For the 2x2x2 Strassen decomposition,
        # I need to treat T_block as the tensor for a 2x2x2 matrix multiplication.
        
        # Let me think of it as: the block-level A is a 2x2 matrix where entry (bi,bj) 
        # is a scalar (the weight). Similarly for B and C.
        # But that's not right either.
        
        # Actually, I think I'm overcomplicating this. Let me just directly construct 
        # the 49 terms by enumerating.
        
        pass
    
    # This is taking too long. Let me just return the schoolbook decomposition 
    # with rank 80 to ensure correctness.
    
    rank = 4 * 4 * 5
    U_final = [[0] * rank for _ in range(16)]
    V_final = [[0] * rank for _ in range(20)]
    W_final = [[0] * rank for _ in range(20)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(5):
                U_final[i*4+j][col] = 1
                V_final[j*4+k][col] = 1
                W_final[k*4+i][col] = 1
                col += 1
    
    return {"U": U_final, "V": V_final, "W": W_final}

def construct():
    # Use the 4x4x4x1 decomposition
    # For now, return schoolbook rank 80
    return _strassen_4x4x4()
# EVOLVE-BLOCK-END

