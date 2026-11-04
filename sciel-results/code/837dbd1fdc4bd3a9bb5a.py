"""Strassen-Winograd rank-49 decomposition for 4x4 matrix multiplication."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def _strassen_2x2_decomp():
    """
    Return the 7 rank-1 terms for Strassen 2x2 matrix multiplication.
    Each term is (u, v, w) where u is in R^4, v in R^4, w in R^4.
    A and B are 2x2 matrices (flattened row-major: [a00, a01, a10, a11]).
    C = AB (flattened row-major: [c00, c01, c10, c11]).
    """
    # Strassen formulas for 2x2:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b01-b11)
    # p4 = a11(b10-b00)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    # c00 = p1+p4-p5+p7
    # c01 = p3+p5
    # c10 = p2+p4
    # c11 = p1-p2+p3+p6
    
    terms = []
    # p1
    u = [1, 0, 0, 1]  # a00 + a11
    v = [1, 0, 0, 1]  # b00 + b11
    w = [1, 0, 0, 1]  # c00 += p1, c11 += p1
    terms.append((u, v, w))
    
    # p2
    u = [0, 0, 1, 1]  # a10 + a11
    v = [1, 0, 0, 0]  # b00
    w = [0, 0, 1, 0]  # c10 += p2
    terms.append((u, v, w))
    
    # p3
    u = [1, 0, 0, 0]  # a00
    v = [0, 1, 0, -1] # b01 - b11
    w = [0, 1, 0, 1]  # c01 += p3, c11 += p3
    terms.append((u, v, w))
    
    # p4
    u = [0, 0, 0, 1]  # a11
    v = [0, 0, 1, -1] # b10 - b00
    w = [1, 0, 1, 0]  # c00 += p4, c10 += p4
    terms.append((u, v, w))
    
    # p5
    u = [1, 1, 0, 0]  # a00 + a01
    v = [0, 0, 0, 1]  # b11
    w = [-1, 0, 0, 1] # c00 -= p5, c11 += p5
    terms.append((u, v, w))
    
    # p6
    u = [0, 0, 1, -1] # a10 - a00
    v = [1, 1, 0, 0]  # b00 + b01
    w = [0, 0, 0, 1]  # c11 += p6
    terms.append((u, v, w))
    
    # p7
    u = [0, 1, 0, -1] # a01 - a11
    v = [0, 0, 1, 1]  # b10 + b11
    w = [1, 0, 0, 0]  # c00 += p7
    terms.append((u, v, w))
    
    return terms

def _split_4x4_blocks(a_flat):
    """Split 4x4 matrix (16 elements, row-major) into 4 2x2 blocks."""
    # Block (0,0): rows 0-1, cols 0-1 -> indices 0,1,4,5
    # Block (0,1): rows 0-1, cols 2-3 -> indices 2,3,6,7
    # Block (1,0): rows 2-3, cols 0-1 -> indices 8,9,12,13
    # Block (1,1): rows 2-3, cols 2-3 -> indices 10,11,14,15
    a00 = [a_flat[0], a_flat[1], a_flat[4], a_flat[5]]
    a01 = [a_flat[2], a_flat[3], a_flat[6], a_flat[7]]
    a10 = [a_flat[8], a_flat[9], a_flat[12], a_flat[13]]
    a11 = [a_flat[10], a_flat[11], a_flat[14], a_flat[15]]
    return a00, a01, a10, a11

def _merge_4x4_blocks(c00, c01, c10, c11):
    """Merge 4 2x2 blocks back into 4x4 matrix (16 elements, row-major)."""
    c = [0]*16
    # Block (0,0): rows 0-1, cols 0-1
    c[0] = c00[0]; c[1] = c00[1]; c[4] = c00[2]; c[5] = c00[3]
    # Block (0,1): rows 0-1, cols 2-3
    c[2] = c01[0]; c[3] = c01[1]; c[6] = c01[2]; c[7] = c01[3]
    # Block (1,0): rows 2-3, cols 0-1
    c[8] = c10[0]; c[9] = c10[1]; c[12] = c10[2]; c[13] = c10[3]
    # Block (1,1): rows 2-3, cols 2-3
    c[10] = c11[0]; c[11] = c11[1]; c[14] = c11[2]; c[15] = c11[3]
    return c

def construct():
    """
    Construct rank-49 Strassen-Winograd decomposition for 4x4 matrix multiplication.
    
    The tensor decomposition computes C = A*B where A is 4x4, B is 4x4, C is 4x4.
    Output convention: W[k*n+i, r] corresponds to C[i,k] (column-major in output).
    
    We use hierarchical Strassen: 7 top-level 2x2 multiplications, each decomposed
    into 7 scalar multiplications, giving 49 total.
    """
    n, m, p = DIMENSIONS  # all 4
    
    strassen_terms = _strassen_2x2_decomp()
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Top-level Strassen: compute C = A*B using 7 products of 2x2 blocks
    # Let A be partitioned into blocks A00, A01, A10, A11 (each 2x2)
    # Let B be partitioned into blocks B00, B01, B10, B11 (each 2x2)
    # 
    # Strassen 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # Each Mi is a 2x2 product, decomposed via Strassen 2x2 into 7 scalar products.
    
    # For each top-level Strassen term (t = 0..6), and each inner Strassen term (s = 0..6)
    # We have a scalar product that contributes to the final result.
    
    for t in range(7):
        u_top, v_top, w_top = strassen_terms[t]
        
        # u_top selects a linear combination of A's 2x2 blocks
        # v_top selects a linear combination of B's 2x2 blocks
        # w_top tells us which C blocks receive this product
        
        # The "A side" of product t is a linear combination of A00, A01, A10, A11
        # The "B side" of product t is a linear combination of B00, B01, B10, B11
        
        # For inner Strassen: we're computing (A_comb) * (B_comb) where A_comb and B_comb are 2x2
        # The inner Strassen terms give us 7 scalar products
        
        for s in range(7):
            u_inner, v_inner, w_inner = strassen_terms[s]
            idx = t * 7 + s
            
            # u_inner is a 4-vector selecting elements of the 2x2 A_comb matrix
            # v_inner is a 4-vector selecting elements of the 2x2 B_comb matrix
            # w_inner is a 4-vector for the 2x2 result
            
            # Now we need to figure out how A_comb and B_comb relate to the full 4x4 matrices.
            # 
            # A_comb is a linear combination of A00, A01, A10, A11 with coefficients from u_top
            # Specifically, A_comb = u_top[0]*A00 + u_top[1]*A01 + u_top[2]*A10 + u_top[3]*A11
            # Similarly for B_comb.
            #
            # The inner Strassen u_inner selects elements of A_comb:
            # A_comb = [a00', a01'; a10', a11'] where each aij' is a linear combo of the 4x4 entries
            #
            # u_inner = [x0, x1, x2, x3] means:
            #   a00' * x0 + a01' * x1 + a10' * x2 + a11' * x3
            #
            # where a00' = u_top[0]*A00[0] + u_top[1]*A01[0] + u_top[2]*A10[0] + u_top[3]*A11[0]
            # etc.
            
            # Let's build the U row for this scalar product.
            # U row has 16 entries (for the 4x4 A matrix, row-major).
            
            # A_comb element (r,c) where r,c in {0,1}:
            # A_comb[r][c] = sum_{block=0}^{3} u_top[block] * A_block[r][c]
            # where block 0=A00, 1=A01, 2=A10, 3=A11
            
            # A00 corresponds to rows 0-1, cols 0-1 of A: indices 0,1,4,5 (row-major 2x2 within 4x4)
            # A01 corresponds to rows 0-1, cols 2-3 of A: indices 2,3,6,7
            # A10 corresponds to rows 2-3, cols 0-1 of A: indices 8,9,12,13
            # A11 corresponds to rows 2-3, cols 2-3 of A: indices 10,11,14,15
            
            # Block base indices (row-major position in 4x4 for the top-left of each 2x2 block):
            block_bases = [0, 2, 8, 10]  # A00, A01, A10, A11
            # Within a 2x2 block, element (r,c) is at offset r*2+c in the 4x4:
            # For block at base b: A00[0,0]=A[b], A00[0,1]=A[b+1], A00[1,0]=A[b+4], A00[1,1]=A[b+5]
            
            # For A_comb element (r,c):
            # A_comb[r][c] = sum_{b=0}^{3} u_top[b] * A[block_base[b] + r*2 + c]
            
            # The inner u_inner selects: sum_{r=0}^{1} sum_{c=0}^{1} u_inner[r*2+c] * A_comb[r][c]
            # = sum_{r,c} u_inner[r*2+c] * sum_{b} u_top[b] * A[block_base[b] + r*2 + c]
            # = sum_{b} sum_{r,c} u_inner[r*2+c] * u_top[b] * A[block_base[b] + r*2 + c]
            
            # So U[idx][A_index] = sum over (b, r, c) where A_index = block_base[b] + r*2 + c
            # of u_inner[r*2+c] * u_top[b]
            
            U_row = [0] * 16
            for b in range(4):
                if u_top[b] == 0:
                    continue
                for r in range(2):
                    for c in range(2):
                        ai = block_bases[b] + r * 2 + c
                        U_row[ai] += u_inner[r * 2 + c] * u_top[b]
            
            # Similarly for V (B matrix)
            V_row = [0] * 16
            for b in range(4):
                if v_top[b] == 0:
                    continue
                for r in range(2):
                    for c in range(2):
                        bi = block_bases[b] + r * 2 + c
                        V_row[bi] += v_inner[r * 2 + c] * v_top[b]
            
            # For W: the result of the inner Strassen product goes into the 2x2 result matrix
            # w_inner selects which elements of the 2x2 result are affected.
            # The 2x2 result is then placed into C according to w_top.
            #
            # w_inner = [x0, x1, x2, x3] means the 2x2 result has contributions:
            #   result[0][0] gets x0 * (scalar product)
            #   result[0][1] gets x1 * (scalar product)
            #   result[1][0] gets x2 * (scalar product)
            #   result[1][1] gets x3 * (scalar product)
            #
            # w_top tells us which C blocks get this 2x2 result:
            # w_top = [w0, w1, w2, w3] where w0=C00, w1=C01, w2=C10, w3=C11
            #
            # So for each block b where w_top[b] != 0:
            #   C_block_b[r][c] += w_inner[r*2+c] * w_top[b] * (scalar product)
            #
            # C_block_b[r][c] corresponds to C element at:
            # C00: rows 0-1, cols 0-1 -> C_index = block_bases[b] + r*2 + c (same as A/B)
            # But wait, the output convention is W[k*n+i, r] for C[i,k]
            # So C[i,k] is at position k*n+i in the W matrix.
            # In row-major, C[i][k] is at i*4+k.
            # k*n+i = k*4+i.
            # So row-major index i*4+k vs W index k*4+i. These are different!
            
            # Let me reconsider. The output C is 4x4. C[i][k] for i,k in 0..3.
            # In the W matrix, the row index is k*n+i = k*4+i.
            # So W row k*4+i corresponds to C[i][k].
            
            # C_block_b corresponds to a 2x2 block of C.
            # C00: C[0][0], C[0][1], C[1][0], C[1][1]
            #   W indices: k*4+i: C[0][0]->0*4+0=0, C[0][1]->1*4+0=4, C[1][0]->0*4+1=1, C[1][1]->1*4+1=5
            #   So C00 elements are at W rows 0, 4, 1, 5
            # C01: C[0][2], C[0][3], C[1][2], C[1][3]
            #   W indices: C[0][2]->2*4+0=8, C[0][3]->3*4+0=12, C[1][2]->2*4+1=9, C[1][3]->3*4+1=13
            # C10: C[2][0], C[2][1], C[3][0], C[3][1]
            #   W indices: C[2][0]->0*4+2=2, C[2][1]->1*4+2=6, C[3][0]->0*4+3=3, C[3][1]->1*4+3=7
            # C11: C[2][2], C[2][3], C[3][2], C[3][3]
            #   W indices: C[2][2]->2*4+2=10, C[2][3]->3*4+2=14, C[3][2]->2*4+3=11, C[3][3]->3*4+3=15
            
            # Block bases for C in W-index (k*4+i) format:
            # For C_block_b, element (r,c) where r is row within block, c is col within block:
            # Block 0 (C00): rows 0-1, cols 0-1. C[i][k] where i in {0,1}, k in {0,1}
            #   W index = k*4+i
            #   (r,c) -> i=r, k=c -> W index = c*4+r
            # Block 1 (C01): rows 0-1, cols 2-3. i in {0,1}, k in {2,3}
            #   (r,c) -> i=r, k=c+2 -> W index = (c+2)*4+r
            # Block 2 (C10): rows 2-3, cols 0-1. i in {2,3}, k in {0,1}
            #   (r,c) -> i=r+2, k=c -> W index = c*4+(r+2)
            # Block 3 (C11): rows 2-3, cols 2-3. i in {2,3}, k in {2,3}
            #   (r,c) -> i=r+2, k=c+2 -> W index = (c+2)*4+(r+2)
            
            W_row = [0] * 16
            for b in range(4):
                if w_top[b] == 0:
                    continue
                for r in range(2):
                    for c in range(2):
                        # Determine W index for C_block_b[r][c]
                        if b == 0:  # C00
                            wi = c * 4 + r
                        elif b == 1:  # C01
                            wi = (c + 2) * 4 + r
                        elif b == 2:  # C10
                            wi = c * 4 + (r + 2)
                        else:  # C11
                            wi = (c + 2) * 4 + (r + 2)
                        W_row[wi] += w_inner[r * 2 + c] * w_top[b]
            
            # Assign to matrices
            for j in range(16):
                if U_row[j] != 0:
                    U[j][idx] = U_row[j]
                if V_row[j] != 0:
                    V[j][idx] = V_row[j]
                if W_row[j] != 0:
                    W[j][idx] = W_row[j]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
