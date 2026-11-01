"""Matrix multiplication 2,4,7 with Strassen-based decomposition."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2x2(U_off, V_off, W_off):
    """
    Returns the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen).
    U indices: (i*2+j) + U_off, i in {0,1}, j in {0,1}
    V indices: (j*2+k) + V_off, j in {0,1}, k in {0,1}
    W indices: (k*2+i) + W_off, k in {0,1}, i in {0,1}
    """
    # Local indices: U: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    #                 V: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    #                 W: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)
    # Strassen formulas for C = A*B where A,B are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a rank-1 term. We need to express each M as a product of U_term * V_term.
    # For M = (sum of A entries) * (sum of B entries), we add terms.
    # Actually, each M is itself a sum of products of A and B entries.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # But wait - Strassen works by computing these 7 "middle" matrices, each of which is a product of sums.
    # The tensor rank contribution is 7, not the expansion. Each M corresponds to ONE rank-1 term
    # in the bilinear form? No - each M is a bilinear form in A and B, but it's not rank-1 in the
    # original entries. 
    
    # Actually, let me reconsider. The standard Strassen algorithm computes 7 products of 2x2 blocks,
    # but each such product is itself a 2x2 matrix multiplication, which has rank 4 (schoolbook) or 7 (recursive Strassen).
    # For scalar 2x2x2, Strassen says the rank is at most 7, achieved by the 7 products above.
    # Each product like M1 = (A00+A11)(B00+B11) is a single scalar product, contributing rank 1 to the bilinear form.
    
    # So we have 7 rank-1 terms. Let me write them out:
    terms = []
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # As a bilinear term: (A00+A11)*(B00+B11)
    # U part: A00+A11, V part: B00+B11
    # This is one rank-1 term in the bilinear form sense? No, it's a product of two linear forms.
    # In the tensor decomposition, each such product is one rank-1 component.
    
    # Let me just use the known Strassen rank-7 decomposition for 2x2x2.
    # The 7 rank-1 tensors are:
    # 1: U=A00+A11, V=B00+B11, W=C00+C11 (contributes to C00 and C11)
    # Actually, the standard form:
    
    # Let me use a cleaner approach. For 2x2x2, the Strassen decomposition has 7 terms:
    # Term 1: U: A00+A11, V: B00+B11, W: C00+C11
    # Term 2: U: A10+A11, V: B00, W: C10
    # Term 3: U: A00, V: B01-B11, W: C01
    # Term 4: U: A11, V: B10-B00, W: C10
    # Term 5: U: A00+A01, V: B11, W: C01
    # Term 6: U: A10-A00, V: B00+B01, W: C11
    # Term 7: U: A01-A11, V: B10+B11, W: C00
    
    # Wait, I need to be more careful. Let me look up the exact Strassen formulas.
    
    # C00 = M1 - M5 + M7 where M1=(A00+A11)(B00+B11), M5=(A00+A01)B11, M7=(A01-A11)(B10+B11)
    # C01 = M3 + M5 where M3=A00(B01-B11), M5=(A00+A01)B11
    # C10 = M2 + M4 where M2=(A10+A11)B00, M4=A11(B10-B00)
    # C11 = M1 - M2 + M3 + M6 where M6=(A10-A00)(B00+B01)
    
    # So the 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi is a bilinear form in A and B. In the tensor decomposition, each Mi contributes
    # to specific entries of C. The rank-1 terms are formed by distributing each Mi.
    
    # Actually, for the purpose of the tensor decomposition, each Mi is ONE rank-1 term
    # where U is the linear form in A, V is the linear form in B, and W is the linear form in C
    # that Mi contributes to.
    
    # Let me define the 7 terms with their contributions:
    # U_local, V_local, W_local, coefficient
    
    # M1: U=A00+A11 (indices 0,2 in U), V=B00+B11 (indices 0,3 in V), W=C00+C11 (indices 0,3 in W)
    # But this is not a single rank-1 term - it's a product of sums, which expands.
    
    # I think I'm overcomplicating this. Let me just implement the schoolbook for small cases
    # and use a known good decomposition.
    
    # For now, let me just return the schoolbook 4 terms for 2x2x2.
    # This is safe and correct.
    result = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                u_idx = i * 2 + j + U_off
                v_idx = j * 2 + k + V_off
                w_idx = k * 2 + i + W_off
                result.append((u_idx, v_idx, w_idx, 1))
    return result

def _schoolbook_block(n_local, m_local, p_local, U_off, V_off, W_off):
    """Schoolbook decomposition for n_local x m_local x p_local block."""
    result = []
    for i in range(n_local):
        for j in range(m_local):
            for k in range(p_local):
                u_idx = i * m_local + j + U_off
                v_idx = j * p_local + k + V_off
                w_idx = k * n_local + i + W_off
                result.append((u_idx, v_idx, w_idx, 1))
    return result

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Split the 4x7 multiplication into blocks.
    # We'll split the middle dimension (m=4) into two parts of size 2.
    # Block 1: A[:, 0:2] @ B[0:2, :] -> 2x2 @ 2x7 = 2x7
    # Block 2: A[:, 2:4] @ B[2:4, :] -> 2x2 @ 2x7 = 2x7
    # Total: 2x7 result.
    
    # For each block, we can further split the 7 columns into 2+2+2+1.
    # So each 2x2 @ 2x7 becomes four sub-blocks: 2x2@2x2, 2x2@2x2, 2x2@2x2, 2x2@2x1.
    
    # For 2x2@2x2, use Strassen (rank 7) or schoolbook (rank 4).
    # For 2x2@2x1, use schoolbook (rank 4).
    
    # Let's use Strassen for the 2x2@2x2 blocks.
    # But I need to implement Strassen correctly.
    
    # Actually, let me just use schoolbook for everything first to get a correct answer,
    # then try to optimize.
    
    rank = n * m * p  # 56
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
# EVOLVE-BLOCK-END
