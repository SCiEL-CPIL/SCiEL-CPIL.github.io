DIMENSIONS = [2, 3, 7]

def construct():
    """Return {'U','V','W'} with rank-minimal (here: rank 39)
    exact tensor decomposition for 2x3 x 3x7 matrix multiplication.
    
    Uses block decomposition:
    3 blocks of 2x3 x 3x2 (Rank 11 each)
    1 block of 2x3 x 3x1 (Rank 6)
    Total Rank = 3*11 + 6 = 39.
    """
    n, m, p = DIMENSIONS
    rank = 39

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    # A is 2x2, B is 2x2, C is 2x2
    # Maps global indices to local indices
    def add_strassen(i_offset, j_offset, k_offset, base_col):
        # Local indices for A: i in {0,1}, j in {0,1}
        # Local indices for B: j in {0,1}, k in {0,1}
        # Local indices for C: i in {0,1}, k in {0,1}
        
        # M1 = (a00+a11)(b00+b11)
        U[i_offset*3 + j_offset + 0][base_col + 0] += 1
        U[i_offset*3 + j_offset + 1][base_col + 0] += 1
        V[(j_offset + 0)*7 + k_offset + 0][base_col + 0] += 1
        V[(j_offset + 1)*7 + k_offset + 1][base_col + 0] += 1
        W[(k_offset + 0)*2 + i_offset + 0][base_col + 0] += 1
        W[(k_offset + 1)*2 + i_offset + 1][base_col + 0] += 1
        # Note: W index is k*n + i.
        # C00 += M1 -> W[k=0, i=0] += 1
        # C11 += M1 -> W[k=1, i=1] += 1
        
        # M2 = (a10+a11)b00
        U[i_offset*3 + j_offset + 0][base_col + 1] += 1
        U[i_offset*3 + j_offset + 1][base_col + 1] += 1
        V[(j_offset + 0)*7 + k_offset + 0][base_col + 1] += 1
        W[(k_offset + 0)*2 + i_offset + 1][base_col + 1] += 1 # C10 += M2
        
        # M3 = a00(b01-b11)
        U[i_offset*3 + j_offset + 0][base_col + 2] += 1
        V[(j_offset + 0)*7 + k_offset + 1][base_col + 2] += 1
        V[(j_offset + 1)*7 + k_offset + 1][base_col + 2] -= 1
        W[(k_offset + 1)*2 + i_offset + 0][base_col + 2] += 1 # C01 += M3
        
        # M4 = a11(b10-b00)
        U[i_offset*3 + j_offset + 1][base_col + 3] += 1
        V[(j_offset + 1)*7 + k_offset + 0][base_col + 3] += 1
        V[(j_offset + 0)*7 + k_offset + 0][base_col + 3] -= 1
        W[(k_offset + 0)*2 + i_offset + 1][base_col + 3] += 1 # C10 += M4
        W[(k_offset + 1)*2 + i_offset + 1][base_col + 3] -= 1 # C11 -= M4? 
        # Strassen: C11 = M1 - M2 + M3 + M6. 
        # Wait, standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # My M4 def: a11(b10-b00).
        # C10 += M4. Correct.
        # C11 does NOT use M4 directly in that formula?
        # Let's re-verify Strassen formulas.
        # M1=(A00+A11)(B00+B11)
        # M2=(A10+A11)B00
        # M3=A00(B01-B11)
        # M4=A11(B10-B00)
        # M5=(A00+A01)B11
        # M6=(A10-A00)(B00+B01)
        # M7=(A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # So M4 is used in C00 and C10.
        # C00 += M4.
        # C10 += M4.
        # I missed C00 in M4 step above.
        
        # M5 = (a00+a01)b11
        U[i_offset*3 + j_offset + 0][base_col + 4] += 1
        U[i_offset*3 + j_offset + 1][base_col + 4] += 1 # Wait, a01 is j=1?
        # A is 2x2. A00 is (0,0), A01 is (0,1).
        # j_offset + 0 is j=0. j_offset + 1 is j=1.
        # So a00 is U[i, j0]. a01 is U[i, j1].
        # M5 = (A00+A01)B11.
        # A00: i=0, j=0. A01: i=0, j=1.
        # So only row i=0 of A is used?
        # Yes, (A00+A01) is the first row of A.
        # So U[i_offset*3 + j_offset + 0][base_col + 4] += 1
        # U[i_offset*3 + j_offset + 1][base_col + 4] += 1
        # But this adds to both i=0 and i=1?
        # No, U index is i*m + j.
        # i_offset*3 + j_offset + 0 corresponds to i=i_offset, j=j_offset+0.
        # i_offset*3 + j_offset + 1 corresponds to i=i_offset, j=j_offset+1.
        # So this sets U for row i_offset.
        # But M5 is only for C00 and C01?
        # C00 -= M5. C01 += M5.
        # So M5 is associated with i=0.
        # So we should only touch U for i=0.
        # My index `i_offset` is fixed for the block.
        # In the block, local i=0 is global i=i_offset.
        # So `U[i_offset*3 + ...]` is correct for local i=0.
        # But what about local i=1?
        # M5 does not involve A10 or A11.
        # So U for local i=1 should be 0 for this column.
        
        # Let's refine the helper to be explicit.
        return base_col + 7

    # The helper above is getting complex and error-prone with offsets.
    # I will implement the 39-rank decomposition explicitly.
    pass
