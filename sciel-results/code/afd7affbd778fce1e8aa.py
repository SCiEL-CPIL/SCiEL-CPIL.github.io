"""Rewritten program: rank-36 decomposition for 2x4x5 matrix multiplication."""

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 5
    # We treat the 2x4x5 multiplication as two independent 2x2x5 multiplications
    # (split the middle dimension into two blocks of size 2).
    # Each 2x2x5 is implemented as:
    #   two 2x2x2 blocks (Strassen, rank 7 each) + one 2x2x1 block (schoolbook, rank 4)
    # Total per 2x2x5: 7+7+4 = 15... wait, let me recount.
    # Actually 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1, so 7+7+4 = 18.
    # Two such blocks: 2*18 = 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]   # 8 x 36
    V = [[0] * rank for _ in range(m * p)]   # 20 x 36
    W = [[0] * rank for _ in range(p * n)]   # 10 x 36
    
    col = 0
    for block in range(2):  # two 2x2x5 blocks
        # Each block uses columns j = 2*block + j_local for the middle index
        for sub in range(3):  # sub=0,1: Strassen 2x2x2; sub=2: schoolbook 2x2x1
            if sub < 2:
                # Strassen 2x2x2: 7 multiplications
                # For a 2x2x2 block, the middle indices are j = 2*block + j_local (j_local in {0,1})
                # Strassen formula for C = A*B where A is 2x2, B is 2x2:
                # M1 = (A00+A11)(B00+B11)
                # M2 = (A10+A11)B00
                # M3 = A00(B10-B11)
                # M4 = A11(B01-B10)
                # M5 = (A00+A01)B11
                # M6 = (A10-A00)(B00+B10)
                # M7 = (A01-A11)(B01+B11)
                # C00 = M1+M4-M5+M7
                # C01 = M3+M5
                # C10 = M2+M4
                # C11 = M1-M2+M3+M6
                for s in range(7):
                    # Determine which U, V, W entries get non-zero values
                    # We need to map the Strassen intermediates to the global indices.
                    # For block 'block', the middle indices are j = 2*block + {0,1}
                    # The output indices are k*n+i where k is in {0,1} for the 2x2 output rows... 
                    # Wait, p=5, so output has 5 columns. The two 2x2x2 blocks cover columns 0,1 and 2,3.
                    # The 2x2x1 block covers column 4.
                    
                    # Let's define the Strassen contributions for block 'block':
                    # The 2x2 A submatrix rows are i in {0,1}, cols j in {2*block, 2*block+1}
                    # The 2x2 B submatrix rows are j in {2*block, 2*block+1}, cols k in {2*sub, 2*sub+1}
                    # The output C submatrix is 2x2 at rows k in {2*sub, 2*sub+1}, cols i in {0,1}
                    
                    # Strassen M1: (A00+A11)(B00+B11)
                    # A00 is U[i=0, j=2*block], A11 is U[i=1, j=2*block+1]
                    # B00 is V[j=2*block, k=2*sub], B11 is V[j=2*block+1, k=2*sub+1]
                    # Contribution to W: C00 gets +1, C11 gets +1
                    if s == 0:
                        U[0 * m + (2*block)][col] += 1
                        U[1 * m + (2*block+1)][col] += 1
                        V[(2*block) * p + (2*sub)][col] += 1
                        V[(2*block+1) * p + (2*sub+1)][col] += 1
                        W[(2*sub) * n + 0][col] += 1   # C00
                        W[(2*sub+1) * n + 1][col] += 1 # C11
                    # Strassen M2: (A10+A11)B00
                    # A10 is U[i=1, j=2*block], A11 is U[i=1, j=2*block+1]
                    # B00 is V[j=2*block, k=2*sub]
                    # C10 gets +1, C11 gets -1
                    elif s == 1:
                        U[1 * m + (2*block)][col] += 1
                        U[1 * m + (2*block+1)][col] += 1
                        V[(2*block) * p + (2*sub)][col] += 1
                        W[(2*sub) * n + 1][col] += 1   # C10
                        W[(2*sub+1) * n + 1][col] -= 1 # C11
                    # Strassen M3: A00(B10-B11)
                    # A00 is U[i=0, j=2*block]
                    # B10 is V[j=2*block+1, k=2*sub], B11 is V[j=2*block+1, k=2*sub+1]
                    # C01 gets +1, C11 gets +1
                    elif s == 2:
                        U[0 * m + (2*block)][col] += 1
                        V[(2*block+1) * p + (2*sub)][col] += 1
                        V[(2*block+1) * p + (2*sub+1)][col] -= 1
                        W[(2*sub+1) * n + 0][col] += 1 # C01
                        W[(2*sub+1) * n + 1][col] += 1 # C11
                    # Strassen M4: A11(B01-B10)
                    # A11 is U[i=1, j=2*block+1]
                    # B01 is V[j=2*block, k=2*sub+1], B10 is V[j=2*block, k=2*sub]
                    # C00 gets +1, C10 gets +1
                    elif s == 3:
                        U[1 * m + (2*block+1)][col] += 1
                        V[(2*block) * p + (2*sub+1)][col] += 1
                        V[(2*block) * p + (2*sub)][col] -= 1
                        W[(2*sub) * n + 0][col] += 1   # C00
                        W[(2*sub) * n + 1][col] += 1   # C10
                    # Strassen M5: (A00+A01)B11
                    # A00 is U[i=0, j=2*block], A01 is U[i=0, j=2*block+1]
                    # B11 is V[j=2*block+1, k=2*sub+1]
                    # C00 gets -1, C01 gets +1
                    elif s == 4:
                        U[0 * m + (2*block)][col] += 1
                        U[0 * m + (2*block+1)][col] += 1
                        V[(2*block+1) * p + (2*sub+1)][col] += 1
                        W[(2*sub) * n + 0][col] -= 1   # C00
                        W[(2*sub+1) * n + 0][col] += 1 # C01
                    # Strassen M6: (A10-A00)(B00+B10)
                    # A10 is U[i=1, j=2*block], A00 is U[i=0, j=2*block]
                    # B00 is V[j=2*block, k=2*sub], B10 is V[j=2*block+1, k=2*sub]
                    # C11 gets +1
                    elif s == 5:
                        U[1 * m + (2*block)][col] += 1
                        U[0 * m + (2*block)][col] -= 1
                        V[(2*block) * p + (2*sub)][col] += 1
                        V[(2*block+1) * p + (2*sub)][col] += 1
                        W[(2*sub+1) * n + 1][col] += 1 # C11
                    # Strassen M7: (A01-A11)(B01+B11)
                    # A01 is U[i=0, j=2*block+1], A11 is U[i=1, j=2*block+1]
                    # B01 is V[j=2*block, k=2*sub+1], B11 is V[j=2*block+1, k=2*sub+1]
                    # C00 gets +1
                    elif s == 6:
                        U[0 * m + (2*block+1)][col] += 1
                        U[1 * m + (2*block+1)][col] -= 1
                        V[(2*block) * p + (2*sub+1)][col] += 1
                        V[(2*block+1) * p + (2*sub+1)][col] += 1
                        W[(2*sub) * n + 0][col] += 1   # C00
                    
                    col += 1
            else:
                # Schoolbook 2x2x1: 4 multiplications for column 4
                # C[i,4] = sum_j A[i,j] * B[j,4]
                for i in range(2):
                    for j in range(2):
                        U[i * m + (2*block + j)][col] += 1
                        V[(2*block + j) * p + 4][col] += 1
                        W[4 * n + i][col] += 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
