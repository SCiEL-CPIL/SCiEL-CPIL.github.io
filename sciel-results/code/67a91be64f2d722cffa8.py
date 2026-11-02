"""Matrix multiplication tensor decomposition for 2x5 by 5x5."""
DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 46
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal current_rank
        # A block: rows i0, i0+1; cols j0, j0+1
        # B block: rows j0, j0+1; cols k0, k0+1
        # C block: rows k0, k0+1; cols i0, i0+1 (Note: C is n x p, so i is col, k is row in W index?)
        # W index is k*n + i. So C[i, k].
        # A[i, j], B[j, k].
        
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i
        
        # M1 = (A11+A22)(B11+B22)
        r = current_rank; current_rank += 1
        # A11, A22
        U[i0*m + j0][r] += 1
        U[(i0+1)*m + (j0+1)][r] += 1
        # B11, B22
        V[j0*p + k0][r] += 1
        V[(j0+1)*p + (k0+1)][r] += 1
        # C11, C22
        W[k0*n + i0][r] += 1
        W[(k0+1)*n + (i0+1)][r] += 1
        
        # M2 = (A21+A22)B11
        r = current_rank; current_rank += 1
        # A21, A22
        U[(i0+1)*m + j0][r] += 1
        U[(i0+1)*m + (j0+1)][r] += 1
        # B11
        V[j0*p + k0][r] += 1
        # C21
        W[k0*n + (i0+1)][r] += 1
        
        # M3 = A11(B12-B22)
        r = current_rank; current_rank += 1
        # A11
        U[i0*m + j0][r] += 1
        # B12, -B22
        V[j0*p + (k0+1)][r] += 1
        V[(j0+1)*p + (k0+1)][r] -= 1
        # C12
        W[(k0+1)*n + i0][r] += 1
        
        # M4 = A22(B21-B11)
        r = current_rank; current_rank += 1
        # A22
        U[(i0+1)*m + (j0+1)][r] += 1
        # B21, -B11
        V[(j0+1)*p + k0][r] += 1
        V[j0*p + k0][r] -= 1
        # C21
        W[k0*n + (i0+1)][r] += 1
        
        # M5 = (A11+A12)B22
        r = current_rank; current_rank += 1
        # A11, A12
        U[i0*m + j0][r] += 1
        U[i0*m + (j0+1)][r] += 1
        # B22
        V[(j0+1)*p + (k0+1)][r] += 1
        # C11, C12
        W[k0*n + i0][r] -= 1
        W[(k0+1)*n + i0][r] += 1
        
        # M6 = (A21-A11)(B11+B12)
        r = current_rank; current_rank += 1
        # A21, -A11
        U[(i0+1)*m + j0][r] += 1
        U[i0*m + j0][r] -= 1
        # B11, B12
        V[j0*p + k0][r] += 1
        V[j0*p + (k0+1)][r] += 1
        # C22
        W[(k0+1)*n + (i0+1)][r] += 1
        
        # M7 = (A12-A22)(B21+B22)
        r = current_rank; current_rank += 1
        # A12, -A22
        U[i0*m + (j0+1)][r] += 1
        U[(i0+1)*m + (j0+1)][r] -= 1
        # B21, B22
        V[(j0+1)*p + k0][r] += 1
        V[(j0+1)*p + (k0+1)][r] += 1
        # C11, C22
        W[k0*n + i0][r] += 1
        W[(k0+1)*n + (i0+1)][r] += 1
        
        # Corrections for C11, C21, C22 based on Strassen formula
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We added M1 to C11, C22.
        # We added M2 to C21.
        # We added M3 to C12.
        # We added M4 to C21.
        # We added -M5 to C11, +M5 to C12.
        # We added M6 to C22.
        # We added M7 to C11, C22.
        
        # Let's verify C11: M1 (from M1) + M4 (from M4) - M5 (from M5) + M7 (from M7). Correct.
        # C12: M3 (from M3) + M5 (from M5). Correct.
        # C21: M2 (from M2) + M4 (from M4). Correct.
        # C22: M1 (from M1) - M2 (from M2) + M3 (from M3) + M6 (from M6).
        # Wait, M2 was added to C21, not C22.
        # In M2 definition, I added to C21.
        # In M2 formula for C22, we need -M2.
        # I did NOT add M2 to C22 in the M2 step.
        # So I need to subtract M2 from C22?
        # Let's re-read Strassen:
        # C22 = M1 - M2 + M3 + M6.
        # My M1 step added +1 to C22.
        # My M2 step added +1 to C21. (Correct for C21).
        # My M3 step added +1 to C12. (Correct for C12).
        # My M6 step added +1 to C22.
        # So C22 currently has M1 + M6.
        # It needs M1 - M2 + M3 + M6.
        # So I am missing -M2 and +M3.
        
        # Let's fix the W assignments.
        
        # Reset W for this block? No, we are adding.
        # I will redo the W assignments carefully.
        
        # Actually, it is safer to define the U, V, W for each M explicitly.
        
        pass # The above logic is getting messy. I will rewrite the helper.

    # Rewrite helper to be precise
    
    def add_strassen_precise(i0, j0, k0):
        nonlocal current_rank
        
        # M1 = (A11+A22)(B11+B22)
        r = current_rank; current_rank += 1
        U[i0*m + j0][r] += 1
        U[(i0+1)*m + (j0+1)][r] += 1
        V[j0*p + k0][r] += 1
        V[(j0+1)*p + (k0+1)][r] += 1
        W[k0*n + i0][r] += 1      # C11
        W[(k0+1)*n + (i0+1)][r] += 1 # C22

        # M2 = (A21+A22)B11
        r = current_rank; current_rank += 1
        U[(i0+1)*m + j0][r] += 1
        U[(i0+1)*m + (j0+1)][r] += 1
        V[j0*p + k0][r] += 1
        W[k0*n + (i0+1)][r] += 1   # C21
        W[(k0+1)*n + (i0+1)][r] -= 1 # C22 (part of -M2)

        # M3 = A11(B12-B22)
        r = current_rank; current_rank += 1
        U[i0*m + j0][r] += 1
        V[j0*p + (k0+1)][r] += 1
        V[(j0+1)*p + (k0+1)][r] -= 1
        W[(k0+1)*n + i0][r] += 1   # C12
        W[(k0+1)*n + (i0+1)][r] += 1 # C22 (part of +M3)

        # M4 = A22(B21-B11)
        r = current_rank; current_rank += 1
        U[(i0+1)*m + (j0+1)][r] += 1
        V[(j0+1)*p + k0][r] += 1
        V[j0*p + k0][r] -= 1
        W[k0*n + (i0+1)][r] += 1   # C21
        W[k0*n + i0][r] += 1       # C11

        # M5 = (A11+A12)B22
        r = current_rank; current_rank += 1
        U[i0*m + j0][r] += 1
        U[i0*m + (j0+1)][r] += 1
        V[(j0+1)*p + (k0+1)][r] += 1
        W[k0*n + i0][r] -= 1       # C11 (part of -M5)
        W[(k0+1)*n + i0][r] += 1   # C12

        # M6 = (A21-A11)(B11+B12)
        r = current_rank; current_rank += 1
        U[(i0+1)*m + j0][r] += 1
        U[i0*m + j0][r] -= 1
        V[j0*p + k0][r] += 1
        V[j0*p + (k0+1)][r] += 1
        W[(k0+1)*n + (i0+1)][r] += 1 # C22

        # M7 = (A12-A22)(B21+B22)
        r = current_rank; current_rank += 1
        U[i0*m + (j0+1)][r] += 1
        U[(i0+1)*m + (j0+1)][r] -= 1
        V[(j0+1)*p + k0][r] += 1
        V[(j0+1)*p + (k0+1)][r] += 1
        W[k0*n + i0][r] += 1       # C11
        W[(k0+1)*n + (i0+1)][r] += 1 # C22

    def add_simple_2x1x2(i0, j0, k0):
        # A (2x1) at col j0. B (1x2) at row j0, cols k0, k0+1.
        # C (2x2) at cols i0, i0+1 and rows k0, k0+1.
        nonlocal current_rank
        # M1 = A1 B1
        r = current_rank; current_rank += 1
        U[i0*m + j0][r] += 1
        V[j0*p + k0][r] += 1
        W[k0*n + i0][r] += 1
        
        # M2 = A1 B2
        r = current_rank; current_rank += 1
        U[i0*m + j0][r] += 1
        V[j0*p + (k0+1)][r] += 1
        W[(k0+1)*n + i0][r] += 1
        
        # M3 = A2 B1
        r = current_rank; current_rank += 1
        U[(i0+1)*m + j0][r] += 1
        V[j0*p + k0][r] += 1
        W[k0*n + (i0+1)][r] += 1
        
        # M4 = A2 B2
        r = current_rank; current_rank += 1
        U[(i0+1)*m + j0][r] += 1
        V[j0*p + (k0+1)][r] += 1
        W[(k0+1)*n + (i0+1)][r] += 1

    def add_simple_2x2x1(i0, j0, k0):
        # A (2x2) at cols j0, j0+1. B (2x1) at rows j0, j0+1, col k0.
        # C (2x1) at col i0, i0+1 and row k0.
        nonlocal current_rank
        # M1 = A11 B1
        r = current_rank; current_rank += 1
        U[i0*m + j0][r] += 1
        V[j0*p + k0][r] += 1
        W[k0*n + i0][r] += 1
        
        # M2 = A12 B2
        r = current_rank; current_rank += 1
        U[i0*m + (j0+1)][r] += 1
        V[(j0+1)*p + k0][r] += 1
        W[k0*n + i0][r] += 1
        
        # M3 = A21 B1
        r = current_rank; current_rank += 1
        U[(i0+1)*m + j0][r] += 1
        V[j0*p + k0][r] += 1
        W[k0*n + (i0+1)][r] += 1
        
        # M4 = A22 B2
        r = current_rank; current_rank += 1
        U[(i0+1)*m + (j0+1)][r] += 1
        V[(j0+1)*p + k0][r] += 1
        W[k0*n + (i0+1)][r] += 1

    def add_simple_2x1x1(i0, j0, k0):
        # A (2x1) at col j0. B (1x1) at row j0, col k0.
        # C (2x1) at col i0, i0+1 and row k0.
        nonlocal current_rank
        # M1 = A1 B1
        r = current_rank; current_rank += 1
        U[i0*m + j0][r] += 1
        V[j0*p + k0][r] += 1
        W[k0*n + i0][r] += 1
        
        # M2 = A2 B1
        r = current_rank; current_rank += 1
        U[(i0+1)*m + j0][r] += 1
        V[j0*p + k0][r] += 1
        W[k0*n + (i0+1)][r] += 1

    # Block v=0 (k=0,1)
    # A(0,1) x B(0,1)
    add_strassen_precise(0, 0, 0) # i0=0, j0=0, k0=0
    # A(2,3) x B(2,3)
    add_strassen_precise(0, 2, 0) # i0=0, j0=2, k0=0
    # A(4) x B(4)
    add_simple_2x1x2(0, 4, 0)     # i0=0, j0=4, k0=0

    # Block v=1 (k=2,3)
    # A(0,1) x B(0,2)
    add_strassen_precise(0, 0, 2) # i0=0, j0=0, k0=2
    # A(2,3) x B(2,3)
    add_strassen_precise(0, 2, 2) # i0=0, j0=2, k0=2
    # A(4) x B(4)
    add_simple_2x1x2(0, 4, 2)     # i0=0, j0=4, k0=2

    # Block v=2 (k=4)
    # A(0,1) x B(0,4)
    add_simple_2x2x1(0, 0, 4)     # i0=0, j0=0, k0=4
    # A(2,3) x B(2,4)
    add_simple_2x2x1(0, 2, 4)     # i0=0, j0=2, k0=4
    # A(4) x B(4)
    add_simple_2x1x1(0, 4, 4)     # i0=0, j0=4, k0=4

    return {"U": U, "V": V, "W": W}
