import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We are targeting rank 66
    # But let's just build it dynamically to be safe.
    # Actually, hardcoding the structure is safer for exactness.
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        # u_idx is (i, j)
        # v_idx is (j, k)
        # w_idx is (k, i)
        i, j = u_idx
        j2, k = v_idx
        k2, i2 = w_idx
        # Check consistency
        # assert i == i2 and j == j2 and k == k2
        U[i * m + j][col] += val
        V[j * p + k][col] += val
        W[k * n + i][col] += val
        
    def add_strassen(i0, j0, k0):
        # A is 2x2 at (i0, j0)
        # B is 2x2 at (j0, k0)
        # C is 2x2 at (i0, k0)
        nonlocal col
        # M1 = (A11+A22)(B11+B22) -> C11, C22
        # M2 = (A21+A22)B11 -> C21
        # M3 = A11(B12-B22) -> C12
        # M4 = A22(B21-B11) -> C21
        # M5 = (A11+A12)B22 -> C11, C12
        # M6 = (A21-A11)(B11+B12) -> C22
        # M7 = (A12-A22)(B21+B22) -> C11, C22
        
        # Term 1
        U[(i0, j0)][col] += 1
        U[(i0+1, j0+1)][col] += 1
        V[(j0, k0)][col] += 1
        V[(j0+1, k0+1)][col] += 1
        W[(k0, i0)][col] += 1
        W[(k0+1, i0+1)][col] += 1
        col += 1
        
        # Term 2
        U[(i0+1, j0)][col] += 1
        U[(i0+1, j0+1)][col] += 1
        V[(j0, k0)][col] += 1
        W[(k0, i0+1)][col] += 1
        col += 1
        
        # Term 3
        U[(i0, j0)][col] += 1
        V[(j0, k0+1)][col] += 1
        V[(j0+1, k0+1)][col] += -1
        W[(k0+1, i0)][col] += 1
        col += 1
        
        # Term 4
        U[(i0+1, j0+1)][col] += 1
        V[(j0+1, k0)][col] += 1
        V[(j0, k0)][col] += -1
        W[(k0, i0+1)][col] += 1
        col += 1
        
        # Term 5
        U[(i0, j0)][col] += 1
        U[(i0, j0+1)][col] += 1
        V[(j0+1, k0+1)][col] += 1
        W[(k0, i0)][col] += -1
        W[(k0+1, i0)][col] += 1
        col += 1
        
        # Term 6
        U[(i0+1, j0)][col] += 1
        U[(i0, j0)][col] += -1
        V[(j0, k0)][col] += 1
        V[(j0, k0+1)][col] += 1
        W[(k0+1, i0+1)][col] += 1
        col += 1
        
        # Term 7
        U[(i0, j0+1)][col] += 1
        U[(i0+1, j0+1)][col] += -1
        V[(j0+1, k0)][col] += 1
        V[(j0+1, k0+1)][col] += 1
        W[(k0, i0)][col] += 1
        W[(k0+1, i0+1)][col] += -1
        col += 1

    def add_block_333(i0, j0, k0):
        # A 3x3 at (i0, j0), B 3x3 at (j0, k0), C 3x3 at (i0, k0)
        nonlocal col
        # Split A into 2x2, 2x1, 1x2, 1x1
        # Split B into 2x2, 2x1, 1x2, 1x1
        # C = A22 B22 + A21 B12 + A12 B21 + A11 B11
        
        # 1. A22 (2x2) * B22 (2x2) -> Strassen
        add_strassen(i0, j0, k0)
        
        # 2. A21 (2x1) * B12 (1x2)
        # A21 is rows i0, i0+1, col j0+2
        # B12 is row j0+2, cols k0, k0+1
        # C contribution to rows i0, i0+1, cols k0, k0+1
        # Terms:
        # (i0, j0+2) * (j0+2, k0) -> (k0, i0)
        U[(i0, j0+2)][col] += 1
        V[(j0+2, k0)][col] += 1
        W[(k0, i0)][col] += 1
        col += 1
        
        U[(i0, j0+2)][col] += 1
        V[(j0+2, k0+1)][col] += 1
        W[(k0+1, i0)][col] += 1
        col += 1
        
        U[(i0+1, j0+2)][col] += 1
        V[(j0+2, k0)][col] += 1
        W[(k0, i0+1)][col] += 1
        col += 1
        
        U[(i0+1, j0+2)][col] += 1
        V[(j0+2, k0+1)][col] += 1
        W[(k0+1, i0+1)][col] += 1
        col += 1
        
        # 3. A12 (1x2) * B21 (2x1)
        # A12 is row i0+2, cols j0, j0+1
        # B21 is rows j0, j0+1, col k0+2
        # C contribution to row i0+2, cols k0+2
        # Terms:
        # (i0+2, j0) * (j0, k0+2) -> (k0+2, i0+2)
        U[(i0+2, j0)][col] += 1
        V[(j0, k0+2)][col] += 1
        W[(k0+2, i0+2)][col] += 1
        col += 1
        
        U[(i0+2, j0+1)][col] += 1
        V[(j0+1, k0+2)][col] += 1
        W[(k0+2, i0+2)][col] += 1
        col += 1
        
        # Wait, A12 is 1x2, B21 is 2x1.
        # C is 1x1.
        # C = A12 B21.
        # Only 1 element in C?
        # No, A12 is 1x2, B21 is 2x1. Product is 1x1.
        # So only W[(k0+2, i0+2)] is involved.
        # But there are 2 terms in the sum: A_{i0+2, j0} B_{j0, k0+2} + A_{i0+2, j0+1} B_{j0+1, k0+2}.
        # So 2 terms.
        # My code above has 2 terms. Correct.
        
        # 4. A11 (1x1) * B11 (1x1)
        # A11 is (i0+2, j0+2)
        # B11 is (j0+2, k0+2)
        # C11 is (k0+2, i0+2)
        U[(i0+2, j0+2)][col] += 1
        V[(j0+2, k0+2)][col] += 1
        W[(k0+2, i0+2)][col] += 1
        col += 1

    def add_block_223(i0, j0, k0):
        # A 2x2 at (i0, j0), B 2x3 at (j0, k0)
        # Split B into 2x1 and 2x2
        nonlocal col
        # 1. A * B_col1 (2x1)
        # Rank 4
        # (i0, j0) * (j0, k0) -> (k0, i0)
        U[(i0, j0)][col] += 1
        V[(j0, k0)][col] += 1
        W[(k0, i0)][col] += 1
        col += 1
        U[(i0, j0+1)][col] += 1
        V[(j0+1, k0)][col] += 1
        W[(k0, i0)][col] += 1
        col += 1
        U[(i0+1, j0)][col] += 1
        V[(j0, k0)][col] += 1
        W[(k0, i0+1)][col] += 1
        col += 1
        U[(i0+1, j0+1)][col] += 1
        V[(j0+1, k0)][col] += 1
        W[(k0, i0+1)][col] += 1
        col += 1
        
        # 2. A * B_col23 (2x2)
        # Strassen
        add_strassen(i0, j0, k0+1)

    def add_block_126(i0, j0, k0):
        # A 1x2 at (i0, j0), B 2x6 at (j0, k0)
        # Rank 12
        nonlocal col
        for jj in range(2):
            for kk in range(6):
                U[(i0, j0+jj)][col] += 1
                V[(j0+jj, k0+kk)][col] += 1
                W[(k0+kk, i0)][col] += 1
                col += 1

    # Construct the full decomposition
    # Block 1: 3x2x6
    # A is 3x2 (i 0..2, j 0..1)
    # B is 2x6 (j 0..1, k 0..5)
    # Split A into 2x2 (i 0..1, j 0..1) and 1x2 (i 2, j 0..1)
    
    # 1.1: 2x2x6
    # Split B into 2x3 (k 0..2) and 2x3 (k 3..5)
    add_block_223(0, 0, 0)
    add_block_223(0, 0, 3)
    
    # 1.2: 1x2x6
    add_block_126(2, 0, 0)
    
    # Block 2: 3x3x6
    # A is 3x3 (i 0..2, j 2..4)
    # B is 3x6 (j 2..4, k 0..5)
    # Split B into 3x3 (k 0..2) and 3x3 (k 3..5)
    add_block_333(0, 2, 0)
    add_block_333(0, 2, 3)
    
    return {"U": U, "V": V, "W": W}
