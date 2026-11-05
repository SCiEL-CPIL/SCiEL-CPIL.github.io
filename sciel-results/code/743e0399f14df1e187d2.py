DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    # Split p into p1=4, p2=3
    p1, p2 = 4, 3
    
    # Part 1: 4x4 x 4x4 using Strassen (rank 28)
    # Part 2: 4x4 x 4x3 using schoolbook (rank 48)
    
    rank = 28 + 48
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x rank
    V = [[0] * rank for _ in range(m * p)]  # 28 x rank
    W = [[0] * rank for _ in range(p * n)]  # 28 x rank
    
    col = 0
    # Strassen for 4x4 x 4x4
    # A indices: i*m+j, i,j in 0..3
    # B indices: j*p+k, j in 0..3, k in 0..3 (for p1 part)
    # C indices: k*n+i, k in 0..3, i in 0..3 (for p1 part)
    
    # 2x2 block indices
    # A11: i=0,1 j=0,1 -> A[0..3]
    # A12: i=0,1 j=2,3 -> A[4..7]
    # A21: i=2,3 j=0,1 -> A[8..11]
    # A22: i=2,3 j=2,3 -> A[12..15]
    
    # B11: j=0,1 k=0,1 -> B[0..3]
    # B12: j=0,1 k=2,3 -> B[4..7]
    # B21: j=2,3 k=0,1 -> B[8..11]
    # B22: j=2,3 k=2,3 -> B[12..15]
    
    # C11: k=0,1 i=0,1 -> W[0..3]
    # C12: k=2,3 i=0,1 -> W[4..7]
    # C21: k=0,1 i=2,3 -> W[8..11]
    # C22: k=2,3 i=2,3 -> W[12..15]
    
    # Strassen products, each is 2x2 x 2x2 (rank 4)
    def add_strassen_product(a_offset, b_offset, c_offset, sign=1):
        nonlocal col
        for ai in range(2):
            for aj in range(2):
                for bk in range(2):
                    # a_index = a_offset + ai*2 + aj (within 2x2 block)
                    # But a_offset is the global index of the top-left of the 2x2 block
                    # For A11: offset=0, indices 0,1,2,3
                    # For A12: offset=4, indices 4,5,6,7
                    # etc.
                    a_idx = a_offset + ai * 2 + aj
                    b_idx = b_offset + aj * 2 + bk  # B block: row j, col k
                    # Wait, B is m x p1 = 4x4, B[j,k], j in 0..3, k in 0..3
                    # B11: j=0,1 k=0,1 -> B[0*4+0]=0, B[0*4+1]=1, B[1*4+0]=4, B[1*4+1]=5
                    # So B index = j*4 + k for the 4x4 part
                    # For B11 (j=0,1, k=0,1): indices 0,1,4,5
                    # For a 2x2 block at (j0, k0): B[j0+aj, k0+bk] = (j0+aj)*4 + (k0+bk)
                    # I need to pass the base indices properly.
                    c_idx = c_offset + bk * 2 + ai  # C[k,i], k in 0..3, i in 0..3
                    # C11: k=0,1 i=0,1 -> C[0*4+0]=0, C[0*4+1]=1, C[1*4+0]=4, C[1*4+1]=5
                    # For C block at (k0, i0): C[k0+bk, i0+ai] = (k0+bk)*4 + (i0+ai)
                    U[a_idx][col] += sign
                    V[b_idx][col] += sign
                    W[c_idx][col] += sign
                    col += 1
    
    # This is getting complicated. Let me use a simpler indexing.
    # I'll rebuild from scratch with clear indexing.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Helper: add a 2x2 x 2x2 schoolbook product
    # A block at rows [i0, i0+1], cols [j0, j0+1]
    # B block at rows [j0, j0+1], cols [k0, k0+1]
    # C block at rows [k0, k0+1], cols [i0, i0+1]
    def add_2x2_product(i0, j0, k0, coeff=1):
        nonlocal col
        for ai in range(2):
            for aj in range(2):
                for bk in range(2):
                    a_idx = (i0 + ai) * m + (j0 + aj)
                    b_idx = (j0 + aj) * p + (k0 + bk)
                    c_idx = (k0 + bk) * n + (i0 + ai)
                    U[a_idx][col] = coeff
                    V[b_idx][col] = coeff
                    W[c_idx][col] = coeff
                    col += 1
    
    # Strassen for 4x4 x 4x4 (first 4 columns of B)
    # M1 = (A11+A22)*(B11+B22)
    # This is a sum of 4 2x2x2x2 products, but I need to represent it as products.
    # (A11+A22) is not a single matrix in my decomposition.
    
    # I think I need to directly construct the rank-1 terms.
    # Let me just use schoolbook for everything but try to be smart.
    
    # Actually, let me just use a pure schoolbook approach but with a different structure
    # to see if I can get below 112.
    
    # For now, let me just return the schoolbook to ensure correctness,
    # then I'll optimize.
    
    # Schoolbook: rank 112
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
