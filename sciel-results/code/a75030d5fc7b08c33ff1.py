import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition for rank 108.
    # Split m=5 into m1=2, m2=3.
    m1, m2 = 2, 3
    
    rank = 108
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Helper to add Strassen 2x2x2 block
    # A is 2x2 at rows i..i+1, cols j..j+1
    # B is 2x2 at rows j..j+1, cols k..k+1
    # C is 2x2 at rows k..k+1, cols i..i+1
    def add_strassen(i, j, k):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        # U: (i,j), (i+1,j+1) -> 1
        # V: (j,k), (j+1,k+1) -> 1
        # W: (k,i), (k+1,i+1) -> 1
        # But Strassen sums these into intermediate matrices.
        # The rank-1 terms are for the products.
        # We need to map the 7 products to the tensor.
        
        # Product 1: (A11+A22) * (B11+B22)
        # U: A11, A22
        # V: B11, B22
        # W: C11, C22 (with signs)
        
        # This manual expansion is error-prone.
        # Better to use the known 7 formulas.
        
        # M1 = (A11 + A22) (B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B12 - B22)
        # M4 = A22 (B21 - B11)
        # M5 = (A11 + A12) B21
        # M6 = (A21 - A11) (B11 + B12)
        # M7 = (A12 - A21) (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We add 7 rank-1 terms.
        
        # 1. (A11+A22)(B11+B22)
        # U: (i,j)=1, (i+1,j+1)=1
        # V: (j,k)=1, (j+1,k+1)=1
        # W: (k,i)=1, (k+1,i+1)=1
        # This is not a single rank-1 term, it's a product of sums.
        # In tensor decomposition, we decompose the product of sums into sums of products?
        # No, the rank-1 terms in Strassen are the 7 matrix multiplications.
        # Each matrix multiplication is a tensor of rank 1? No, rank of M_2,2,2 is 7.
        # The 7 "terms" in Strassen are actually 7 scalar multiplications of 2x2 matrices.
        # Each scalar multiplication of 2x2 matrices corresponds to 4 scalar multiplications in schoolbook, but Strassen reduces this.
        # Wait.
        # Strassen computes 7 products of 2x2 matrices.
        # Each product is a matrix multiplication.
        # The tensor for 2x2x2 multiplication has rank 7.
        # So each "Strassen term" M_i is itself a rank-7 tensor?
        # No.
        # The decomposition of M_2,2,2 into 7 rank-1 tensors is the Strassen algorithm.
        # So we need to add 7 rank-1 terms for each 2x2x2 block.
        
        # Let's define the 7 rank-1 terms for a 2x2x2 block.
        # The 7 terms are:
        # t1 = (A11+A22) * (B11+B22) -> This is not rank 1.
        # The rank-1 terms are:
        # u1 = A11+A22, v1 = B11+B22, w1 = C11+C22? No.
        
        # The standard Strassen decomposition:
        # M1 = (A11+A22)(B11+B22)
        # This M1 is a matrix.
        # The contribution to C is linear in M1.
        # The tensor decomposition is:
        # T = sum_{r=1}^7 u_r \otimes v_r \otimes w_r
        # where u_r, v_r, w_r are 2x2 matrices (flattened).
        
        # The 7 vectors are:
        # u1 = [[1,0],[1,1]], v1 = [[1,1],[0,1]], w1 = [[1,0],[0,1]] ? No.
        
        # Let's look up the exact 7 rank-1 tensors for 2x2x2.
        # They are:
        # 1. (A11+A22) * (B11+B22)
        #    U: [[1,0],[1,1]]? No.
        #    The vectors are:
        #    u1 = A11+A22 = [[1,0],[0,1]] + [[0,0],[1,0]]?
        #    
        #    Actually, it's easier to just hardcode the 7 terms for 2x2x2.
        #    Term 1:
        #    U: A11+A22 -> [1,0,0,1] (flattened row-major)
        #    V: B11+B22 -> [1,0,0,1]
        #    W: C11+C22 -> [1,0,0,1]
        #    
        #    Term 2:
        #    U: A21+A22 -> [0,0,1,1]
        #    V: B11 -> [1,0,0,0]
        #    W: C21 -> [0,0,1,0]
        #    
        #    Term 3:
        #    U: A11 -> [1,0,0,0]
        #    V: B12-B22 -> [0,1,0,-1]
        #    W: C12 -> [0,1,0,0]
        #    
        #    Term 4:
        #    U: A22 -> [0,0,0,1]
        #    V: B21-B11 -> [0,0,1,-1]
        #    W: C22 -> [0,0,0,1]
        #    
        #    Term 5:
        #    U: A11+A12 -> [1,1,0,0]
        #    V: B21 -> [0,0,1,0]
        #    W: C11-C21? No.
        #    
        #    This is getting complicated.
        #    
        #    Alternative: Just use the schoolbook for 2x2x2 (8 terms) if Strassen is too hard to code correctly on the fly.
        #    But we know Strassen reduces 8 to 7.
        #    
        #    Let's try to implement the 7 terms explicitly.
        
        # Flattened indices for 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # Term 1: (A11+A22)(B11+B22)
        # U: [1, 0, 0, 1]
        # V: [1, 0, 0, 1]
        # W: [1, 0, 0, 1]
        add_vec(U, i*m+j, [1, 0, 0, 1], col)
        add_vec(V, j*p+k, [1, 0, 0, 1], col)
        add_vec(W, k*n+i, [1, 0, 0, 1], col)
        col += 1
        
        # Term 2: (A21+A22)B11
        # U: [0, 0, 1, 1]
        # V: [1, 0, 0, 0]
        # W: [0, 0, 1, 0]
        add_vec(U, i*m+j, [0, 0, 1, 1], col)
        add_vec(V, j*p+k, [1, 0, 0, 0], col)
        add_vec(W, k*n+i, [0, 0, 1, 0], col)
        col += 1
        
        # Term 3: A11(B12-B22)
        # U: [1, 0, 0, 0]
        # V: [0, 1, 0, -1]
        # W: [0, 1, 0, 0]
        add_vec(U, i*m+j, [1, 0, 0, 0], col)
        add_vec(V, j*p+k, [0, 1, 0, -1], col)
        add_vec(W, k*n+i, [0, 1, 0, 0], col)
        col += 1
        
        # Term 4: A22(B21-B11)
        # U: [0, 0, 0, 1]
        # V: [0, 0, 1, -1]
        # W: [0, 0, 0, 1]
        add_vec(U, i*m+j, [0, 0, 0, 1], col)
        add_vec(V, j*p+k, [0, 0, 1, -1], col)
        add_vec(W, k*n+i, [0, 0, 0, 1], col)
        col += 1
        
        # Term 5: (A11+A12)B21
        # U: [1, 1, 0, 0]
        # V: [0, 0, 1, 0]
        # W: [1, 0, -1, 0] ?
        # C11 = M1 + M4 - M5 + M7
        # C21 = M2 + M4
        # So M5 contributes to C11 with -1 and C21 with 0?
        # Wait, M5 = (A11+A12)B21.
        # Result is 2x2 matrix.
        # (A11+A12) is [[a11+a12, 0], [0, 0]]? No.
        # A11 is top-left. A12 is top-right.
        # A11+A12 is [[a11, a12], [0, 0]]? No.
        # A = [[A11, A12], [A21, A22]].
        # A11+A12 = [[a11, a12], [a21, a22]]? No.
        # A11 is a scalar? No, A11 is a 1x1 block?
        # In 2x2 Strassen, A11 is 1x1.
        # So A11+A12 is a 1x1 matrix? No, A11 and A12 are scalars.
        # So (A11+A12) is a scalar.
        # B21 is a scalar.
        # Product is a scalar.
        # But we are multiplying 2x2 matrices.
        # The blocks A11 etc are 1x1.
        # So M5 is a scalar.
        # It contributes to C11 and C21?
        # C11 = M1 + M4 - M5 + M7
        # C21 = M2 + M4
        # So M5 contributes -1 to C11.
        # W: [-1 at (0,0), 0 elsewhere] -> [-1, 0, 0, 0]
        add_vec(U, i*m+j, [1, 1, 0, 0], col)
        add_vec(V, j*p+k, [0, 0, 1, 0], col)
        add_vec(W, k*n+i, [-1, 0, 0, 0], col)
        col += 1
        
        # Term 6: (A21-A11)(B11+B12)
        # U: [0, 0, 1, -1] ?
        # A21 is bottom-left. A11 is top-left.
        # A21-A11 = [[-a11, 0], [a21, 0]]?
        # Flattened: [0, 0, 1, -1] is wrong.
        # A21 is at (1,0). A11 is at (0,0).
        # So vector is [-1, 0, 1, 0].
        # V: B11+B12. B11 at (0,0), B12 at (0,1).
        # Vector [1, 1, 0, 0].
        # W: C22 = M1 - M2 + M3 + M6.
        # M6 contributes +1 to C22.
        # Vector [0, 0, 0, 1].
        add_vec(U, i*m+j, [-1, 0, 1, 0], col)
        add_vec(V, j*p+k, [1, 1, 0, 0], col)
        add_vec(W, k*n+i, [0, 0, 0, 1], col)
        col += 1
        
        # Term 7: (A12-A21)(B21+B22)
        # U: A12 at (0,1), A21 at (1,0).
        # Vector [0, 1, -1, 0].
        # V: B21 at (1,0), B22 at (1,1).
        # Vector [0, 0, 1, 1].
        # W: C11 = M1 + M4 - M5 + M7.
        # M7 contributes +1 to C11.
        # Vector [1, 0, 0, 0].
        add_vec(U, i*m+j, [0, 1, -1, 0], col)
        add_vec(V, j*p+k, [0, 0, 1, 1], col)
        add_vec(W, k*n+i, [1, 0, 0, 0], col)
        col += 1

    def add_vec(mat, offset, vec, col_idx):
        for i, val in enumerate(vec):
            if val != 0:
                mat[offset + i][col_idx] = val

    # 1. A2 B2 (4x2 * 2x6)
    # Split A2 into A21 (2x2), A22 (2x2)
    # A21 is rows 0-1, cols 0-1
    # A22 is rows 2-3, cols 0-1
    # B2 is rows 0-1, cols 0-5
    
    # A21 B2: 2x2 * 2x6
    # Split B2 into B21 (2x4), B22 (2x2)
    # B21 is cols 0-3
    # B22 is cols 4-5
    
    # A21 B21: 2x2 * 2x4
    # Split B21 into B21a (2x2), B21b (2x2)
    # B21a cols 0-1
    # B21b cols 2-3
    
    add_strassen(0, 0, 0) # A21 (0,0) * B21a (0,0) -> C (0,0)
    add_strassen(0, 2, 0) # A21 (0,0) * B21b (2,0) -> C (0,2)
    
    # A21 B22: 2x2 * 2x2
    add_strassen(0, 4, 0) # A21 (0,0) * B22 (4,0) -> C (0,4)
    
    # A22 B2: 2x2 * 2x6
    # A22 is rows 2-3, cols 0-1
    # A22 B21: 2x2 * 2x4
    add_strassen(2, 0, 0) # A22 (2,0) * B21a (0,0) -> C (2,0)
    add_strassen(2, 2, 0) # A22 (2,0) * B21b (2,0) -> C (2,2)
    
    # A22 B22: 2x2 * 2x2
    add_strassen(2, 4, 0) # A22 (2,0) * B22 (4,0) -> C (2,4)
    
    # 2. A3 B3 (4x3 * 3x6)
    # Split A3 into A31 (2x3), A32 (2x3)
    # A31 rows 0-1, cols 2-4
    # A32 rows 2-3, cols 2-4
    # B3 rows 2-4, cols 0-5
    
    # A31 B3: 2x3 * 3x6
    # Split B3 into B31 (2x6), B32 (1x6)
    # B31 rows 2-3, cols 0-5
    # B32 row 4, cols 0-5
    
    # A31 B31: 2x2 * 2x6 (using first 2 cols of A31 and first 2 rows of B31)
    # A31 cols 2-3 (2 cols), B31 rows 2-3 (2 rows)
    add_strassen(0, 2, 0) # A31 (0,2) * B31 (2,0) -> C (0,0)
    add_strassen(0, 4, 0) # A31 (0,2) * B31 (2,2) -> C (0,2)
    
    # A31 B32: 2x1 * 1x6
    # A31 col 4 (1 col), B32 row 4 (1 row)
    # 2x1 * 1x6 is outer product.
    # A31 is 2x1. B32 is 1x6.
    # 12 terms.
    for r in range(2):
        for c in range(6):
            add_term((0 + r)*m + 4, (2 + 2)*p + c, c*n + (0 + r), 1)
            # A index: row 0-1, col 4.
            # V index: row 4, col 0-5.
            # W index: row 0-5, col 0-1.
            # U: (0+r, 4) -> (0+r)*5 + 4
            # V: (4, c) -> 4*6 + c
            # W: (c, 0+r) -> c*4 + (0+r)
            
    # A32 B3: 2x3 * 3x6
    # A32 rows 2-3, cols 2-4
    # A32 B31: 2x2 * 2x6
    add_strassen(2, 2, 0) # A32 (2,2) * B31 (2,0) -> C (2,0)
    add_strassen(2, 4, 0) # A32 (2,2) * B31 (2,2) -> C (2,2)
    
    # A32 B32: 2x1 * 1x6
    for r in range(2):
        for c in range(6):
            add_term((2 + r)*m + 4, (2 + 2)*p + c, c*n + (2 + r), 1)
            
    return {"U": U, "V": V, "W": W}

# Helper function needed inside construct
def add_vec(mat, offset, vec, col_idx):
    for i, val in enumerate(vec):
        if val != 0:
            mat[offset + i][col_idx] = val

# The add_vec function is defined outside, but used inside.
# In Python, this works if add_vec is global.

