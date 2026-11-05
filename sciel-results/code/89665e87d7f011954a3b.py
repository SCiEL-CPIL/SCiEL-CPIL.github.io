import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # We split the p=8 dimension into 3, 3, 2.
    # For the first 3 columns, we use Laderman's rank 23 algorithm.
    # For the next 3 columns, we use Laderman's rank 23 algorithm.
    # For the last 2 columns, we use schoolbook rank 18.
    # Total rank = 23 + 23 + 18 = 64.
    
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman's 23 terms for 3x3x3
    laderman_terms = [
        ((1,0,0), (1,0,0), (1,0,0)),
        ((0,1,0), (1,0,0), (0,0,1)),
        ((0,0,1), (1,0,0), (0,1,0)),
        ((1,0,0), (0,1,0), (0,1,0)),
        ((0,1,0), (0,1,0), (1,0,0)),
        ((0,0,1), (0,1,0), (1,1,0)),
        ((1,0,0), (0,0,1), (0,0,1)),
        ((0,1,0), (0,0,1), (0,1,1)),
        ((0,0,1), (0,0,1), (1,1,1)),
        ((1,0,0), (1,1,0), (0,0,0)),
        ((0,1,0), (1,1,0), (0,0,0)),
        ((0,0,1), (1,1,0), (0,0,0)),
        ((1,0,0), (0,1,1), (0,0,0)),
        ((0,1,0), (0,1,1), (0,0,0)),
        ((0,0,1), (0,1,1), (0,0,0)),
        ((1,0,0), (1,0,1), (0,0,0)),
        ((0,1,0), (1,0,1), (0,0,0)),
        ((0,0,1), (1,0,1), (0,0,0)),
        ((1,1,0), (0,0,1), (0,0,0)),
        ((1,0,1), (0,0,1), (0,0,0)),
        ((0,1,1), (0,0,1), (0,0,0)),
        ((1,1,0), (1,0,1), (0,0,0)),
        ((1,0,1), (1,1,0), (0,0,0))
    ]
    
    col = 0
    # First block: columns 0, 1, 2 of B and C
    for u, v, w in laderman_terms:
        for i in range(3):
            U[i * 3 + 0][col] = u[i] # A is 3x3, so row index is i, col index is 0? No.
            # U index is i*m + j. Here m=3.
            # The term u is a vector for A. A is 3x3.
            # u[i] corresponds to A[i, ?].
            # In tensor decomposition, U[i*m+j] is the coefficient for A[i,j].
            # So u must be a 3x3 matrix flattened? No, u is a vector of length 3?
            # Wait. The tensor is T_{ijk}.
            # Decomposition: T_{ijk} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
            # U is (n*m, rank). Index i*m+j.
            # V is (m*p, rank). Index j*p+k.
            # W is (p*n, rank). Index k*n+i.
            
            # For Laderman, the vectors u, v, w are length 3.
            # u[i] is the coefficient for A[i, ?].
            # Actually, the linear forms are on the entries of A, B, C.
            # A is 3x3. The linear form u is a vector in R^9? No.
            # Laderman's algorithm uses linear forms on the 9 entries of A.
            # So u should be a 9-element vector?
            # No, the standard presentation uses 3x3 matrices for the linear forms?
            # Let's check the dimension.
            # If u is length 3, it only depends on one row or column?
            # No, Laderman's terms are of the form (A_{i0}+A_{i1}) (B_{j0}+B_{j1}) ...
            # The vectors u, v, w in the 23-term list I recalled are likely for the 2x2 case or I am misremembering the dimension.
            # For 3x3, the linear forms are on 9 variables.
            # So u, v, w should be length 9.
            
            # My recalled list has length 3. This is likely for 2x2x2 Strassen or similar.
            # I cannot use this list for 3x3x3.
            
            pass # This block is invalid.
        
        col += 1
    
    # Since I cannot reliably recall the 9-dim Laderman vectors, I will fall back to a simpler, guaranteed improvement.
    # I will use schoolbook for the first 3 columns (rank 27) and schoolbook for the rest? No, that's 72.
    # I will use Strassen for the 2x2x2 part of the 3x3x3 multiplication.
    # But as calculated, that might not be better than schoolbook for the whole 3x3x3.
    
    # Let's try a different split.
    # Use schoolbook for all, but optimize the 3x3x8 by noticing structure? No.
    
    # I will return the schoolbook solution to ensure validity, but this scores 1.0.
    # I need to find a valid improvement.
    
    # What if I use the known rank 23 for 3x3x3?
    # I will hardcode the 23 terms for 3x3x3 from a reliable source I can derive?
    # No, I will use the **Bordagaray** algorithm which is also rank 23.
    # But I don't have the coefficients.
    
    # I will stick to the schoolbook solution but with a small optimization:
    # None available that is guaranteed to be better without complex coefficients.
    
    # However, I can use the fact that R(3,3,4) <= 25.
    # If I can implement R(3,3,4) = 25, I get 50.
    # But I don't have the coefficients.
    
    # I will return the schoolbook solution.
    rank = n * m * p
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
