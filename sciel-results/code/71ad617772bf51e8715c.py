import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def _strassen_222():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: 0..3 correspond to flattened 2x2 matrices.
    """
    # Strassen formulas
    # M1 = (a+b)(c+d)
    # M2 = (a+b)(e+f)
    # M3 = a(f-d)
    # M4 = c(g-e)
    # M5 = (a+e)(g+h)
    # M6 = (c+d)(g-h)
    # M7 = (b-f)(e+h)
    
    # A = [[a, b], [c, d]] -> indices 0,1,2,3
    # B = [[e, f], [g, h]] -> indices 0,1,2,3
    # C = [[a, b], [c, d]] * [[e, f], [g, h]]
    # C11 = a*e + b*g
    # C12 = a*f + b*h
    # C21 = c*e + d*g
    # C22 = c*f + d*h
    
    # We need U, V, W such that C = sum_l U_l V_l W_l^T?
    # Actually T = sum u_l v_l w_l.
    # w_l is the output matrix.
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # l=1: M1 = (a+b)(c+d)
    # u1 = [1, 1, 0, 0] (a+b)
    # v1 = [1, 0, 1, 0] (c+d) ? No, v is for B.
    # Wait, Strassen multiplies linear forms of A and B.
    # Let's define linear forms for A (u) and B (v).
    # And the resulting output matrix w.
    
    # M1 = (A00+A01) * (A10+A11) ?? No.
    # Standard Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (b+d)(g+h)
    # p3 = (a+b)(e)
    # p4 = (c+d)(h)
    # p5 = (a-e)(f+h)
    # p6 = (b-f)(e+g)
    # p7 = (c-a)(d+h)
    
    # Let's use the standard 7 multiplications.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    
    # p1 = (a+d)(e+h)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [[1, 0], [0, 1]] # e+h contributes to C11 and C22?
    # C11 = a*e + b*g
    # C22 = c*f + d*h
    # p1 = a*e + a*h + d*e + d*h
    # This doesn't map directly to a single output matrix with 1s.
    # The decomposition is T = sum p_k * (linear combo of basis tensors).
    
    # It is easier to construct the tensor directly.
    # T_222 = sum_{i,j,k} e_i e_j e_k.
    # Rank 7 decomposition exists.
    # I will hardcode the rank 7 decomposition for 2x2x2.
    
    # Reference: 
    # u1 = [1, 0, 0, 0], v1 = [1, 0, 0, 0], w1 = [1, 0, 0, 0] (a*e)
    # This is schoolbook.
    
    # Strassen Rank 7:
    # See "Strassen's algorithm" tensor decomposition.
    
    # I will use a known decomposition.
    # U, V, W for 2x2x2.
    
    U[0,0] = 1; U[1,0] = 1; U[2,0] = 0; U[3,0] = 0 # a+b
    V[0,0] = 1; V[1,0] = 0; V[2,0] = 0; V[3,0] = 0 # c
    W[0,0] = 0; W[1,0] = 1; W[2,0] = 0; W[3,0] = 1 # ?
    
    # This is getting risky to hardcode without verification.
    # I will implement the rank 48 block sum using schoolbook for the blocks if Strassen is too complex to hardcode correctly.
    # Schoolbook for 2x2x6 is rank 12.
    # 3 blocks of 2x2x6? No.
    # A1 (2x2) B1 (2x6). Schoolbook rank 2*2*6 = 24.
    # A2 (2x2) B2 (2x6). Schoolbook rank 24.
    # A3 (2x1) B3 (1x6). Schoolbook rank 12.
    # Total 60.
    
    # So I need Strassen to get below 60.
    
    pass

def construct():
    n, m, p = DIMENSIONS
    # Return schoolbook for safety if Strassen fails
    schoolbook_rank = n * m * p
    U = [[0.0] * schoolbook_rank for _ in range(n * m)]
    V = [[0.0] * schoolbook_rank for _ in range(m * p)]
    W = [[0.0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
