"""Vectorized decomposition of 2x2 by 2x3 matrix multiplication tensor."""
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=3
    # A is 2x2: indices a (row0,col0), b (row0,col1), c (row1,col0), d (row1,col1)
    # B is 2x3: indices e (row0,col0), f (row0,col1), g (row0,col2), h (row1,col0), i (row1,col1), j (row1,col2)
    # C is 2x3: W[k*n+i, r] where k is col of C, i is row of C
    # C[0,0]=ae+bg, C[0,1]=af+bh, C[0,2]=ag+bj
    # C[1,0]=ce+dg, C[1,1]=cf+dh, C[1,2]=cg+dj
    
    # U: shape (n*m, rank) = (4, 11)
    # V: shape (m*p, rank) = (6, 11)
    # W: shape (p*n, rank) = (6, 11)
    
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Index mapping for U (A matrix, 2x2):
    # a = U[0], b = U[1], c = U[2], d = U[3]
    # U rows: 0->a, 1->b, 2->c, 3->d
    
    # Index mapping for V (B matrix, 2x3):
    # e = V[0], f = V[1], g = V[2], h = V[3], i = V[4], j = V[5]
    # V rows: 0->e, 1->f, 2->g, 3->h, 4->i, 5->j
    
    # Index mapping for W (C matrix, 2x3), W[k*n+i, r]:
    # k=0 (col 0 of C): W[0]=C[0,0], W[1]=C[1,0]
    # k=1 (col 1 of C): W[2]=C[0,1], W[3]=C[1,1]
    # k=2 (col 2 of C): W[4]=C[0,2], W[5]=C[1,2]
    
    # Strassen terms for C[:, 0:2]
    # M1 = (a+b)(e+g)
    U[0, 0] = 1.0; U[1, 0] = 1.0
    V[0, 0] = 1.0; V[2, 0] = 1.0
    # C00 += M1, C10 += M1
    W[0, 0] = 1.0; W[1, 0] = 1.0
    
    # M2 = (a+b)(f+h)
    U[0, 1] = 1.0; U[1, 1] = 1.0
    V[1, 1] = 1.0; V[3, 1] = 1.0
    # C00 -= M2, C01 += M2
    W[0, 1] = -1.0; W[2, 1] = 1.0
    
    # M3 = (c+d)(e+g)
    U[2, 2] = 1.0; U[3, 2] = 1.0
    V[0, 2] = 1.0; V[2, 2] = 1.0
    # C01 -= M3, C10 -= M3
    W[2, 2] = -1.0; W[1, 2] = -1.0
    
    # M4 = (c+d)(f+h)
    U[2, 3] = 1.0; U[3, 3] = 1.0
    V[1, 3] = 1.0; V[3, 3] = 1.0
    # C00 += M4, C01 += M4, C11 += M4
    W[0, 3] = 1.0; W[2, 3] = 1.0; W[3, 3] = 1.0
    
    # M5 = (a-c)(e-g)
    U[0, 4] = 1.0; U[2, 4] = -1.0
    V[0, 4] = 1.0; V[2, 4] = -1.0
    # C00 -= M5, C10 += M5
    W[0, 4] = -1.0; W[1, 4] = 1.0
    
    # M6 = (c-d)(g+h)
    U[2, 5] = 1.0; U[3, 5] = -1.0
    V[2, 5] = 1.0; V[3, 5] = 1.0
    # C11 += M6
    W[3, 5] = 1.0
    
    # M7 = (a-b)(f+h)
    U[0, 6] = 1.0; U[1, 6] = -1.0
    V[1, 6] = 1.0; V[3, 6] = 1.0
    # C01 -= M7, C10 -= M7
    W[2, 6] = -1.0; W[1, 6] = -1.0
    
    # Direct terms for C[:, 2]
    # M8 = a*x (where x = g, col 2 row 0 of B)
    U[0, 7] = 1.0
    V[2, 7] = 1.0
    # C02 += M8
    W[4, 7] = 1.0
    
    # M9 = b*y (where y = j, col 2 row 1 of B)
    U[1, 8] = 1.0
    V[5, 8] = 1.0
    # C02 += M9
    W[4, 8] = 1.0
    
    # M10 = c*x
    U[2, 9] = 1.0
    V[2, 9] = 1.0
    # C12 += M10
    W[5, 9] = 1.0
    
    # M11 = d*y
    U[3, 10] = 1.0
    V[5, 10] = 1.0
    # C12 += M11
    W[5, 10] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

