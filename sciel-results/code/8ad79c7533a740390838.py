"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal (or near-minimal) exact tensor decomposition for 3x4 * 4x5 multiplication.
    
    Strategy: 
    The matrix multiplication tensor T_{n,m,p} has rank at most n*m*p (schoolbook).
    For small dimensions, we can use known lower bounds and constructions.
    
    For 3x4x5:
    - Schoolbook rank = 3*4*5 = 60
    - We can potentially do better by exploiting structure.
    
    However, finding the exact minimal rank is an open problem in general.
    Let's try a construction that combines Strassen-like ideas.
    
    Approach: Decompose the 3x4 and 4x5 matrices into blocks and apply
    Strassen's algorithm where possible, combined with schoolbook for the rest.
    
    Let's try:
    - Split the 3 rows into 1+2
    - Split the 4 columns into 2+2
    - Split the 5 columns into 2+3
    
    For the 2x2x2 sub-block, use Strassen (rank 7)
    For other blocks, use schoolbook.
    
    Actually, let me think more carefully. The tensor is T_{i,j,k} = delta_{j,k} for
    (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    
    The standard schoolbook decomposition has rank n*m*p with:
    U_{(i,j), c} = 1 if c corresponds to (i,j)
    V_{(j,k), c} = 1 if c corresponds to (j,k)  
    W_{(k,i), c} = 1 if c corresponds to (k,i)
    
    For Strassen on 2x2 matrices, we use rank 7 instead of 8.
    
    Let's try a hybrid approach:
    - For the top-left 2x2 block of the result (rows 0-1, cols 0-1), use Strassen
    - For the rest, use schoolbook
    
    But this gets complicated. Let me just use schoolbook for now to ensure correctness,
    and the rank will be 60.
    
    Actually, let me try to implement a proper Strassen-based decomposition.
    
    For multiplying a 2x2 matrix A by a 2x2 matrix B:
    C = A*B where A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    Strassen's 7 multiplications:
    m1 = (a+d)(e+h)
    m2 = (c+d)e
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)h
    m6 = (c-a)(e+f)
    m7 = (b-d)(g+h)
    
    C11 = m1 + m4 - m5 + m7
    C12 = m3 + m5
    C21 = m2 + m4
    C22 = m1 - m2 + m3 + m6
    
    For the tensor decomposition, each multiplication m_i corresponds to a rank-1 term.
    
    Let me construct the full decomposition for 3x4 * 4x5 using a mix of Strassen and schoolbook.
    
    This is complex. Let me just use schoolbook for correctness first.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
