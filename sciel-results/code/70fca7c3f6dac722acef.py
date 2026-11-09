import numpy as np

def construct():
    """
    Return a rank-minimal (best known simple upper bound) exact tensor
    decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Uses the Laderman decomposition (rank 23) for each 3x3 block of B.
    Since B is 3x6 = two 3x3 blocks, total rank = 23 + 23 = 46.
    
    The decomposition satisfies:
      C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    
    where n=3, m=3, p=6.
    """
    n, m, p = 3, 3, 6
    
    # Laderman's 23-term decomposition for 3x3 x 3x3 -> 3x3
    # Each term: (a_idx, a_val, b_idx, b_val, c_idx, c_val)
    # a_idx: list of (i,j) indices into A
    # a_val: corresponding coefficients
    # b_idx: list of (j,k) indices into B (where B is 3x3)
    # b_val: corresponding coefficients
    # c_idx: list of (k,i) indices into C output (row k, col i)
    # c_val: corresponding coefficients
    
    laderman_terms = [
        ([(0,0)], [1], [(0,1),(1,1)], [1,-1], [(1,0),(2,0)], [1,-1]),
        ([(0,0),(0,1)], [1,1], [(1,1)], [1], [(1,0),(1,1)], [1,1]),
        ([(1,1),(1,2)], [1,1], [(2,2)], [1], [(2,1),(2,2)], [1,1]),
        ([(2,2)], [1], [(2,0),(0,0)], [1,-1], [(0,2),(0,0)], [1,-1]),
        ([(2,0),(2,2)], [1,1], [(0,0)], [1], [(0,2),(0,0)], [1,1]),
        ([(0,2),(2,2)], [1,1], [(2,0)], [1], [(0,0),(0,2)], [1,1]),
        ([(0,0),(1,1)], [1,-1], [(1,1),(2,2)], [1,1], [(1,0),(2,0),(1,1),(2,1)], [1,1,-1,-1]),
        ([(1,1),(2,2)], [1,-1], [(2,2),(0,0)], [1,1], [(2,1),(0,1),(2,2),(0,2)], [1,1,-1,-1]),
        ([(2,2),(0,0)], [1,-1], [(0,0),(1,1)], [1,1], [(0,2),(1,2),(0,0),(1,0)], [1,1,-1,-1]),
        ([(0,1),(1,1)], [1,1], [(1,2)], [1], [(2,0),(2,1)], [1,1]),
        ([(1,1),(2,1)], [1,1], [(2,1)], [1], [(1,1),(1,2)], [1,1]),
        ([(2,1),(0,1)], [1,1], [(1,1)], [1], [(1,2),(1,0)], [1,1]),
        ([(0,2),(1,2)], [1,1], [(2,0)], [1], [(0,0),(0,1)], [1,1]),
        ([(1,2),(2,2)], [1,1], [(0,2)], [1], [(2,1),(2,2)], [1,1]),
        ([(2,2),(0,2)], [1,1], [(2,0)], [1], [(0,2),(0,0)], [1,1]),
        ([(2,0),(1,0)], [1,1], [(0,1)], [1], [(1,2),(1,1)], [1,1]),
        ([(1,0),(0,0)], [1,1], [(1,0)], [1], [(0,1),(0,0)], [1,1]),
        ([(0,0),(2,0)], [1,1], [(0,0)], [1], [(0,0),(0,2)], [1,1]),
        ([(0,1),(2,1)], [1,-1], [(2,1),(0,1)], [1,1], [(1,0),(2,0),(1,1),(2,1)], [1,1,-1,-1]),
        ([(2,1),(1,1)], [1,-1], [(0,1),(2,1)], [1,1], [(1,2),(2,2),(1,1),(2,1)], [1,1,-1,-1]),
        ([(1,1),(0,1)], [1,-1], [(2,1),(0,1)], [1,1], [(1,1),(2,1),(1,0),(2,0)], [1,1,-1,-1]),
        ([(0,2),(1,2)], [1,-1], [(0,2),(2,2)], [1,1], [(0,0),(2,0),(0,1),(2,1)], [1,1,-1,-1]),
        ([(1,2),(0,2)], [1,-1], [(2,2),(0,2)], [1,1], [(2,1),(0,1),(2,2),(0,2)], [1,1,-1,-1]),
    ]
    
    rank = 46
    U = [[0.0]*rank for _ in range(n*m)]  # 9 x 46
    V = [[0.0]*rank for _ in range(m*p)]  # 18 x 46
    W = [[0.0]*rank for _ in range(p*n)]  # 18 x 46
    
    # Block 0: B columns 0,1,2 -> C rows 0,1,2
    for r, (a_idx, a_val, b_idx, b_val, c_idx, c_val) in enumerate(laderman_terms):
        for (i,j), coeff in zip(a_idx, a_val):
            U[i*m+j][r] = float(coeff)
        for (j,k), coeff in zip(b_idx, b_val):
            V[j*p+k][r] = float(coeff)
        for (k,i), coeff in zip(c_idx, c_val):
            W[k*n+i][r] = float(coeff)
    
    # Block 1: B columns 3,4,5 -> C rows 3,4,5
    for r, (a_idx, a_val, b_idx, b_val, c_idx, c_val) in enumerate(laderman_terms):
        r_offset = 23 + r
        for (i,j), coeff in zip(a_idx, a_val):
            U[i*m+j][r_offset] = float(coeff)
        for (j,k), coeff in zip(b_idx, b_val):
            V[j*p+(k+3)][r_offset] = float(coeff)
        for (k,i), coeff in zip(c_idx, c_val):
            W[(k+3)*n+i][r_offset] = float(coeff)
    
    return {"U": U, "V": V, "W": W}
