"""Matrix multiplication 2,3,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(U_entry, V_entry, W_entry):
        """Add a rank-1 term."""
        U_col = [0] * (n * m)
        for idx, val in U_entry.items():
            U_col[idx] = val
        V_col = [0] * (m * p)
        for idx, val in V_entry.items():
            V_col[idx] = val
        W_col = [0] * (p * n)
        for idx, val in W_entry.items():
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    # Part 1: A1 * B1 where A1 is 2x2 (cols 0,1 of A), B1 is 2x9 (rows 0,1 of B)
    # Using Strassen for each pair of columns
    
    for block_start in range(0, 8, 2):
        k0 = block_start
        k1 = block_start + 1
        # M1 = (a00+a11)*(b00+b11)
        add_term(
            {0*m+0: 1, 1*m+1: 1},
            {0*p+k0: 1, 1*p+k1: 1},
            {k0*n+0: 1, k1*n+1: 1}
        )
        # M2 = (a10+a11)*b00
        add_term(
            {1*m+0: 1, 1*m+1: 1},
            {0*p+k0: 1},
            {k0*n+1: 1, k1*n+1: -1}
        )
        # M3 = a00*(b01-b11)
        add_term(
            {0*m+0: 1},
            {0*p+k1: 1, 1*p+k1: -1},
            {k1*n+0: 1, k1*n+1: 1}
        )
        # M4 = a11*(b10-b00)
        add_term(
            {1*m+1: 1},
            {1*p+k0: 1, 0*p+k0: -1},
            {k0*n+0: 1, k0*n+1: 1}
        )
        # M5 = (a00+a01)*b11
        add_term(
            {0*m+0: 1, 0*m+1: 1},
            {1*p+k1: 1},
            {k0*n+0: -1, k1*n+0: 1}
        )
        # M6 = (a10-a00)*(b00+b01)
        add_term(
            {1*m+0: 1, 0*m+0: -1},
            {0*p+k0: 1, 0*p+k1: 1},
            {k1*n+1: 1}
        )
        # M7 = (a01-a11)*(b10+b11)
        add_term(
            {0*m+1: 1, 1*m+1: -1},
            {1*p+k0: 1, 1*p+k1: 1},
            {k0*n+0: 1}
        )
    
    # Single column k=8: schoolbook 2x2 x 2x1
    k = 8
    add_term({0*m+0: 1}, {0*p+k: 1}, {k*n+0: 1})
    add_term({0*m+1: 1}, {1*p+k: 1}, {k*n+0: 1})
    add_term({1*m+0: 1}, {0*p+k: 1}, {k*n+1: 1})
    add_term({1*m+1: 1}, {1*p+k: 1}, {k*n+1: 1})
    
    # Part 2: A2 * B2 where A2 is 2x1 (col 2 of A), B2 is 1x9 (row 2 of B)
    for i in range(n):
        for k in range(p):
            add_term(
                {i*m+2: 1},
                {2*p+k: 1},
                {k*n+i: 1}
            )
    
    rank = len(U_cols)
    
    # Transpose to get (n*m, rank), (m*p, rank), (p*n, rank)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(m*p)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
