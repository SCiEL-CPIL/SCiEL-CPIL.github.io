import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    # U shape: (n*m, rank) = (6, 39)
    # V shape: (m*p, rank) = (21, 39)
    # W shape: (p*n, rank) = (14, 39)
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    # u_indices: list of (index, coeff) for U
    # v_indices: list of (index, coeff) for V
    # w_indices: list of (index, coeff) for W
    def add_term(u_idx, v_idx, w_idx):
        nonlocal col
        for idx, coeff in u_idx:
            U[idx][col] += coeff
        for idx, coeff in v_idx:
            V[idx][col] += coeff
        for idx, coeff in w_idx:
            W[idx][col] += coeff
        col += 1
        
    # Part 1: A_R B_R (Rank 14)
    # A_R is col 2 of A (indices 2, 5 in flattened U? No, U is n*m=6. A is 2x3.
    # A[i,j] index in U is i*m + j.
    # A_R: i in {0,1}, j=2. Indices: 0*3+2=2, 1*3+2=5.
    # B_R is row 2 of B (indices 14..20 in flattened V? V is m*p=21. B is 3x7.
    # B[j,k] index in V is j*p + k.
    # B_R: j=2, k in 0..6. Indices: 2*7+k = 14+k.
    # Output P[i,k]. Index in W is k*n + i.
    
    for i in range(n):
        for k in range(p):
            u_idx = [(i * m + 2, 1)]
            v_idx = [(2 * p + k, 1)]
            w_idx = [(k * n + i, 1)]
            add_term(u_idx, v_idx, w_idx)
            
    # Part 2: A_L B_L (Rank 25)
    # A_L is cols 0,1 of A. Indices:
    # a = A[0,0] -> 0
    # b = A[0,1] -> 1
    # c = A[1,0] -> 3
    # d = A[1,1] -> 4
    
    # B_L is rows 0,1 of B.
    # Split B_L into 3 blocks of 2 cols (0,1), (2,3), (4,5) and 1 block of 1 col (6).
    
    # Strassen blocks
    for block in range(3):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # B indices
        # e = B[0,k0] -> 0*7+k0 = k0
        # f = B[0,k1] -> k1
        # g = B[1,k0] -> 7+k0
        # h = B[1,k1] -> 7+k1
        
        # W indices
        # M00: k0*n + 0 = 2*k0
        # M01: k1*n + 0 = 2*k1
        # M10: k0*n + 1 = 2*k0+1
        # M11: k1*n + 1 = 2*k1+1
        
        # Term 1: (a+d)(e+h)
        add_term(
            [(0, 1), (4, 1)],
            [(k0, 1), (7+k1, 1)],
            [(2*k0, 1), (2*k1+1, 1)]
        )
        
        # Term 2: (c+d)e
        add_term(
            [(3, 1), (4, 1)],
            [(k0, 1)],
            [(2*k0+1, 1), (2*k1+1, -1)]
        )
        
        # Term 3: a(f-h)
        add_term(
            [(0, 1)],
            [(k1, 1), (7+k1, -1)],
            [(2*k1, 1), (2*k1+1, 1)]
        )
        
        # Term 4: d(g-e)
        add_term(
            [(4, 1)],
            [(7+k0, 1), (k0, -1)],
            [(2*k0, 1), (2*k0+1, 1)]
        )
        
        # Term 5: (a+b)h
        add_term(
            [(0, 1), (1, 1)],
            [(7+k1, 1)],
            [(2*k0, -1), (2*k1, 1)]
        )
        
        # Term 6: (c-a)(e+f)
        add_term(
            [(3, 1), (0, -1)],
            [(k0, 1), (k1, 1)],
            [(2*k1+1, 1)]
        )
        
        # Term 7: (b-d)(g+h)
        add_term(
            [(1, 1), (4, -1)],
            [(7+k0, 1), (7+k1, 1)],
            [(2*k0, 1)]
        )
        
    # Last block of B_L: col 6
    k = 6
    # e = B[0,6] -> 6
    # g = B[1,6] -> 13
    
    # Term 1: a*e
    add_term(
        [(0, 1)],
        [(6, 1)],
        [(2*6, 1)] # M00
    )
    # Term 2: b*g
    add_term(
        [(1, 1)],
        [(13, 1)],
        [(2*6, 1)] # M00
    )
    # Term 3: c*e
    add_term(
        [(3, 1)],
        [(6, 1)],
        [(2*6+1, 1)] # M10
    )
    # Term 4: d*g
    add_term(
        [(4, 1)],
        [(13, 1)],
        [(2*6+1, 1)] # M10
    )
    
    return {"U": U, "V": V, "W": W}
