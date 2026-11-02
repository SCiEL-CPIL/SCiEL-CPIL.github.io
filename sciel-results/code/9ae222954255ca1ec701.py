"""Rewritten program for Matrix multiplication 2,2,13"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 140  # 10 seconds before the 150s limit
    
    n, m, p = DIMENSIONS
    # Strategy: Block decomposition using Strassen for 2x2x2 blocks and schoolbook for 2x2x1.
    # p=13 -> 6 blocks of 2 columns + 1 column of 1.
    # Rank = 6 * 7 + 4 = 46.
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u, v, w):
        nonlocal col
        for i, val in u:
            U[i][col] += val
        for i, val in v:
            V[i][col] += val
        for i, val in w:
            W[i][col] += val
        col += 1

    # Strassen terms for 2x2x2 block at columns k0, k1
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: j*p + k
    # W indices: k*n + i
    
    for block_idx in range(6):
        k0 = block_idx * 2
        k1 = k0 + 1
        
        # M1: (A00+A11)(B00+B11)
        add_term(
            [(0, 1), (3, 1)], 
            [(k0, 1), (13+k1, 1)], 
            [(2*k0, 1), (2*k1+1, 1)]
        )
        
        # M2: (A10+A11)B00
        add_term(
            [(2, 1), (3, 1)], 
            [(k0, 1)], 
            [(2*k0+1, 1), (2*k1+1, -1)]
        )
        
        # M3: A00(B01-B11)
        add_term(
            [(0, 1)], 
            [(k1, 1), (13+k1, -1)], 
            [(2*k1, 1), (2*k1+1, 1)]
        )
        
        # M4: A11(B10-B00)
        add_term(
            [(3, 1)], 
            [(13+k0, 1), (k0, -1)], 
            [(2*k0, 1), (2*k0+1, 1)]
        )
        
        # M5: (A00+A01)B11
        add_term(
            [(0, 1), (1, 1)], 
            [(13+k1, 1)], 
            [(2*k0, -1), (2*k1, 1)]
        )
        
        # M6: (A10-A00)(B00+B01)
        add_term(
            [(2, 1), (0, -1)], 
            [(k0, 1), (k1, 1)], 
            [(2*k1+1, 1)]
        )
        
        # M7: (A01-A11)(B10+B11)
        add_term(
            [(1, 1), (3, -1)], 
            [(13+k0, 1), (13+k1, 1)], 
            [(2*k0, 1)]
        )

    # Last column k=12 (Schoolbook)
    k = 12
    add_term([(0, 1)], [(k, 1)], [(2*k, 1)])
    add_term([(1, 1)], [(13+k, 1)], [(2*k, 1)])
    add_term([(2, 1)], [(k, 1)], [(2*k+1, 1)])
    add_term([(3, 1)], [(13+k, 1)], [(2*k+1, 1)])

    if time.time() > deadline:
        # Return what we have so far (should be complete by now)
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
