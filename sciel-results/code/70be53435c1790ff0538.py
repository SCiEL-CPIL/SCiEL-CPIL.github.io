"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _add_strassen_2x2x2(U, V, W, u_offset, v_offset, w_offset, rank_offset):
    """
    Adds a 2x2x2 Strassen decomposition to U, V, W.
    U rows: i in {0,1}, j in {0,1} -> indices u_offset + i*2 + j
    V rows: j in {0,1}, k in {0,1} -> indices v_offset + j*2 + k
    W rows: k in {0,1}, i in {0,1} -> indices w_offset + k*2 + i
    """
    # Strassen 7 products
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a10)b11
    # M6 = (a01-a00)(b00+b01)
    # M7 = (a11-a10)(b10+b11)
    
    # U, V, W contributions for each product
    # Product 1
    U[u_offset + 0][rank_offset] = 1
    U[u_offset + 3][rank_offset] = 1 # a11
    V[v_offset + 0][rank_offset] = 1
    V[v_offset + 3][rank_offset] = 1 # b11
    W[w_offset + 0][rank_offset] = 1 # c00
    W[w_offset + 2][rank_offset] = 1 # c10
    W[w_offset + 1][rank_offset] = 1 # c01
    W[w_offset + 3][rank_offset] = 1 # c11
    # Wait, W indices are k*n+i.
    # c00 -> k=0, i=0 -> idx 0
    # c01 -> k=1, i=0 -> idx 2
    # c10 -> k=0, i=1 -> idx 1
    # c11 -> k=1, i=1 -> idx 3
    
    # Correct W mapping for Strassen:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Let's define the coefficients for W for each M
    # M1: c00(+1), c11(+1)
    # M2: c10(+1), c11(-1)
    # M3: c01(+1), c11(+1)
    # M4: c00(+1), c10(+1)
    # M5: c00(-1), c01(+1)
    # M6: c11(+1)
    # M7: c00(+1)
    
    # Re-implementing properly
    U[u_offset + 0][rank_offset] = 1 # a00
    U[u_offset + 3][rank_offset] = 1 # a11
    V[v_offset + 0][rank_offset] = 1 # b00
    V[v_offset + 3][rank_offset] = 1 # b11
    W[w_offset + 0][rank_offset] = 1 # c00
    W[w_offset + 3][rank_offset] = 1 # c11

    rank_offset += 1
    
    # M2 = (a10+a11)b00
    U[u_offset + 2][rank_offset] = 1 # a10
    U[u_offset + 3][rank_offset] = 1 # a11
    V[v_offset + 0][rank_offset] = 1 # b00
    W[w_offset + 1][rank_offset] = 1 # c10
    W[w_offset + 3][rank_offset] = -1 # c11

    rank_offset += 1
    
    # M3 = a00(b01-b11)
    U[u_offset + 0][rank_offset] = 1 # a00
    V[v_offset + 1][rank_offset] = 1 # b01
    V[v_offset + 3][rank_offset] = -1 # b11
    W[w_offset + 2][rank_offset] = 1 # c01
    W[w_offset + 3][rank_offset] = 1 # c11

    rank_offset += 1
    
    # M4 = a11(b10-b00)
    U[u_offset + 3][rank_offset] = 1 # a11
    V[v_offset + 2][rank_offset] = 1 # b10
    V[v_offset + 0][rank_offset] = -1 # b00
    W[w_offset + 0][rank_offset] = 1 # c00
    W[w_offset + 1][rank_offset] = 1 # c10

    rank_offset += 1
    
    # M5 = (a00+a10)b11
    U[u_offset + 0][rank_offset] = 1 # a00
    U[u_offset + 2][rank_offset] = 1 # a10
    V[v_offset + 3][rank_offset] = 1 # b11
    W[w_offset + 0][rank_offset] = -1 # c00
    W[w_offset + 2][rank_offset] = 1 # c01

    rank_offset += 1
    
    # M6 = (a01-a00)(b00+b01)
    U[u_offset + 1][rank_offset] = 1 # a01
    U[u_offset + 0][rank_offset] = -1 # a00
    V[v_offset + 0][rank_offset] = 1 # b00
    V[v_offset + 1][rank_offset] = 1 # b01
    W[w_offset + 3][rank_offset] = 1 # c11

    rank_offset += 1
    
    # M7 = (a11-a10)(b10+b11)
    U[u_offset + 3][rank_offset] = 1 # a11
    U[u_offset + 2][rank_offset] = -1 # a10
    V[v_offset + 2][rank_offset] = 1 # b10
    V[v_offset + 3][rank_offset] = 1 # b11
    W[w_offset + 0][rank_offset] = 1 # c00
    
    return rank_offset

def _add_outer_2x1x2(U, V, W, u_offset, v_offset, w_offset, rank_offset):
    """
    Adds a 2x1x2 outer product decomposition (schoolbook).
    A is 2x1 (col 2 of A), B is 1x2 (row 2 of B).
    U rows: i in {0,1}, j=2 -> u_offset + i*3 + 2
    V rows: j=2, k in {0,1} -> v_offset + 2*2 + k
    W rows: k in {0,1}, i in {0,1} -> w_offset + k*2 + i
    """
    # c00 = a02 b20
    U[u_offset + 2][rank_offset] = 1
    V[v_offset + 4][rank_offset] = 1
    W[w_offset + 0][rank_offset] = 1
    rank_offset += 1
    
    # c01 = a02 b21
    U[u_offset + 2][rank_offset] = 1
    V[v_offset + 5][rank_offset] = 1
    W[w_offset + 2][rank_offset] = 1
    rank_offset += 1
    
    # c10 = a12 b20
    U[u_offset + 5][rank_offset] = 1
    V[v_offset + 4][rank_offset] = 1
    W[w_offset + 1][rank_offset] = 1
    rank_offset += 1
    
    # c11 = a12 b21
    U[u_offset + 5][rank_offset] = 1
    V[v_offset + 5][rank_offset] = 1
    W[w_offset + 3][rank_offset] = 1
    rank_offset += 1
    
    return rank_offset

def _add_schoolbook_2x3x1(U, V, W, u_offset, v_offset, w_offset, rank_offset):
    """
    Adds a 2x3x1 schoolbook decomposition.
    A is 2x3, B is 3x1.
    U rows: i in {0,1}, j in {0,1,2}
    V rows: j in {0,1,2}, k=0
    W rows: k=0, i in {0,1}
    """
    for i in range(2):
        for j in range(3):
            U[u_offset + i*3 + j][rank_offset] = 1
            V[v_offset + j*1 + 0][rank_offset] = 1
            W[w_offset + 0*2 + i][rank_offset] = 1
            rank_offset += 1
    return rank_offset

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Total rank estimate:
    # Block 1 (k=0,1): 11
    # Block 2 (k=2,3): 11
    # Block 3 (k=4,5,6): 11 (for k=4,5) + 6 (for k=6) = 17
    # Total: 39
    
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    rank_offset = 0
    
    # Block 1: k=0,1
    # u_offset = 0, v_offset = 0, w_offset = 0
    rank_offset = _add_strassen_2x2x2(U, V, W, 0, 0, 0, rank_offset)
    rank_offset = _add_outer_2x1x2(U, V, W, 0, 0, 0, rank_offset)
    
    # Block 2: k=2,3
    # v_offset = 2*2 = 4 (j*2+k, so j=0..2, k=2..3 -> offset 4 for j=0,k=2)
    # Actually V index is j*p + k. p=7.
    # For block 1, k=0,1. V rows used: j*7 + 0, j*7 + 1.
    # For block 2, k=2,3. V rows used: j*7 + 2, j*7 + 3.
    # W index is k*n + i.
    # For block 1, k=0,1. W rows used: 0*2+i, 1*2+i.
    # For block 2, k=2,3. W rows used: 2*2+i, 3*2+i.
    
    # Update offsets
    # U is same for all blocks (A is same).
    # V offset for j=0 is 0*7 + 2 = 2.
    # W offset for k=2 is 2*2 + 0 = 4.
    
    rank_offset = _add_strassen_2x2x2(U, V, W, 0, 2, 4, rank_offset)
    rank_offset = _add_outer_2x1x2(U, V, W, 0, 2, 4, rank_offset)
    
    # Block 3: k=4,5,6
    # Split into k=4,5 (rank 11) and k=6 (rank 6)
    
    # Sub-block 3a: k=4,5
    # V offset: j*7 + 4.
    # W offset: k*2 + i. k=4 -> 8.
    
    rank_offset = _add_strassen_2x2x2(U, V, W, 0, 4, 8, rank_offset)
    rank_offset = _add_outer_2x1x2(U, V, W, 0, 4, 8, rank_offset)
    
    # Sub-block 3b: k=6
    # V offset: j*7 + 6.
    # W offset: k*2 + i. k=6 -> 12.
    
    rank_offset = _add_schoolbook_2x3x1(U, V, W, 0, 6, 12, rank_offset)
    
    # Verify rank_offset matches rank
    assert rank_offset == rank, f"Rank mismatch: {rank_offset} vs {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
