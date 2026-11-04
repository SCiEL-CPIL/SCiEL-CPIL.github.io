"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-83 exact tensor decomposition for M_{3,4,7}.
    
    Strategy: Start with the trivial rank-84 decomposition, then replace
    the 8 trivial terms corresponding to the 2x2x2 sub-block (i,j,k in {0,1})
    with Strassen's 7-term algorithm for 2x2 matrix multiplication.
    This reduces the rank by 1 to 83.
    
    The decomposition is exact: each C_{i,k} = sum_j A_{i,j} B_{j,k}
    is reproduced by the sum of rank-1 tensor products.
    """
    n, m, p = DIMENSIONS
    # n=3 rows of A, m=4 cols of A / rows of B, p=7 cols of B
    
    # Trivial rank would be n*m*p = 84
    # We replace 8 terms (2x2x2 sub-block) with 7 Strassen terms
    # Total rank = 84 - 8 + 7 = 83
    
    rank = 83
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 12 x 83
    V = [[0] * rank for _ in range(m * p)]  # 28 x 83
    W = [[0] * rank for _ in range(p * n)]  # 21 x 83
    
    # First, fill in the trivial terms, skipping the 2x2x2 sub-block
    # The sub-block is i in {0,1}, j in {0,1}, k in {0,1}
    skipped = set()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                skipped.add((i, j, k))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in skipped:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Now add the 7 Strassen terms for the 2x2x2 sub-block
    # Strassen's algorithm for C = A * B where A, B are 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For our indexing:
    # A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1] -> rows 0,1 of A, cols 0,1
    # B is 2x2: B[0,0], B[0,1], B[1,0], B[1,1] -> rows 0,1 of B, cols 0,1
    # C is 2x2: C[0,0], C[0,1], C[1,0], C[1,1] -> rows 0,1 of C, cols 0,1
    
    # In our tensor:
    # U index: i*m + j where i in {0,1}, j in {0,1} -> indices 0,1,4,5
    # V index: j*p + k where j in {0,1}, k in {0,1} -> indices 0,1,7,8
    # W index: k*n + i where k in {0,1}, i in {0,1} -> indices 0,1,3,4
    
    # Strassen terms:
    # M1 = (A00 + A11)(B00 + B11)
    #   U: A00=1, A11=1 -> U[0]=1, U[5]=1
    #   V: B00=1, B11=1 -> V[0]=1, V[8]=1
    #   W: C00=1, C11=1 -> W[0]=1, W[4]=1
    U[0][col] = 1; U[5][col] = 1
    V[0][col] = 1; V[8][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A10 + A11)B00
    #   U: A10=1, A11=1 -> U[4]=1, U[5]=1
    #   V: B00=1 -> V[0]=1
    #   W: C10=1 -> W[3]=1
    U[4][col] = 1; U[5][col] = 1
    V[0][col] = 1
    W[3][col] = 1
    col += 1
    
    # M3 = A00(B01 - B11)
    #   U: A00=1 -> U[0]=1
    #   V: B01=1, B11=-1 -> V[1]=1, V[8]=-1
    #   W: C01=1 -> W[1]=1
    U[0][col] = 1
    V[1][col] = 1; V[8][col] = -1
    W[1][col] = 1
    col += 1
    
    # M4 = A11(B10 - B00)
    #   U: A11=1 -> U[5]=1
    #   V: B10=1, B00=-1 -> V[7]=1, V[0]=-1
    #   W: C10=1 -> W[3]=1
    U[5][col] = 1
    V[7][col] = 1; V[0][col] = -1
    W[3][col] = 1
    col += 1
    
    # M5 = (A00 + A01)B11
    #   U: A00=1, A01=1 -> U[0]=1, U[1]=1
    #   V: B11=1 -> V[8]=1
    #   W: C11=1 -> W[4]=1
    U[0][col] = 1; U[1][col] = 1
    V[8][col] = 1
    W[4][col] = 1
    col += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    #   U: A10=1, A00=-1 -> U[4]=1, U[0]=-1
    #   V: B00=1, B01=1 -> V[0]=1, V[1]=1
    #   W: C11=1 -> W[4]=1
    U[4][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    #   U: A01=1, A11=-1 -> U[1]=1, U[5]=-1
    #   V: B10=1, B11=1 -> V[7]=1, V[8]=1
    #   W: C00=1 -> W[0]=1
    U[1][col] = 1; U[5][col] = -1
    V[7][col] = 1; V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    # Verify that we used exactly rank columns
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Quick verification that the decomposition is correct
    # Check all valid (i,j,k) triples give sum = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if s != 1:
                    raise ValueError(f"Invalid decomposition at (i={i}, j={j}, k={k}): got {s}")
    
    # Check that invalid combinations give 0
    # Sample some invalid combinations for speed
    import random
    random.seed(42)
    for _ in range(1000):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_c = c // n
        i_c = c % n
        
        is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
        
        s = 0
        for r in range(rank):
            s += U[a][r] * V[b][r] * W[c][r]
        
        if is_valid and s != 1:
            raise ValueError(f"Invalid decomposition at valid triple: got {s}")
        if not is_valid and s != 0:
            raise ValueError(f"Invalid decomposition at invalid triple: got {s}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
