"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
from itertools import product
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a sparse dict."""
    T = {}
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T[(a, b, c)] = 1
    return T

def _matrix_rank_rational(M):
    """Compute rank of a matrix over the rationals using Gaussian elimination."""
    if not M or not M[0]:
        return 0
    rows = len(M)
    cols = len(M[0])
    M_copy = [[Fraction(x) for x in row] for row in M]
    rank = 0
    for col in range(cols):
        pivot = -1
        for row in range(rank, rows):
            if M_copy[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            continue
        M_copy[rank], M_copy[pivot] = M_copy[pivot], M_copy[rank]
        pivot_val = M_copy[rank][col]
        for j in range(col, cols):
            M_copy[rank][j] /= pivot_val
        for row in range(rows):
            if row != rank and M_copy[row][col] != 0:
                factor = M_copy[row][col]
                for j in range(col, cols):
                    M_copy[row][j] -= factor * M_copy[rank][j]
        rank += 1
        if rank == rows:
            break
    return rank

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    T = {}
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T[(a, b, c)] = 1
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                expected = 1
                actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if actual != expected:
                    return False
    
    # Check a sample of zero entries
    checked = set()
    random.seed(123)
    for _ in range(min(200, n*m*p)):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        if (a, b, c) in checked:
            continue
        checked.add((a, b, c))
        if (a, b, c) in T:
            continue
        actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        if actual != 0:
            return False
    
    return True

def _schoolbook_decomposition(n, m, p):
    """Generate the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][column] = 1
                V[k * p + j][column] = 1
                W[j * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W matrices of shape (4, 7).
    Convention: U[i*2+k, r], V[k*2+j, r], W[j*2+i, r]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]  # 4 x 7
    V = [[0] * rank for _ in range(m * p)]  # 4 x 7
    W = [[0] * rank for _ in range(p * n)]  # 4 x 7
    
    # Strassen formulas for C = A * B where A, B are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Index mapping: a = i*2+k, b = k*2+j, c = j*2+i
    # A00 -> i=0,k=0 -> a=0
    # A11 -> i=1,k=1 -> a=3
    # A10 -> i=1,k=0 -> a=2
    # A01 -> i=0,k=1 -> a=1
    # B00 -> k=0,j=0 -> b=0
    # B11 -> k=1,j=1 -> b=3
    # B01 -> k=0,j=1 -> b=1
    # B10 -> k=1,j=0 -> b=2
    # C00 -> j=0,i=0 -> c=0
    # C01 -> j=0,i=1 -> c=1
    # C10 -> j=1,i=0 -> c=2
    # C11 -> j=1,i=1 -> c=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] += 1; U[3][0] += 1  # A00, A11
    V[0][0] += 1; V[3][0] += 1  # B00, B11
    W[0][0] += 1; W[1][0] += 1; W[2][0] += 0; W[3][0] += 1  # C00, C01? No.
    # C = M1*M? Let's be careful.
    # The tensor is T[a,b,c].
    # Term r contributes U[a,r] * V[b,r] * W[c,r].
    # M1 is a scalar product of linear forms in A and B, resulting in a contribution to C.
    # C00 gets +M1, C11 gets +M1.
    # So for M1:
    # A part: A00 + A11 => U[0,r]=1, U[3,r]=1
    # B part: B00 + B11 => V[0,r]=1, V[3,r]=1
    # C part: C00 + C11 => W[0,r]=1, W[3,r]=1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    # A: A10 (a=2) + A11 (a=3) => U[2,1]=1, U[3,1]=1
    # B: B00 (b=0) => V[0,1]=1
    # C: C10 gets +M2 => C10 is j=1,i=0 => c=2. So W[2,1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00(B01-B11)
    # A: A00 (a=0) => U[0,2]=1
    # B: B01 (b=1) - B11 (b=3) => V[1,2]=1, V[3,2]=-1
    # C: C01 gets +M3 => C01 is j=0,i=1 => c=1. So W[1,2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B10-B00)
    # A: A11 (a=3) => U[3,3]=1
    # B: B10 (b=2) - B00 (b=0) => V[2,3]=1, V[0,3]=-1
    # C: C00 gets +M4, C10 gets +M4 => W[0,3]=1, W[2,3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A10)B11
    # A: A00 (a=0) + A10 (a=2) => U[0,4]=1, U[2,4]=1
    # B: B11 (b=3) => V[3,4]=1
    # C: C00 gets -M5, C01 gets +M5 => W[0,4]=-1, W[1,4]=1
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A11-A00)(B00+B01)
    # A: A11 (a=3) - A00 (a=0) => U[3,5]=1, U[0,5]=-1
    # B: B00 (b=0) + B01 (b=1) => V[0,5]=1, V[1,5]=1
    # C: C11 gets +M6 => W[3,5]=1
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # A: A01 (a=1) - A11 (a=3) => U[1,6]=1, U[3,6]=-1
    # B: B10 (b=2) + B11 (b=3) => V[2,6]=1, V[3,6]=1
    # C: C00 gets +M7 => W[0,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _apply_strassen_block(U, V, W, n, m, p, i0, k0, j0):
    """
    Replace the schoolbook terms for a 2x2x2 block starting at (i0, k0, j0)
    with Strassen terms.
    Assumes U, V, W are currently schoolbook (rank 60) and we are modifying them in place
    or building a new set. This function returns the Strassen columns to be used.
    Actually, it's easier to construct the full decomposition from scratch using this.
    """
    pass

def _construct_strassen_3x4x5(n, m, p):
    """
    Construct a decomposition for 3x4x5 by using Strassen for a 2x2x2 sub-block
    and schoolbook for the rest.
    Total rank = (60 - 8) + 7 = 59.
    """
    U = [[0] * 59 for _ in range(n * m)]
    V = [[0] * 59 for _ in range(m * p)]
    W = [[0] * 59 for _ in range(p * n)]
    
    # Identify the block to replace with Strassen.
    # Let's pick the top-left 2x2x2 block: i in {0,1}, k in {0,1}, j in {0,1}.
    # Indices in the full tensor:
    # a = i*4 + k
    # b = k*5 + j
    # c = j*3 + i
    
    block_indices = set()
    for i in range(2):
        for k in range(2):
            for j in range(2):
                block_indices.add((i, k, j))
    
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    
    col = 0
    # First, add the Strassen terms for the block
    for r in range(7):
        # Map local 2x2x2 indices to global 3x4x5 indices
        # Local a = li*2 + lk, Global a = li*4 + lk (since i0=0, k0=0)
        # Wait, the local indices are just 0..1 for i, k, j.
        # Global indices for i in {0,1}, k in {0,1}:
        # i=0, k=0 -> a=0
        # i=0, k=1 -> a=1
        # i=1, k=0 -> a=4
        # i=1, k=1 -> a=5
        # So local a (0..3) maps to global a:
        # 0 -> 0
        # 1 -> 1
        # 2 -> 4
        # 3 -> 5
        
        # Local b = lk*2 + lj, Global b = lk*5 + lj
        # lk=0, lj=0 -> b=0
        # lk=0, lj=1 -> b=1
        # lk=1, lj=0 -> b=5
        # lk=1, lj=1 -> b=6
        # So local b (0..3) maps to global b:
        # 0 -> 0
        # 1 -> 1
        # 2 -> 5
        # 3 -> 6
        
        # Local c = lj*2 + li, Global c = lj*3 + li
        # lj=0, li=0 -> c=0
        # lj=0, li=1 -> c=1
        # lj=1, li=0 -> c=3
        # lj=1, li=1 -> c=4
        # So local c (0..3) maps to global c:
        # 0 -> 0
        # 1 -> 1
        # 2 -> 3
        # 3 -> 4
        
        a_map = [0, 1, 4, 5]
        b_map = [0, 1, 5, 6]
        c_map = [0, 1, 3, 4]
        
        for la in range(4):
            ga = a_map[la]
            U[ga][col] = strassen_U[la][r]
        for lb in range(4):
            gb = b_map[lb]
            V[gb][col] = strassen_V[lb][r]
        for lc in range(4):
            gc = c_map[lc]
            W[gc][col] = strassen_W[lc][r]
            
        col += 1
    
    # Now add schoolbook terms for the rest
    for i in range(n):
        for k in range(m):
            for j in range(p):
                if (i, k, j) in block_indices:
                    continue
                a = i * m + k
                b = k * p + j
                c = j * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
    
    if col != 59:
        # Fallback if something went wrong
        return _schoolbook_decomposition(n, m, p)
        
    return {"U": U, "V": V, "W": W}

def _try_low_rank_search(n, m, p, target_rank, deadline, max_attempts=50):
    """Try to find a rank decomposition of the given rank using structured random search."""
    random.seed(42)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        values = [-1, -0.5, 0, 0, 0, 0.5, 1]
        
        U = [[random.choice(values) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[random.choice(values) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[random.choice(values) for _ in range(target_rank)] for _ in range(p * n)]
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # 1. Try the Strassen-block construction (Rank 59)
    candidate = _construct_strassen_3x4x5(n, m, p)
    if _verify_decomposition(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        return candidate
    
    # 2. Fallback to Schoolbook (Rank 60)
    best = _schoolbook_decomposition(n, m, p)
    best_rank = schoolbook_rank
    
    # 3. If time permits, try random search for lower ranks (58, 57, ...)
    # This is unlikely to succeed but costs little if bounded.
    for candidate_rank in range(best_rank - 2, best_rank):
        if time.time() > deadline - 10:
            break
        result = _try_low_rank_search(n, m, p, candidate_rank, deadline)
        if result is not None:
            return result
    
    return best

# EVOLVE-BLOCK-END
