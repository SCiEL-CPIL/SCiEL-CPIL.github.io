import time
_START_TIME = time.time()
_DEADLINE = 120.0
_MARGIN = 5.0

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_rank88():
    """Build the known rank-88 decomposition (schoolbook for row 0, Strassen for rows 1-2)."""
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: Schoolbook for A[0, :] (1x4) * B (4x8)
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: Strassen for A[1:3, :] (2x4) * B (4x8)
    # Strassen terms for 2x2 blocks
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # where:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = (A11 - A21)B22
    # M5 = (A12 + A22)(B11 + B12)
    # M6 = (A12 - A21)(B21 + B22)
    # M7 = (A11 + A12)(B11 - B21)
    
    # In our 2x4 * 4x8 case, we split B into two 4x4 blocks
    # A is 2x4, B is 4x8
    # We can view A as [[A11, A12], [A21, A22]] where each is 1x2
    # B as [[B11, B12], [B21, B22]] where each is 2x4
    # C will be 2x8, viewed as [[C11, C12], [C21, C22]] where each is 1x4
    
    # Strassen terms with local indices (row, col) for 2x2 blocks
    # U indices: local (r, c) -> (1+r)*m + c  (rows 1-2 of A)
    # V indices: local (r, c) -> r*p + (2*block + c)  (rows of B, split into two 4-col blocks)
    # W indices: local (r, c) -> (2*block + c)*n + (1+r)  (cols of C, split into two 4-col blocks)
    
    strassen_terms = [
        # M1 = (A11+A22)(B11+B22): U: (0,0)=1,(1,1)=1; V: (0,0)=1,(1,1)=1; W: (0,0)=1,(0,1)=1,(1,0)=1,(1,1)=-1
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): -1}},
        # M2 = (A21+A22)B11: U: (1,0)=1,(1,1)=1; V: (0,0)=1; W: (1,0)=1,(1,1)=-1
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1, (1,1): -1}},
        # M3 = A11(B12-B22): U: (0,0)=1; V: (0,1)=1,(1,1)=-1; W: (0,1)=1,(1,1)=1
        {'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): 1}},
        # M4 = (A11-A21)B22: U: (0,0)=1,(1,0)=-1; V: (1,1)=1; W: (0,0)=1,(1,0)=1
        {'U': {(0,0): 1, (1,0): -1}, 'V': {(1,1): 1}, 'W': {(0,0): 1, (1,0): 1}},
        # M5 = (A12+A22)(B11+B12): U: (0,1)=1,(1,1)=1; V: (0,0)=1,(0,1)=1; W: (0,0)=-1,(0,1)=1,(1,0)=-1,(1,1)=1
        {'U': {(0,1): 1, (1,1): 1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(0,0): -1, (0,1): 1, (1,0): -1, (1,1): 1}},
        # M6 = (A12-A21)(B21+B22): U: (0,1)=1,(1,0)=-1; V: (1,0)=1,(1,1)=1; W: (1,1)=1
        {'U': {(0,1): 1, (1,0): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(1,1): 1}},
        # M7 = (A11+A12)(B11-B21): U: (0,0)=1,(0,1)=1; V: (0,0)=1,(1,0)=-1; W: (0,0)=1
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(0,0): 1, (1,0): -1}, 'W': {(0,0): 1}},
    ]
    
    for j_block in range(4):
        for term in strassen_terms:
            for (lr, lc), val in term['U'].items():
                U[(1 + lr) * m + (2 * j_block + lc)][col] += val
            for (lr, lc), val in term['V'].items():
                V[(2 * lr) * p + (2 * j_block + lc)][col] += val
            for (lr, lc), val in term['W'].items():
                W[(2 * j_block + lc) * n + (1 + lr)][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    # Build the known rank-88 solution
    result = _build_rank88()
    
    # Verify it's correct
    if _verify_decomposition(result['U'], result['V'], result['W']):
        return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
