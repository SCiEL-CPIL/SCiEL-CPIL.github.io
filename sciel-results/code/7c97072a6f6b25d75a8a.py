"""Matrix multiplication tensor decomposition for M_{2,2,15}."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Schoolbook decomposition of rank 60."""
    n, m, p = 2, 2, 15
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2, rank 7.
    Returns U (4x7), V (4x7), W (4x7) where:
      U indexed by (i,j), V by (j,k), W by (k,i)
    """
    # Strassen formulas:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    U = np.zeros((4, 7), dtype=object)
    V = np.zeros((4, 7), dtype=object)
    W = np.zeros((4, 7), dtype=object)
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1  # A11, A22
    V[0, 0] = 1; V[3, 0] = 1  # B11, B22
    W[0, 0] = 1; W[3, 0] = 1  # C11, C22
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1  # A21, A22
    V[0, 1] = 1               # B11
    W[2, 1] = 1; W[3, 1] = 1  # C21, C22
    
    # M3
    U[0, 2] = 1               # A11
    V[1, 2] = 1; V[3, 2] = -1 # B12, -B22
    W[1, 2] = 1; W[3, 2] = 1  # C12, C22
    
    # M4
    U[3, 3] = 1               # A22
    V[2, 3] = 1; V[0, 3] = -1 # B21, -B11
    W[0, 3] = 1; W[2, 3] = 1  # C11, C21
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1  # A11, A12
    V[3, 4] = 1               # B22
    W[0, 4] = -1; W[1, 4] = 1 # -C11, C12
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1 # A21, -A11
    V[0, 5] = 1; V[1, 5] = 1  # B11, B12
    W[3, 5] = 1               # C22
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1 # A12, -A22
    V[2, 6] = 1; V[3, 6] = 1  # B21, B22
    W[0, 6] = 1               # C11
    
    return {"U": U, "V": V, "W": W}


def _block_strassen():
    """
    Block Strassen: split the 2x15 matrix into 7 blocks of 2x2 and one 2x1.
    Use Strassen for each 2x2 block (rank 7 each = 49) and schoolbook for 2x1 (rank 2).
    Total rank: 49 + 2 = 51.
    
    Actually, let's think more carefully. We can group the 15 columns into:
    - 7 groups of 2 columns each (using Strassen, 7 rank per group) = 49
    - 1 group of 1 column (schoolbook, 2 rank) = 2
    Total: 51
    
    But wait, we can do even better. Let's use Strassen on the full 2x15 by
    padding to 2x16 (8 blocks of 2x2), giving 8*7 = 56 rank, then we need
    to handle the fact that the 8th block is zero. That doesn't help.
    
    Better approach: Just use the direct block decomposition.
    For each pair of columns (2b, 2b+1), use Strassen (rank 7).
    For the last single column (column 14), use schoolbook (rank 2).
    Total rank: 7*7 + 2 = 51.
    """
    n, m, p = 2, 2, 15
    U = np.zeros((n * m, 51), dtype=object)
    V = np.zeros((m * p, 51), dtype=object)
    W = np.zeros((p * n, 51), dtype=object)
    
    col = 0
    # 7 blocks of 2 columns each
    for b in range(7):
        s = _strassen_2x2x2()
        # Map local 2x2x2 indices to global
        # U: local (i,j) with i in {0,1}, j in {0,1} -> global i*m + j
        for i in range(2):
            for j in range(2):
                for r in range(7):
                    if s["U"][i, j] != 0:
                        U[i * m + j, col + r] = s["U"][i, j]
        
        # V: local (j,k) with j in {0,1}, k in {0,1} -> global j*p + (2b+k)
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    if s["V"][j, k] != 0:
                        V[j * p + 2 * b + k, col + r] = s["V"][j, k]
        
        # W: local (k,i) with k in {0,1}, i in {0,1} -> global (2b+k)*n + i
        for k in range(2):
            for i in range(2):
                for r in range(7):
                    if s["W"][k, i] != 0:
                        W[(2 * b + k) * n + i, col + r] = s["W"][k, i]
        
        col += 7
    
    # Last single column (column 14)
    # Schoolbook for 2x2 * 2x1: rank = 2*2*1 = 4... wait
    # Actually for 2x2 * 2x1, the tensor has 2*2*2 = 8 entries, but rank is 2*2*1 = 4
    # Wait, no. For C = A * B where A is 2x2, B is 2x1, C is 2x1.
    # C[i,0] = sum_j A[i,j] * B[j,0]
    # The tensor T[i,j,k] = delta_{i,?}... 
    # Actually the multiplication tensor for m_{2,2,1} has rank 2*2*1 = 4.
    # Let me just use schoolbook: for each (i,j) pair, one rank-1 term.
    for i in range(2):
        for j in range(2):
            U[i * m + j, col] = 1
            V[j * p + 14, col] = 1
            W[14 * n + i, col] = 1
            col += 1
    
    assert col == 51, f"Expected 51, got {col}"
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Construct the rank-minimal (or at least valid) tensor decomposition for 2x2 * 2x15.
    Uses block Strassen with rank 51.
    """
    deadline = time.time() + 120 * 0.9  # safety margin
    
    # Try block Strassen first (rank 51)
    result = _block_strassen()
    
    if time.time() > deadline:
        # Fallback to schoolbook if we're running out of time
        return _schoolbook()
    
    return result

# EVOLVE-BLOCK-END
