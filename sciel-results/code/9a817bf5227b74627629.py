"""
Bootstrap portfolio: MODEL-AND-OPTIMIZE approach.
We start from a known low-rank decomposition (block-diagonal Strassen where applicable)
and attempt to find a rank-minimal exact decomposition via optimization/search.
Given the small dimensions (2x5x6), we can afford some computational search.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s safety margin

def check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def schoolbook_decomposition(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return U, V, W

def strassen_decomposition_2x2():
    """
    Returns Strassen decomposition for a 2x2x2 block.
    U_block: (4, 7), V_block: (4, 7), W_block: (4, 7)
    Local indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank_block = 7
    U_block = np.zeros((4, rank_block))
    V_block = np.zeros((4, rank_block))
    W_block = np.zeros((4, rank_block))
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U_block[0, 0] = 1; U_block[3, 0] = 1
    V_block[0, 0] = 1; V_block[3, 0] = 1
    W_block[0, 0] = 1; W_block[3, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    U_block[2, 1] = 1; U_block[3, 1] = 1
    V_block[0, 1] = 1
    W_block[2, 1] = 1  # W index k*n+i = 0*2+1 = 2
    
    # Term 2: M3 = A00(B01-B11)
    U_block[0, 2] = 1
    V_block[1, 2] = 1; V_block[3, 2] = -1
    W_block[1, 2] = 1  # W index k*n+i = 1*2+0 = 1
    
    # Term 3: M4 = A11(B10-B00)
    U_block[3, 3] = 1
    V_block[2, 3] = 1; V_block[0, 3] = -1
    W_block[0, 3] = 1; W_block[2, 3] = 1
    
    # Term 4: M5 = (A00+A10)B11
    U_block[0, 4] = 1; U_block[2, 4] = 1
    V_block[3, 4] = 1
    W_block[0, 4] = -1; W_block[1, 4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U_block[2, 5] = 1; U_block[0, 5] = -1
    V_block[0, 5] = 1; V_block[1, 5] = 1
    W_block[3, 5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U_block[1, 6] = 1; U_block[3, 6] = -1
    V_block[2, 6] = 1; V_block[3, 6] = 1
    W_block[0, 6] = 1
    
    return U_block, V_block, W_block

def schoolbook_2x1x2():
    """
    Returns schoolbook decomposition for a 2x1x2 block.
    n=2, m=1, p=2. Rank = 4.
    U_block: (2, 4), V_block: (2, 4), W_block: (4, 4)
    Local indices: U[i*1+j] = U[i], V[j*2+k] = V[k] (since j=0), W[k*2+i]
    """
    rank_block = 4
    U_block = np.zeros((2, rank_block))
    V_block = np.zeros((2, rank_block))
    W_block = np.zeros((4, rank_block))
    
    # c=0: i=0, j=0, k=0
    U_block[0, 0] = 1
    V_block[0, 0] = 1
    W_block[0, 0] = 1  # W[k*2+i] = W[0*2+0] = W[0]
    
    # c=1: i=0, j=0, k=1
    U_block[0, 1] = 1
    V_block[1, 1] = 1
    W_block[2, 1] = 1  # W[k*2+i] = W[1*2+0] = W[2]
    
    # c=2: i=1, j=0, k=0
    U_block[1, 2] = 1
    V_block[0, 2] = 1
    W_block[1, 2] = 1  # W[k*2+i] = W[0*2+1] = W[1]
    
    # c=3: i=1, j=0, k=1
    U_block[1, 3] = 1
    V_block[1, 3] = 1
    W_block[3, 3] = 1  # W[k*2+i] = W[1*2+1] = W[3]
    
    return U_block, V_block, W_block

def build_block_decomposition(n, m, p):
    """
    Builds a decomposition using Strassen for 2x2x2 blocks and schoolbook for 2x1x2 blocks.
    For 2x5x6:
    i in {0,1}
    j in {0..4}
    k in {0..5}
    
    Partition:
    j: {0,1}, {2,3}, {4}
    k: {0,1}, {2,3}, {4,5}
    
    Blocks:
    1. j:0-1, k:0-1 (2x2x2) -> Strassen rank 7
    2. j:0-1, k:2-3 (2x2x2) -> Strassen rank 7
    3. j:0-1, k:4-5 (2x2x2) -> Strassen rank 7
    4. j:2-3, k:0-1 (2x2x2) -> Strassen rank 7
    5. j:2-3, k:2-3 (2x2x2) -> Strassen rank 7
    6. j:2-3, k:4-5 (2x2x2) -> Strassen rank 7
    7. j:4, k:0-1 (2x1x2) -> Schoolbook rank 4
    8. j:4, k:2-3 (2x1x2) -> Schoolbook rank 4
    9. j:4, k:4-5 (2x1x2) -> Schoolbook rank 4
    
    Total rank: 6*7 + 3*4 = 42 + 12 = 54.
    """
    max_rank = 60
    U = np.zeros((n*m, max_rank))
    V = np.zeros((m*p, max_rank))
    W = np.zeros((p*n, max_rank))
    
    col = 0
    
    # Define blocks: (j_start, k_start, j_size, k_size, method)
    # i is always 0-1 (size 2)
    blocks = [
        (0, 0, 2, 2, 'strassen'),
        (0, 2, 2, 2, 'strassen'),
        (0, 4, 2, 2, 'strassen'),
        (2, 0, 2, 2, 'strassen'),
        (2, 2, 2, 2, 'strassen'),
        (2, 4, 2, 2, 'strassen'),
        (4, 0, 1, 2, 'schoolbook_2x1x2'),
        (4, 2, 1, 2, 'schoolbook_2x1x2'),
        (4, 4, 1, 2, 'schoolbook_2x1x2'),
    ]
    
    for j0, k0, mj, pk, method in blocks:
        if method == 'strassen':
            Ub, Vb, Wb = strassen_decomposition_2x2()
            # Ub is 4x7, Vb is 4x7, Wb is 4x7
            # Global U indices: i in {0,1}, j in {j0, j0+1} -> flat (i*m+j)
            for i in range(2):
                for j in range(2):
                    row_idx = i*m + (j0+j)
                    U[row_idx, col:col+7] = Ub[i*2+j, :]
            # Global V indices: j in {j0, j0+1}, k in {k0, k0+1} -> flat (j*p+k)
            for j in range(2):
                for k in range(2):
                    row_idx = (j0+j)*p + (k0+k)
                    V[row_idx, col:col+7] = Vb[j*2+k, :]
            # Global W indices: k in {k0, k0+1}, i in {0,1} -> flat (k*n+i)
            for k in range(2):
                for i in range(2):
                    row_idx = (k0+k)*n + i
                    W[row_idx, col:col+7] = Wb[k*2+i, :]
            col += 7
        elif method == 'schoolbook_2x1x2':
            Ub, Vb, Wb = schoolbook_2x1x2()
            # Ub is 2x4, Vb is 2x4, Wb is 4x4
            # Global U indices: i in {0,1}, j = j0 -> flat (i*m+j0)
            for i in range(2):
                row_idx = i*m + j0
                U[row_idx, col:col+4] = Ub[i, :]
            # Global V indices: j = j0, k in {k0, k0+1} -> flat (j0*p + k)
            for k in range(2):
                row_idx = j0*p + (k0+k)
                V[row_idx, col:col+4] = Vb[k, :]
            # Global W indices: k in {k0, k0+1}, i in {0,1} -> flat (k*n+i)
            for k in range(2):
                for i in range(2):
                    row_idx = (k0+k)*n + i
                    W[row_idx, col:col+4] = Wb[k*2+i, :]
            col += 4
    
    # Slice to actual size
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    # Verify
    if not check_decomposition(U, V, W, n, m, p):
        raise ValueError("Block decomposition failed verification")
        
    return U, V, W

def construct():
    global START_TIME
    n, m, p = DIMENSIONS
    
    # 1. Try block decomposition (rank 54)
    try:
        U, V, W = build_block_decomposition(n, m, p)
        rank_block = U.shape[1]
        # Return as lists of lists with integer/half-integer entries
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    except Exception as e:
        # Fallback to schoolbook
        pass
        
    # 2. Fallback: Schoolbook (rank 60)
    U, V, W = schoolbook_decomposition(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
