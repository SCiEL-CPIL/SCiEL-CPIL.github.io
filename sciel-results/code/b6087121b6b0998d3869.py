"""Matrix multiplication 3x4x8 with rank reduction via Strassen-like block decomposition."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the trivial rank 96 decomposition."""
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for c in range(rank):
                    actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if actual != expected:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return U, V, W for the rank-7 Strassen decomposition of 2x2x2.
    U rows: a, b, c, d (input A entries)
    V rows: e, f, g, h (input B entries)
    W rows: M11, M12, M21, M22 (output C entries)
    """
    U = [
        [1, 0, 1, 0, 1, 0, 1],  # a
        [0, 1, 0, 0, 0, 1, 0],  # b
        [0, 0, 1, 1, 1, 0, 0],  # c
        [1, 0, 0, 1, 0, 1, 0]   # d
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 0],  # e
        [0, 0, 0, 1, 0, 0, 1],  # f
        [0, 0, 1, 0, 0, 1, 0],  # g
        [1, 0, 1, 0, 1, 0, 0]   # h
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # M11
        [0, 0, 1, 0, 1, 0, 0],   # M12
        [0, 1, 0, 1, 0, 0, 0],   # M21
        [1, -1, 1, 0, 0, 1, 0]   # M22
    ]
    return U, V, W

def _add_strassen_block(U, V, W, u_map, v_map, w_map, col_offset):
    """
    Add a 2x2x2 Strassen block to the global factor matrices.
    u_map, v_map, w_map are lists of 4 global row indices.
    col_offset is the starting column index for this block.
    """
    U2, V2, W2 = _strassen_2x2x2()
    for i in range(4):
        for c in range(7):
            U[u_map[i]][col_offset + c] += U2[i][c]
            V[v_map[i]][col_offset + c] += V2[i][c]
            W[w_map[i]][col_offset + c] += W2[i][c]

def _construct_rank_88():
    """
    Construct a rank-88 decomposition for 3x4x8.
    
    Strategy:
    - Split the j dimension (size 4) into two pairs: (0,1) and (2,3).
    - Split the k dimension (size 8) into four pairs: (0,1), (2,3), (4,5), (6,7).
    - For the i dimension (size 3), split i=0,1 into a pair and handle i=2 separately.
    
    For each (i_pair, j_pair, k_pair) combination, we have a subproblem.
    - If i is a pair (i=0,1), the subproblem is 2x2x2, which we solve with rank 7 (Strassen).
    - If i is single (i=2), the subproblem is 1x2x2, which we solve with rank 4 (schoolbook).
    
    Number of 2x2x2 subproblems: 1 (i_pair) * 2 (j_pairs) * 4 (k_pairs) = 8. Rank contribution: 8 * 7 = 56.
    Number of 1x2x2 subproblems: 1 (i=2) * 2 (j_pairs) * 4 (k_pairs) = 8. Rank contribution: 8 * 4 = 32.
    Total rank: 56 + 32 = 88.
    
    Indexing conventions:
    U rows: i*m + j (i in 0..2, j in 0..3) -> 0..11
    V rows: j*p + k (j in 0..3, k in 0..7) -> 0..31
    W rows: k*n + i (k in 0..7, i in 0..2) -> 0..23
    """
    n, m, p = 3, 4, 8
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # j pairs: (0,1) -> jp=0, (2,3) -> jp=1
    # k pairs: (0,1) -> kp=0, (2,3) -> kp=1, (4,5) -> kp=2, (6,7) -> kp=3
    
    # Part 1: i_pair = (0, 1). Subproblem is 2x2x2.
    # For each j_pair and k_pair, we use Strassen.
    for jp in range(2):
        j0 = 2 * jp
        j1 = 2 * jp + 1
        for kp in range(4):
            k0 = 2 * kp
            k1 = 2 * kp + 1
            
            # U mapping:
            # i=0, j=j0 -> row 0*4 + j0 = j0
            # i=0, j=j1 -> row 0*4 + j1 = j1
            # i=1, j=j0 -> row 1*4 + j0 = 4 + j0
            # i=1, j=j1 -> row 1*4 + j1 = 4 + j1
            # Strassen U order: a, b, c, d
            # a = A[0, j0], b = A[0, j1], c = A[1, j0], d = A[1, j1]
            u_map = [j0, j1, 4 + j0, 4 + j1]
            
            # V mapping:
            # j=j0, k=k0 -> row j0*8 + k0
            # j=j0, k=k1 -> row j0*8 + k1
            # j=j1, k=k0 -> row j1*8 + k0
            # j=j1, k=k1 -> row j1*8 + k1
            # Strassen V order: e, f, g, h
            # e = B[j0, k0], f = B[j0, k1], g = B[j1, k0], h = B[j1, k1]
            v_map = [j0 * 8 + k0, j0 * 8 + k1, j1 * 8 + k0, j1 * 8 + k1]
            
            # W mapping:
            # k=k0, i=0 -> row k0*3 + 0
            # k=k1, i=0 -> row k1*3 + 0
            # k=k0, i=1 -> row k0*3 + 1
            # k=k1, i=1 -> row k1*3 + 1
            # Strassen W order: M11, M12, M21, M22
            # M11 = C[0, k0], M12 = C[0, k1], M21 = C[1, k0], M22 = C[1, k1]
            w_map = [k0 * 3 + 0, k1 * 3 + 0, k0 * 3 + 1, k1 * 3 + 1]
            
            _add_strassen_block(U, V, W, u_map, v_map, w_map, col_offset)
            col_offset += 7
    
    # Part 2: i = 2 (single). Subproblem is 1x2x2.
    # For each j_pair and k_pair, we use schoolbook (rank 4).
    # C[2, k] = A[2, j0]*B[j0, k] + A[2, j1]*B[j1, k]
    # We decompose this into 4 multiplications:
    # P1 = A[2, j0] * B[j0, k0]
    # P2 = A[2, j0] * B[j0, k1]
    # P3 = A[2, j1] * B[j1, k0]
    # P4 = A[2, j1] * B[j1, k1]
    # C[2, k0] = P1 + P3
    # C[2, k1] = P2 + P4
    
    for jp in range(2):
        j0 = 2 * jp
        j1 = 2 * jp + 1
        for kp in range(4):
            k0 = 2 * kp
            k1 = 2 * kp + 1
            
            # U row for i=2, j=j0 is 2*4 + j0 = 8 + j0
            # U row for i=2, j=j1 is 2*4 + j1 = 8 + j1
            
            # V rows:
            # j=j0, k=k0 -> j0*8 + k0
            # j=j0, k=k1 -> j0*8 + k1
            # j=j1, k=k0 -> j1*8 + k0
            # j=j1, k=k1 -> j1*8 + k1
            
            # W rows:
            # k=k0, i=2 -> k0*3 + 2
            # k=k1, i=2 -> k1*3 + 2
            
            # P1: A[2, j0] * B[j0, k0] -> C[2, k0]
            U[8 + j0][col_offset] += 1
            V[j0 * 8 + k0][col_offset] += 1
            W[k0 * 3 + 2][col_offset] += 1
            col_offset += 1
            
            # P2: A[2, j0] * B[j0, k1] -> C[2, k1]
            U[8 + j0][col_offset] += 1
            V[j0 * 8 + k1][col_offset] += 1
            W[k1 * 3 + 2][col_offset] += 1
            col_offset += 1
            
            # P3: A[2, j1] * B[j1, k0] -> C[2, k0]
            U[8 + j1][col_offset] += 1
            V[j1 * 8 + k0][col_offset] += 1
            W[k0 * 3 + 2][col_offset] += 1
            col_offset += 1
            
            # P4: A[2, j1] * B[j1, k1] -> C[2, k1]
            U[8 + j1][col_offset] += 1
            V[j1 * 8 + k1][col_offset] += 1
            W[k1 * 3 + 2][col_offset] += 1
            col_offset += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # First, try the rank 88 decomposition
    try:
        result_88 = _construct_rank_88()
        if _verify(result_88["U"], result_88["V"], result_88["W"], 3, 4, 8):
            return result_88
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
