import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    time_limit = 145.0

    def _time_exceeded():
        return (time.time() - t_start) > time_limit

    n, m, p = 2, 3, 8
    
    # We construct the rank-44 decomposition for 2x3 * 3x8.
    # Partition A (2x3) -> A_top (2x2), A_bot (2x1)
    # Partition B (3x8) -> B_left (2x8), B_right (1x8)
    # Total Rank = Rank(2x2x8) + Rank(2x1x8)
    # Rank(2x2x8) = 28 (4 Strassen blocks)
    # Rank(2x1x8) = 16 (Schoolbook)
    
    rank = 44
    # Allocate zero matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # --- Part 1: 2x2x8 using 4 Strassen blocks ---
    # Strassen 2x2x2 core
    # Indices: 0:A00, 1:A01, 2:A10, 3:A11
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_w = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    col_offset = 0
    for block_idx in range(4):
        k_start = block_idx * 2
        for s in range(7):
            if _time_exceeded():
                # Fallback: return schoolbook rank 48
                return _schoolbook_fallback(n, m, p)
            
            # Map Strassen indices to global U, V, W indices
            # A part: rows 0,1 of A, cols 0,1 of B (which correspond to a=0,1)
            # A is 2x3. Rows i=0,1. Cols a=0,1.
            # U index: i*m + a
            for i in range(2):
                for a in range(2):
                    val = strassen_u[s][i*2 + a]
                    if val != 0:
                        U[i*m + a, col_offset] = val

            # B part: Rows a=0,1 of B, Cols k=k_start, k_start+1
            # B is 3x8. Rows a=0,1 (since we picked top 2 cols of B? No, B is 3x8, A is 2x3.
            # Wait. A is 2x3. B is 3x8.
            # A = [ A_left (2x2) | A_right (2x1) ]
            # B = [ B_top (2x8) ; B_bottom (1x8) ]
            # Product = A_left B_top + A_right B_bottom
            # So for the first part, we use A_left (cols 0,1 of A) and B_top (rows 0,1 of B).
            # U index for A: i*m + a. i in {0,1}, a in {0,1}.
            # V index for B: a*p + k. a in {0,1}, k in {k_start, k_start+1}.
            
            for a in range(2):
                for k in range(2):
                    val = strassen_v[s][a*2 + k]
                    if val != 0:
                        V[a*p + (k_start + k), col_offset] = val

            # W part: Output C (2x8). Index k*n + i.
            # k is output col (0..7), i is output row (0..1).
            for k in range(2):
                for i in range(2):
                    val = strassen_w[s][k*2 + i]
                    if val != 0:
                        W[(k_start + k)*n + i, col_offset] = val

            col_offset += 1
            
    # --- Part 2: 2x1x8 using Schoolbook ---
    # A_right (2x1) * B_bottom (1x8)
    # A_right: Rows i=0,1. Col a=2.
    # B_bottom: Row a=2. Cols k=0..7.
    # Output: C[i,k] = A[i,2] * B[2,k]
    
    for i in range(2):
        for k in range(8):
            if _time_exceeded():
                return _schoolbook_fallback(n, m, p)
            
            U[i*m + 2, col_offset] = 1
            V[2*p + k, col_offset] = 1
            W[k*n + i, col_offset] = 1
            col_offset += 1

    # Convert to lists of lists as per interface requirement (if strict, though np arrays often work, task says "integer/half-integer entries")
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook_fallback(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for a in range(m):
            for k in range(p):
                U[i*m + a, col] = 1
                V[a*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
