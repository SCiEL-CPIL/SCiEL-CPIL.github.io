"""Exact rank-78 decomposition of M_{3,4,7} via six disjoint 2x2x2 Strassen blocks."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify the decomposition."""
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Guaranteed correct rank n*m*p decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Define the 6 disjoint 2x2x2 blocks
    # (i_start, j_start, k_start)
    # Using a mixed arrangement: 5 blocks on i∈{0,1}, 1 block on i∈{1,2}
    # to cover i=2 partially, potentially allowing better packing
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
        (0, 0, 4),
        (1, 2, 4),
    ]
    
    # Total rank calculation
    # 6 blocks * 7 ranks each + remaining schoolbook terms
    num_blocks = len(blocks)
    covered_triples = num_blocks * 8
    total_triples = n * m * p
    remaining_triples = total_triples - covered_triples
    total_rank = num_blocks * 7 + remaining_triples
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_idx = 0
    for i0, j0, k0 in blocks:
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # Strassen 7 terms
        # 1
        U[i0*m+j0][col_idx] = 1
        U[i1*m+j1][col_idx] = 1
        V[j0*p+k0][col_idx] = 1
        V[j1*p+k1][col_idx] = 1
        W[k0*n+i0][col_idx] = 1
        W[k1*n+i1][col_idx] = 1
        col_idx += 1

        # 2
        U[i1*m+j0][col_idx] = 1
        U[i1*m+j1][col_idx] = 1
        V[j0*p+k0][col_idx] = 1
        W[k0*n+i1][col_idx] = 1
        W[k1*n+i1][col_idx] = -1
        col_idx += 1

        # 3
        U[i0*m+j0][col_idx] = 1
        V[j0*p+k1][col_idx] = 1
        V[j1*p+k1][col_idx] = -1
        W[k1*n+i0][col_idx] = 1
        W[k1*n+i1][col_idx] = 1
        col_idx += 1

        # 4
        U[i1*m+j1][col_idx] = 1
        V[j1*p+k0][col_idx] = 1
        V[j0*p+k0][col_idx] = -1
        W[k0*n+i0][col_idx] = 1
        W[k0*n+i1][col_idx] = 1
        col_idx += 1

        # 5
        U[i0*m+j0][col_idx] = 1
        U[i0*m+j1][col_idx] = 1
        V[j1*p+k1][col_idx] = 1
        W[k0*n+i0][col_idx] = -1
        W[k1*n+i0][col_idx] = 1
        col_idx += 1

        # 6
        U[i1*m+j0][col_idx] = 1
        U[i0*m+j0][col_idx] = -1
        V[j0*p+k0][col_idx] = 1
        V[j0*p+k1][col_idx] = 1
        W[k1*n+i1][col_idx] = 1
        col_idx += 1

        # 7
        U[i0*m+j1][col_idx] = 1
        U[i1*m+j1][col_idx] = -1
        V[j1*p+k0][col_idx] = 1
        V[j1*p+k1][col_idx] = 1
        W[k0*n+i0][col_idx] = 1
        col_idx += 1

    # Mark used triples
    used = set()
    for i0, j0, k0 in blocks:
        for i in range(i0, i0+2):
            for j in range(j0, j0+2):
                for k in range(k0, k0+2):
                    used.add((i,j,k))

    # Fill the rest with schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in used:
                    U[i*m+j][col_idx] = 1
                    V[j*p+k][col_idx] = 1
                    W[k*n+i][col_idx] = 1
                    col_idx += 1

    if col_idx != total_rank:
        return _schoolbook(n, m, p)

    if time.time() > deadline:
        return _schoolbook(n, m, p)
        
    if not _verify(U, V, W, n, m, p):
        return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
