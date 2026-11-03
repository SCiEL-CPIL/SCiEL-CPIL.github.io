"""Rewritten program for 3x5x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_222():
    """
    Return (U, V, W) for 2x2x2 Strassen multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10 += M2
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C01 += M3, C11 += M3
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C00 += M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # C00 -= M5, C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00 += M7
    
    return U, V, W


def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
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
    return U, V, W


def _build_block_decomposition(i_blocks, j_blocks, k_blocks, n, m, p):
    """
    Build a block decomposition using Strassen for 2x2x2 blocks and schoolbook otherwise.
    Returns (U, V, W, rank) or None if verification fails.
    """
    su, sv, sw = _strassen_222()
    
    # Calculate total rank
    total_rank = 0
    for i_set in i_blocks:
        for j_set in j_blocks:
            for k_set in k_blocks:
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    total_rank += 7
                else:
                    total_rank += len(i_set) * len(j_set) * len(k_set)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    for i_set in i_blocks:
        for j_set in j_blocks:
            for k_set in k_blocks:
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    # Use Strassen
                    for c in range(7):
                        for li in range(2):
                            gi = i_set[li]
                            for lj in range(2):
                                gj = j_set[lj]
                                U[gi * m + gj][col + c] = su[li * 2 + lj][c]
                        for lj in range(2):
                            gj = j_set[lj]
                            for lk in range(2):
                                gk = k_set[lk]
                                V[gj * p + gk][col + c] = sv[lj * 2 + lk][c]
                        for lk in range(2):
                            gk = k_set[lk]
                            for li in range(2):
                                gi = i_set[li]
                                W[gk * n + gi][col + c] = sw[lk * 2 + li][c]
                    col += 7
                else:
                    # Schoolbook
                    for gi in i_set:
                        for gj in j_set:
                            for gk in k_set:
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
    
    if col != total_rank:
        return None
    
    # Verify using einsum for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape: U[i*m+j, r] -> U3[i, j, r], V[j*p+k, r] -> V3[j, k, r], W[k*n+i, r] -> W3[k, i, r]
    U3 = U_np.reshape(n, m, -1)
    V3 = V_np.reshape(m, p, -1)
    W3 = W_np.reshape(p, n, -1)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=1e-10):
        return None
    
    return U, V, W, total_rank


def _try_partition(i_blocks, j_blocks, k_blocks, n, m, p):
    """Try a specific partition and return (U, V, W, rank) if valid."""
    try:
        result = _build_block_decomposition(i_blocks, j_blocks, k_blocks, n, m, p)
        if result is not None:
            return result
    except Exception:
        pass
    return None


def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Generate all possible contiguous partitions for each dimension
    # For size 3: partitions into blocks of size 1 or 2
    i_partitions = [
        ((0,), (1,), (2,)),
        ((0,), (1, 2)),
        ((0, 1), (2,)),
    ]
    
    # For size 5: partitions into blocks of size 1 or 2
    j_partitions = [
        ((0,), (1,), (2,), (3,), (4,)),
        ((0,), (1,), (2,), (3, 4)),
        ((0,), (1,), (2, 3), (4,)),
        ((0,), (1, 2), (3,), (4,)),
        ((0, 1), (2,), (3,), (4,)),
        ((0,), (1, 2), (3, 4)),
        ((0, 1), (2, 3), (4,)),
        ((0, 1), (2,), (3, 4)),
    ]
    
    k_partitions = [
        ((0,), (1,), (2,), (3,), (4,)),
        ((0,), (1,), (2,), (3, 4)),
        ((0,), (1,), (2, 3), (4,)),
        ((0,), (1, 2), (3,), (4,)),
        ((0, 1), (2,), (3,), (4,)),
        ((0,), (1, 2), (3, 4)),
        ((0, 1), (2, 3), (4,)),
        ((0, 1), (2,), (3, 4)),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    # Generate all combinations and sort by estimated rank
    combinations = []
    for i_blocks in i_partitions:
        for j_blocks in j_partitions:
            for k_blocks in k_partitions:
                # Count Strassen blocks and estimate rank
                strassen_count = 0
                est_rank = 0
                for i_set in i_blocks:
                    for j_set in j_blocks:
                        for k_set in k_blocks:
                            if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                                strassen_count += 1
                                est_rank += 7
                            else:
                                est_rank += len(i_set) * len(j_set) * len(k_set)
                combinations.append((est_rank, -strassen_count, i_blocks, j_blocks, k_blocks))
    
    # Sort by estimated rank (ascending), then by more Strassen blocks
    combinations.sort(key=lambda x: (x[0], x[1]))
    
    for est_rank, neg_strassen, i_blocks, j_blocks, k_blocks in combinations:
        if time.time() > deadline:
            break
        
        # Since combinations are sorted by est_rank, if est_rank >= best_rank,
        # all remaining will also be >= best_rank, so we can break
        if est_rank >= best_rank:
            break
        
        result = _try_partition(i_blocks, j_blocks, k_blocks, n, m, p)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook
        U, V, W = _schoolbook_factors(n, m, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
