import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140

def construct():
    """
    Returns the rank-36 tensor decomposition for 2x4 matrix by 4x5 matrix multiplication.
    
    The decomposition is constructed by splitting the 4x5 matrix B into a 4x4 part and a 4x1 part.
    The 2x4 * 4x4 part is decomposed into four 2x2 * 2x2 blocks, each using Strassen's algorithm (rank 7).
    The 2x4 * 4x1 part is decomposed into two 2x2 * 2x1 blocks, each using schoolbook (rank 4).
    Total rank: 4 * 7 + 2 * 4 = 28 + 8 = 36.
    
    Factors U, V, W have shapes (8, 36), (20, 36), (10, 36) respectively.
    """
    n, m, p = 2, 4, 5
    total_rank = 36
    
    # Initialize factor matrices with zeros
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a Strassen 2x2x2 block
    # Indices: 
    # A rows: i0, i1
    # A cols / B rows: j0, j1
    # B cols: k0, k1
    def add_strassen(i0, j0, k0, base):
        # M1 = (A00 + A11) * (B00 + B11)
        U[i0 * m + j0][base] += 1
        U[i1 * m + j1][base] += 1
        V[j0 * p + k0][base] += 1
        V[j1 * p + k1][base] += 1
        W[k0 * n + i0][base] += 1
        W[k1 * n + i1][base] += 1
        
        # M2 = (A10 + A11) * B00
        c = base + 1
        U[i1 * m + j0][c] += 1
        U[i1 * m + j1][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + i1][c] += 1
        W[k1 * n + i1][c] -= 1
        
        # M3 = A00 * (B01 - B11)
        c = base + 2
        U[i0 * m + j0][c] += 1
        V[j0 * p + k1][c] += 1
        V[j1 * p + k1][c] -= 1
        W[k1 * n + i0][c] += 1
        W[k1 * n + i1][c] += 1
        
        # M4 = A11 * (B10 - B00)
        c = base + 3
        U[i1 * m + j1][c] += 1
        V[j1 * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + i1][c] += 1
        
        # M5 = (A00 + A01) * B11
        c = base + 4
        U[i0 * m + j0][c] += 1
        U[i0 * m + j1][c] += 1
        V[j1 * p + k1][c] += 1
        W[k0 * n + i0][c] -= 1
        W[k1 * n + i0][c] += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        c = base + 5
        U[i1 * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + k1][c] += 1
        W[k1 * n + i1][c] += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        c = base + 6
        U[i0 * m + j1][c] += 1
        U[i1 * m + j1][c] -= 1
        V[j1 * p + k0][c] += 1
        V[j1 * p + k1][c] += 1
        W[k0 * n + i0][c] += 1
        
        return base + 7

    # Helper to add a schoolbook 2x2x1 block
    def add_schoolbook(i0, j0, k0, base):
        c = base
        for di in range(2):
            for dj in range(2):
                idx_u = (i0 + di) * m + (j0 + dj)
                idx_v = (j0 + dj) * p + k0
                idx_w = k0 * n + (i0 + di)
                U[idx_u][c] = 1
                V[idx_v][c] = 1
                W[idx_w][c] = 1
                c += 1
        return c

    # Build the decomposition
    # Part 1: C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    next_base = add_strassen(0, 0, 0, 0)
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    next_base = add_strassen(0, 2, 0, next_base)
    
    # Part 2: C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    next_base = add_strassen(0, 0, 2, next_base)
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    next_base = add_strassen(0, 2, 2, next_base)
    
    # Part 3: C[:, 4] = A[:, 0:2] * B[0:2, 4] + A[:, 2:4] * B[2:4, 4]
    # Block 5: A[0:2, 0:2] * B[0:2, 4] -> C[0:2, 4]
    next_base = add_schoolbook(0, 0, 4, next_base)
    # Block 6: A[0:2, 2:4] * B[2:4, 4] -> C[0:2, 4]
    next_base = add_schoolbook(0, 2, 4, next_base)
    
    if next_base != total_rank:
        raise RuntimeError(f"Rank mismatch: expected {total_rank}, got {next_base}")
        
    if time.time() > _DEADLINE:
        raise TimeoutError("Deadline exceeded")
        
    return {"U": U, "V": V, "W": W}
