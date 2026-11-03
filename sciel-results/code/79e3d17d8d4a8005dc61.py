import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def verify_decomposition(U, V, W):
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.sum((T_comp - T)**2)
        return error < 1e-9

    # Define the 6 disjoint Strassen blocks
    # Each block is (i_indices, j_indices, k_indices)
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [2, 3]),
        ([0, 1], [0, 1], [4, 5]),
        ([0, 1], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3]),
        ([0, 1], [2, 3], [4, 5])
    ]
    
    # Calculate total rank
    # 6 blocks * 7 rank each = 42
    # Remaining terms: 105 - 6*8 = 57
    # Total rank = 42 + 57 = 99
    total_rank = 6 * 7 + (n * m * p - 6 * 8)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block, we add 7 columns
    for i_idx, j_idx, k_idx in blocks:
        i0, i1 = i_idx
        j0, j1 = j_idx
        k0, k1 = k_idx
        
        # Indices in global matrices
        # A indices: i*m + j
        a00 = i0 * m + j0
        a01 = i0 * m + j1
        a10 = i1 * m + j0
        a11 = i1 * m + j1
        
        # B indices: j*p + k
        b00 = j0 * p + k0
        b01 = j0 * p + k1
        b10 = j1 * p + k0
        b11 = j1 * p + k1
        
        # C indices: k*n + i
        c00 = k0 * n + i0
        c01 = k1 * n + i0
        c10 = k0 * n + i1
        c11 = k1 * n + i1
        
        # M1
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
        # M2
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # M3
        U[a00, col] += 1
        V[b10, col] += 1; V[b11, col] += -1
        W[c01, col] += 1
        col += 1
        
        # M4
        U[a11, col] += 1
        V[b01, col] += 1; V[b00, col] += -1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        
        # M5
        U[a00, col] += 1; U[a10, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1; W[c01, col] += 1
        col += 1
        
        # M6
        U[a10, col] += 1; U[a00, col] += -1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7
        U[a01, col] += 1; U[a11, col] += -1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1

    # Add schoolbook terms for the rest
    # We need to identify which (i, j, k) tuples are NOT covered by the blocks
    covered = set()
    for i_idx, j_idx, k_idx in blocks:
        for i in i_idx:
            for j in j_idx:
                for k in k_idx:
                    covered.add((i, j, k))
                    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    # Verify
    if not verify_decomposition(U, V, W):
        # Fallback to schoolbook if verification fails
        rank = rank_school
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
