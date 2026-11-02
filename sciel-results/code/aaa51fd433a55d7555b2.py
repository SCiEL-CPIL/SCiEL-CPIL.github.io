"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a rank-50 exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    Uses Strassen's algorithm for 2x2x2 blocks and schoolbook for 2x2x1 blocks.
    """
    n, m, p = 2, 4, 7
    
    # Total rank calculation:
    # We partition the 4x7 multiplication into blocks based on j and k.
    # j is split into two pairs: (0,1) and (2,3).
    # For each j-pair, k (size 7) is split into three pairs (0,1), (2,3), (4,5) and one single (6).
    # Each 2x2x2 block (i in {0,1}, j_pair, k_pair) uses Strassen (rank 7).
    # Each 2x2x1 block (i in {0,1}, j_pair, k_single) uses Schoolbook (rank 4).
    # Per j-pair: 3 * 7 + 1 * 4 = 25.
    # Total: 2 * 25 = 50.
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    current_col = 0

    def add_strassen_block(i_base, j_base, k_base):
        """Adds a 2x2x2 Strassen block at offsets i_base, j_base, k_base."""
        nonlocal current_col
        # Indices for A (2x2): rows i_base, i_base+1; cols j_base, j_base+1
        a00 = i_base * m + j_base
        a01 = i_base * m + j_base + 1
        a10 = (i_base + 1) * m + j_base
        a11 = (i_base + 1) * m + j_base + 1
        
        # Indices for B (2x2): rows j_base, j_base+1; cols k_base, k_base+1
        b00 = j_base * p + k_base
        b01 = j_base * p + k_base + 1
        b10 = (j_base + 1) * p + k_base
        b11 = (j_base + 1) * p + k_base + 1
        
        # Indices for C (2x2): rows k_base, k_base+1; cols i_base, i_base+1
        # Note: W index is k*n + i. 
        # The sub-tensor corresponds to C[k, i] for k in {k_base, k_base+1} and i in {i_base, i_base+1}.
        c00 = k_base * n + i_base
        c01 = k_base * n + i_base + 1
        c10 = (k_base + 1) * n + i_base
        c11 = (k_base + 1) * n + i_base + 1

        # M1 = (A00 + A11) * (B00 + B11)
        U[a00, current_col] += 1; U[a11, current_col] += 1
        V[b00, current_col] += 1; V[b11, current_col] += 1
        W[c00, current_col] += 1; W[c11, current_col] += 1
        current_col += 1

        # M2 = (A10 + A11) * B00
        U[a10, current_col] += 1; U[a11, current_col] += 1
        V[b00, current_col] += 1
        W[c10, current_col] += 1; W[c11, current_col] -= 1
        current_col += 1

        # M3 = A00 * (B01 - B11)
        U[a00, current_col] += 1
        V[b01, current_col] += 1; V[b11, current_col] -= 1
        W[c01, current_col] += 1; W[c11, current_col] += 1
        current_col += 1

        # M4 = A11 * (B10 - B00)
        U[a11, current_col] += 1
        V[b10, current_col] += 1; V[b00, current_col] -= 1
        W[c00, current_col] += 1; W[c10, current_col] += 1
        current_col += 1

        # M5 = (A00 + A01) * B11
        U[a00, current_col] += 1; U[a01, current_col] += 1
        V[b11, current_col] += 1
        W[c00, current_col] -= 1; W[c01, current_col] += 1
        current_col += 1

        # M6 = (A10 - A00) * (B00 + B01)
        U[a10, current_col] += 1; U[a00, current_col] -= 1
        V[b00, current_col] += 1; V[b01, current_col] += 1
        W[c11, current_col] += 1
        current_col += 1

        # M7 = (A01 - A11) * (B10 + B11)
        U[a01, current_col] += 1; U[a11, current_col] -= 1
        V[b10, current_col] += 1; V[b11, current_col] += 1
        W[c00, current_col] += 1
        current_col += 1

    def add_schoolbook_block(i_base, j_base, k_base):
        """Adds a 2x2x1 schoolbook block at offsets i_base, j_base, k_base."""
        nonlocal current_col
        a00 = i_base * m + j_base
        a01 = i_base * m + j_base + 1
        a10 = (i_base + 1) * m + j_base
        a11 = (i_base + 1) * m + j_base + 1
        
        b0 = j_base * p + k_base
        b1 = (j_base + 1) * p + k_base
        
        c0 = k_base * n + i_base
        c1 = k_base * n + i_base + 1

        # P1 = A00 * B0 -> C0
        U[a00, current_col] = 1; V[b0, current_col] = 1; W[c0, current_col] = 1
        current_col += 1
        
        # P2 = A01 * B1 -> C0
        U[a01, current_col] = 1; V[b1, current_col] = 1; W[c0, current_col] = 1
        current_col += 1
        
        # P3 = A10 * B0 -> C1
        U[a10, current_col] = 1; V[b0, current_col] = 1; W[c1, current_col] = 1
        current_col += 1
        
        # P4 = A11 * B1 -> C1
        U[a11, current_col] = 1; V[b1, current_col] = 1; W[c1, current_col] = 1
        current_col += 1

    # Iterate over j-pairs
    for j_base in [0, 2]:
        # Iterate over k-pairs
        for k_base in [0, 2, 4]:
            add_strassen_block(0, j_base, k_base)
        # Handle the remaining k=6
        add_schoolbook_block(0, j_base, 6)

    if current_col != rank:
        # Fallback: should not happen
        return _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    """Fallback standard schoolbook decomposition."""
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
