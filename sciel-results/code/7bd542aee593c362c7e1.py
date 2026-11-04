import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 factors for 2x2 matrix multiplication
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Rows of U2 correspond to A00, A01, A10, A11 (index 0..3)
    # Rows of V2 correspond to B00, B01, B10, B11 (index 0..3)
    # Rows of W2 correspond to C00, C01, C10, C11 (index 0..3)
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 1, 0, 1, 0, 0, 0],  # C01
        [0, 0, 1, 0, 1, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ], dtype=int)

    # Construct 4x4 factors using Kronecker product
    # Rank = 7 * 7 = 49
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    # The Kronecker product np.kron(A, B) produces a matrix where row R
    # corresponds to the outer product of row R//4 of A and row R%4 of B.
    # For 4x4 matrix multiplication, we split the 4x4 matrix into 2x2 blocks of 2x2.
    # Let the linear index of the 4x4 matrix entry be S.
    # For U (A matrix): S = i*4 + j.
    #   i = 2*i_hi + i_lo
    #   j = 2*j_hi + j_lo
    # The Kronecker row index K is formed by the "outer" 2x2 index and "inner" 2x2 index.
    # Outer index corresponds to the block position (i_hi, j_hi) -> L_out = i_hi*2 + j_hi
    # Inner index corresponds to the position within the block (i_lo, j_lo) -> L_in = i_lo*2 + j_lo
    # K = L_out * 4 + L_in
    
    # We need to permute the rows of U4_kron such that:
    # U4_fixed[S, :] = U4_kron[K(S), :]
    
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        # Decompose S into row and col indices of the 4x4 matrix
        row = S // 4
        col = S % 4
        
        # Decompose row and col into high and low bits (2x2 block structure)
        r_hi = row // 2
        r_lo = row % 2
        c_hi = col // 2
        c_lo = col % 2
        
        # Calculate Kronecker index K
        # Outer part: (r_hi, c_hi)
        # Inner part: (r_lo, c_lo)
        L_out = r_hi * 2 + c_hi
        L_in = r_lo * 2 + c_lo
        K = L_out * 4 + L_in
        
        perm[S] = K

    # Apply permutation to align Kronecker product with standard row-major ordering
    U4 = U4_kron[perm, :]
    V4 = V4_kron[perm, :]
    W4 = W4_kron[perm, :]

    # Safety check for time
    if time.time() - start_time > deadline:
        # Fallback to naive rank 64 if something went wrong (should not happen)
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
