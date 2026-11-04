import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 factors for 2x2 matrix multiplication
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Rows of U2 correspond to A00, A01, A10, A11
    # Rows of V2 correspond to B00, B01, B10, B11
    # Rows of W2 correspond to C00, C01, C10, C11
    
    # Standard Strassen formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

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
    # The rank will be 7 * 7 = 49
    # U4 shape: (4*4, 49) = (16, 49)
    # V4 shape: (4*4, 49) = (16, 49)
    # W4 shape: (4*4, 49) = (16, 49)
    
    # The Kronecker product of the tensor decompositions gives a decomposition
    # for the larger matrix multiplication.
    # Indices for 4x4:
    # i, j, k in 0..3
    # i = 2*i_hi + i_lo
    # j = 2*j_hi + j_lo
    # k = 2*k_hi + k_lo
    
    # U index: i*4 + j
    # V index: j*4 + k
    # W index: k*4 + i (as per prompt "W[k*n+i, r]")
    
    # We can build U4, V4, W4 by iterating over the 2x2 blocks.
    # Alternatively, use np.kron on the factor matrices?
    # np.kron(U2, U2) gives a 16x49 matrix.
    # Let's check the index ordering.
    # np.kron(A, B) where A is 4x7, B is 4x7.
    # Result is 16x49.
    # Row index r = i1*4 + i2.
    # U2 row i1 corresponds to A_hi.
    # U2 row i2 corresponds to A_lo.
    # So row i1*4 + i2 of kron(U2, U2) corresponds to A_{i1, i2}?
    # Wait. A is 4x4. A[i, j].
    # i = 2*i_hi + i_lo. j = 2*j_hi + j_lo.
    # U[i*4+j, r].
    # In the Kronecker product structure:
    # U[i*4+j, r] = U2[i_hi*2+j_hi, a] * U2[i_lo*2+j_lo, b] where r = a*7 + b.
    
    # Let's verify if np.kron(U2, U2) matches this.
    # np.kron(U2, U2)[i1*4 + i2, a*7 + b] = U2[i1, a] * U2[i2, b].
    # Here i1 is the high part of the row index of U4?
    # U4 row index is I = i*4 + j.
    # This is not simply i_hi*4 + i_lo.
    # i*4 + j = (2*i_hi + i_lo)*4 + (2*j_hi + j_lo) = 8*i_hi + 4*i_lo + 2*j_hi + j_lo.
    # This does not match the standard Kronecker index structure directly if we just map i->i_hi, i_lo.
    
    # So we must construct it manually or use a reshaping trick.
    # Manual construction is safer given the index mapping.
    
    n = 4
    rank = 49
    
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Fill U
    for i in range(4):
        for j in range(4):
            i_hi = i // 2
            i_lo = i % 2
            j_hi = j // 2
            j_lo = j % 2
            
            u_high = U2[i_hi*2 + j_hi]
            u_low = U2[i_lo*2 + j_lo]
            
            # Outer product of u_high and u_low
            # r = a*7 + b
            # U[i*4+j, a*7+b] = u_high[a] * u_low[b]
            U[i*4+j, :] = np.outer(u_high, u_low).flatten()
            
    # Fill V
    for j in range(4):
        for k in range(4):
            j_hi = j // 2
            j_lo = j % 2
            k_hi = k // 2
            k_lo = k % 2
            
            v_high = V2[j_hi*2 + k_hi]
            v_low = V2[j_lo*2 + k_lo]
            
            V[j*4+k, :] = np.outer(v_high, v_low).flatten()

    # Fill W
    # Index is k*4 + i
    for k in range(4):
        for i in range(4):
            k_hi = k // 2
            k_lo = k % 2
            i_hi = i // 2
            i_lo = i % 2
            
            # W2 row index for high part: k_hi*2 + i_hi
            # W2 row index for low part: k_lo*2 + i_lo
            w_high = W2[k_hi*2 + i_hi]
            w_low = W2[k_lo*2 + i_lo]
            
            W[k*4+i, :] = np.outer(w_high, w_low).flatten()

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
