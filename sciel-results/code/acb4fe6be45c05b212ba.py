import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    The multiplication tensor decomposes as a direct sum of 7 independent 2x2 * 2x2 tensors.
    Using Strassen's rank-7 decomposition for each block gives total rank 49.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28

    num_blocks = p // 2  # 7 blocks of 2x2
    rank = num_blocks * 7  # 49

    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    # Strassen coefficients for 2x2 * 2x2
    # Local indices for A: 0=a00, 1=a01, 2=a10, 3=a11
    # Local indices for B: 0=b00, 1=b01, 2=b10, 3=b11
    # Local indices for C: 0=c00, 1=c10, 2=c01, 3=c11 
    # Note: C local index mapping in W vector is [C_00, C_10, C_01, C_11] 
    # based on global index k*n+i with k=0,1 local.
    # k=0 (even global col): i=0 -> idx 0, i=1 -> idx 1.
    # k=1 (odd global col): i=0 -> idx 2, i=1 -> idx 3.
    
    # Each term is (U_vec, V_vec, W_vec)
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A10+A11)B00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # M3 = A00(B10-B00)
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]),
        # M4 = A11(B01-B10)
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 1, 0, 0]),
        # M5 = (A00+A01)B11
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # M6 = (A10-A00)(B00+B01)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (A01-A11)(B10+B11)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    for t in range(num_blocks):
        # Global index offsets
        # B indices: j*p + k
        # Block t corresponds to k in {2t, 2t+1}
        # b00: j=0, k=2t -> 2t
        # b01: j=0, k=2t+1 -> 2t+1
        # b10: j=1, k=2t -> 14+2t
        # b11: j=1, k=2t+1 -> 15+2t
        b_offsets = [2*t, 2*t+1, 14+2*t, 15+2*t]
        
        # C indices: k*n + i
        # c00: k=2t, i=0 -> 4t
        # c10: k=2t, i=1 -> 4t+1
        # c01: k=2t+1, i=0 -> 4t+2
        # c11: k=2t+1, i=1 -> 4t+3
        c_offsets = [4*t, 4*t+1, 4*t+2, 4*t+3]
        
        for s in range(7):
            col = t * 7 + s
            u_vec, v_vec, w_vec = strassen_terms[s]
            
            # Fill U
            # A indices are always 0..3
            U[0, col] = u_vec[0]
            U[1, col] = u_vec[1]
            U[2, col] = u_vec[2]
            U[3, col] = u_vec[3]
            
            # Fill V
            V[b_offsets[0], col] = v_vec[0]
            V[b_offsets[1], col] = v_vec[1]
            V[b_offsets[2], col] = v_vec[2]
            V[b_offsets[3], col] = v_vec[3]
            
            # Fill W
            W[c_offsets[0], col] = w_vec[0]
            W[c_offsets[1], col] = w_vec[1]
            W[c_offsets[2], col] = w_vec[2]
            W[c_offsets[3], col] = w_vec[3]

    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                rx = round(x)
                if abs(x - rx) < 1e-9:
                    new_row.append(int(rx))
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
