import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 4, 4, 8

    # Strassen 2x2x2 factors (rank 7)
    # U[i,j,r], V[i,j,r], W[i,j,r]
    # Indices for 2x2: 0, 1
    U2 = np.zeros((2, 2, 7), dtype=float)
    V2 = np.zeros((2, 2, 7), dtype=float)
    W2 = np.zeros((2, 2, 7), dtype=float)

    # r=0: (A00+A11)(B00+B11)
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1

    # r=1: (A10+A11)B00
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = 1; W2[1, 1, 1] = -1

    # r=2: A00(B01-B11)
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    W2[1, 0, 2] = 1; W2[1, 1, 2] = 1

    # r=3: A11(B10-B00)
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    W2[0, 0, 3] = 1; W2[0, 1, 3] = 1

    # r=4: (A00+A01)B11
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = -1; W2[1, 0, 4] = 1

    # r=5: (A10-A00)(B00+B01)
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    W2[1, 1, 5] = 1

    # r=6: (A01-A11)(B10+B11)
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    W2[0, 0, 6] = 1

    # Construct 4x4x4 factors (rank 49)
    U4 = np.zeros((4, 4, 49), dtype=float)
    V4 = np.zeros((4, 4, 49), dtype=float)
    W4 = np.zeros((4, 4, 49), dtype=float)
    
    rank_indices = np.unravel_index(np.arange(49), (7, 7))
    s_hi_arr, r_lo_arr = rank_indices
    
    for R in range(49):
        s_hi = s_hi_arr[R]
        r_lo = r_lo_arr[R]
        
        M1_u = U2[:, :, s_hi]
        M0_u = U2[:, :, r_lo]
        U4[:, :, R] = np.kron(M1_u, M0_u)
        
        M1_v = V2[:, :, s_hi]
        M0_v = V2[:, :, r_lo]
        V4[:, :, R] = np.kron(M1_v, M0_v)
        
        M1_w = W2[:, :, s_hi]
        M0_w = W2[:, :, r_lo]
        W4[:, :, R] = np.kron(M1_w, M0_w)
        
        if time.time() > deadline - 5.0:
            break

    # Flatten 4x4x49 to (16, 49)
    U4_flat = U4.reshape(16, 49)
    V4_flat = V4.reshape(16, 49)
    W4_flat = W4.reshape(16, 49)

    # Construct final factors for 4x4x8 (rank 98)
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # U is shared
    U[:, :49] = U4_flat
    U[:, 49:] = U4_flat
    
    # V: Rows are j*8+k. 
    # Block 1 (r<49): k in 0..3.
    # Block 2 (r>=49): k in 4..7.
    j_indices = np.repeat(np.arange(4), 8)
    k_indices = np.tile(np.arange(8), 4)
    
    mask_k1 = k_indices < 4
    valid_rows_1 = np.arange(32)[mask_k1]
    j_vals_1 = j_indices[mask_k1]
    k_vals_1 = k_indices[mask_k1]
    idx4_1 = j_vals_1 * 4 + k_vals_1
    
    V[valid_rows_1, :49] = V4_flat[idx4_1, :]
    
    mask_k2 = k_indices >= 4
    valid_rows_2 = np.arange(32)[mask_k2]
    j_vals_2 = j_indices[mask_k2]
    k_vals_2 = k_indices[mask_k2]
    k_prime_2 = k_vals_2 - 4
    idx4_2 = j_vals_2 * 4 + k_prime_2
    
    V[valid_rows_2, 49:] = V4_flat[idx4_2, :]
    
    # W: Rows are k*4+i.
    # Block 1 (r<49): k in 0..3. Rows 0..15.
    W[0:16, :49] = W4_flat
    
    # Block 2 (r>=49): k in 4..7. Rows 16..31.
    W[16:32, 49:] = W4_flat

    # Verification
    if time.time() < deadline - 10.0:
        U3 = U.reshape(4, 4, 98)
        V3 = V.reshape(4, 8, 98)
        W3 = W.reshape(8, 4, 98)
        
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
        
        if not np.allclose(result, 1.0, atol=1e-10):
             rank_school = n * m * p
             U_s = np.zeros((n * m, rank_school))
             V_s = np.zeros((m * p, rank_school))
             W_s = np.zeros((p * n, rank_school))
             col = 0
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         U_s[i * m + j, col] = 1
                         V_s[j * p + k, col] = 1
                         W_s[k * n + i, col] = 1
                         col += 1
             return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
