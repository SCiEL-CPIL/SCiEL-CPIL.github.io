import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 factors for 2x2
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Build rank-49 decomposition via Kronecker product of Strassen factors
    # U4[i*4+j, c1*7+c2] = U2[i//2*2 + j//2, c1] * U2[i%2*2 + j%2, c2]
    # V4[j*4+k, c1*7+c2] = V2[j//2*2 + k//2, c1] * V2[j%2*2 + k%2, c2]
    # W4[k*4+i, c1*7+c2] = W2[k//2*2 + i//2, c1] * W2[k%2*2 + i%2, c2]

    # Vectorized construction using broadcasting
    # For U: row index is i*4+j where i,j in 0..3
    # Split into high and low bits: ih=i//2, il=i%2, jh=j//2, jl=j%2
    # U2 row for (ih,jh) is U2[ih*2+jh, :]
    # U2 row for (il,jl) is U2[il*2+jl, :]
    # Column index is c1*7+c2

    # Precompute the 16x7 matrices for each factor
    # For U: for each (i,j), the 7-vector is outer product of U2[ih*2+jh,:] and U2[il*2+jl,:]
    # This gives a 7x7 matrix, flattened to 49

    # Build using numpy operations
    # Create index arrays
    i_vals = np.repeat(np.arange(4), 4)  # i for each row of U (16 rows)
    j_vals = np.tile(np.arange(4), 4)    # j for each row of U

    # High and low bits
    i_hi = (i_vals // 2).astype(int)
    i_lo = (i_vals % 2).astype(int)
    j_hi = (j_vals // 2).astype(int)
    j_lo = (j_vals % 2).astype(int)

    # For each row, get the 7-vector from U2
    # U2[ih*2+jh, :] for high part, U2[il*2+jl, :] for low part
    u_hi_vecs = U2[i_hi * 2 + j_hi]  # shape (16, 7)
    u_lo_vecs = U2[i_lo * 2 + j_lo]  # shape (16, 7)

    # Outer product for each row: (16, 7, 7)
    u_outer = u_hi_vecs[:, :, np.newaxis] * u_lo_vecs[:, np.newaxis, :]  # (16, 7, 7)
    u_mat = u_outer.reshape(16, 49)

    # For V: row index is j*4+k
    j_vals_v = np.repeat(np.arange(4), 4)  # j for each row of V
    k_vals_v = np.tile(np.arange(4), 4)    # k for each row of V

    j_hi_v = (j_vals_v // 2).astype(int)
    j_lo_v = (j_vals_v % 2).astype(int)
    k_hi_v = (k_vals_v // 2).astype(int)
    k_lo_v = (k_vals_v % 2).astype(int)

    v_hi_vecs = V2[j_hi_v * 2 + k_hi_v]  # shape (16, 7)
    v_lo_vecs = V2[j_lo_v * 2 + k_lo_v]  # shape (16, 7)

    v_outer = v_hi_vecs[:, :, np.newaxis] * v_lo_vecs[:, np.newaxis, :]  # (16, 7, 7)
    v_mat = v_outer.reshape(16, 49)

    # For W: row index is k*4+i
    k_vals_w = np.repeat(np.arange(4), 4)  # k for each row of W
    i_vals_w = np.tile(np.arange(4), 4)    # i for each row of W

    k_hi_w = (k_vals_w // 2).astype(int)
    k_lo_w = (k_vals_w % 2).astype(int)
    i_hi_w = (i_vals_w // 2).astype(int)
    i_lo_w = (i_vals_w % 2).astype(int)

    w_hi_vecs = W2[k_hi_w * 2 + i_hi_w]  # shape (16, 7)
    w_lo_vecs = W2[k_lo_w * 2 + i_lo_w]  # shape (16, 7)

    w_outer = w_hi_vecs[:, :, np.newaxis] * w_lo_vecs[:, np.newaxis, :]  # (16, 7, 7)
    w_mat = w_outer.reshape(16, 49)

    # Verification: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should equal 1 for all i,j,k
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                prod = np.sum(u_mat[i * 4 + j] * v_mat[j * 4 + k] * w_mat[k * 4 + i])
                if prod != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid or time.time() - start_time > deadline:
        # Fallback: rank 64 schoolbook
        n, m, p = 4, 4, 4
        rank = n * m * p
        U_f = [[0] * rank for _ in range(n * m)]
        V_f = [[0] * rank for _ in range(m * p)]
        W_f = [[0] * rank for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_f[ii * m + jj][col] = 1
                    V_f[jj * p + kk][col] = 1
                    W_f[kk * n + ii][col] = 1
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    return {
        "U": u_mat.tolist(),
        "V": v_mat.tolist(),
        "W": w_mat.tolist()
    }
