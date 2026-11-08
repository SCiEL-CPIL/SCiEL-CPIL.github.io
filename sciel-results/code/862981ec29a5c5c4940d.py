import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 terms definition
    # u_vec: coeffs for X00, X01, X10, X11
    # v_vec: coeffs for Y00, Y01, Y10, Y11
    # w_vec: coeffs for Z00, Z01, Z10, Z11
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    def add_block(a_col_off, b_row_off, b_col_off, c_col_off, start_rank):
        for t_idx, (u_coeffs, v_coeffs, w_coeffs) in enumerate(strassen_terms):
            r = start_rank + t_idx
            
            # U: A[0..1, a_col_off..a_col_off+1]
            # Indices in U: row*4 + col
            # X00: A[0, a_col_off] -> idx 0*4 + a_col_off
            # X01: A[0, a_col_off+1] -> idx 0*4 + a_col_off+1
            # X10: A[1, a_col_off] -> idx 1*4 + a_col_off
            # X11: A[1, a_col_off+1] -> idx 1*4 + a_col_off+1
            U[0*4 + a_col_off, r] += u_coeffs[0]
            U[0*4 + a_col_off + 1, r] += u_coeffs[1]
            U[1*4 + a_col_off, r] += u_coeffs[2]
            U[1*4 + a_col_off + 1, r] += u_coeffs[3]
            
            # V: B[b_row_off..b_row_off+1, b_col_off..b_col_off+1]
            # Indices in V: row*4 + col
            # Y00: B[b_row_off, b_col_off]
            # Y01: B[b_row_off, b_col_off+1]
            # Y10: B[b_row_off+1, b_col_off]
            # Y11: B[b_row_off+1, b_col_off+1]
            V[b_row_off*4 + b_col_off, r] += v_coeffs[0]
            V[b_row_off*4 + b_col_off + 1, r] += v_coeffs[1]
            V[(b_row_off+1)*4 + b_col_off, r] += v_coeffs[2]
            V[(b_row_off+1)*4 + b_col_off + 1, r] += v_coeffs[3]
            
            # W: C[0..1, c_col_off..c_col_off+1]
            # Indices in W: col*2 + row (k*n+i)
            # Z00: C[0, c_col_off] -> idx c_col_off*2 + 0
            # Z01: C[0, c_col_off+1] -> idx (c_col_off+1)*2 + 0
            # Z10: C[1, c_col_off] -> idx c_col_off*2 + 1
            # Z11: C[1, c_col_off+1] -> idx (c_col_off+1)*2 + 1
            W[c_col_off*2 + 0, r] += w_coeffs[0]
            W[(c_col_off+1)*2 + 0, r] += w_coeffs[1]
            W[c_col_off*2 + 1, r] += w_coeffs[2]
            W[(c_col_off+1)*2 + 1, r] += w_coeffs[3]

    # 1. A0 * B0,0: A cols 0,1; B rows 0,1 cols 0,1; C cols 0,1
    add_block(0, 0, 0, 0, 0)
    # 2. A0 * B0,1: A cols 0,1; B rows 0,1 cols 2,3; C cols 2,3
    add_block(0, 0, 2, 2, 7)
    # 3. A1 * B1,0: A cols 2,3; B rows 2,3 cols 0,1; C cols 0,1
    add_block(2, 2, 0, 0, 14)
    # 4. A1 * B1,1: A cols 2,3; B rows 2,3 cols 2,3; C cols 2,3
    add_block(2, 2, 2, 2, 21)
    
    # Verify correctness
    def verify(U_mat, V_mat, W_mat):
        A = np.arange(n * m).reshape(n, m) + 1.0
        B = np.arange(m * p).reshape(m, p) + 2.0
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(U_mat.shape[1]):
                    for jj in range(m):
                        acc += U_mat[ii * m + jj, cc] * V_mat[jj * p + kk, cc] * W_mat[kk * n + ii, cc]
                Cf[ii, kk] = acc
        return np.allclose(C, Cf)
    
    if not verify(U, V, W):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
