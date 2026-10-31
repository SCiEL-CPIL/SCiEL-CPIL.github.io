import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 tensor decomposition for 4x4 matrix multiplication using recursive Strassen.
    U: (16, 49) - U[i*4+j, r] for A[i,j]
    V: (16, 49) - V[j*4+k, r] for B[j,k]
    W: (16, 49) - W[k*4+i, r] for C[i,k] (trace convention)
    """
    n, m, p = DIMENSIONS
    
    a_coeffs = np.array([
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]],
    ], dtype=np.int64)

    b_coeffs = np.array([
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]],
    ], dtype=np.int64)

    c_coeffs = np.array([
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]],
    ], dtype=np.int64)

    rank = 49
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((16, rank), dtype=np.int64)
    W = np.zeros((16, rank), dtype=np.int64)
    
    for s in range(7):
        for t in range(7):
            col = 7 * s + t
            for I in range(2):
                for J in range(2):
                    as_val = a_coeffs[s, I, J]
                    if as_val == 0:
                        continue
                    for ip in range(2):
                        for jp in range(2):
                            at_val = a_coeffs[t, ip, jp]
                            if at_val == 0:
                                continue
                            coeff = as_val * at_val
                            i = I * 2 + ip
                            j = J * 2 + jp
                            U[i * 4 + j, col] = coeff
            for J in range(2):
                for K in range(2):
                    bs_val = b_coeffs[s, J, K]
                    if bs_val == 0:
                        continue
                    for jp in range(2):
                        for kp in range(2):
                            bt_val = b_coeffs[t, jp, kp]
                            if bt_val == 0:
                                continue
                            coeff = bs_val * bt_val
                            j = J * 2 + jp
                            k = K * 2 + kp
                            V[j * 4 + k, col] = coeff
            for I in range(2):
                for K in range(2):
                    cs_val = c_coeffs[s, I, K]
                    if cs_val == 0:
                        continue
                    for ip in range(2):
                        for kp in range(2):
                            ct_val = c_coeffs[t, ip, kp]
                            if ct_val == 0:
                                continue
                            coeff = cs_val * ct_val
                            i = I * 2 + ip
                            k = K * 2 + kp
                            W[k * 4 + i, col] = coeff
    
    # Quick verification
    Uf = U.astype(np.float64)
    Vf = V.astype(np.float64)
    Wf = W.astype(np.float64)
    
    A_test = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], dtype=np.float64)
    B_test = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        for k in range(4):
            for r in range(rank):
                u_val = 0.0
                for j in range(4):
                    u_val += Uf[i*4+j, r] * A_test[i, j]
                v_val = 0.0
                for j in range(4):
                    v_val += Vf[j*4+k, r] * B_test[j, k]
                C_computed[i, k] += u_val * v_val * Wf[k*4+i, r]
    
    if np.max(np.abs(C_computed - C_expected)) > 1e-6:
        # Fallback: schoolbook rank 64
        rank_sb = 64
        U_sb = np.zeros((16, rank_sb), dtype=np.int64)
        V_sb = np.zeros((16, rank_sb), dtype=np.int64)
        W_sb = np.zeros((16, rank_sb), dtype=np.int64)
        for j in range(4):
            for i in range(4):
                for k in range(4):
                    col = j * 16 + i * 4 + k
                    U_sb[i*4+j, col] = 1
                    V_sb[j*4+k, col] = 1
                    W_sb[k*4+i, col] = 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
