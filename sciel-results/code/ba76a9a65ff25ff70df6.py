import numpy as np
import time

def construct():
    n, m, p = 3, 4, 7
    
    # Slice-and-dice construction:
    # Partition i (3 rows) into blocks: [0,1] and [2]
    # Partition j (4 cols) into blocks: [0,1] and [2,3]
    # k (7 cols) stays whole
    #
    # This gives 4 sub-tensors:
    #   2x2x7 (i in {0,1}, j in {0,1}, k in 0..6)
    #   2x2x7 (i in {0,1}, j in {2,3}, k in 0..6)
    #   1x2x7 (i in {2},    j in {0,1}, k in 0..6)
    #   1x2x7 (i in {2},    j in {2,3}, k in 0..6)
    #
    # For each sub-tensor, use schoolbook decomposition.
    # Total rank = 2*2*7 + 2*2*7 + 1*2*7 + 1*2*7 = 28+28+14+14 = 84
    # (Same as global schoolbook, but structured for potential future improvements)
    
    # Sub-tensor definitions: (i_indices, j_indices, k_indices)
    sub_tensors = [
        ([0, 1], [0, 1], list(range(7))),   # 2x2x7
        ([0, 1], [2, 3], list(range(7))),   # 2x2x7
        ([2],    [0, 1], list(range(7))),   # 1x2x7
        ([2],    [2, 3], list(range(7))),   # 1x2x7
    ]
    
    # Compute total rank (schoolbook for each sub-tensor)
    total_rank = sum(len(ibs) * len(jbs) * len(kbs) for ibs, jbs, kbs in sub_tensors)
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for ibs, jbs, kbs in sub_tensors:
        for i in ibs:
            for j in jbs:
                for k in kbs:
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    # Verify correctness
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    U_t = np.array(U)
    V_t = np.array(V)
    W_t = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The tensor entry for matrix mult is 1 (it's the identity tensor)
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should be 1
                val = np.sum(U_t[i*m+j, :] * V_t[j*p+k, :] * W_t[k*n+i, :])
                if abs(val - 1.0) > 1e-10:
                    # Fallback to schoolbook if verification fails
                    rank = n * m * p
                    U_f = np.zeros((n * m, rank), dtype=float)
                    V_f = np.zeros((m * p, rank), dtype=float)
                    W_f = np.zeros((p * n, rank), dtype=float)
                    c = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_f[ii * m + jj, c] = 1.0
                                V_f[jj * p + kk, c] = 1.0
                                W_f[kk * n + ii, c] = 1.0
                                c += 1
                    return {
                        "U": U_f.tolist(),
                        "V": V_f.tolist(),
                        "W": W_f.tolist(),
                    }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
