import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the 7 rank-1 terms for Strassen's 2x2x2 matrix multiplication.
    Each term is (U_row, V_row, W_row) where each row has 4 entries.
    U index: [a11, a12, a21, a22] -> (i*m+j) for i in {0,1}, j in {0,1}
    V index: [b11, b12, b21, b22] -> (j*p+k) for j in {0,1}, k in {0,1}
    W index: [c11, c12, c21, c22] -> (k*n+i) for k in {0,1}, i in {0,1}
    """
    # M1 = (a11+a22)(b11+b22) -> c11 += M1, c22 += M1
    # M2 = (a21+a22)b11 -> c21 += M2, c22 -= M2
    # M3 = a11(b12-b22) -> c12 += M3, c22 += M3
    # M4 = a22(b21-b11) -> c11 += M4, c21 += M4
    # M5 = (a11+a12)b22 -> c11 -= M5, c12 += M5
    # M6 = (a21-a11)(b11+b12) -> c22 += M6
    # M7 = (a12-a22)(b21+b22) -> c11 += M7
    
    return [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]

def _strassen_4x4():
    """
    Return the 28 rank-1 terms for 4x4x4 matrix multiplication using
    recursive Strassen (partition 4 = 2 + 2).
    Each term is (U_row, V_row, W_row) where each row has 16 entries.
    """
    # We build this by combining the 2x2 Strassen terms with the block structure.
    # For 4x4, we partition into 2x2 blocks. There are 4 block positions for each matrix.
    # The 4x4x4 tensor is the sum over 4 block triples (I,J,K) of the 2x2x2 tensor at that position.
    # Since Strassen is bilinear, we can just place each 2x2x2 term at the appropriate block.
    
    factors_2x2 = _strassen_2x2()
    result = []
    
    # For each of the 4 block positions (I, J, K) where I, J, K in {0, 1}
    # (0 = top-left, 1 = bottom-right in the 2x2 block partition)
    for I in range(2):
        for J in range(2):
            for K in range(2):
                for (u2, v2, w2) in factors_2x2:
                    u4 = [0.0] * 16
                    v4 = [0.0] * 16
                    w4 = [0.0] * 16
                    
                    # Place u2 into u4 at block (I, J)
                    # u4 index: (i*m + j) where i in 0..3, j in 0..3
                    # Block (I, J): i in {2I, 2I+1}, j in {2J, 2J+1}
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            global_i = 2 * I + li
                            global_j = 2 * J + lj
                            u4[global_i * 4 + global_j] = u2[local_idx]
                    
                    # Place v2 into v4 at block (J, K)
                    # v4 index: (j*p + k) where j in 0..3, k in 0..3
                    for lj in range(2):
                        for lk in range(2):
                            local_idx = lj * 2 + lk
                            global_j = 2 * J + lj
                            global_k = 2 * K + lk
                            v4[global_j * 4 + global_k] = v2[local_idx]
                    
                    # Place w2 into w4 at block (K, I)
                    # w4 index: (k*n + i) where k in 0..3, i in 0..3
                    for lk in range(2):
                        for li in range(2):
                            local_idx = lk * 2 + li
                            global_k = 2 * K + lk
                            global_i = 2 * I + li
                            w4[global_k * 4 + global_i] = w2[local_idx]
                    
                    result.append((u4, v4, w4))
    
    return result

def _construct_5x5x5():
    """
    Construct a valid decomposition for 5x5x5 matrix multiplication.
    Uses block partitioning 5 = 4 + 1.
    Applies recursive Strassen (rank 28) to 4x4x4 blocks, schoolbook to others.
    """
    n, m, p = 5, 5, 5
    
    # Block sizes for partitioning 5 = 4 + 1
    block_sizes = [4, 1]
    
    # Block offsets
    offsets = [0, 4]
    
    strassen_4x4 = _strassen_4x4()
    
    # Total rank calculation
    total_rank = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 4 and mj == 4 and pk == 4:
                    total_rank += 28
                else:
                    total_rank += ni * mj * pk
    
    # Allocate matrices
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 4 and mj == 4 and pk == 4:
                    # Use recursive Strassen for 4x4x4
                    for (uf, vf, wf) in strassen_4x4:
                        for idx in range(16):
                            if uf[idx] != 0:
                                U[offsets[i] * m + offsets[j] + idx][col] += uf[idx]
                            if vf[idx] != 0:
                                V[offsets[j] * p + offsets[k] + idx][col] += vf[idx]
                            if wf[idx] != 0:
                                W[offsets[k] * n + offsets[i] + idx][col] += wf[idx]
                        col += 1
                else:
                    # Schoolbook for this block
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                V_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                W_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[U_idx][col] = 1.0
                                V[V_idx][col] = 1.0
                                W[W_idx][col] = 1.0
                                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    return {"U": U, "V": V, "W": W}, total_rank

def _verify(n, m, p, U, V, W):
    """
    Verify that the decomposition exactly reconstructs the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """
    Main entry point. Returns a valid tensor decomposition for 5x5x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    result, rank = _construct_5x5x5()
    
    # Verify the decomposition
    if _verify(n, m, p, result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: Pure schoolbook (rank 125)
    U = [[0] * 125 for _ in range(25)]
    V = [[0] * 125 for _ in range(25)]
    W = [[0] * 125 for _ in range(25)]
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
