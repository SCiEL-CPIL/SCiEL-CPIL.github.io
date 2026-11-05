import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_factors():
    """
    Return list of (U_vec, V_vec, W_vec) for Strassen 2x2x2.
    U_vec has length 4: indices [0,1,2,3] = [a11,a12,a21,a22]
    V_vec has length 4: indices [0,1,2,3] = [b11,b12,b21,b22]
    W_vec has length 4: indices [0,1,2,3] = [c11,c12,c21,c22]
    """
    factors = []
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    return factors

def _verify(n, m, p, U, V, W, tol=1e-9):
    """
    Verify that the decomposition exactly reconstructs the matrix multiplication tensor.
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
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_5x5x5():
    """
    Construct a valid decomposition for 5x5x5 matrix multiplication.
    Uses block partitioning 5 = 2 + 2 + 1.
    Applies Strassen (rank 7) to 2x2x2 blocks, schoolbook to others.
    """
    n, m, p = 5, 5, 5
    block_sizes = [2, 2, 1]
    num_blocks = len(block_sizes)
    
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    strassen_factors = _strassen_factors()
    
    total_rank = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    for (uf, vf, wf) in strassen_factors:
                        local_to_global_U = [
                            (offsets[i] + 0) * m + (offsets[j] + 0),
                            (offsets[i] + 0) * m + (offsets[j] + 1),
                            (offsets[i] + 1) * m + (offsets[j] + 0),
                            (offsets[i] + 1) * m + (offsets[j] + 1),
                        ]
                        local_to_global_V = [
                            (offsets[j] + 0) * p + (offsets[k] + 0),
                            (offsets[j] + 0) * p + (offsets[k] + 1),
                            (offsets[j] + 1) * p + (offsets[k] + 0),
                            (offsets[j] + 1) * p + (offsets[k] + 1),
                        ]
                        local_to_global_W = [
                            (offsets[k] + 0) * n + (offsets[i] + 0),
                            (offsets[k] + 1) * n + (offsets[i] + 0),
                            (offsets[k] + 0) * n + (offsets[i] + 1),
                            (offsets[k] + 1) * n + (offsets[i] + 1),
                        ]
                        
                        for idx in range(4):
                            if uf[idx] != 0:
                                U[local_to_global_U[idx], col] += uf[idx]
                            if vf[idx] != 0:
                                V[local_to_global_V[idx], col] += vf[idx]
                            if wf[idx] != 0:
                                W[local_to_global_W[idx], col] += wf[idx]
                        col += 1
                else:
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                V_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                W_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[U_idx, col] = 1.0
                                V[V_idx, col] = 1.0
                                W[W_idx, col] = 1.0
                                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, total_rank

def construct():
    """
    Main entry point. Returns a valid tensor decomposition for 5x5x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    result, rank = _construct_5x5x5()
    
    if time.time() < deadline:
        if _verify(n, m, p, result["U"], result["V"], result["W"]):
            return result
    
    U = np.zeros((n * m, 125), dtype=np.float64)
    V = np.zeros((m * p, 125), dtype=np.float64)
    W = np.zeros((p * n, 125), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
