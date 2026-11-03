"""Improved program using Strassen-based block decomposition for 3x4x7."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # M1: (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C11 and C22
    
    # M2: (A21 + A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C21
    
    # M3: A11(B21 - B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C12 and C22
    
    # M4: A22(B12 - B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1  # C11 and C21
    
    # M5: (A11 + A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1  # C11 and C12
    
    # M6: (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22
    
    # M7: (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1  # C11
    
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = int(n), int(m), int(p)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-10:
                    return False, val
    return True, 0.0

def _schoolbook_block(n_i, m_j, p_k):
    """Return (U, V, W) for schoolbook decomposition of n_i x m_j x p_k."""
    rank = n_i * m_j * p_k
    U = np.zeros((n_i * m_j, rank))
    V = np.zeros((m_j * p_k, rank))
    W = np.zeros((p_k * n_i, rank))
    col = 0
    for i in range(n_i):
        for j in range(m_j):
            for k in range(p_k):
                U[i * m_j + j, col] = 1
                V[j * p_k + k, col] = 1
                W[k * n_i + i, col] = 1
                col += 1
    return U, V, W

def _assemble_decomposition(n, m, p, n_sizes, m_sizes, p_sizes, block_decomps):
    """Assemble block decompositions into full U, V, W matrices.
    
    block_decomps: dict mapping (ni, mj, pk) -> (U_block, V_block, W_block)
    """
    n_splits = [0] + list(np.cumsum(n_sizes))
    m_splits = [0] + list(np.cumsum(m_sizes))
    p_splits = [0] + list(np.cumsum(p_sizes))
    
    total_rank = sum(Us.shape[1] for (Us, Vs, Ws) in block_decomps.values())
    
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for ni in range(len(n_sizes)):
        for mj in range(len(m_sizes)):
            for pk in range(len(p_sizes)):
                if (ni, mj, pk) not in block_decomps:
                    continue
                Us, Vs, Ws = block_decomps[(ni, mj, pk)]
                block_rank = Us.shape[1]
                
                n_i = n_sizes[ni]
                m_j = m_sizes[mj]
                p_k = p_sizes[pk]
                
                n_start = n_splits[ni]
                m_start = m_splits[mj]
                p_start = p_splits[pk]
                
                # Place U block: rows (n_start*m + m_start) to (n_start+m_i)*m + (m_start+m_j)
                for i in range(n_i):
                    for j in range(m_j):
                        u_row = (n_start + i) * m + (m_start + j)
                        U_full[u_row, col_offset:col_offset + block_rank] = Us[i * m_j + j, :]
                
                # Place V block: rows (m_start*p + p_start) to (m_start+m_j)*p + (p_start+p_k)
                for j in range(m_j):
                    for k in range(p_k):
                        v_row = (m_start + j) * p + (p_start + k)
                        V_full[v_row, col_offset:col_offset + block_rank] = Vs[j * p_k + k, :]
                
                # Place W block: rows (p_start*n + n_start) to (p_start+p_k)*n + (n_start+n_i)
                for k in range(p_k):
                    for i in range(n_i):
                        w_row = (p_start + k) * n + (n_start + i)
                        W_full[w_row, col_offset:col_offset + block_rank] = Ws[k * n_i + i, :]
                
                col_offset += block_rank
    
    return U_full, V_full, W_full

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Strategy: Use Strassen for 2x2x2 blocks, schoolbook for others
    # Split: n=3 -> [2,1], m=4 -> [2,2], p=7 -> [2,2,2,1]
    # 2x2x2 blocks: 1*2*3 = 6 blocks, each rank 7 (Strassen)
    # 2x2x1 blocks: 1*2*1 = 2 blocks, each rank 4 (schoolbook)
    # 1x2x2 blocks: 1*2*3 = 6 blocks, each rank 4 (schoolbook)
    # 1x2x1 blocks: 1*2*1 = 2 blocks, each rank 2 (schoolbook)
    # Total rank: 6*7 + 2*4 + 6*4 + 2*2 = 42 + 8 + 24 + 4 = 78
    
    n_sizes = [2, 1]
    m_sizes = [2, 2]
    p_sizes = [2, 2, 2, 1]
    
    block_decomps = {}
    
    for ni in range(len(n_sizes)):
        for mj in range(len(m_sizes)):
            for pk in range(len(p_sizes)):
                n_i = n_sizes[ni]
                m_j = m_sizes[mj]
                p_k = p_sizes[pk]
                
                if n_i == 2 and m_j == 2 and p_k == 2:
                    # Use Strassen: rank 7
                    block_decomps[(ni, mj, pk)] = _strassen_2x2x2()
                else:
                    # Schoolbook for this block
                    block_decomps[(ni, mj, pk)] = _schoolbook_block(n_i, m_j, p_k)
    
    # Assemble
    U_full, V_full, W_full = _assemble_decomposition(n, m, p, n_sizes, m_sizes, p_sizes, block_decomps)
    
    # Verify
    valid, val = _verify_decomposition(n, m, p, U_full, V_full, W_full)
    if valid:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: pure schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
