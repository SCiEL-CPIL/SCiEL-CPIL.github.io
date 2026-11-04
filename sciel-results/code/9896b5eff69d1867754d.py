"""Matrix multiplication 4,4,9 tensor decomposition - Deterministic block Strassen + Winograd."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    n_dim = n * m  # 16
    m_dim = m * p  # 36
    p_dim = p * n  # 36
    
    # Build target tensor for verification
    target = np.zeros(n_dim * m_dim * p_dim, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                flat_idx = a_idx * (m_dim * p_dim) + b_idx * p_dim + c_idx
                target[flat_idx] = 1.0
    
    def verify_decomposition(U, V, W, rank):
        """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == target tensor."""
        T = np.zeros((n_dim, m_dim, p_dim))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(T.flatten(), target, atol=1e-10)
    
    # Strassen 2x2x2 decomposition
    strassen_2x2 = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def add_strassen_4x4x4(U_list, V_list, W_list, col_offset):
        """Add 4x4x4 Strassen decomposition (rank 49) at column offset."""
        for u_c, v_c, w_c in strassen_2x2:
            for li in range(2):
                for lj in range(2):
                    if li * 2 + lj not in u_c:
                        continue
                    for bj in range(2):
                        for bk in range(2):
                            if bj * 2 + bk not in v_c:
                                continue
                            for gi in range(2):
                                for gk in range(2):
                                    if gi * 2 + gk not in w_c:
                                        continue
                                    
                                    u_vec = [0] * n_dim
                                    v_vec = [0] * m_dim
                                    w_vec = [0] * p_dim
                                    
                                    u_val = u_c[li * 2 + lj]
                                    v_val = v_c[bj * 2 + bk]
                                    w_val = w_c[gi * 2 + gk]
                                    
                                    u_idx = (gi * 2 + li) * m + (bj * 2 + lj)
                                    v_idx = (bj * 2 + lj) * p + (col_offset + gi * 2 + gk)
                                    w_idx = (col_offset + gi * 2 + gk) * n + (gi * 2 + li)
                                    
                                    u_vec[u_idx] = u_val
                                    v_vec[v_idx] = v_val
                                    w_vec[w_idx] = w_val
                                    
                                    U_list.append(u_vec)
                                    V_list.append(v_vec)
                                    W_list.append(w_vec)
    
    # Build the decomposition
    U_list = []
    V_list = []
    W_list = []
    
    # Block 1: 4x4x4 at columns 0-3
    add_strassen_4x4x4(U_list, V_list, W_list, 0)
    
    # Block 2: 4x4x4 at columns 4-7
    add_strassen_4x4x4(U_list, V_list, W_list, 4)
    
    # Block 3: 4x4x1 at column 8 (schoolbook, rank 16)
    for i in range(4):
        for j in range(4):
            u_vec = [0] * n_dim
            v_vec = [0] * m_dim
            w_vec = [0] * p_dim
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
    
    rank = len(U_list)
    
    # Convert to numpy arrays
    U = np.zeros((n_dim, rank))
    V = np.zeros((m_dim, rank))
    W = np.zeros((p_dim, rank))
    
    for r in range(rank):
        for i in range(n_dim):
            U[i, r] = U_list[r][i]
        for i in range(m_dim):
            V[i, r] = V_list[r][i]
        for i in range(p_dim):
            W[i, r] = W_list[r][i]
    
    # Verify
    if not verify_decomposition(U, V, W, rank):
        raise ValueError("Decomposition verification failed")
    
    # Convert to list format
    U_out = [[U[row, col] for col in range(rank)] for row in range(n_dim)]
    V_out = [[V[row, col] for col in range(rank)] for row in range(m_dim)]
    W_out = [[W[row, col] for col in range(rank)] for row in range(p_dim)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
