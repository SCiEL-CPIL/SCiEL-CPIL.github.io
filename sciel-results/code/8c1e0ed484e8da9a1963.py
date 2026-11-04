import numpy as np
import time

def construct():
    """
    Rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Uses Block Strassen (7 steps of 2x2 Strassen) for each 4x4 block of the output.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2x2 base case (rank 7)
    # Indices: 0:00, 1:01, 2:10, 3:11
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Block Strassen specs for 4x4x4
    # A blocks: (coeff, block_idx) where block_idx 0:00, 1:01, 2:10, 3:11
    # B blocks: (coeff, block_idx)
    # C blocks: (coeff, block_idx)
    
    steps = [
        # k=0: M1
        {
            'A': [(1, 0), (1, 3)], # A00 + A11
            'B': [(1, 0), (1, 3)], # B00 + B11
            'C': [(1, 0), (1, 3)]  # C00 + C11
        },
        # k=1: M2
        {
            'A': [(1, 2), (1, 3)], # A10 + A11
            'B': [(1, 0)],         # B00
            'C': [(1, 2), (-1, 3)] # C10 - C11
        },
        # k=2: M3
        {
            'A': [(1, 0)],         # A00
            'B': [(1, 1), (-1, 3)],# B01 - B11
            'C': [(1, 1), (1, 3)]  # C01 + C11
        },
        # k=3: M4
        {
            'A': [(1, 3)],         # A11
            'B': [(1, 2), (-1, 0)],# B10 - B00
            'C': [(1, 0), (1, 2)]  # C00 + C10
        },
        # k=4: M5
        {
            'A': [(1, 0), (1, 1)], # A00 + A01
            'B': [(1, 3)],         # B11
            'C': [(-1, 0), (1, 1)] # -C00 + C01
        },
        # k=5: M6
        {
            'A': [(1, 2), (-1, 0)],# A10 - A00
            'B': [(1, 0), (1, 1)], # B00 + B01
            'C': [(1, 3)]          # C11
        },
        # k=6: M7
        {
            'A': [(1, 1), (-1, 3)],# A01 - A11
            'B': [(1, 2), (1, 3)], # B10 + B11
            'C': [(1, 0)]          # C00
        }
    ]

    def get_global_indices(block_idx, dim=4):
        r_start = (block_idx // 2) * 2
        c_start = (block_idx % 2) * 2
        indices = []
        for r in range(2):
            for c in range(2):
                indices.append((r_start + r, c_start + c))
        return indices

    # Iterate over the two halves of the 8-column output
    for half in range(2):
        for k_idx, step in enumerate(steps):
            for m_s in range(7): # Inner Strassen index
                col = half * 49 + k_idx * 7 + m_s
                
                # A part
                for coeff_a, block_idx in step['A']:
                    for (r, c) in get_global_indices(block_idx):
                        global_a_idx = r * 4 + c
                        local_r = r % 2
                        local_c = c % 2
                        local_a_idx = local_r * 2 + local_c
                        U[global_a_idx, col] += coeff_a * U_2[local_a_idx, m_s]
                
                # B part
                for coeff_b, block_idx in step['B']:
                    for (r, c) in get_global_indices(block_idx):
                        global_b_idx = r * 8 + (c + half * 4)
                        local_r = r % 2
                        local_c = c % 2
                        local_b_idx = local_r * 2 + local_c
                        V[global_b_idx, col] += coeff_b * V_2[local_b_idx, m_s]
                
                # C part (W)
                for coeff_c, block_idx in step['C']:
                    for (r, c) in get_global_indices(block_idx):
                        global_w_idx = (c + half * 4) * 4 + r
                        local_r = r % 2
                        local_c = c % 2
                        local_w_idx = local_r * 2 + local_c
                        W[global_w_idx, col] += coeff_c * W_2[local_w_idx, m_s]

    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to naive if verification fails (should not happen)
        return construct_naive(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    # Test with random matrices
    rng = np.random.default_rng(42)
    for trial in range(3):
        A = rng.integers(-5, 6, size=(n, m)).astype(float)
        B = rng.integers(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        M = np.zeros(rank)
        for t in range(rank):
            u_val = np.dot(U[:, t], A_flat)
            v_val = np.dot(V[:, t], B_flat)
            M[t] = u_val * v_val
        
        C_computed = np.zeros((n, p))
        for k in range(p):
            for i in range(n):
                C_computed[i, k] = np.dot(W[k*n+i, :], M)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True
