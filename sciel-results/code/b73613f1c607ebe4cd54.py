"""Matrix multiplication 4x4x8 with Strassen-based rank-98 decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _build_2x2x2_strassen():
    """Build the rank-7 Strassen decomposition for 2x2x2."""
    # Standard Strassen terms
    # U index: i*2+j, V index: j*2+k, W index: k*2+i
    terms = []
    
    # M1 = (a-d)(e+h)
    terms.append(([1, 0, 0, -1], [1, 0, 0, 1], [1, 0, 0, 1]))
    
    # M2 = (c+d)e
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    
    # M3 = a(f-h)
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    
    # M4 = d(g-e)
    terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    
    # M5 = (a+b)h
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    
    # M6 = (c-d)(f+h)
    terms.append(([0, 0, 1, -1], [0, 1, 0, 1], [0, 0, 0, 1]))
    
    # M7 = (a-c)(e+g)
    terms.append(([1, 0, -1, 0], [1, 0, 1, 0], [1, 0, 0, 0]))
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    
    return U, V, W

def _build_4x4x4_strassen2():
    """
    Build a rank-49 decomposition for 4x4x4 using Strassen applied to 2x2 blocks.
    """
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    U2, V2, W2 = _build_2x2x2_strassen()
    
    # Block operations for Strassen on 2x2 blocks
    block_ops = [
        ([1, 0, 0, -1], [1, 0, 0, 1]),    # M1
        ([0, 0, 1, 1], [1, 0, 0, 0]),     # M2
        ([1, 0, 0, 0], [0, 1, 0, -1]),    # M3
        ([0, 0, 0, 1], [0, 0, 1, -1]),    # M4
        ([1, 1, 0, 0], [0, 0, 0, 1]),     # M5
        ([0, 0, 1, -1], [0, 1, 0, 1]),    # M6
        ([1, 0, -1, 0], [1, 0, 1, 0]),    # M7
    ]
    
    c_combos = [
        [1, 0, 0, 1, -1, 0, 1],    # C00
        [0, 0, 1, 0, 1, 0, 0],     # C01
        [0, 1, 0, 1, 0, 0, 0],     # C10
        [1, -1, 1, 0, 0, 1, 0],    # C11
    ]
    
    for op_idx in range(7):
        lc, rc = block_ops[op_idx]
        for t in range(7):
            col = op_idx * 7 + t
            # Fill U
            for a in range(4):
                if lc[a] != 0:
                    a_row = a // 2
                    a_col = a % 2
                    for li in range(2):
                        for lj in range(2):
                            u_idx = (a_row * 2 + li) * 4 + (a_col * 2 + lj)
                            U[u_idx, col] += lc[a] * U2[2 * li + lj, t]
            
            # Fill V
            for b in range(4):
                if rc[b] != 0:
                    b_row = b // 2
                    b_col = b % 2
                    for lj in range(2):
                        for lk in range(2):
                            v_idx = (b_row * 2 + lj) * 4 + (b_col * 2 + lk)
                            V[v_idx, col] += rc[b] * V2[2 * lj + lk, t]
            
            # Fill W
            for c_blk in range(4):
                c_coeff = c_combos[c_blk][op_idx]
                if c_coeff == 0:
                    continue
                c_row = c_blk // 2
                c_col = c_blk % 2
                for lk in range(2):
                    for li in range(2):
                        w_idx = (c_row * 2 + lk) * 4 + (c_col * 2 + li)
                        W[w_idx, col] += c_coeff * W2[2 * lk + li, t]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Build 4x4x4 rank-49 decomposition
    U44, V44, W44 = _build_4x4x4_strassen2()
    
    # For 4x4x8: C = A * B where A is 4x4, B is 4x8
    # Split B into B1 (4x4) and B2 (4x4)
    # C = [C1 | C2] where C1 = A*B1, C2 = A*B2
    
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # First half: C1 = A * B1
    U[:, 0:49] = U44
    V[0:16, 0:49] = V44
    W[0:16, 0:49] = W44
    
    # Second half: C2 = A * B2
    U[:, 49:98] = U44
    V[16:32, 49:98] = V44
    W[16:32, 49:98] = W44
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
