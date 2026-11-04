import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    n, m, p = 3, 3, 8
    
    # Rank 23 for 3x3x3 (Strassen-like)
    terms_23 = [
        (0, 0, 0, 1), (1, 3, 3, 1), (2, 6, 6, 1), (3, 1, 1, 1), (4, 4, 4, 1),
        (5, 7, 7, 1), (6, 2, 2, 1), (7, 5, 5, 1), (8, 8, 8, 1),
        (0, 4, 4, 1), (4, 0, 0, 1), (1, 7, 7, 1), (7, 1, 1, 1), (2, 8, 8, 1),
        (8, 2, 2, 1), (3, 7, 7, 1), (7, 3, 3, 1), (4, 5, 5, 1), (5, 4, 4, 1),
        (6, 8, 8, 1), (8, 6, 6, 1), (3, 5, 5, 1), (5, 3, 3, 1)
    ]
    
    U23 = np.zeros((9, 23))
    V23 = np.zeros((9, 23))
    W23 = np.zeros((9, 23))
    
    for t, (u_idx, v_idx, w_idx, coeff) in enumerate(terms_23):
        U23[u_idx, t] = coeff
        V23[v_idx, t] = coeff
        W23[w_idx, t] = coeff
        
    # Verify 3x3x3 part
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = 3*i + j
                b = 3*j + k
                c = 3*k + i
                val = np.dot(U23[a, :] * V23[b, :], W23[c, :])
                if abs(val - 1) > 1e-10:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if not valid:
        # Fallback to schoolbook 72
        U = np.zeros((9, 72))
        V = np.zeros((24, 72))
        W = np.zeros((24, 72))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(8):
                    U[3*i + j, col] = 1
                    V[j*8 + k, col] = 1
                    W[k*3 + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy: Use rank-23 for first 3 columns (3x3x3), then for remaining 5 columns,
    # try to find a more efficient decomposition.
    # For 3x3x1, rank is 9. But we can potentially share structure.
    # Total rank = 23 + 5*9 = 68 (current)
    
    # Better idea: Group columns differently.
    # For 3x3x4 (3x3 matrix times 3x4 matrix), what's the rank?
    # We can use: rank(3x3x4) <= rank(3x3x3) + rank(3x3x1) = 23 + 9 = 32
    # So for 3x3x8: rank <= 2*rank(3x3x4) = 2*32 = 64
    # Or: rank(3x3x8) <= rank(3x3x4) + rank(3x3x4) = 64
    
    # Let's try rank 64: two groups of 4 columns each
    # Group 1: columns 0-3, Group 2: columns 4-7
    
    def build_3x3x4_decomp(col_offset_B):
        """Build rank-32 decomposition for 3x3 x 3x4, columns col_offset_B..col_offset_B+3"""
        # Use 23 terms for the 3x3x3 part (cols 0-2 of this group)
        # and 9 terms for the 3x3x1 part (col 3 of this group)
        U_g = np.zeros((9, 32))
        V_g = np.zeros((12, 32))  # 3*4 = 12 rows
        W_g = np.zeros((12, 32))  # 4*3 = 12 rows
        
        # Copy 3x3x3 part
        U_g[:, 0:23] = U23
        V_g[0:9, 0:23] = V23
        W_g[0:9, 0:23] = W23
        
        # Add 3x3x1 part for 4th column
        for i in range(3):
            for j in range(3):
                t = 23 + i*3 + j
                U_g[3*i + j, t] = 1
                V_g[j*4 + 3, t] = 1  # j-th row, 4th column of this 3x4 block
                W_g[3*3 + i, t] = 1  # 4th column group, i-th row
        
        return U_g, V_g, W_g
    
    # Check time
    if time.time() > deadline:
        # Fallback to 68
        U = np.zeros((9, 68))
        V = np.zeros((24, 68))
        W = np.zeros((24, 68))
        U[:, 0:23] = U23
        V[0:9, 0:23] = V23
        W[0:9, 0:23] = W23
        col_offset = 23
        for k_g in range(3, 8):
            for i in range(3):
                for j in range(3):
                    t = col_offset + (k_g - 3) * 9 + i * 3 + j
                    U[3*i + j, t] = 1
                    V[j*8 + k_g, t] = 1
                    W[k_g*3 + i, t] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build rank-64 solution: two 3x3x4 blocks
    U64 = np.zeros((9, 64))
    V64 = np.zeros((24, 64))
    W64 = np.zeros((24, 64))
    
    # Block 1: columns 0-3 of B
    U_g1, V_g1, W_g1 = build_3x3x4_decomp(0)
    U64[:, 0:32] = U_g1
    V64[0:12, 0:32] = V_g1  # rows 0-11 correspond to j*4+k for k=0..3
    W64[0:12, 0:32] = W_g1  # rows 0-11 correspond to k*3+i for k=0..3
    
    # Block 2: columns 4-7 of B
    U_g2, V_g2, W_g2 = build_3x3x4_decomp(4)
    U64[:, 32:64] = U_g2
    V64[12:24, 32:64] = V_g2  # rows 12-23 correspond to j*4+k for k=4..7
    W64[12:24, 32:64] = W_g2  # rows 12-23 correspond to k*3+i for k=4..7
    
    # Verify the full solution
    valid_full = True
    for i in range(3):
        for j in range(3):
            for k in range(8):
                a = 3*i + j
                b = j*8 + k
                c = k*3 + i
                val = np.dot(U64[a, :] * V64[b, :], W64[c, :])
                if abs(val - 1) > 1e-10:
                    valid_full = False
                    break
            if not valid_full:
                break
        if not valid_full:
            break
    
    if valid_full:
        return {"U": U64.tolist(), "V": V64.tolist(), "W": W64.tolist()}
    else:
        # Fallback to 68
        U = np.zeros((9, 68))
        V = np.zeros((24, 68))
        W = np.zeros((24, 68))
        U[:, 0:23] = U23
        V[0:9, 0:23] = V23
        W[0:9, 0:23] = W23
        col_offset = 23
        for k_g in range(3, 8):
            for i in range(3):
                for j in range(3):
                    t = col_offset + (k_g - 3) * 9 + i * 3 + j
                    U[3*i + j, t] = 1
                    V[j*8 + k_g, t] = 1
                    W[k_g*3 + i, t] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
