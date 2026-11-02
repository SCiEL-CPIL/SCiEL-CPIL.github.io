import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The rank of M_{2,4,4} is 28.
    # This is derived from the disjoint support property in the A dimension:
    # The multiplication can be decomposed into 4 independent 2x2 matrix multiplications
    # (Strassen, rank 7 each) because the A-blocks (left/right) have disjoint indices
    # and the B-blocks (top/bottom) have disjoint indices relative to the A-blocks usage.
    # Specifically:
    # C_L = A_L * B_TL + A_R * B_BL
    # C_R = A_L * B_TR + A_R * B_BR
    # Since A_L and A_R have disjoint indices, the tensors for (A_L * B_TL) and (A_R * B_BL)
    # have disjoint supports in the U (A) mode, so their ranks add.
    # Same for the other pair.
    # Total Rank = 4 * 7 = 28.
    
    rank = 28
    
    # Using lists for direct construction, then converting to numpy if needed by evaluator
    # The evaluator expects {'U': list, 'V': list, 'W': list}
    
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    # Strassen 2x2 x 2x2 decomposition parameters
    # For A (2x2) and B (2x2) producing C (2x2)
    # Local indices 0..3 for 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # Strassen products M1..M7
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # Output C reconstruction:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Define the 7 rank-1 terms for a 2x2 multiplication.
    # Each term is a tuple (u_indices, v_indices, w_indices)
    # Indices are local 0..3. Coefficients are implicit in the list structure (1 or -1).
    # We store as list of (local_idx, coeff)
    
    strassen_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    def map_u_a(a_side, local_idx):
        """
        Map local 2x2 index to global U index (A matrix 2x4).
        A_side: 'L' for cols 0,1; 'R' for cols 2,3.
        Local idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        Global A index: row * m + col = row * 4 + col
        """
        r = local_idx // 2
        c = local_idx % 2
        if a_side == 'L':
            global_c = c
        else:
            global_c = 2 + c
        return r * 4 + global_c

    def map_v_b(b_block, local_idx):
        """
        Map local 2x2 index to global V index (B matrix 4x4).
        b_block: 'TL', 'TR', 'BL', 'BR'
        Local idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        Global B index: row * p + col = row * 4 + col
        """
        r = local_idx // 2
        c = local_idx % 2
        
        if b_block == 'TL':
            global_r = r
            global_c = c
        elif b_block == 'TR':
            global_r = r
            global_c = 2 + c
        elif b_block == 'BL':
            global_r = 2 + r
            global_c = c
        elif b_block == 'BR':
            global_r = 2 + r
            global_c = 2 + c
            
        return global_r * 4 + global_c

    def map_w_c(c_side, local_idx):
        """
        Map local 2x2 index to global W index (C matrix 2x4).
        Trace convention: W[k*n + i, r] where k=col, i=row.
        c_side: 'L' for cols 0,1; 'R' for cols 2,3.
        Local idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        Local (r, c) corresponds to global row r, global col c.
        W index = global_col * n + global_row = global_col * 2 + r
        """
        r = local_idx // 2
        c = local_idx % 2
        
        if c_side == 'L':
            global_c = c
        else:
            global_c = 2 + c
            
        return global_c * 2 + r

    def add_strassen_block(a_side, b_block, c_side, start_col):
        for t in strassen_terms:
            u_local, v_local, w_local = t
            col_idx = start_col
            
            for idx, coeff in u_local:
                g_idx = map_u_a(a_side, idx)
                U[g_idx][col_idx] += coeff
            
            for idx, coeff in v_local:
                g_idx = map_v_b(b_block, idx)
                V[g_idx][col_idx] += coeff
            
            for idx, coeff in w_local:
                g_idx = map_w_c(c_side, idx)
                W[g_idx][col_idx] += coeff
                
            start_col += 1
        
        return start_col

    # Apply the 4 blocks
    # 1. A_L * B_TL -> C_L
    col = add_strassen_block('L', 'TL', 'L', 0)
    
    # 2. A_R * B_BL -> C_L
    col = add_strassen_block('R', 'BL', 'L', col)
    
    # 3. A_L * B_TR -> C_R
    col = add_strassen_block('L', 'TR', 'R', col)
    
    # 4. A_R * B_BR -> C_R
    col = add_strassen_block('R', 'BR', 'R', col)
    
    return {"U": U, "V": V, "W": W}
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
