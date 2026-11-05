import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0  # seconds


def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5

    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def add_term(i, k, j, coeff_u=1.0, coeff_v=1.0, coeff_w=1.0):
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        u_vec[i * m + k] = coeff_u
        v_vec[k * p + j] = coeff_v
        w_vec[j * n + i] = coeff_w
        add_rank1(u_vec, v_vec, w_vec)

    # The 4x5x5 matrix multiplication can be decomposed as:
    # A (4x5) * B (5x5) = C (4x5)
    # Split A into A_left (4x4) and A_right (4x1)
    # Split B into B_top (5x4) and B_bottom (5x1)
    # C = A_left * B_top + A_right * B_bottom
    
    # Wait, that's not right. Let me think again.
    # A is 4x5, B is 5x5.
    # We can split the 5 columns of A and 5 rows of B.
    # Let's split at column/row 4 (the last one).
    # A = [A0 | A1] where A0 is 4x4, A1 is 4x1
    # B = [B0; B1] where B0 is 4x5, B1 is 1x5
    # C = A0*B0 + A1*B1
    
    # A0*B0 is 4x4 * 4x5 -> 4x5. This is not square.
    # Let's split differently.
    # A is 4x5. B is 5x5.
    # Let's split the inner dimension 5 into 4+1.
    # A = [A_left (4x4) | A_right (4x1)]
    # B = [B_top (4x5); B_bottom (1x5)]
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top: 4x4 * 4x5 -> 4x5
    # A_right * B_bottom: 4x1 * 1x5 -> 4x5
    
    # The term A_right * B_bottom has rank 4*1*5 = 20 (schoolbook).
    # The term A_left * B_top is 4x4 * 4x5.
    # We can further split B_top into B_top_left (4x4) and B_top_right (4x1).
    # A_left * B_top = A_left * B_top_left + A_left * B_top_right
    
    # A_left * B_top_left: 4x4 * 4x4 -> 4x4. Rank 28 (Strassen 4x4).
    # A_left * B_top_right: 4x4 * 4x1 -> 4x1. Rank 16 (schoolbook).
    # A_right * B_bottom: 4x1 * 1x5 -> 4x5. Rank 20 (schoolbook).
    
    # Total rank: 28 + 16 + 20 = 64.
    
    # Let's implement this.
    
    # Part 1: A_left * B_top_left (4x4 * 4x4)
    # Indices: A[i,k] for i,k in 0..3 -> U[i*5+k]
    #          B[k,j] for k,j in 0..3 -> V[k*5+j]
    #          C[i,j] for i,j in 0..3 -> W[j*4+i]
    
    # Use the 28-term 4x4x4 decomposition.
    # I will use the known 28-term decomposition for 4x4x4.
    
    # The 28-term decomposition for 4x4x4 matrix multiplication:
    # This is a known result. I'll implement it explicitly.
    
    # Let's use the following 28-term scheme for 4x4x4:
    # (I'll hardcode the 28 bilinear forms)
    
    # Actually, implementing the 28-term 4x4x4 decomposition correctly is complex.
    # Let's use the 49-term nested Strassen for 4x4x4 for simplicity and correctness,
    # which gives 49 + 16 + 20 = 85.
    
    # But we want to achieve 64. Let's try to implement the 28-term 4x4x4.
    
    # Here is a known 28-term decomposition for 4x4x4:
    # (From the literature, e.g., Strassen's 1969 paper or later works)
    
    # I'll use a different approach: use the 49-term nested Strassen but see if I can
    # reduce the other parts.
    
    # Actually, let's just use the 49-term approach for now and see if we can improve.
    
    # Define the 7 Strassen products for 4x4x4 (outer level)
    strassen_products = [
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}},  # M1
        {'A': {2: 1, 3: 1}, 'B': {0: 1}},          # M2
        {'A': {0: 1}, 'B': {1: 1, 3: -1}},         # M3
        {'A': {3: 1}, 'B': {2: 1, 0: -1}},         # M4
        {'A': {0: 1, 1: 1}, 'B': {3: 1}},          # M5
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}},   # M6
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}},   # M7
    ]
    
    c_contributions = [
        {0: 1, 3: 1},   # M1
        {2: 1, 3: -1},  # M2
        {1: 1, 3: 1},   # M3
        {0: 1, 2: 1},   # M4
        {0: -1, 1: 1},  # M5
        {3: 1},         # M6
        {0: 1},         # M7
    ]
    
    def a_global(block, li, lk):
        if block == 0:
            return li, lk
        elif block == 1:
            return li, lk + 2
        elif block == 2:
            return li + 2, lk
        elif block == 3:
            return li + 2, lk + 2
    
    def b_global(block, lk, lj):
        if block == 0:
            return lk, lj
        elif block == 1:
            return lk, lj + 2
        elif block == 2:
            return lk + 2, lj
        elif block == 3:
            return lk + 2, lj + 2
    
    def c_global(block, li, lj):
        if block == 0:
            return li, lj
        elif block == 1:
            return li, lj + 2
        elif block == 2:
            return li + 2, lj
        elif block == 3:
            return li + 2, lj + 2
    
    inner_strassen = [
        {'A': [(0,0,1),(1,1,1)], 'B': [(0,0,1),(1,1,1)], 'C': [(0,0,1),(1,1,1)]},
        {'A': [(1,0,1),(1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1),(1,1,-1)]},
        {'A': [(0,0,1)], 'B': [(0,1,1),(1,1,-1)], 'C': [(0,1,1),(1,1,1)]},
        {'A': [(1,1,1)], 'B': [(1,0,1),(0,0,-1)], 'C': [(0,0,1),(1,0,1)]},
        {'A': [(0,0,1),(0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1),(0,1,1)]},
        {'A': [(1,0,1),(0,0,-1)], 'B': [(0,0,1),(0,1,1)], 'C': [(1,1,1)]},
        {'A': [(0,1,1),(1,1,-1)], 'B': [(1,0,1),(1,1,1)], 'C': [(0,0,1)]},
    ]
    
    for prod_idx, prod in enumerate(strassen_products):
        a_form = prod['A']
        b_form = prod['B']
        c_contrib = c_contributions[prod_idx]
        
        for inner_idx, inner in enumerate(inner_strassen):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            for a_block, a_sign in a_form.items():
                for li, lk, inner_sign in inner['A']:
                    gi, gk = a_global(a_block, li, lk)
                    u_vec[gi * m + gk] += a_sign * inner_sign
            
            for b_block, b_sign in b_form.items():
                for lk, lj, inner_sign in inner['B']:
                    gk, gj = b_global(b_block, lk, lj)
                    v_vec[gk * p + gj] += b_sign * inner_sign
            
            for c_block, c_sign in c_contrib.items():
                for li, lj, inner_sign in inner['C']:
                    gi, gj = c_global(c_block, li, lj)
                    w_vec[gj * n + gi] += c_sign * inner_sign
            
            add_rank1(u_vec, v_vec, w_vec)
    
    # Part 2: A_left * B_top_right (4x4 * 4x1)
    # A[i,k] for i,k in 0..3 -> U[i*5+k]
    # B[k,4] for k in 0..3 -> V[k*5+4]
    # C[i,4] for i in 0..3 -> W[4*4+i] = W[16+i]
    
    for i in range(4):
        for k in range(4):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            u_vec[i * m + k] = 1.0
            v_vec[k * p + 4] = 1.0
            w_vec[4 * n + i] = 1.0
            add_rank1(u_vec, v_vec, w_vec)
    
    # Part 3: A_right * B_bottom (4x1 * 1x5)
    # A[i,4] for i in 0..3 -> U[i*5+4]
    # B[4,j] for j in 0..4 -> V[4*5+j] = V[20+j]
    # C[i,j] for i,j in 0..4 -> W[j*4+i]
    
    for i in range(4):
        for j in range(5):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            u_vec[i * m + 4] = 1.0
            v_vec[4 * p + j] = 1.0
            w_vec[j * n + i] = 1.0
            add_rank1(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    
    if rank == 0:
        add_term(0, 0, 0, 1.0, 1.0, 1.0)
        rank = 1
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    # Deadline guard
    elapsed = time.time() - _START
    if elapsed > _DEADLINE - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
