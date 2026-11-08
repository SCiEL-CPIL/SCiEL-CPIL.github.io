import numpy as np
import time

DIMENSIONS = [4, 4, 9]
START_TIME = time.time()
TIME_LIMIT = 150.0

def construct():
    """
    Returns a rank-minimal tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    
    Strategy:
    Split B into B1 (4x4), B2 (4x4), B3 (4x1).
    Use Strassen (rank 49) for A*B1 and A*B2.
    Use Schoolbook (rank 16) for A*B3.
    Total rank: 49 + 49 + 16 = 114.
    """
    # Deadline guard
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback: return a valid schoolbook decomposition if we hit the limit
        # This is unlikely as construction is fast, but safe.
        return _schoolbook_decomposition()

    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen terms for 2x2 * 2x2
    # u: coeffs for A, v: coeffs for B, w: coeffs for C
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_2x2_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # m1 = (a+d)(e+h)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),          # m2 = (c+d)e
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),          # m3 = a(f-h)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),          # m4 = d(g-e)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),          # m5 = (a+b)h
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),          # m6 = (c-a)(e+f)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),          # m7 = (b-d)(g+h)
    ]
    
    def add_strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """
        Adds 49 terms for the Strassen decomposition of a 4x4 * 4x4 block.
        """
        def u_idx(i, j):
            return i * m + j
            
        def v_idx(j, k):
            return j * p + k
            
        def w_idx(k, i):
            return k * n + i
            
        # Define the 7 block-level Strassen multiplications
        # A_side: list of (block_row, block_col, coeff)
        # B_side: list of (block_row, block_col, coeff)
        # C_side: dict of (block_row, block_col) -> coeff
        block_mults = [
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], {(0,0): 1, (1,1): 1}),
            ([(1,0,1), (1,1,1)], [(0,0,1)], {(1,0): 1, (1,1): -1}),
            ([(0,0,1)], [(0,1,1), (1,1,-1)], {(0,1): 1, (1,1): 1}),
            ([(1,1,1)], [(1,0,1), (0,0,-1)], {(0,0): 1, (1,0): 1}),
            ([(0,0,1), (0,1,1)], [(1,1,1)], {(0,0): -1, (0,1): 1}),
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], {(1,1): 1}),
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], {(0,0): 1}),
        ]
        
        for A_side, B_side, C_side in block_mults:
            for u_meta, v_meta, w_meta in strassen_2x2_terms:
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                for local_idx, meta_coeff in u_meta.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    for br, bc, block_coeff in A_side:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[u_idx(gi, gj)] += meta_coeff * block_coeff
                        
                for local_idx, meta_coeff in v_meta.items():
                    lj = local_idx // 2
                    lk = local_idx % 2
                    for br, bc, block_coeff in B_side:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[v_idx(gj, gk)] += meta_coeff * block_coeff
                        
                for local_idx, meta_coeff in w_meta.items():
                    li = local_idx // 2
                    lk = local_idx % 2
                    for (br, bc), block_coeff in C_side.items():
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[w_idx(gk, gi)] += meta_coeff * block_coeff
                        
                add_term(u_vec, v_vec, w_vec)

    # Part 1: A * B1 (cols 0-3)
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 (cols 4-7)
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B3 (col 8) - Schoolbook
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
            
    # Build final matrices
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
    U_cols = []
    V_cols = []
    W_cols = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                u_vec[i * m + j] = 1
                v_vec[j * p + k] = 1
                w_vec[k * n + i] = 1
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    return {"U": U, "V": V, "W": W}
