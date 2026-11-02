import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact tensor decomposition for 3x4x7 matrix multiplication.
    Uses Strassen's algorithm on 2x2x2 sub-blocks to reduce rank from 84 to 78.
    """
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # We will build U, V, W as lists of lists to allow easy appending.
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def make_vec(size, indices_values):
        vec = [0.0] * size
        for idx, val in indices_values:
            vec[idx] = val
        return vec

    # Define Strassen 2x2 block
    # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 (local)
    # B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 (local)
    # C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 (local)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Local to Global mapping for a block defined by I, J, K
    # I: list of 2 row indices in A
    # J: list of 2 col indices in A (row indices in B)
    # K: list of 2 col indices in B
    
    def add_strassen_block(I, J, K):
        # I[0], I[1] are global i
        # J[0], J[1] are global j
        # K[0], K[1] are global k
        
        # Local indices in U (size n*m): i*m + j
        u_00 = I[0]*m + J[0]
        u_01 = I[0]*m + J[1]
        u_10 = I[1]*m + J[0]
        u_11 = I[1]*m + J[1]
        
        # Local indices in V (size m*p): j*p + k
        v_00 = J[0]*p + K[0]
        v_01 = J[0]*p + K[1]
        v_10 = J[1]*p + K[0]
        v_11 = J[1]*p + K[1]
        
        # Local indices in W (size p*n): k*n + i
        w_00 = K[0]*n + I[0]
        w_01 = K[1]*n + I[0]
        w_10 = K[0]*n + I[1]
        w_11 = K[1]*n + I[1]
        
        # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        add_term(
            make_vec(n*m, [(u_00, 1), (u_11, 1)]),
            make_vec(m*p, [(v_00, 1), (v_11, 1)]),
            make_vec(p*n, [(w_00, 1), (w_11, 1)])
        )
        
        # M2: (A10+A11)B00 -> C10 += 1
        add_term(
            make_vec(n*m, [(u_10, 1), (u_11, 1)]),
            make_vec(m*p, [(v_00, 1)]),
            make_vec(p*n, [(w_10, 1)])
        )
        
        # M3: A00(B10-B11) -> C01 += 1
        add_term(
            make_vec(n*m, [(u_00, 1)]),
            make_vec(m*p, [(v_10, 1), (v_11, -1)]),
            make_vec(p*n, [(w_01, 1)])
        )
        
        # M4: A11(B01-B00) -> C00 += 1, C10 += 1
        add_term(
            make_vec(n*m, [(u_11, 1)]),
            make_vec(m*p, [(v_01, 1), (v_00, -1)]),
            make_vec(p*n, [(w_00, 1), (w_10, 1)])
        )
        
        # M5: (A00+A01)B11 -> C00 -= 1, C01 += 1
        add_term(
            make_vec(n*m, [(u_00, 1), (u_01, 1)]),
            make_vec(m*p, [(v_11, 1)]),
            make_vec(p*n, [(w_00, -1), (w_01, 1)])
        )
        
        # M6: (A10-A00)(B00+B01) -> C11 += 1
        add_term(
            make_vec(n*m, [(u_10, 1), (u_00, -1)]),
            make_vec(m*p, [(v_00, 1), (v_01, 1)]),
            make_vec(p*n, [(w_11, 1)])
        )
        
        # M7: (A01-A11)(B10+B11) -> C00 += 1, C11 -= 1
        add_term(
            make_vec(n*m, [(u_01, 1), (u_11, -1)]),
            make_vec(m*p, [(v_10, 1), (v_11, 1)]),
            make_vec(p*n, [(w_00, 1), (w_11, -1)])
        )

    # Define the blocks
    # I: {0, 1}
    # J: {0, 1}, {2, 3}
    # K: {0, 1}, {2, 3}, {4, 5}
    
    I_block = [0, 1]
    J_blocks = [[0, 1], [2, 3]]
    K_blocks = [[0, 1], [2, 3], [4, 5]]
    
    for J in J_blocks:
        for K in K_blocks:
            add_strassen_block(I_block, J, K)
            
    # Add remaining schoolbook terms
    # Terms where i=2
    for j in range(m):
        for k in range(p):
            add_term(
                make_vec(n*m, [(2*m + j, 1)]),
                make_vec(m*p, [(j*p + k, 1)]),
                make_vec(p*n, [(k*n + 2, 1)])
            )
            
    # Terms where i in {0,1} and k=6
    for i in [0, 1]:
        for j in range(m):
            add_term(
                make_vec(n*m, [(i*m + j, 1)]),
                make_vec(m*p, [(j*p + 6, 1)]),
                make_vec(p*n, [(6*n + i, 1)])
            )
            
    # Construct final matrices
    R = len(U_cols)
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    for r in range(R):
        u_vec = U_cols[r]
        v_vec = V_cols[r]
        w_vec = W_cols[r]
        for idx, val in enumerate(u_vec):
            if val != 0:
                U[idx][r] = val
        for idx, val in enumerate(v_vec):
            if val != 0:
                V[idx][r] = val
        for idx, val in enumerate(w_vec):
            if val != 0:
                W[idx][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
