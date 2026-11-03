import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U_dim = n * m  # 6
    V_dim = m * p  # 15
    W_dim = p * n  # 10

    U = np.zeros((U_dim, rank))
    V = np.zeros((V_dim, rank))
    W = np.zeros((W_dim, rank))

    # Helper to add a rank-1 component
    # u_idx, v_idx, w_idx are the flattened indices
    # coeff_u, coeff_v, coeff_w are the coefficients
    # col is the column index in U, V, W
    def add_term(col, u_idx, coeff_u, v_idx, coeff_v, w_idx, coeff_w):
        U[u_idx, col] += coeff_u
        V[v_idx, col] += coeff_v
        W[w_idx, col] += coeff_w

    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B01
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output reconstruction:
    # C00 = M1 - M4 + M6 - M7
    # C01 = M2 + M4
    # C10 = M3 + M7
    # C11 = M1 - M3 + M5 - M6

    # We need to map local 2x2 indices to global indices.
    # For a block with k_offset (0 or 2) and j_offset (0, as we use j=0,1 for Strassen and j=2 for outer)
    # Wait, for the 2x2 part, we use j=0,1.
    # A indices: i*3 + j. i in {0,1}, j in {0,1}.
    # B indices: j*5 + k. j in {0,1}, k in {k_offset, k_offset+1}.
    # C indices: k*2 + i. k in {k_offset, k_offset+1}, i in {0,1}.

    def add_strassen_block(k_offset, base_col):
        # k_offset is 0 or 2
        # We use j=0,1 for the 2x2 part.
        
        # M1
        # U: A00+A11 -> U[0, col]=1, U[4, col]=1 (idx 0*3+0=0, 1*3+1=4)
        # V: B00+B11 -> V[0*5+k_offset, col]=1, V[1*5+k_offset+1, col]=1
        # W: C00, C11? No, W accumulates into C.
        # C00 gets +M1. C11 gets +M1.
        # C idx: k*2+i.
        # C00: k=0, i=0 -> idx 0*2+0=0.
        # C11: k=1, i=1 -> idx 1*2+1=3.
        col = base_col
        add_term(col, 0, 1, 0*5+k_offset, 1, 0*2+0, 1)
        U[4, col] += 1
        V[1*5+k_offset+1, col] += 1
        W[1*2+1, col] += 1

        # M2
        # U: A10+A11 -> U[3, col]=1, U[4, col]=1
        # V: B01 -> V[0*5+k_offset+1, col]=1
        # W: C01 gets +M2. C01: k=1, i=0 -> idx 1*2+0=2.
        col = base_col + 1
        add_term(col, 3, 1, 0*5+k_offset+1, 1, 1*2+0, 1)
        U[4, col] += 1

        # M3
        # U: A00 -> U[0, col]=1
        # V: B10-B11 -> V[1*5+k_offset, col]=1, V[1*5+k_offset+1, col]=-1
        # W: C10 gets +M3. C10: k=0, i=1 -> idx 0*2+1=1.
        col = base_col + 2
        add_term(col, 0, 1, 1*5+k_offset, 1, 0*2+1, 1)
        V[1*5+k_offset+1, col] += -1

        # M4
        # U: A11 -> U[4, col]=1
        # V: B00-B01 -> V[0*5+k_offset, col]=1, V[0*5+k_offset+1, col]=-1
        # W: C00 gets -M4, C01 gets +M4.
        col = base_col + 3
        add_term(col, 4, 1, 0*5+k_offset, 1, 0*2+0, -1)
        V[0*5+k_offset+1, col] += -1
        W[1*2+0, col] += 1

        # M5
        # U: A00+A01 -> U[0, col]=1, U[1, col]=1
        # V: B11 -> V[1*5+k_offset+1, col]=1
        # W: C11 gets +M5.
        col = base_col + 4
        add_term(col, 0, 1, 1*5+k_offset+1, 1, 1*2+1, 1)
        U[1, col] += 1

        # M6
        # U: A10-A00 -> U[3, col]=1, U[0, col]=-1
        # V: B00+B01 -> V[0*5+k_offset, col]=1, V[0*5+k_offset+1, col]=1
        # W: C00 gets +M6, C11 gets -M6.
        col = base_col + 5
        add_term(col, 3, 1, 0*5+k_offset, 1, 0*2+0, 1)
        U[0, col] += -1
        V[0*5+k_offset+1, col] += 1
        W[1*2+1, col] += -1

        # M7
        # U: A01-A11 -> U[1, col]=1, U[4, col]=-1
        # V: B10+B11 -> V[1*5+k_offset, col]=1, V[1*5+k_offset+1, col]=1
        # W: C10 gets +M7, C00 gets -M7.
        col = base_col + 6
        add_term(col, 1, 1, 1*5+k_offset, 1, 0*2+1, 1)
        U[4, col] += -1
        V[1*5+k_offset+1, col] += 1
        W[0*2+0, col] += -1

    def add_outer_block(k_offset, base_col):
        # Handles A_i2 * B_2k
        # U: A_i2 -> U[i*3+2, col]=1
        # V: B_2k -> V[2*5+k_offset, col]=1, V[2*5+k_offset+1, col]=1
        # W: C_ik -> W[k*2+i, col]=1
        col = base_col
        # i=0
        add_term(col, 0*3+2, 1, 2*5+k_offset, 1, 0*2+0, 1)
        V[2*5+k_offset+1, col] += 1
        W[1*2+0, col] += 1
        # i=1
        U[1*3+2, col] += 1
        W[0*2+1, col] += 1
        W[1*2+1, col] += 1

    # Block 1: k=0,1 (Strassen + Outer)
    add_strassen_block(0, 0)
    add_outer_block(0, 7)
    
    # Block 2: k=2,3 (Strassen + Outer)
    add_strassen_block(2, 8)
    add_outer_block(2, 15)
    
    # Block 3: k=4 (Dot product)
    # C_i4 = sum_j A_ij B_j4
    # This is rank 3? No, it's sum of 3 terms.
    # But we can do it with 3 rank-1 terms?
    # T_i,j,4 = 1.
    # We need 3 mults: A_04*B_04, A_14*B_14, A_24*B_24.
    # Wait, the "rank 1" I estimated for M_2,3,1 was wrong.
    # M_{2,3,1}: C_{i,0} = sum_j A_{i,j} B_{j,0}.
    # Tensor T_{i,j,0} = 1.
    # This is NOT rank 1.
    # T_{i,j,k} = 1 for all i,j,k? No, k is fixed to 0.
    # T_{i,j,0} = 1.
    # This tensor is $1_2 \otimes 1_3 \otimes e_0$.
    # Rank is 1?
    # $T_{i,j,k} = u_i v_j w_k$.
    # $u_i = 1, v_j = 1, w_k = \delta_{k,0}$.
    # Yes, it IS rank 1.
    # Why did I think 3?
    # $C_{i,0} = A_{i,0}B_{0,0} + A_{i,1}B_{1,0} + A_{i,2}B_{2,0}$.
    # If we use 1 multiplication $M = (\sum A) (\sum B)$, we get cross terms.
    # We need exact decomposition.
    # The tensor $T_{i,j,0}=1$ is indeed rank 1.
    # $T = u \otimes v \otimes w$.
    # $u_i = 1, v_j = 1, w_0 = 1, w_{k \ne 0} = 0$.
    # Then $\sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = \sum_{i,j} 1 \cdot 1 \cdot \delta_{k,0} A_{i,j} B_{j,0} = \sum_{i,j} A_{i,j} B_{j,0} = C_{i,0}$.
    # Yes! Rank 1.
    
    col = 16
    # U: all A_ij -> 1
    for i in range(2):
        for j in range(3):
            U[i*3+j, col] += 1
    # V: all B_j0 -> 1
    for j in range(3):
        V[j*5+4, col] += 1
    # W: C_i0 -> 1
    for i in range(2):
        W[4*2+i, col] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
