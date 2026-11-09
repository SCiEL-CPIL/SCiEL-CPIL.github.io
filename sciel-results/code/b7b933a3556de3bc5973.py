import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank calculation based on block decomposition
    # Block 1: A(3x2) * B(2x6)
    # Split A into 1x2 and 2x2.
    # 1x2 * 2x6: 1x2x2 blocks. 3 blocks of 2x2 in B. 3 * (1*2*2) = 12? 
    # Wait, 1x2 * 2x2 is rank 4 schoolbook. 3 such blocks -> 12.
    # 2x2 * 2x6: 2x2x2 blocks. 3 blocks. 3 * 7 (Strassen) = 21.
    # Total Block 1: 33.
    
    # Block 2: A(3x3) * B(3x6)
    # Split A into 1x3 and 2x3.
    # 1x3 * 3x6: 1x3x6 schoolbook. Rank 18.
    # 2x3 * 3x6: Split B into 1x6 and 2x6.
    #   2x1 * 1x6: 2x1x6 schoolbook. Rank 12.
    #   2x2 * 2x6: Split B into 2x2 blocks. 3 blocks. 3 * 7 (Strassen) = 21.
    # Total Block 2: 18 + 12 + 21 = 51.
    
    total_rank = 33 + 51
    rank = total_rank
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def get_u_idx(i, j):
        return i * m + j
        
    def get_v_idx(j, k):
        return j * p + k
        
    def get_w_idx(k, i):
        return k * n + i

    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val
        col += 1

    def add_schoolbook(n_b, m_b, p_b, i_off, j_off, k_off):
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u_idx = get_u_idx(i_off + i, j_off + j)
                    v_idx = get_v_idx(j_off + j, k_off + k)
                    w_idx = get_w_idx(k_off + k, i_off + i)
                    add_term(u_idx, v_idx, w_idx, 1.0, 1.0, 1.0)

    def add_strassen(i_off, j_off, k_off):
        # A indices
        u11 = get_u_idx(i_off, j_off)
        u12 = get_u_idx(i_off, j_off + 1)
        u21 = get_u_idx(i_off + 1, j_off)
        u22 = get_u_idx(i_off + 1, j_off + 1)
        
        # B indices
        v11 = get_v_idx(j_off, k_off)
        v12 = get_v_idx(j_off, k_off + 1)
        v21 = get_v_idx(j_off + 1, k_off)
        v22 = get_v_idx(j_off + 1, k_off + 1)
        
        # C (W) indices
        # C11 -> W(k_off, i_off)
        w11 = get_w_idx(k_off, i_off)
        # C12 -> W(k_off+1, i_off)
        w12 = get_w_idx(k_off + 1, i_off)
        # C21 -> W(k_off, i_off+1)
        w21 = get_w_idx(k_off, i_off + 1)
        # C22 -> W(k_off+1, i_off+1)
        w22 = get_w_idx(k_off + 1, i_off + 1)
        
        # M1 = (A11 + A22)(B11 + B22)
        # C11 += M1, C22 += M1
        add_term(u11, v11, w11, 1.0, 1.0, 1.0)
        add_term(u22, v11, w11, 1.0, 1.0, 0.0) # Wait, this adds to existing col? No, add_term increments col.
        
        # The helper `add_term` increments `col`.
        # Strassen requires combining terms into the SAME column? No.
        # Strassen creates 7 new terms (columns).
        # Each column corresponds to one multiplication $M_k$.
        # The U, V, W vectors for that column contain the coefficients.
        # So for M1, we need ONE column where U has 1 at u11 and 1 at u22, V has 1 at v11 and 1 at v22, W has 1 at w11 and 1 at w22.
        
        # My `add_term` function adds to `U[u_idx, col]` and then increments `col`.
        # This is suitable for schoolbook where each term is distinct.
        # For Strassen, I need to construct the vectors for the 7 multiplications and then assign them to 7 new columns.
        
        # Let's refactor.
        # Instead of `add_term` which advances col, I'll build the column vectors for the 7 Strassen terms and then append them.
        
        # But `add_schoolbook` uses `add_term` which works because each (i,j,k) is a unique term.
        # For Strassen, I will manually construct the 7 columns.
        
        # Reset logic for this function:
        # I need to write to U, V, W at the current `col` .. `col+6`.
        
        # M1
        c = col
        U[u11, c] += 1.0
        U[u22, c] += 1.0
        V[v11, c] += 1.0
        V[v22, c] += 1.0
        W[w11, c] += 1.0
        W[w22, c] += 1.0
        col += 1
        
        # M2 = (A21 + A22)B11
        c = col
        U[u21, c] += 1.0
        U[u22, c] += 1.0
        V[v11, c] += 1.0
        W[w21, c] += 1.0
        col += 1
        
        # M3 = A11(B21 - B22)
        c = col
        U[u11, c] += 1.0
        V[v21, c] += 1.0
        V[v22, c] -= 1.0
        W[w12, c] += 1.0
        W[w22, c] += 1.0
        col += 1
        
        # M4 = A22(B12 - B11)
        c = col
        U[u22, c] += 1.0
        V[v12, c] += 1.0
        V[v11, c] -= 1.0
        W[w11, c] += 1.0
        W[w21, c] += 1.0
        col += 1
        
        # M5 = (A11 + A12)B22
        c = col
        U[u11, c] += 1.0
        U[u12, c] += 1.0
        V[v22, c] += 1.0
        W[w11, c] -= 1.0
        W[w12, c] += 1.0
        col += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        c = col
        U[u21, c] += 1.0
        U[u11, c] -= 1.0
        V[v11, c] += 1.0
        V[v12, c] += 1.0
        W[w22, c] += 1.0
        col += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        c = col
        U[u12, c] += 1.0
        U[u22, c] -= 1.0
        V[v21, c] += 1.0
        V[v22, c] += 1.0
        W[w11, c] += 1.0
        col += 1

    # Block 1: A(3x2) * B(2x6)
    # A1: rows 0, cols 0-1. B1: rows 0-1, cols 0-5.
    # Split A1 into 1x2 (row 0) and 2x2 (rows 1-2).
    
    # A1a (1x2) * B1 (2x6)
    # i_off=0, j_off=0.
    # B1 split into 3 blocks of 2x2.
    for k_block in range(3):
        add_schoolbook(1, 2, 2, 0, 0, k_block * 2)
        
    # A1b (2x2) * B1 (2x6)
    # i_off=1, j_off=0.
    for k_block in range(3):
        add_strassen(1, 0, k_block * 2)

    # Block 2: A(3x3) * B(3x6)
    # A2: rows 0-2, cols 2-4. B2: rows 2-4, cols 0-5.
    # Split A2 into 1x3 (row 0) and 2x3 (rows 1-2).
    
    # A2a (1x3) * B2 (3x6)
    # i_off=0, j_off=2.
    add_schoolbook(1, 3, 6, 0, 2, 0)
    
    # A2b (2x3) * B2 (3x6)
    # i_off=1, j_off=2.
    # Split B2 into 1x6 (row 2) and 2x6 (rows 3-4).
    
    # A2b1 (2x1) * B21 (1x6)
    # i_off=1, j_off=2.
    add_schoolbook(2, 1, 6, 1, 2, 0)
    
    # A2b2 (2x2) * B22 (2x6)
    # i_off=1, j_off=3.
    for k_block in range(3):
        add_strassen(1, 3, k_block * 2)

    if col != rank:
        print(f"Rank mismatch: expected {rank}, got {col}")
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
