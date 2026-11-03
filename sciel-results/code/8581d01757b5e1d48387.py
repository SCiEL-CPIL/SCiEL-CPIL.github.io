"""Tensor decomposition search for 3x5x5 matrix multiplication."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=5, p=5
    
    # We decompose the multiplication into 4 blocks:
    # A is 3x5, split into A_L (3x3) and A_R (3x2)
    # B is 5x5, split into B_LL (3x3), B_LR (3x2), B_RL (2x3), B_RR (2x2)
    # C is 3x5, split into C_LL (3x3), C_LR (3x2), C_RL (3x3), C_RR (3x2)
    # Note: C = A B = [A_L A_R] [B_LL B_LR; B_RL B_RR] = [A_L B_LL + A_R B_RL, A_L B_LR + A_R B_RR]
    # C_LL = A_L B_LL + A_R B_RL (3x3)
    # C_LR = A_L B_LR + A_R B_RR (3x2)
    
    # We will construct the decomposition for each part and sum them.
    # Since the blocks are independent in terms of indices (disjoint support in U, V, W),
    # we can just concatenate the columns of U, V, W.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx):
        # u_idx is (i, j) in A
        # v_idx is (j, k) in B
        # w_idx is (k, i) in C
        i, j = u_idx
        j_v, k = v_idx
        k_w, i_w = w_idx
        # Verify consistency
        if i != i_w or j != j_v or k != k_w:
            raise ValueError("Index mismatch")
        
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        u_vec[i * m + j] = 1.0
        v_vec[j * p + k] = 1.0
        w_vec[k * n + i] = 1.0
        
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Block 1: A_L (3x3) x B_LL (3x3) -> C_LL (3x3)
    # Schoolbook: 27 terms
    for i in range(3):
        for j in range(3):
            for k in range(3):
                add_term((i, j), (j, k), (k, i))

    # Block 2: A_L (3x3) x B_LR (3x2) -> C_LR (3x2)
    # Dot product: 3 terms (one for each j in 0..2)
    # C_ik = sum_j A_ij B_jk
    # Term j: U picks A_ij, V picks B_jk, W picks C_ik
    # V picks B_jk for k in 3..4. So V has 1s at (j,3) and (j,4).
    # W picks C_ik for k in 3..4. So W has 1s at (3,i) and (4,i).
    for j in range(3):
        # We need to represent the term that multiplies A_ij and B_jk and contributes to C_ik
        # This is a single rank-1 term in the tensor space?
        # Yes, for fixed j, the tensor is T_{i,j,k} = 1 for all i, k.
        # This is the outer product of (1,1,1) in i, (delta_j) in j, (1,1) in k?
        # No, the term is:
        # u_i = 1 for all i? No, u depends on i.
        # The term is: sum_i A_ij * sum_k B_jk * C_ik?
        # No, the decomposition is:
        # C_ik = sum_j (A_ij * B_jk)
        # This is sum_j (Term_j)
        # Term_j = A_ij * B_jk * 1
        # So U_{i,j} = 1, V_{j,k} = 1, W_{k,i} = 1.
        # But V_{j,k} is 1 for k=3,4.
        # So V vector has 1s at indices j*5+3 and j*5+4.
        # W vector has 1s at indices 3*3+i and 4*3+i.
        # U vector has 1s at indices i*5+j for all i.
        
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for i in range(3):
            u_vec[i * m + j] = 1.0
            w_vec[3 * n + i] = 1.0
            w_vec[4 * n + i] = 1.0
            
        v_vec[j * p + 3] = 1.0
        v_vec[j * p + 4] = 1.0
        
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Block 3: A_R (3x2) x B_RL (2x3) -> C_RL (3x3)
    # Dot product: 2 terms (j in 3..4)
    # C_ik = sum_j A_ij B_jk
    # j is 3,4. k is 0..2. i is 0..2.
    for j in range(3, 5):
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for i in range(3):
            u_vec[i * m + j] = 1.0
            for k in range(3):
                w_vec[k * n + i] = 1.0
                v_vec[j * p + k] = 1.0
        
        # Wait, v_vec[j*p+k] = 1 for all k?
        # Yes, B_jk is picked.
        # And w_vec[k*n+i] = 1 for all k, i?
        # Yes, C_ik is the output.
        
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Block 4: A_R (3x2) x B_RR (2x2) -> C_RR (3x2)
    # Dot product: 2 terms (j in 3..4)
    # C_ik = sum_j A_ij B_jk
    # j is 3,4. k is 3..4. i is 0..2.
    for j in range(3, 5):
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for i in range(3):
            u_vec[i * m + j] = 1.0
            w_vec[3 * n + i] = 1.0
            w_vec[4 * n + i] = 1.0
            
        v_vec[j * p + 3] = 1.0
        v_vec[j * p + 4] = 1.0
        
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Convert lists of vectors to matrices
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    # Return as lists of lists to match interface? 
    # The interface says 'U', 'V', 'W' factor matrices.
    # The seed returned lists of lists.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
