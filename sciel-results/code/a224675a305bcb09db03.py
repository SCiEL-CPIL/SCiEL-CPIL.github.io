import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # 1. Define 2x2 Strassen coefficients
    # alpha[t][i, j] for A, beta[t][j, k] for B
    # t from 0 to 6
    alpha = []
    beta = []
    
    # P1 = (A00 + A11)(B00 + B11)
    alpha.append(np.array([[1, 0], [0, 1]]))
    beta.append(np.array([[1, 0], [0, 1]]))
    
    # P2 = (A10 + A11)B00
    alpha.append(np.array([[0, 0], [1, 1]]))
    beta.append(np.array([[1, 0], [0, 0]]))
    
    # P3 = A00(B01 - B11)
    alpha.append(np.array([[1, 0], [0, 0]]))
    beta.append(np.array([[0, 1], [0, -1]]))
    
    # P4 = A11(B10 - B00)
    alpha.append(np.array([[0, 0], [0, 1]]))
    beta.append(np.array([[-1, 0], [1, 0]]))
    
    # P5 = (A00 + A01)B11
    alpha.append(np.array([[1, 1], [0, 0]]))
    beta.append(np.array([[0, 0], [0, 1]]))
    
    # P6 = (A10 - A00)(B00 + B01)
    alpha.append(np.array([[-1, 0], [1, 0]]))
    beta.append(np.array([[1, 1], [0, 0]]))
    
    # P7 = (A01 - A11)(B10 + B11)
    alpha.append(np.array([[0, 1], [0, -1]]))
    beta.append(np.array([[0, 0], [1, 1]]))
    
    # Pattern matrices for reconstruction of 2x2 product
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    patterns = []
    patterns.append(np.array([[1, 0], [0, 1]])) # P1: C00, C11
    patterns.append(np.array([[0, 0], [1, -1]])) # P2: C10, C11
    patterns.append(np.array([[0, 1], [0, 1]])) # P3: C01, C11
    patterns.append(np.array([[1, 0], [1, 0]])) # P4: C00, C10
    patterns.append(np.array([[-1, 1], [0, 0]])) # P5: C00, C01
    patterns.append(np.array([[0, 0], [0, 1]])) # P6: C11
    patterns.append(np.array([[1, 0], [0, 0]])) # P7: C00

    # 2. Define Block Strassen coefficients for 4x4
    # Block indices: 00, 01, 10, 11
    # A_blocks: A00, A01, A10, A11
    # B_blocks: B00, B01, B10, B11
    
    # M1 = (A00 + A11)(B00 + B11)
    block_alpha = []
    block_beta = []
    
    block_alpha.append(np.array([[1, 0], [0, 1]])) # A00 + A11
    block_beta.append(np.array([[1, 0], [0, 1]])) # B00 + B11
    
    block_alpha.append(np.array([[0, 0], [1, 1]])) # A10 + A11
    block_beta.append(np.array([[1, 0], [0, 0]])) # B00
    
    block_alpha.append(np.array([[1, 0], [0, 0]])) # A00
    block_beta.append(np.array([[0, 1], [0, -1]])) # B01 - B11
    
    block_alpha.append(np.array([[0, 0], [0, 1]])) # A11
    block_beta.append(np.array([[-1, 0], [1, 0]])) # B10 - B00
    
    block_alpha.append(np.array([[1, 1], [0, 0]])) # A00 + A01
    block_beta.append(np.array([[0, 0], [0, 1]])) # B11
    
    block_alpha.append(np.array([[-1, 0], [1, 0]])) # A10 - A00
    block_beta.append(np.array([[1, 1], [0, 0]])) # B00 + B01
    
    block_alpha.append(np.array([[0, 1], [0, -1]])) # A01 - A11
    block_beta.append(np.array([[0, 0], [1, 1]])) # B10 + B11
    
    # Block reconstruction patterns
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    block_patterns = []
    block_patterns.append(np.array([[1, 0], [0, 1]])) # M1
    block_patterns.append(np.array([[0, 0], [1, -1]])) # M2
    block_patterns.append(np.array([[0, 1], [0, 1]])) # M3
    block_patterns.append(np.array([[1, 0], [1, 0]])) # M4
    block_patterns.append(np.array([[-1, 1], [0, 0]])) # M5
    block_patterns.append(np.array([[0, 0], [0, 1]])) # M6
    block_patterns.append(np.array([[1, 0], [0, 0]])) # M7

    # 3. Construct U, V, W
    col = 0
    for s in range(7):
        for t in range(7):
            # Compute U vector (coefficients for A)
            # U is length 16. Index i*4+j
            # U[i*4+j] = sum_{IJ} block_alpha[s][I,J] * alpha[t][i_local, j_local]
            # where I,J are block indices, i_local, j_local are local indices within block
            # Global i = 2*I + i_local, Global j = 2*J + j_local
            
            u_vec = np.zeros(16)
            for I in range(2):
                for J in range(2):
                    coeff_block = block_alpha[s][I, J]
                    if coeff_block != 0:
                        for i_loc in range(2):
                            for j_loc in range(2):
                                coeff_local = alpha[t][i_loc, j_loc]
                                if coeff_local != 0:
                                    i_global = 2 * I + i_loc
                                    j_global = 2 * J + j_loc
                                    u_vec[i_global * 4 + j_global] += coeff_block * coeff_local
            U[:, col] = u_vec
            
            # Compute V vector (coefficients for B)
            # V[j*4+k]
            v_vec = np.zeros(16)
            for J in range(2):
                for K in range(2):
                    coeff_block = block_beta[s][J, K]
                    if coeff_block != 0:
                        for j_loc in range(2):
                            for k_loc in range(2):
                                coeff_local = beta[t][j_loc, k_loc]
                                if coeff_local != 0:
                                    j_global = 2 * J + j_loc
                                    k_global = 2 * K + k_loc
                                    v_vec[j_global * 4 + k_global] += coeff_block * coeff_local
            V[:, col] = v_vec
            
            # Compute W vector (coefficients for C)
            # W[k*4+i] (Seed convention)
            # Standard C index is i*4+k
            # W_std[i*4+k] = sum_{KL} block_patterns[s][K,L] * patterns[t][k_loc, l_loc]
            # where K,L are block indices, k_loc, l_loc are local indices
            # Global i = 2*K + k_loc, Global k = 2*L + l_loc
            
            w_vec_std = np.zeros(16)
            for K in range(2):
                for L in range(2):
                    coeff_block = block_patterns[s][K, L]
                    if coeff_block != 0:
                        for k_loc in range(2):
                            for l_loc in range(2):
                                coeff_local = patterns[t][k_loc, l_loc]
                                if coeff_local != 0:
                                    i_global = 2 * K + k_loc
                                    k_global = 2 * L + l_loc
                                    w_vec_std[i_global * 4 + k_global] += coeff_block * coeff_local
            
            # Permute W_std to W_seed
            # W_seed[k*4+i] = W_std[i*4+k]
            w_vec_seed = np.zeros(16)
            for i in range(4):
                for k in range(4):
                    w_vec_seed[k * 4 + i] = w_vec_std[i * 4 + k]
            
            W[:, col] = w_vec_seed
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
