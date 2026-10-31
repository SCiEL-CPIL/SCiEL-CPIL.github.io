import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (rank 49) exact tensor decomposition for 4x4 matrix multiplication.
    
    The decomposition uses a 7x7 Strassen-like structure where each of the 7x7=49
    rank-1 terms corresponds to a block-level multiplication with 2x2 Strassen
    coefficients.
    
    U: shape (16, 49) - factors for A matrix (4x4 = 16 entries)
    V: shape (16, 49) - factors for B matrix (4x4 = 16 entries)
    W: shape (16, 49) - factors for C matrix (4x4 = 16 entries), indexed as W[k*4+i]
    
    The tensor identity: C[i,k] = sum_r U[i*4+k, r] * V[... , r] * W[k*4+i, r]
    should hold exactly for integer entries.
    """
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = DIMENSIONS
    rank = 49
    
    # 2x2 Strassen coefficients for the 7 multiplications
    # For 2x2 matrices, we have 7 rank-1 terms with coefficients in {-1, 0, 1}
    
    # U2[row, t]: coefficient for A element at (i,j) in t-th Strassen product
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V2[row, t]: coefficient for B element at (j,k) in t-th Strassen product
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W2[row, t]: coefficient for C element at (i,k) in t-th Strassen product
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # Standard 2x2 Strassen algorithm:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # t=0 (M1)
    U2[0, 0] = 1  # A00
    U2[3, 0] = 1  # A11
    V2[0, 0] = 1  # B00
    V2[3, 0] = 1  # B11
    W2[0, 0] = 1  # C00
    W2[3, 0] = 1  # C11
    
    # t=1 (M2)
    U2[2, 1] = 1  # A10
    U2[3, 1] = 1  # A11
    V2[0, 1] = 1  # B00
    W2[2, 1] = 1  # C10
    W2[3, 1] = -1  # C11
    
    # t=2 (M3)
    U2[0, 2] = 1  # A00
    V2[1, 2] = 1  # B01
    V2[3, 2] = -1  # B11
    W2[1, 2] = 1  # C01
    W2[3, 2] = 1  # C11
    
    # t=3 (M4)
    U2[3, 3] = 1  # A11
    V2[2, 3] = 1  # B10
    V2[0, 3] = -1  # B00
    W2[0, 3] = -1  # C00
    W2[2, 3] = 1  # C10
    
    # t=4 (M5)
    U2[0, 4] = 1  # A00
    U2[1, 4] = 1  # A01
    V2[3, 4] = 1  # B11
    W2[0, 4] = -1  # C00
    W2[1, 4] = 1  # C01
    
    # t=5 (M6)
    U2[2, 5] = 1  # A10
    U2[0, 5] = -1  # A00
    V2[0, 5] = 1  # B00
    V2[1, 5] = 1  # B01
    W2[3, 5] = 1  # C11
    
    # t=6 (M7)
    U2[1, 6] = 1  # A01
    U2[3, 6] = -1  # A11
    V2[2, 6] = 1  # B10
    V2[3, 6] = 1  # B11
    W2[0, 6] = 1  # C00
    
    # For 4x4 matrices, we use block decomposition:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Each block multiplication Cij = sum_k Aik * Bkj is done via 7 Strassen products
    # So total rank = 7*7 = 49
    
    # Block-level Strassen coefficients:
    # For block (I,J) of A, (J,K) of B, (I,K) of C:
    # The 7 block products are:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # beta[s][I][J]: coefficient for block (I,J) of A in s-th block product
    # gamma[s][J][K]: coefficient for block (J,K) of B in s-th block product
    # delta[s][I][K]: coefficient for block (I,K) of C in s-th block product
    
    beta = np.zeros((7, 2, 2), dtype=int)
    gamma = np.zeros((7, 2, 2), dtype=int)
    delta = np.zeros((7, 2, 2), dtype=int)
    
    # s=0 (M1): (A00+A11)*(B00+B11)
    beta[0, 0, 0] = 1
    beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1
    gamma[0, 1, 1] = 1
    delta[0, 0, 0] = 1
    delta[0, 1, 1] = 1
    
    # s=1 (M2): (A10+A11)*B00
    beta[1, 1, 0] = 1
    beta[1, 1, 1] = 1
    gamma[1, 0, 0] = 1
    delta[1, 1, 0] = 1
    delta[1, 1, 1] = -1
    
    # s=2 (M3): A00*(B01-B11)
    beta[2, 0, 0] = 1
    gamma[2, 0, 1] = 1
    gamma[2, 1, 1] = -1
    delta[2, 0, 1] = 1
    delta[2, 1, 1] = 1
    
    # s=3 (M4): A11*(B10-B00)
    beta[3, 1, 1] = 1
    gamma[3, 1, 0] = 1
    gamma[3, 0, 0] = -1
    delta[3, 0, 0] = -1
    delta[3, 1, 0] = 1
    
    # s=4 (M5): (A00+A01)*B11
    beta[4, 0, 0] = 1
    beta[4, 0, 1] = 1
    gamma[4, 1, 1] = 1
    delta[4, 0, 0] = -1
    delta[4, 0, 1] = 1
    
    # s=5 (M6): (A10-A00)*(B00+B01)
    beta[5, 1, 0] = 1
    beta[5, 0, 0] = -1
    gamma[5, 0, 0] = 1
    gamma[5, 0, 1] = 1
    delta[5, 1, 1] = 1
    
    # s=6 (M7): (A01-A11)*(B10+B11)
    beta[6, 0, 1] = 1
    beta[6, 1, 1] = -1
    gamma[6, 1, 0] = 1
    gamma[6, 1, 1] = 1
    delta[6, 0, 0] = 1
    
    # Build U, V, W matrices
    # U has shape (16, 49): rows indexed by (i,j) where i,j in 0..3
    # V has shape (16, 49): rows indexed by (j,k)
    # W has shape (16, 49): rows indexed by (k,i) per convention W[k*n+i]
    
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    for s in range(7):
        for t in range(7):
            c = 7 * s + t
            
            # Fill U: for each element (i,j) in A
            for i in range(4):
                for j in range(4):
                    I = i // 2  # block row
                    i_local = i % 2  # local row within block
                    J = j // 2  # block col
                    j_local = j % 2  # local col within block
                    
                    block_coeff = beta[s, I, J]
                    local_coeff = U2[i_local * 2 + j_local, t]
                    coeff = block_coeff * local_coeff
                    if coeff != 0:
                        U[i * 4 + j, c] = coeff
            
            # Fill V: for each element (j,k) in B
            for j in range(4):
                for k in range(4):
                    J = j // 2
                    j_local = j % 2
                    K = k // 2
                    k_local = k % 2
                    
                    block_coeff = gamma[s, J, K]
                    local_coeff = V2[j_local * 2 + k_local, t]
                    coeff = block_coeff * local_coeff
                    if coeff != 0:
                        V[j * 4 + k, c] = coeff
            
            # Fill W: for each element (k,i) in C, indexed as W[k*4+i]
            for k in range(4):
                for i in range(4):
                    K = k // 2
                    k_local = k % 2
                    I = i // 2
                    i_local = i % 2
                    
                    block_coeff = delta[s, I, K]
                    local_coeff = W2[k_local * 2 + i_local, t]
                    coeff = block_coeff * local_coeff
                    if coeff != 0:
                        W[k * 4 + i, c] = coeff
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
