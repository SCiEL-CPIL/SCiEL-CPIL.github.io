"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # We use a block-stride approach: partition the 6x6 matrices into 3x3 blocks of 2x2 sub-matrices.
    # For each block triple (I, J, K) in {0,1,2}^3, we compute the 2x2 matrix multiplication A_IJ * B_JK -> C_KI.
    # We use Strassen's algorithm for the 2x2 multiplication, which has rank 7.
    # Total rank = 3 * 3 * 3 * 7 = 189.
    
    n, m, p = DIMENSIONS
    num_blocks = 3  # since 6 / 2 = 3
    block_size = 2
    
    rank = num_blocks * num_blocks * num_blocks * 7  # 27 * 7 = 189
    
    # U shape: (n*m, rank) = (36, 189)
    # V shape: (m*p, rank) = (36, 189)
    # W shape: (p*n, rank) = (36, 189)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen terms: for each term t in 0..6, define:
    # - U coefficients for A entries (a00, a01, a10, a11)
    # - V coefficients for B entries (b00, b01, b10, b11)
    # - W coefficients for C entries (c00, c01, c10, c11)
    
    # A entries indices in 2x2: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B entries indices in 2x2: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C entries indices in 2x2: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    strassen_terms = []
    
    # M1 = (A00 + A11) * (B00 + B11)
    strassen_terms.append({
        'u': {0: 1, 3: 1},   # A00, A11
        'v': {0: 1, 3: 1},   # B00, B11
        'w': {0: 1, 3: 1}    # C00, C11 (from C00=M1+... and C11=M1-...)
    })
    
    # M2 = (A10 + A11) * B00
    strassen_terms.append({
        'u': {2: 1, 3: 1},   # A10, A11
        'v': {0: 1},         # B00
        'w': {2: 1, 3: -1}   # C10=M2+..., C11=...-M2+...
    })
    
    # M3 = A00 * (B01 - B11)
    strassen_terms.append({
        'u': {0: 1},         # A00
        'v': {1: 1, 3: -1},  # B01, -B11
        'w': {1: 1, 3: 1}    # C01=M3+..., C11=...+M3+...
    })
    
    # M4 = A11 * (B10 - B00)
    strassen_terms.append({
        'u': {3: 1},         # A11
        'v': {2: 1, 0: -1},  # B10, -B00
        'w': {0: 1, 2: 1}    # C00=...+M4-..., C10=M2+M4
    })
    
    # M5 = (A00 + A01) * B11
    strassen_terms.append({
        'u': {0: 1, 1: 1},   # A00, A01
        'v': {3: 1},         # B11
        'w': {0: -1, 1: 1}   # C00=...-M5+..., C01=M3+M5
    })
    
    # M6 = (A10 - A00) * (B00 + B01)
    strassen_terms.append({
        'u': {2: 1, 0: -1},  # A10, -A00
        'v': {0: 1, 1: 1},   # B00, B01
        'w': {3: 1}          # C11=...+M6
    })
    
    # M7 = (A01 - A11) * (B10 + B11)
    strassen_terms.append({
        'u': {1: 1, 3: -1},  # A01, -A11
        'v': {2: 1, 3: 1},   # B10, B11
        'w': {0: 1}          # C00=M1+M4-M5+M7
    })
    
    col = 0
    for I in range(num_blocks):
        for J in range(num_blocks):
            for K in range(num_blocks):
                for t in range(7):
                    term = strassen_terms[t]
                    
                    # A block IJ: rows I*2, I*2+1; cols J*2, J*2+1
                    # U row index: row * m + col = row * 6 + col
                    # A entry (r, c) in block IJ -> global row = I*2 + r, global col = J*2 + c
                    # U row = (I*2 + r) * 6 + (J*2 + c)
                    
                    for a_idx, coeff in term['u'].items():
                        r, c = divmod(a_idx, 2)  # a_idx 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                        u_row = (I * block_size + r) * m + (J * block_size + c)
                        U[u_row, col] += coeff
                    
                    # B block JK: rows J*2, J*2+1; cols K*2, K*2+1
                    # V row index: row * p + col = row * 6 + col
                    # B entry (r, c) in block JK -> global row = J*2 + r, global col = K*2 + c
                    # V row = (J*2 + r) * 6 + (K*2 + c)
                    
                    for b_idx, coeff in term['v'].items():
                        r, c = divmod(b_idx, 2)
                        v_row = (J * block_size + r) * p + (K * block_size + c)
                        V[v_row, col] += coeff
                    
                    # C block KI: rows K*2, K*2+1; cols I*2, I*2+1
                    # W row index: row * n + col = row * 6 + col
                    # C entry (r, c) in block KI -> global row = K*2 + r, global col = I*2 + c
                    # W row = (K*2 + r) * 6 + (I*2 + c)
                    
                    for c_idx, coeff in term['w'].items():
                        r, c = divmod(c_idx, 2)
                        w_row = (K * block_size + r) * n + (I * block_size + c)
                        W[w_row, col] += coeff
                    
                    col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
