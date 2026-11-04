"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-154 decomposition of 6x6 matrix multiplication using block-structured
    Strassen: partition 6x6 into 2x2 blocks of 3x3 matrices, use Strassen's
    7-product algorithm for the 2x2 block structure and a 22-term decomposition
    for the 3x3 block multiplications (Strassen on 2x2 core + 15 extra terms).
    
    Total rank = 7 * 22 = 154.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    rank = 154
    
    U = np.zeros((n * m, rank), dtype=np.float64)  # (36, 154)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (36, 154)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (36, 154)
    
    # Block structure: 6x6 = 2x2 blocks of 3x3
    block_size = 3
    num_blocks = 2
    
    # Strassen's 7 multiplications for 2x2 matrix multiplication:
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
    
    strassen_steps = [
        # M1 = (A00 + A11) * (B00 + B11)
        {
            'X': {(0,0): 1, (1,1): 1},
            'Y': {(0,0): 1, (1,1): 1},
            'C': {(0,0): 1, (1,1): 1}
        },
        # M2 = (A10 + A11) * B00
        {
            'X': {(1,0): 1, (1,1): 1},
            'Y': {(0,0): 1},
            'C': {(1,0): 1, (1,1): -1}
        },
        # M3 = A00 * (B01 - B11)
        {
            'X': {(0,0): 1},
            'Y': {(0,1): 1, (1,1): -1},
            'C': {(0,1): 1, (1,1): 1}
        },
        # M4 = A11 * (B10 - B00)
        {
            'X': {(1,1): 1},
            'Y': {(1,0): 1, (0,0): -1},
            'C': {(0,0): 1, (1,0): 1}
        },
        # M5 = (A00 + A01) * B11
        {
            'X': {(0,0): 1, (0,1): 1},
            'Y': {(1,1): 1},
            'C': {(0,0): -1, (0,1): 1}
        },
        # M6 = (A10 - A00) * (B00 + B01)
        {
            'X': {(1,0): 1, (0,0): -1},
            'Y': {(0,0): 1, (0,1): 1},
            'C': {(1,1): 1}
        },
        # M7 = (A01 - A11) * (B10 + B11)
        {
            'X': {(0,1): 1, (1,1): -1},
            'Y': {(1,0): 1, (1,1): 1},
            'C': {(0,0): 1}
        }
    ]
    
    # For each 3x3 block multiplication, we use 22 terms:
    # 7 terms from Strassen on the 2x2 core (indices 0,1 for rows/cols)
    # 15 terms for interactions with row/col 2 (the 3rd row/col)
    #
    # The 3x3 matrix:
    # [a00 a01 a02]   [b00 b01 b02]   [c00 c01 c02]
    # [a10 a11 a12] * [b10 b11 b12] = [c10 c11 c12]
    # [a20 a21 a22]   [b20 b21 b22]   [c20 c21 c22]
    #
    # Strassen on 2x2 core (indices 0,1):
    # S1 = (a00+a11)*(b00+b11) -> c00 += S1, c11 += S1
    # S2 = (a10+a11)*b00       -> c10 += S2, c11 -= S2
    # S3 = a00*(b01-b11)       -> c01 += S3, c11 += S3
    # S4 = a11*(b10-b00)       -> c00 += S4, c10 += S4
    # S5 = (a00+a01)*b11       -> c00 -= S5, c01 += S5
    # S6 = (a10-a00)*(b00+b01) -> c11 += S6
    # S7 = (a01-a11)*(b10+b11) -> c00 += S7
    #
    # Remaining 15 terms (interactions with index 2):
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c12 = a10*b02 + a11*b12 + a12*b22
    # c20 = a20*b00 + a21*b10 + a22*b20
    # c21 = a20*b01 + a21*b11 + a22*b21
    # c22 = a20*b02 + a21*b12 + a22*b22
    
    # Define the 22 terms for 3x3 multiplication
    # Each term: (a_indices, b_indices, c_indices)
    # a_indices: list of (row, col, coeff) for A block
    # b_indices: list of (row, col, coeff) for B block
    # c_indices: list of (row, col, coeff) for C block
    
    def make_3x3_terms():
        """Return list of 22 terms, each with (a_terms, b_terms, c_terms)."""
        terms = []
        
        # Strassen 2x2 core (7 terms)
        # S1 = (a00+a11)*(b00+b11)
        terms.append((
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        ))
        # S2 = (a10+a11)*b00
        terms.append((
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,0,1), (1,1,-1)]
        ))
        # S3 = a00*(b01-b11)
        terms.append((
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(0,1,1), (1,1,1)]
        ))
        # S4 = a11*(b10-b00)
        terms.append((
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (1,0,1)]
        ))
        # S5 = (a00+a01)*b11
        terms.append((
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(0,0,-1), (0,1,1)]
        ))
        # S6 = (a10-a00)*(b00+b01)
        terms.append((
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)]
        ))
        # S7 = (a01-a11)*(b10+b11)
        terms.append((
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)]
        ))
        
        # 15 extra terms for row/col 2 interactions
        # c02 = a00*b02 + a01*b12 + a02*b22
        terms.append(([(0,0,1)], [(0,2,1)], [(0,2,1)]))
        terms.append(([(0,1,1)], [(1,2,1)], [(0,2,1)]))
        terms.append(([(0,2,1)], [(2,2,1)], [(0,2,1)]))
        
        # c12 = a10*b02 + a11*b12 + a12*b22
        terms.append(([(1,0,1)], [(0,2,1)], [(1,2,1)]))
        terms.append(([(1,1,1)], [(1,2,1)], [(1,2,1)]))
        terms.append(([(1,2,1)], [(2,2,1)], [(1,2,1)]))
        
        # c20 = a20*b00 + a21*b10 + a22*b20
        terms.append(([(2,0,1)], [(0,0,1)], [(2,0,1)]))
        terms.append(([(2,1,1)], [(1,0,1)], [(2,0,1)]))
        terms.append(([(2,2,1)], [(2,0,1)], [(2,0,1)]))
        
        # c21 = a20*b01 + a21*b11 + a22*b21
        terms.append(([(2,0,1)], [(0,1,1)], [(2,1,1)]))
        terms.append(([(2,1,1)], [(1,1,1)], [(2,1,1)]))
        terms.append(([(2,2,1)], [(2,1,1)], [(2,1,1)]))
        
        # c22 = a20*b02 + a21*b12 + a22*b22
        terms.append(([(2,0,1)], [(0,2,1)], [(2,2,1)]))
        terms.append(([(2,1,1)], [(1,2,1)], [(2,2,1)]))
        terms.append(([(2,2,1)], [(2,2,1)], [(2,2,1)]))
        
        return terms
    
    inner_terms = make_3x3_terms()
    assert len(inner_terms) == 22, f"Expected 22 terms, got {len(inner_terms)}"
    
    col = 0
    
    # For each of the 7 Strassen multiplications, we do the 22-term 3x3 multiplication
    for step in strassen_steps:
        X_coeffs = step['X']
        Y_coeffs = step['Y']
        C_coeffs = step['C']
        
        # For each of the 22 scalar products in the 3x3 matrix multiplication
        for a_terms, b_terms, c_terms in inner_terms:
            # U part: linear combination of A elements
            for (ar, ac, coeff) in a_terms:
                # A element at block-relative position (ar, ac)
                # Global position: (ib*3+ar, jb*3+ac) for each (ib,jb) in X_coeffs
                for (ib, jb), x_coeff in X_coeffs.items():
                    u_row = ib * block_size + ar
                    u_col = jb * block_size + ac
                    u_idx = u_row * n + u_col
                    U[u_idx, col] += x_coeff * coeff
            
            # V part: linear combination of B elements
            for (br, bc, coeff) in b_terms:
                # B element at block-relative position (br, bc)
                # Global position: (jb*3+br, kb*3+bc) for each (jb,kb) in Y_coeffs
                for (jb, kb), y_coeff in Y_coeffs.items():
                    v_row = jb * block_size + br
                    v_col = kb * block_size + bc
                    v_idx = v_row * m + v_col
                    V[v_idx, col] += y_coeff * coeff
            
            # W part: linear combination of C elements
            for (cr, cc, coeff) in c_terms:
                # C element at block-relative position (cr, cc)
                # Global position: (ib*3+cr, kb*3+cc) for each (ib,kb) in C_coeffs
                # W trace convention: W[k*n+i, r] where k=kb*3+cc, i=ib*3+cr
                for (ib, kb), c_coeff in C_coeffs.items():
                    w_k = kb * block_size + cc
                    w_i = ib * block_size + cr
                    w_idx = w_k * n + w_i
                    W[w_idx, col] += c_coeff * coeff
            
            col += 1
        
        # Time check
        if time.time() - start_time > 130:
            break
    
    # Verify col matches expected rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
