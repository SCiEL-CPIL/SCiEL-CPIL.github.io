"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a 3x3 block decomposition where each block is 2x2.
    # Number of blocks along each dimension
    blocks = 3
    block_size = 2
    
    # Total rank is blocks^3 * rank_2x2
    rank_2x2 = 7
    total_rank = blocks * blocks * blocks * rank_2x2
    
    # Initialize matrices
    # U shape: (n*m, total_rank) = (36, 63)
    # V shape: (m*p, total_rank) = (36, 63)
    # W shape: (p*n, total_rank) = (36, 63)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Define the 7 Strassen products for 2x2 matrices
    # Each product is defined by linear combinations of A and B elements.
    # A elements: A00, A01, A10, A11
    # B elements: B00, B01, B10, B11
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)(B00)
    # M3 = (A00)(B10 - B11)
    # M4 = (A11)(B01 - B00)
    # M5 = (A00 + A01)(B11)
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We represent the coefficients for U (A side) and V (B side) for each of the 7 multiplications.
    # U coeff for M1: A00:1, A11:1
    # V coeff for M1: B00:1, B11:1
    # ... and so on.
    
    # Structure: list of 7 tuples (U_coeffs, V_coeffs)
    # U_coeffs is a list of (local_row, coeff)
    # V_coeffs is a list of (local_col, coeff)
    
    # Local indices in 2x2 block:
    # 0: 0,0  1: 0,1
    # 2: 1,0  3: 1,1
    
    strassen_steps = [
        # M1
        {
            'U': [(0, 1), (3, 1)],   # A00 + A11
            'V': [(0, 1), (3, 1)]    # B00 + B11
        },
        # M2
        {
            'U': [(2, 1), (3, 1)],   # A10 + A11
            'V': [(0, 1)]            # B00
        },
        # M3
        {
            'U': [(0, 1)],           # A00
            'V': [(2, 1), (3, -1)]   # B10 - B11
        },
        # M4
        {
            'U': [(3, 1)],           # A11
            'V': [(1, 1), (0, -1)]   # B01 - B00
        },
        # M5
        {
            'U': [(0, 1), (1, 1)],   # A00 + A01
            'V': [(3, 1)]            # B11
        },
        # M6
        {
            'U': [(2, 1), (0, -1)],  # A10 - A00
            'V': [(0, 1), (1, 1)]    # B00 + B01
        },
        # M7
        {
            'U': [(1, 1), (3, -1)],  # A01 - A11
            'V': [(2, 1), (3, 1)]    # B10 + B11
        }
    ]
    
    # W coefficients for each output element C_ij in terms of M1..M7
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    w_coeffs = {
        0: [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00
        1: [(2, 1), (4, 1)],                   # C01
        2: [(1, 1), (3, 1)],                   # C10
        3: [(0, 1), (1, -1), (2, 1), (5, 1)]   # C11
    }
    
    current_rank = 0
    for I in range(blocks):
        for J in range(blocks):
            for K in range(blocks):
                for step_idx in range(7):
                    # Determine global indices for U, V, W
                    # U index: row in A, col in A
                    # A is n x m. Block I (row), J (col)
                    # Global row of block I: I * block_size
                    # Global col of block J: J * block_size
                    
                    # V index: row in B, col in B
                    # B is m x p. Block J (row), K (col)
                    # Global row of block J: J * block_size
                    # Global col of block K: K * block_size
                    
                    # W index: row in C, col in C
                    # C is p x n. Block K (row), I (col)
                    # Global row of block K: K * block_size
                    # Global col of block I: I * block_size
                    
                    step = strassen_steps[step_idx]
                    
                    # Update U
                    for local_row, coeff in step['U']:
                        # Local row in 2x2 block: 0->0, 1->1, 2->2, 3->3
                        # Actually, local_row 0 is (0,0), 1 is (0,1), 2 is (1,0), 3 is (1,1)
                        # We need to map local_row to (r, c) within the block
                        lr, lc = divmod(local_row, 2)
                        
                        global_row = I * block_size + lr
                        global_col = J * block_size + lc
                        
                        # U is (n*m, rank). Index is global_row * m + global_col
                        u_idx = global_row * m + global_col
                        U[u_idx][current_rank] += coeff
                    
                    # Update V
                    for local_col, coeff in step['V']:
                        # local_col is index in B (m x p)
                        # B block is J (row), K (col)
                        # local_col 0 is (0,0), 1 is (0,1), 2 is (1,0), 3 is (1,1)
                        lr, lc = divmod(local_col, 2)
                        
                        global_row = J * block_size + lr
                        global_col = K * block_size + lc
                        
                        # V is (m*p, rank). Index is global_row * p + global_col
                        v_idx = global_row * p + global_col
                        V[v_idx][current_rank] += coeff
                    
                    # Update W
                    # We need to add contributions to all 4 output elements of the C block (K, I)
                    # C block is K (row), I (col)
                    for local_out_idx, w_terms in w_coeffs.items():
                        # local_out_idx 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                        lr, lc = divmod(local_out_idx, 2)
                        
                        global_row = K * block_size + lr
                        global_col = I * block_size + lc
                        
                        # W is (p*n, rank). Index is global_row * n + global_col
                        w_idx = global_row * n + global_col
                        
                        # For this specific rank column (current_rank), we add the coefficient
                        # corresponding to this Strassen step (step_idx)
                        # The term in w_terms is (m_idx, coeff). We only care if m_idx == step_idx
                        for m_idx, coeff in w_terms:
                            if m_idx == step_idx:
                                W[w_idx][current_rank] += coeff
                                break # Only one match possible per step_idx
                    
                    current_rank += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
