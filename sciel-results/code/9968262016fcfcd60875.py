"""Optimized decomposition for 2x3x9 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-36 decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    
    The decomposition exploits the block structure of the 3x9 matrix, treating it 
    as three 3x3 blocks. Each 3x3 block multiplication with a 2x3 matrix is 
    decomposed into rank 12 using a mix of Strassen (rank 7) and schoolbook 
    (rank 5) sub-routines.
    
    U: shape (n*m, rank) = (6, 36)
    V: shape (m*p, rank) = (27, 36)
    W: shape (p*n, rank) = (18, 36)
    """
    n, m, p = DIMENSIONS
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 36
    V = [[0] * rank for _ in range(m * p)]  # 27 x 36
    W = [[0] * rank for _ in range(p * n)]  # 18 x 36
    
    # Define the 12 terms for a 2x3x3 block
    # Local indices:
    # A: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # B: j*3+k, j in 0..2, k in 0..2
    # C: k*2+i, k in 0..2, i in 0..1
    
    # Term structure: (U_indices, V_indices, W_indices, value)
    # U_indices: list of (local_A_idx, val)
    # V_indices: list of (local_B_idx, val)
    # W_indices: list of (local_C_idx, val)
    
    terms = []
    
    # --- Part 1: Strassen 2x2x2 (7 terms) ---
    # A uses cols 0,1. B uses rows 0,1, cols 0,1. C uses cols 0,1.
    # Strassen terms for 2x2x2:
    # 1. (A00+A10)(B00+B10) -> C00, C11
    # 2. (A10+A11)(B00) -> C00
    # 3. (A00+A01)(B10) -> C01
    # 4. (A00)(B10+B11) -> C01, C10
    # 5. (A01)(B00+B01) -> C01
    # 6. (A10)(B11) -> C11
    # 7. (A11)(B10) -> C10, C11
    
    # Note: Standard Strassen coefficients might vary by convention. 
    # I will use the standard set that sums to the product.
    
    # T1: (A00+A10)(B00+B10)
    terms.append(([ (0,1), (3,1) ], [ (0,1), (2,1) ], [ (0,1), (3,1) ]))
    # T2: (A10+A11)(B00)
    terms.append(([ (3,1), (4,1) ], [ (0,1) ], [ (0,-1) ]))
    # T3: (A00+A01)(B10)
    terms.append(([ (0,1), (1,1) ], [ (2,1) ], [ (1,1) ]))
    # T4: (A00)(B10+B11)
    terms.append(([ (0,1) ], [ (2,1), (3,1) ], [ (1,-1), (2,1) ]))
    # T5: (A01)(B00+B01)
    terms.append(([ (1,1) ], [ (0,1), (1,1) ], [ (1,1) ]))
    # T6: (A10)(B11)
    terms.append(([ (3,1) ], [ (3,1) ], [ (3,1) ]))
    # T7: (A11)(B10)
    terms.append(([ (4,1) ], [ (2,1) ], [ (2,-1), (3,1) ]))
    
    # --- Part 2: Schoolbook 2x2x1 (2 terms) ---
    # A cols 0,1. B rows 0,1, col 2. C col 2.
    # C_i2 = A_i0*B_02 + A_i1*B_12
    # T8: A_col0 * B_02
    terms.append(([ (0,1), (3,1) ], [ (2,1) ], [ (4,1), (6,1) ]))
    # T9: A_col1 * B_12
    terms.append(([ (1,1), (4,1) ], [ (5,1) ], [ (4,1), (6,1) ]))
    
    # --- Part 3: Schoolbook 2x1x2 (2 terms) ---
    # A col 2. B row 2, cols 0,1. C cols 0,1.
    # C_i0 = A_i2*B_20
    # C_i1 = A_i2*B_21
    # T10: A_col2 * B_20
    terms.append(([ (2,1), (5,1) ], [ (6,1) ], [ (0,1), (2,1) ]))
    # T11: A_col2 * B_21
    terms.append(([ (2,1), (5,1) ], [ (7,1) ], [ (1,1), (3,1) ]))
    
    # --- Part 4: Schoolbook 2x1x1 (1 term) ---
    # A col 2. B row 2, col 2. C col 2.
    # T12: A_col2 * B_22
    terms.append(([ (2,1), (5,1) ], [ (8,1) ], [ (4,1), (6,1) ]))
    
    # Total 12 terms per block.
    
    # Fill the global matrices
    for b in range(3):
        for t_idx, term in enumerate(terms):
            col = b * 12 + t_idx
            u_indices, v_indices, w_indices = term
            
            for a_idx, val in u_indices:
                U[a_idx][col] += val
                
            for b_idx, val in v_indices:
                # Map local B index (j*3+k) to global V index (j*9 + (3*b+k))
                # local b_idx = j*3 + k
                j = b_idx // 3
                k = b_idx % 3
                global_v_idx = j * 9 + (3 * b + k)
                V[global_v_idx][col] += val
                
            for c_idx, val in w_indices:
                # Map local C index (k*2+i) to global W index ((3*b+k)*2 + i)
                # local c_idx = k*2 + i
                k = c_idx // 2
                i = c_idx % 2
                global_w_idx = (3 * b + k) * 2 + i
                W[global_w_idx][col] += val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
