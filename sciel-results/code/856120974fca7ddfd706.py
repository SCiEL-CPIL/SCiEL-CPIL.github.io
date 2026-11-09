"""Matrix multiplication tensor decomposition for 2x4 @ 4x8 with rank 56 (Strassen-based)."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time

def construct():
    """
    Returns a rank-56 tensor decomposition for 2x4 @ 4x8 matrix multiplication.
    Uses Strassen's rank-7 algorithm for 2x2x2, applied to 8 independent sub-blocks.
    
    The 2x4x8 multiplication is viewed as 4 output-column blocks (size 2) × 2 
    intermediate-row blocks (size 2), giving 8 independent 2x2x2 problems,
    each solved with rank 7, for total rank 56.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # Each product: (u_coeff_dict, v_coeff_dict, w_coeff_dict)
    # u_dict maps (i,j) -> coeff for A[i,j]
    # v_dict maps (j,k) -> coeff for B[j,k]  
    # w_dict maps (k,i) -> coeff for C[k,i]... wait, C is 2x2, indexed [i,k]
    # Actually W[k*n+i, r] convention means output index is k*n+i where k is col, i is row
    # For 2x2: W[k*2+i, r]
    
    # Let me re-derive Strassen carefully.
    # C[0,0] = P1 + P4 - P5 - P7
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P2 + P3 + P6
    #
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    strassen_products = [
        # P1: u={(0,0):1,(1,1):1}, v={(0,0):1,(1,1):1}, w={(0,0):1,(1,1):-1}
        # C[0,0]+=1*P1, C[1,1]-=1*P1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
        # P2: u={(1,0):1,(1,1):1}, v={(0,0):1}, w={(1,0):1,(1,1):-1}
        # C[1,0]+=1*P2, C[1,1]-=1*P2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # P3: u={(0,0):1}, v={(0,1):1,(1,1):-1}, w={(0,1):1,(1,1):1}
        # C[0,1]+=1*P3, C[1,1]+=1*P3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # P4: u={(1,1):1}, v={(1,0):1,(0,0):-1}, w={(0,0):1,(1,0):1}
        # C[0,0]+=1*P4, C[1,0]+=1*P4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # P5: u={(0,0):1,(0,1):1}, v={(1,1):1}, w={(0,0):-1,(0,1):1}
        # C[0,0]-=1*P5, C[0,1]+=1*P5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # P6: u={(1,0):1,(0,0):-1}, v={(0,0):1,(0,1):1}, w={(1,1):1}
        # C[1,1]+=1*P6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7: u={(0,1):1,(1,1):-1}, v={(1,0):1,(1,1):1}, w={(0,0):-1}
        # C[0,0]-=1*P7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1}),
    ]
    
    rank = 56
    U = np.zeros((n * m, rank))  # 8 x 56
    V = np.zeros((m * p, rank))  # 32 x 56
    W = np.zeros((p * n, rank))  # 16 x 56
    
    col_idx = 0
    # 4 blocks of columns in B (size 2 each): p//2 = 4
    # 2 blocks of rows in B / cols in A (size 2 each): m//2 = 2
    for b_col_block in range(p // 2):
        for b_row_block in range(m // 2):
            # Offsets
            a_col_off = b_row_block * 2   # which 2 cols of A
            b_row_off = b_row_block * 2   # which 2 rows of B
            b_col_off = b_col_block * 2   # which 2 cols of B
            w_col_off = b_col_block * 2   # which 2 cols of C (output)
            
            for u_dict, v_dict, w_dict in strassen_products:
                # Fill U: U[i*m + j, col_idx]
                for (i, j), val in u_dict.items():
                    idx_u = i * m + (a_col_off + j)
                    U[idx_u, col_idx] += val
                
                # Fill V: V[j*p + k, col_idx]
                for (j, k), val in v_dict.items():
                    idx_v = (b_row_off + j) * p + (b_col_off + k)
                    V[idx_v, col_idx] += val
                
                # Fill W: W[k*n + i, col_idx]  (official trace convention)
                for (k, i), val in w_dict.items():
                    idx_w = (w_col_off + k) * n + i
                    W[idx_w, col_idx] += val
                
                col_idx += 1
    
    # Convert to list of lists with clean values
    def to_clean_list(mat):
        return [[float(round(val, 10)) for val in row] for row in mat.tolist()]
    
    return {
        "U": to_clean_list(U),
        "V": to_clean_list(V),
        "W": to_clean_list(W)
    }
# EVOLVE-BLOCK-END
