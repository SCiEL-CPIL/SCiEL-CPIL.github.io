"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the tensor decomposition of matrix multiplication.
    Dimensions: n x m * m x p = n x p
    n=2, m=4, p=8
    Rank: 28 (using Strassen decomposition for 2x2x2 blocks)
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Each entry is a list of (index, coeff) for U, V, W
    # U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_terms = [
        { # M1
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (3, 1)],
            'W': [(0, 1), (3, 1)]
        },
        { # M2
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(2, 1), (3, -1)]
        },
        { # M3
            'U': [(0, 1)],
            'V': [(1, 1), (3, -1)],
            'W': [(1, 1), (3, 1)]
        },
        { # M4
            'U': [(3, 1)],
            'V': [(2, 1), (0, -1)],
            'W': [(0, 1), (2, 1)]
        },
        { # M5
            'U': [(0, 1), (1, 1)],
            'V': [(3, 1)],
            'W': [(0, -1), (1, 1)]
        },
        { # M6
            'U': [(2, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(3, 1)]
        },
        { # M7
            'U': [(1, 1), (3, -1)],
            'V': [(2, 1), (3, 1)],
            'W': [(0, 1), (3, 1)]
        }
    ]
    
    col = 0
    # Define the 4 blocks
    # Block (ai, bj) where ai in {0,1} selects A cols 2*ai..2*ai+1
    # bj in {0,1,2,3} selects B rows 2*bj//2? No.
    # B is 4x8. Split into 2x4 blocks vertically?
    # B = [B1; B2] where B1 is rows 0,1 and B2 is rows 2,3.
    # B1 is split into B11 (cols 0,1) and B12 (cols 2,3).
    # B2 is split into B21 (cols 0,1) and B22 (cols 2,3).
    
    # A is 2x4. Split into A1 (cols 0,1) and A2 (cols 2,3).
    
    blocks = [
        # (A_col_offset, B_row_offset, B_col_offset)
        (0, 0, 0), # A1, B11 -> C11
        (0, 0, 2), # A1, B12 -> C12
        (2, 2, 0), # A2, B21 -> C21
        (2, 2, 2), # A2, B22 -> C22
    ]
    
    for a_off, b_row_off, b_col_off in blocks:
        for term in strassen_terms:
            # Add to U
            for u_idx, coeff in term['U']:
                # u_idx 0..3 corresponds to local (i, j)
                li = u_idx // 2
                lj = u_idx % 2
                global_i = li
                global_j = a_off + lj
                U[global_i * m + global_j][col] += coeff
                
            # Add to V
            for v_idx, coeff in term['V']:
                # v_idx 0..3 corresponds to local (j, k)
                lj = v_idx // 2
                lk = v_idx % 2
                global_j = b_row_off + lj
                global_k = b_col_off + lk
                V[global_j * p + global_k][col] += coeff
                
            # Add to W
            for w_idx, coeff in term['W']:
                # w_idx 0..3 corresponds to local (i, k)
                li = w_idx // 2
                lk = w_idx % 2
                global_i = li
                global_k = b_col_off + lk
                # W index is k*n + i
                W[global_k * n + global_i][col] += coeff
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
