"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 49 for 4x4x4 (Strassen) + Rank 16 for 4x4x1 (Schoolbook) = 65
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # --- Part 2: Schoolbook for column 4 (k=4) ---
    # A (4x4) * b (4x1)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # U[i*m+j, col], V[j*p+4, col], W[4*n+i, col]
    col_offset = rank_444
    for i in range(n):
        for j in range(m):
            col = col_offset + i * m + j
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
    
    # --- Part 1: Strassen 4x4x4 ---
    # Define the 7 outer Strassen linear forms for A and B
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # For each k in 1..7, U_k and V_k are 2x2 matrices of linear forms.
    # We represent U_k as a 2x2x4x4 tensor? No, just 2x2 matrices of lists of (index, coeff).
    # Or simpler: U_k is a 2x2 matrix where each entry is a vector of length 16 (coeffs for A).
    
    def get_a_block_indices(block_id):
        # block_id 0: (0,0) to (1,1)
        # block_id 1: (0,2) to (1,3)
        # block_id 2: (2,0) to (3,1)
        # block_id 3: (2,2) to (3,3)
        r_start = (block_id // 2) * 2
        c_start = (block_id % 2) * 2
        idxs = []
        for dr in range(2):
            for dc in range(2):
                idxs.append((r_start + dr) * m + (c_start + dc))
        return idxs

    def get_b_block_indices(block_id):
        # B is 4x4 for this part (cols 0-3)
        # block_id 0: (0,0) to (1,1)
        # block_id 1: (0,2) to (1,3)
        # block_id 2: (2,0) to (3,1)
        # block_id 3: (2,2) to (3,3)
        r_start = (block_id // 2) * 2
        c_start = (block_id % 2) * 2
        idxs = []
        for dr in range(2):
            for dc in range(2):
                idxs.append((r_start + dr) * p + (c_start + dc))
        return idxs

    # Outer Strassen definitions
    # k=1: U=A00+A11, V=B00+B11
    # k=2: U=A10+A11, V=B00
    # k=3: U=A00, V=B01-B11
    # k=4: U=A11, V=B10-B00
    # k=5: U=A00+A01, V=B11
    # k=6: U=A10-A00, V=B00+B01
    # k=7: U=A01-A11, V=B10+B11
    
    outer_defs = [
        { # k=1
            'U_blocks': [(0, 1.0), (3, 1.0)],
            'V_blocks': [(0, 1.0), (3, 1.0)]
        },
        { # k=2
            'U_blocks': [(2, 1.0), (3, 1.0)],
            'V_blocks': [(0, 1.0)]
        },
        { # k=3
            'U_blocks': [(0, 1.0)],
            'V_blocks': [(1, 1.0), (3, -1.0)]
        },
        { # k=4
            'U_blocks': [(3, 1.0)],
            'V_blocks': [(2, 1.0), (0, -1.0)]
        },
        { # k=5
            'U_blocks': [(0, 1.0), (1, 1.0)],
            'V_blocks': [(3, 1.0)]
        },
        { # k=6
            'U_blocks': [(2, 1.0), (0, -1.0)],
            'V_blocks': [(0, 1.0), (1, 1.0)]
        },
        { # k=7
            'U_blocks': [(1, 1.0), (3, -1.0)],
            'V_blocks': [(2, 1.0), (3, 1.0)]
        }
    ]
    
    # Inner Rank-7 decomposition of 2x2 multiplication
    # u_r, v_r, w_r for r=0..6
    # Indices: 0:00, 1:01, 2:10, 3:11
    u_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_forms = [
        [1, 1, 0, -1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Output combination coefficients for C blocks
    # C00: M1+M4-M5+M7
    # C01: M3+M5
    # C10: M2+M4
    # C11: M1-M2+M3+M6
    # w_coeffs[k][block_id]
    w_coeffs = [
        {0: 1.0, 3: 1.0, 1: -1.0, 2: 1.0}, # k=1: C00(1), C11(1) ... wait
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # k=1 (M1): C00(+1), C11(+1)
        {0: 1.0, 3: 1.0},
        # k=2 (M2): C10(+1), C11(-1)
        {2: 1.0, 3: -1.0},
        # k=3 (M3): C01(+1), C11(+1)
        {1: 1.0, 3: 1.0},
        # k=4 (M4): C00(+1), C10(+1)
        {0: 1.0, 2: 1.0},
        # k=5 (M5): C00(-1), C01(+1)
        {0: -1.0, 1: 1.0},
        # k=6 (M6): C11(+1)
        {3: 1.0},
        # k=7 (M7): C00(+1)
        {0: 1.0}
    ]
    
    col = 0
    for k in range(7):
        # Build U_k and V_k as 2x2 matrices of linear forms
        # U_k[i][j] is a vector of length 16 (coeffs for A)
        # V_k[i][j] is a vector of length 16 (coeffs for B)
        
        U_k = np.zeros((2, 2, n*m))
        V_k = np.zeros((2, 2, m*p))
        
        for b_idx, coeff in outer_defs[k]['U_blocks']:
            idxs = get_a_block_indices(b_idx)
            for i in range(2):
                for j in range(2):
                    U_k[i, j, idxs[i*2+j]] = coeff
        
        for b_idx, coeff in outer_defs[k]['V_blocks']:
            idxs = get_b_block_indices(b_idx)
            for i in range(2):
                for j in range(2):
                    V_k[i, j, idxs[i*2+j]] = coeff
        
        for r in range(7):
            u_r = u_forms[r]
            v_r = v_forms[r]
            w_r = w_forms[r]
            
            # Construct U_global
            # U_global[idx] = sum_{i,j} u_r[i*2+j] * U_k[i, j, idx]
            U_global = np.zeros(n*m)
            for i in range(2):
                for j in range(2):
                    U_global += u_r[i*2+j] * U_k[i, j]
            
            # Construct V_global
            V_global = np.zeros(m*p)
            for i in range(2):
                for j in range(2):
                    V_global += v_r[i*2+j] * V_k[i, j]
            
            # Construct W_global
            W_global = np.zeros(p*n)
            # w_r has 4 entries: 00, 01, 10, 11
            # These correspond to local positions in the 2x2 block M_k
            # M_k contributes to C blocks defined in w_coeffs[k]
            # Local pos (li, lj) corresponds to global pos:
            # Global Block (bi, bj) from w_coeffs keys
            # Global Row = bi*2 + li
            # Global Col = bj*2 + lj
            # W index = Global Col * n + Global Row
            
            for li in range(2):
                for lj in range(2):
                    w_val = w_r[li*2+lj]
                    if w_val == 0:
                        continue
                    for c_block, c_coeff in w_coeffs[k].items():
                        # c_block is 0,1,2,3
                        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                        bi = c_block // 2
                        bj = c_block % 2
                        global_row = bi * 2 + li
                        global_col = bj * 2 + lj
                        w_idx = global_col * n + global_row
                        W_global[w_idx] += c_coeff * w_val
            
            # Assign to U, V, W
            U[:, col] = U_global
            V[:, col] = V_global
            W[:, col] = W_global
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
