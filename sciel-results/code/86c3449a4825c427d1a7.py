"""Combined: explicit Strassen 4x4x4 (49) + 4x(2x2x3) (44) construction for 4x4x7."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0

    # Precompute index arrays for efficient verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        u_vals = Ua[u_idx, :]
        v_vals = Va[v_idx, :]
        w_vals = Wa[w_idx, :]
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        # The tensor for matrix multiplication has 1s at (i,j,k) and 0 elsewhere.
        # The decomposition should sum to 1 for all valid (i,j,k) tuples?
        # No, the tensor T[i,j,k] = 1 if the product contributes to C[i,k] via A[i,j]B[j,k].
        # Actually, the condition is:
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k.
        # Wait, the standard tensor decomposition of matrix multiplication T_{ijk} is:
        # T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
        # Where T_{ijk} is 1 if the indices correspond to the multiplication term A_{ij}B_{jk} contributing to C_{ik}.
        # In the standard basis, T_{ijk} = 1 for all i,j,k.
        # So result should be 1.0 everywhere.
        return np.max(np.abs(result - 1.0))

    # Strassen 2x2x2: 7 terms.
    # Each term is (A_sub, B_sub, C_sub)
    # A_sub, B_sub are 2x2 sparse dicts {(row,col): coeff}
    # C_sub is 2x2 sparse dict {(row,col): coeff}
    strassen_2x2x2 = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),   # M1
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),           # M2
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),           # M3
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),           # M4
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),           # M5
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),           # M6
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),           # M7
    ]

    total_rank = 93
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    current_rank = 0

    # Top-level Strassen 4x4x4 terms
    # A blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) in block coords (br, bc) -> index 2*br+bc
    # The outer Strassen combines these 4 blocks.
    # M1: A(0,0)+A(1,1) * B(0,0)+B(1,1) -> C(0,0)+C(1,1)
    # M2: A(1,0)+A(1,1) * B(0,0) -> C(1,0)-C(1,1) ... wait, standard Strassen on 4x4 blocks.
    
    # Let's use the explicit block coefficients from the inspiration which is verified.
    # Outer terms for 4x4x4 (Rank 49)
    # A_block_indices, B_block_indices, C_block_indices
    # Block index: 0:0-1, 1:2-3 (row), 0:0-1, 1:2-3 (col)
    # So block 0 is top-left, 1 is top-right, 2 is bottom-left, 3 is bottom-right.
    
    top_terms = [
        # M1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # M2
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}), # C(1,0) is block 2? No.
        # Let's stick to the row/col block indices.
        # Block (br, bc) where br, bc in {0,1}.
        # M1: A(0,0)+A(1,1) * B(0,0)+B(1,1) -> C(0,0)+C(1,1)
        # M2: A(1,0)+A(1,1) * B(0,0) -> C(1,0) - C(1,1) is wrong.
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Mapping to block indices:
        # A11=0, A12=1, A21=2, A22=3
        # B11=0, B12=1, B21=2, B22=3
        # C11=0, C12=1, C21=2, C22=3
        
        # M1: A{0,3} B{0,3} C{0,3}
        # M2: A{2,3} B{0} C{2,-3} -> C(2,0) and C(3,-1)?
        # C = M1 + M4 - M5 + M7 (for C11)
        # C = M3 + M5 (for C12)
        # C = M2 + M4 (for C21)
        # C = M1 - M2 + M6 + M7 (for C22)
        
        # So the C block contributions:
        # M1: C0:1, C3:1
        # M2: C2:1, C3:-1
        # M3: C1:1, C3:1
        # M4: C0:1, C2:1
        # M5: C0:-1, C1:1
        # M6: C3:1
        # M7: C0:1
    ]
    
    # Correct Top Terms based on standard Strassen formulas:
    top_terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}), # M1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),      # M2
        ({0:1}, {2:1, 3:-1}, {1:1, 3:1}),      # M3
        ({3:1}, {1:1, 0:-1}, {0:1, 2:1}),      # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),      # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),      # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),      # M7
    ]

    for m_idx in range(7):
        A_blocks, B_blocks, C_blocks = top_terms[m_idx]
        for t_idx in range(7):
            A_sub, B_sub, C_sub = strassen_2x2x2[t_idx]
            r = current_rank
            current_rank += 1
            
            # U: Combine A_block and A_sub
            for (li, lj), c_sub in A_sub.items():
                for b_idx, c_blk in A_blocks.items():
                    br = b_idx // 2
                    bc = b_idx % 2
                    i = 2 * br + li
                    j = 2 * bc + lj
                    U[i * m + j, r] += c_sub * c_blk
                    
            # V: Combine B_block and B_sub
            for (li, lj), c_sub in B_sub.items():
                for b_idx, c_blk in B_blocks.items():
                    br = b_idx // 2
                    bc = b_idx % 2
                    i = 2 * br + li
                    j = 2 * bc + lj
                    V[i * p + j, r] += c_sub * c_blk
                    
            # W: Combine C_block and C_sub
            for (li, lj), c_sub in C_sub.items():
                for b_idx, c_blk in C_blocks.items():
                    br = b_idx // 2
                    bc = b_idx % 2
                    i = 2 * br + li
                    j = 2 * bc + lj
                    W[j * n + i, r] += c_sub * c_blk

    assert current_rank == 49, f"Expected 49, got {current_rank}"

    # Part 2: 4x4x3 (cols 4,5,6)
    # We need to handle C_top (rows 0-1, cols 4-6) and C_bot (rows 2-3, cols 4-6)
    # C_top = A_00 B_top + A_01 B_bot
    # C_bot = A_10 B_top + A_11 B_bot
    
    def add_2x2x3(a_row, a_col, b_row, c_row):
        """
        Adds decomposition for A[a_row:a_row+2, a_col:a_col+2] * B[b_row:b_row+2, 4:7] -> C[c_row:c_row+2, 4:7]
        """
        nonlocal current_rank
        
        # Strassen 2x2x2 for cols 4,5 (indices 0,1 in local B/C)
        for t_idx in range(7):
            A_sub, B_sub, C_sub = strassen_2x2x2[t_idx]
            r = current_rank
            current_rank += 1
            
            for (li, lj), c in A_sub.items():
                i = a_row + li
                j = a_col + lj
                U[i * m + j, r] += c
                
            for (li, lj), c in B_sub.items():
                i = b_row + li
                j = 4 + lj # Global col 4, 5
                V[i * p + j, r] += c
                
            for (li, lj), c in C_sub.items():
                i = c_row + li
                j = 4 + lj # Global col 4, 5
                W[j * n + i, r] += c

        # Schoolbook for col 6 (index 2 in local B/C)
        # C[i, 6] = sum_j A[i, j] B[j, 6]
        # Local indices: i in 0,1; j in 0,1
        for li in range(2):
            for lj in range(2):
                r = current_rank
                current_rank += 1
                
                i_a = a_row + li
                j_a = a_col + lj
                U[i_a * m + j_a, r] = 1
                
                i_b = b_row + lj # B row corresponds to A col in schoolbook?
                # Wait. C[i,k] = sum_j A[i,j] B[j,k].
                # Here k=6. j is the summation index.
                # A is 2x2. Rows li, Cols lj.
                # B is 2x3. Rows lj (since A col lj matches B row lj), Cols 4,5,6.
                # So B row is b_row + lj.
                V[i_b * p + 6, r] = 1
                
                i_c = c_row + li
                W[6 * n + i_c, r] = 1

    # A_00 (0,0) * B_top (0) -> C_top (0)
    add_2x2x3(a_row=0, a_col=0, b_row=0, c_row=0)
    # A_01 (0,1) * B_bot (1) -> C_top (0)
    # A_01 is rows 0-1, cols 2-3. So a_row=0, a_col=1 (block index) -> a_col=2.
    # B_bot is rows 2-3. So b_row=1 (block index) -> b_row=2.
    # C_top is rows 0-1. So c_row=0 (block index) -> c_row=0.
    add_2x2x3(a_row=0, a_col=2, b_row=2, c_row=0)
    # A_10 (1,0) * B_top (0) -> C_bot (1)
    # A_10 is rows 2-3, cols 0-1. a_row=2, a_col=0.
    # B_top is rows 0-1. b_row=0.
    # C_bot is rows 2-3. c_row=2.
    add_2x2x3(a_row=2, a_col=0, b_row=0, c_row=2)
    # A_11 (1,1) * B_bot (1) -> C_bot (1)
    # A_11 is rows 2-3, cols 2-3. a_row=2, a_col=2.
    # B_bot is rows 2-3. b_row=2.
    # C_bot is rows 2-3. c_row=2.
    add_2x2x3(a_row=2, a_col=2, b_row=2, c_row=2)

    assert current_rank == 93, f"Expected 93, got {current_rank}"

    # Verify
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    err = verify(U_list, V_list, W_list, 93)
    if err < 1e-9:
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If verification fails, it's a bug in construction.
    # For safety, return schoolbook (Rank 112)
    rank = n * m * p
    U_s = [[0] * rank for _ in range(n * m)]
    V_s = [[0] * rank for _ in range(m * p)]
    W_s = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][col] = 1
                V_s[j * p + k][col] = 1
                W_s[k * n + i][col] = 1
                col += 1
    return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
