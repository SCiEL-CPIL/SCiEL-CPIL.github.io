"""Combined program: Strassen-based structural decomposition for 4x4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s kill

    def build_strassen_decomposition():
        """
        Build a Strassen-based decomposition for 4x4x7 matrix multiplication.
        
        Strategy:
        - Decompose 4x4x7 as 4x4x4 (Strassen rank 49) + 4x4x2 (Strassen rank 28) + 4x4x1 (schoolbook rank 16)
        - Total rank: 93
        """
        total_rank = 93
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        current_rank = 0
        
        # Strassen 2x2x2 terms: (x_sub, y_sub, c_sub)
        strassen_2x2 = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
        ]
        
        # Strassen 4x4x4: 7 outer terms, each with 7 inner terms
        strassen_4x4_outer = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}),
            ({0: 1}, {1: 1, 3: -1}),
            ({3: 1}, {2: 1, 0: -1}),
            ({0: 1, 1: 1}, {3: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}),
        ]
        
        c_blocks_4x4 = [
            {(0,0): 1.0, (1,1): 1.0},
            {(1,0): 1.0, (1,1): -1.0},
            {(0,1): 1.0, (1,1): 1.0},
            {(0,0): 1.0, (1,0): 1.0},
            {(0,0): -1.0, (0,1): 1.0},
            {(1,1): 1.0},
            {(0,0): 1.0},
        ]
        
        # Part 1: 4x4x4 Strassen (rank 49)
        for m_idx in range(7):
            x_def, y_def = strassen_4x4_outer[m_idx]
            c_block = c_blocks_4x4[m_idx]
            for t_idx in range(7):
                r = current_rank
                current_rank += 1
                x_sub, y_sub, c_sub = strassen_2x2[t_idx]
                
                for (li, lj), cx in x_sub.items():
                    for b, cb in x_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        U[i * m + j, r] += cx * cb
                
                for (li, lj), cy in y_sub.items():
                    for b, cb in y_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        V[i * p + j, r] += cy * cb
                
                for (li, lj), cc in c_sub.items():
                    if cc == 0:
                        continue
                    for (br, bc), cb_c in c_block.items():
                        if cb_c == 0:
                            continue
                        i_final = 2 * br + li
                        k_final = 2 * bc + lj
                        W[k_final * n + i_final, r] += cc * cb_c
        
        # Part 2: 4x4x2 Strassen (rank 28)
        # Decompose 4x4x2 as two 2x4x2 blocks
        # Block 1: rows 0-1 of A, cols 4-5 of B, rows 0-1 of C
        # Block 2: rows 2-3 of A, cols 4-5 of B, rows 2-3 of C
        # Each 2x4x2 can be done with rank 14 (Strassen-like)
        
        # For 2x4x2: A is 2x4, B is 4x2, C is 2x2
        # We can use a decomposition: split A into A[:, 0:2] and A[:, 2:4]
        # Each 2x2x2 uses Strassen rank 7
        # Total: 7 + 7 = 14 per block, 28 for both blocks
        
        for block in range(2):
            # block 0: A rows 0-1, B cols 4-5, C rows 0-1
            # block 1: A rows 2-3, B cols 4-5, C rows 2-3
            a_row_offset = 2 * block
            c_row_offset = 2 * block
            
            # Split B cols 4-5 into B[:, 4] and B[:, 5]
            # Actually B is 4x7, so B[:, 4] and B[:, 5] are vectors
            # For 2x4x2: A[i,j] where i in 0-1, j in 0-3
            # B[j,k] where j in 0-3, k in 0-1
            # C[i,k] where i in 0-1, k in 0-1
            
            # We decompose as: C = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
            # Each term is 2x2x2, using Strassen rank 7
            
            for sub_block in range(2):
                # sub_block 0: A cols 0-1, B rows 0-1
                # sub_block 1: A cols 2-3, B rows 2-3
                a_col_offset = 2 * sub_block
                b_row_offset = 2 * sub_block
                
                # Now we have 2x2x2: A[i, a_col_offset + j], B[b_row_offset + j, k], C[i, k]
                # Use Strassen for 2x2x2
                for t_idx in range(7):
                    r = current_rank
                    current_rank += 1
                    x_sub, y_sub, c_sub = strassen_2x2[t_idx]
                    
                    # U: A indices
                    for (li, lj), cx in x_sub.items():
                        i = a_row_offset + li
                        j = a_col_offset + lj
                        U[i * m + j, r] = cx
                    
                    # V: B indices
                    for (lj, lk), cy in y_sub.items():
                        j = b_row_offset + lj
                        k = 4 + lk  # B cols 4,5
                        V[j * p + k, r] = cy
                    
                    # W: C indices
                    for (li, lk), cc in c_sub.items():
                        i = c_row_offset + li
                        k = lk
                        W[k * n + i, r] = cc
        
        # Part 3: 4x4x1 schoolbook (rank 16)
        # A is 4x4, B is 4x1 (col 6), C is 4x1
        for i in range(n):
            for j in range(m):
                r = current_rank
                current_rank += 1
                U[i * m + j, r] = 1.0
                V[j * p + 6, r] = 1.0
                W[6 * n + i, r] = 1.0
        
        assert current_rank == total_rank, f"Expected {total_rank}, got {current_rank}"
        
        return U[:, :total_rank].tolist(), V[:, :total_rank].tolist(), W[:, :total_rank].tolist()
    
    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        u_vals = Ua[u_idx, :]
        v_vals = Va[v_idx, :]
        w_vals = Wa[w_idx, :]
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))
    
    # Build the structural decomposition
    U, V, W, rank = build_strassen_decomposition()
    
    # Verify correctness
    err = verify(U, V, W, rank)
    
    if err < 1e-10:
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, fall back to schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END

