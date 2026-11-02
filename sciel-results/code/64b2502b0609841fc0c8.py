"""
Rank-minimal tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Strategy: Start with Strassen's rank-7 algorithm on 2x2 blocks (rank 42),
then attempt to reduce rank by finding linear dependencies using tensor
slicing and rank-revealing linear algebra over the rationals.
"""

import time
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 6
    nm = n * m  # 8
    mp = m * p  # 24
    pn = p * n  # 12

    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # ---------------------------------------------------------------
    # Strassen 2x2@2x2 rank-7 decomposition
    # ---------------------------------------------------------------
    strassen_U = [
        {0: 1, 3: 1},       # P1: a00 + a11
        {2: 1, 3: 1},       # P2: a10 + a11
        {0: 1},             # P3: a00
        {3: 1},             # P4: a11
        {0: 1, 1: 1},       # P5: a00 + a01
        {0: -1, 2: 1},      # P6: a10 - a00
        {1: 1, 3: -1},      # P7: a01 - a11
    ]
    strassen_V = [
        {0: 1, 3: 1},       # b00 + b11
        {0: 1},             # b00
        {1: 1, 3: -1},      # b01 - b11
        {2: 1, 0: -1},      # b10 - b00
        {3: 1},             # b11
        {0: 1, 1: 1},       # b00 + b01
        {2: 1, 3: 1},       # b10 + b11
    ]
    strassen_W = [
        {0: 1, 3: 1},       # P1: c00 + c11
        {2: 1, 3: -1},      # P2: c10 - c11
        {1: 1, 3: 1},       # P3: c01 + c11
        {0: 1, 2: 1},       # P4: c00 + c10
        {0: -1, 1: 1},      # P5: -c00 + c01
        {3: 1},             # P6: c11
        {0: 1},             # P7: c00
    ]

    # ---------------------------------------------------------------
    # Build the rank-42 decomposition
    # ---------------------------------------------------------------
    rank_total = 42
    U = [[0] * rank_total for _ in range(nm)]
    V = [[0] * rank_total for _ in range(mp)]
    W = [[0] * rank_total for _ in range(pn)]

    col_offset = 0
    for j_block in range(3):
        for side in ['L', 'R']:
            if side == 'L':
                u_col_offset = 0
                v_row_offset = 0
            else:
                u_col_offset = 2
                v_row_offset = 2

            for prod_idx in range(7):
                col = col_offset + prod_idx

                for local_idx, sign in strassen_U[prod_idx].items():
                    i_local = local_idx // 2
                    j_local = local_idx % 2
                    u_global = i_local * m + (u_col_offset + j_local)
                    U[u_global][col] += sign

                for local_idx, sign in strassen_V[prod_idx].items():
                    j_local = local_idx // 2
                    k_local = local_idx % 2
                    v_global = (v_row_offset + j_local) * p + (2 * j_block + k_local)
                    V[v_global][col] += sign

                for local_idx, sign in strassen_W[prod_idx].items():
                    i_local = local_idx // 2
                    k_local = local_idx % 2
                    w_global = (2 * j_block + k_local) * n + i_local
                    W[w_global][col] += sign

            col_offset += 7

    # ---------------------------------------------------------------
    # Verify the rank-42 decomposition
    # ---------------------------------------------------------------
    def verify(U_mat, V_mat, W_mat, rank):
        U_arr = np.array(U_mat, dtype=np.float64)
        V_arr = np.array(V_mat, dtype=np.float64)
        W_arr = np.array(W_mat, dtype=np.float64)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.sum(U_arr[idx_u] * V_arr[idx_v] * W_arr[idx_w])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err

    err_42 = verify(U, V, W, rank_total)
    if err_42 > 1e-9:
        # Fallback: schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(nm)]
        V_s = [[0] * rank_school for _ in range(mp)]
        W_s = [[0] * rank_school for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}

    best_U, best_V, best_W = U, V, W
    best_rank = rank_total

    # ---------------------------------------------------------------
    # Tensor slicing method: analyze the structure of the tensor
    # ---------------------------------------------------------------
    # The multiplication tensor T[i,j,k] = 1 for all valid i,j,k
    # We want to find if we can reduce the rank by finding
    # linear dependencies in the slice space.
    
    # For each "slice" (fixing one index), analyze the rank structure
    # Slice by i: T[i,j,k] for fixed i gives a 4x6 matrix
    # Slice by j: T[i,j,k] for fixed j gives a 2x6 matrix  
    # Slice by k: T[i,j,k] for fixed k gives a 2x4 matrix
    
    # The key insight: if we can find a non-trivial linear combination
    # of the rank-1 tensors that sums to zero, we can reduce the rank.
    
    # Convert to numpy for efficient operations
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # ---------------------------------------------------------------
    # Method 1: Check for linear dependencies using exact arithmetic
    # ---------------------------------------------------------------
    # For each pair of columns, check if there exist alpha, beta
    # such that the combined column maintains the tensor equation.
    
    if time.time() < deadline - 30:
        # Try greedy elimination: for each column, check if it's
        # in the span of the others in a way that preserves the tensor.
        
        for col_to_remove in range(rank_total - 1, -1, -1):
            if time.time() > deadline - 15:
                break
            
            # Check if we can remove this column
            # by finding if the tensor equation still holds
            U_others = np.delete(U_arr, col_to_remove, axis=1)
            V_others = np.delete(V_arr, col_to_remove, axis=1)
            W_others = np.delete(W_arr, col_to_remove, axis=1)
            
            # Check if the column is "redundant" by seeing if we can
            # express it as a combination of others that preserves the tensor
            U_col = U_arr[:, col_to_remove]
            V_col = V_arr[:, col_to_remove]
            W_col = W_arr[:, col_to_remove]
            
            # Check rank conditions
            rank_U_others = np.linalg.matrix_rank(U_others)
            rank_V_others = np.linalg.matrix_rank(V_others)
            rank_W_others = np.linalg.matrix_rank(W_others)
            
            rank_U_with = np.linalg.matrix_rank(np.column_stack([U_others, U_col]))
            rank_V_with = np.linalg.matrix_rank(np.column_stack([V_others, V_col]))
            rank_W_with = np.linalg.matrix_rank(np.column_stack([W_others, W_col]))
            
            # If all three are in the span, we might be able to remove
            if rank_U_with == rank_U_others and rank_V_with == rank_V_others and rank_W_with == rank_W_others:
                # Try removing and verify
                U_new = U_others
                V_new = V_others
                W_new = W_others
                
                new_rank = rank_total - 1
                err = verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), new_rank)
                if err < 1e-9:
                    best_rank = new_rank
                    best_U = U_new.tolist()
                    best_V = V_new.tolist()
                    best_W = W_new.tolist()
                    U_arr = U_new
                    V_arr = V_new
                    W_arr = W_new
                    rank_total = new_rank
                    # Continue trying to remove more columns
                    col_to_remove = min(col_to_remove, rank_total - 1)
    
    # ---------------------------------------------------------------
    # Method 2: Try to find a lower-rank decomposition by
    # optimizing the tensor network structure
    # ---------------------------------------------------------------
    if time.time() < deadline - 20 and best_rank == 42:
        # Try to find a rank-40 or lower decomposition by
        # searching for a subset of columns that still spans the tensor
        
        for target_rank in [41, 40, 39, 38]:
            if time.time() > deadline - 15:
                break
            
            # Try random subsets of columns
            np.random.seed(42)
            found = False
            for attempt in range(200):
                if time.time() > deadline - 15:
                    break
                
                # Random subset of columns
                cols = np.random.choice(rank_total, target_rank, replace=False)
                cols.sort()
                
                U_sub = U_arr[:, cols]
                V_sub = V_arr[:, cols]
                W_sub = W_arr[:, cols]
                
                # Check if this sub-decomposition still works
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            val = np.sum(U_sub[idx_u] * V_sub[idx_v] * W_sub[idx_w])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-9:
                    best_rank = target_rank
                    best_U = U_sub.tolist()
                    best_V = V_sub.tolist()
                    best_W = W_sub.tolist()
                    found = True
                    break
            
            if found:
                break
    
    # ---------------------------------------------------------------
    # Method 3: Try a direct search for a rank-40 decomposition
    # using a different block structure
    # ---------------------------------------------------------------
    if time.time() < deadline - 10 and best_rank == 42:
        # Try to find a more efficient decomposition by
        # combining blocks differently
        
        # Idea: instead of 3 independent 2x2@2x2 blocks,
        # try to find a rank-14 decomposition for 2x4@4x4
        # and a rank-14 for 2x4@4x2, sharing some structure
        
        # For now, try a few different random initializations
        # and use gradient descent to minimize the error
        
        # This is a heuristic approach; we'll try a few random
        # rank-40 decompositions and see if any work
        
        np.random.seed(123)
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            # Random rank-40 decomposition
            U_rand = np.random.randn(nm, 40)
            V_rand = np.random.randn(mp, 40)
            W_rand = np.random.randn(pn, 40)
            
            # Scale to match the expected magnitude
            U_rand *= 0.5
            V_rand *= 0.5
            W_rand *= 0.5
            
            # Check if this decomposition works
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        val = np.sum(U_rand[idx_u] * V_rand[idx_v] * W_rand[idx_w])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-9:
                best_rank = 40
                best_U = U_rand.tolist()
                best_V = V_rand.tolist()
                best_W = W_rand.tolist()
                break
    
    # ---------------------------------------------------------------
    # Return the best decomposition found
    # ---------------------------------------------------------------
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
