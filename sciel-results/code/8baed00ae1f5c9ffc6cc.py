"""Matrix multiplication 2,5,6 rank-minimal tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Target Rank 54 based on Strassen decomposition of 2x2 blocks
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        """
        Adds a rank-7 Strassen decomposition for a 2x2 * 2x2 block.
        """
        nonlocal col
        
        # M0: (A00+A11)(B00+B11)
        U[i_base * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        U[(i_base + 1) * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] -= 1
        W[(k_base + 1) * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base + 1) * n + i_base][col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        U[(i_base + 1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        U[i_base * m + (j_base + 1)][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] -= 1
        V[(j_base + 1) * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        """
        Adds a rank-4 naive decomposition for a 2x1 * 1x2 block.
        """
        nonlocal col
        
        # 1: A00 B00 -> C00
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        
        # 2: A00 B01 -> C01
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + i_base][col] += 1
        col += 1
        
        # 3: A10 B00 -> C10
        U[(i_base + 1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1
        
        # 4: A10 B01 -> C11
        U[(i_base + 1) * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1

    # Apply Strassen blocks for j pairs (0,1) and (2,3)
    # j_base = 0
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # j_base = 2
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # Apply naive blocks for j=4
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)
    
    # Verify col == 54
    if col != 54:
        # Fallback to schoolbook if logic error
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Verify the decomposition using NumPy
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
                
    T_dec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    if not np.allclose(T_dec, T, atol=1e-10, rtol=1e-10):
        # If verification fails, return schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Attempt rank reduction via dual-variable elimination
    # Check if any rank-1 term can be eliminated while preserving the decomposition
    
    U_mat = np.array(U, dtype=np.float64)  # (10, 54)
    V_mat = np.array(V, dtype=np.float64)  # (30, 54)
    W_mat = np.array(W, dtype=np.float64)  # (12, 54)
    
    # Build matrix of flattened rank-1 tensors: each column is vec(u_r v_r w_r)
    # Shape: (60, 54)
    flat_cols = []
    for r in range(54):
        term = np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        flat_cols.append(term.flatten())
    flat_matrix = np.array(flat_cols).T  # (60, 54)
    
    # For each term r, check if vec(term_r) is in the column space of the other 53 terms
    # If yes, we can eliminate term r and reduce rank to 53
    
    # Due to time constraints, try at most 10 candidate terms
    # Prioritize terms that might be redundant (check norm or similarity)
    
    best_reduced = None
    best_rank = 54
    
    # Try to eliminate each term
    for r_to_remove in range(54):
        if time.time() > deadline - 20:  # Safety margin
            break
        
        # Build matrix of other 53 terms
        other_indices = [i for i in range(54) if i != r_to_remove]
        A = flat_matrix[:, other_indices]  # (60, 53)
        b = flat_matrix[:, r_to_remove]  # (60,)
        
        # Check if b is in column space of A
        # Solve A x = b using least squares
        try:
            x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
            # Check if the residual is small
            residual_norm = np.linalg.norm(A @ x - b)
            b_norm = np.linalg.norm(b)
            if b_norm > 1e-10 and residual_norm / b_norm < 1e-8:
                # Term r_to_remove can be eliminated
                new_rank = 53
                if new_rank < best_rank:
                    # Build reduced decomposition
                    U_new = [U_mat[:, i] for i in other_indices]
                    V_new = [V_mat[:, i] for i in other_indices]
                    W_new = [W_mat[:, i] for i in other_indices]
                    
                    # Verify the reduced decomposition
                    U_new_arr = np.array(U_new).T  # (10, 53)
                    V_new_arr = np.array(V_new).T  # (30, 53)
                    W_new_arr = np.array(W_new).T  # (12, 53)
                    
                    T_dec_new = np.einsum('ar,br,cr->abc', U_new_arr, V_new_arr, W_new_arr)
                    if np.allclose(T_dec_new, T, atol=1e-10, rtol=1e-10):
                        best_reduced = (U_new_arr, V_new_arr, W_new_arr)
                        best_rank = new_rank
        except:
            pass
    
    # If we found a reduced decomposition, return it
    if best_reduced is not None:
        U_r, V_r, W_r = best_reduced
        # Convert to list format with integer/half-integer entries
        U_out = [[float(x) for x in row] for row in U_r.T]
        V_out = [[float(x) for x in row] for row in V_r.T]
        W_out = [[float(x) for x in row] for row in W_r.T]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Otherwise return the original rank-54 decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
