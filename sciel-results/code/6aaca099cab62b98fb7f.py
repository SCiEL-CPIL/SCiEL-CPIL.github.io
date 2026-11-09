"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # We use a block decomposition.
    # Split p=10 into 5 blocks of size 2.
    # Each block is M_{2,3,2}.
    # Rank of M_{2,3,2} is 11.
    # Total rank = 5 * 11 = 55.
    
    num_blocks = p // 2
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Pre-define the rank 11 decomposition for M_{2,3,2}
    # Indices for A: i in {0,1}, j in {0,1,2}
    # Indices for B: j in {0,1,2}, k in {0,1} (local k)
    # Indices for C: i in {0,1}, k in {0,1} (local k)
    # W index: k * n + i
    
    # Strassen part (7 terms) for A[:, 0:2] and B[0:2, :]
    # Local indices: i, j in {0,1}, k in {0,1}
    strassen_terms = [
        # M1: (A00+A11)(B00+B11)
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)] # C00, C11
        },
        # M2: (A10+A11)B00
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)] # C10 +=, C11 -=
        },
        # M3: A00(B01-B11)
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)] # C01 +=, C11 +=
        },
        # M4: A11(B10-B00)
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)] # C00 +=, C10 +=
        },
        # M5: (A00+A01)B11
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)] # C00 -=, C01 +=
        },
        # M6: (A10-A00)(B00+B01)
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)] # C11 +=
        },
        # M7: (A01-A11)(B10+B11)
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)] # C00 +=
        }
    ]
    
    # Outer product part (4 terms) for A[:, 2] and B[2, :]
    # Local indices: i in {0,1}, j=2, k in {0,1}
    outer_terms = [
        # i=0, k=0
        {
            'U': [(0,2,1)],
            'V': [(2,0,1)],
            'W': [(0,0,1)]
        },
        # i=0, k=1
        {
            'U': [(0,2,1)],
            'V': [(2,1,1)],
            'W': [(1,0,1)]
        },
        # i=1, k=0
        {
            'U': [(1,2,1)],
            'V': [(2,0,1)],
            'W': [(0,1,1)]
        },
        # i=1, k=1
        {
            'U': [(1,2,1)],
            'V': [(2,1,1)],
            'W': [(1,1,1)]
        }
    ]
    
    all_terms = strassen_terms + outer_terms
    
    col_offset = 0
    for b in range(num_blocks):
        for term in all_terms:
            if time.time() > deadline:
                break
            col = col_offset
            # Fill U
            for i, j, val in term['U']:
                U[i * m + j][col] += val
            # Fill V
            for j, k_local, val in term['V']:
                k_global = b * block_size + k_local
                V[j * p + k_global][col] += val
            # Fill W
            for k_local, i, val in term['W']:
                k_global = b * block_size + k_local
                W[k_global * n + i][col] += val
            col_offset += 1
        if time.time() > deadline:
            break
            
    best_rank = total_rank
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by finding linear dependencies among the rank-1 terms
    if time.time() < deadline - 10:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        def _verify(U_mat, V_mat, W_mat, n, m, p):
            rank = U_mat.shape[1]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = 0.0
                        for c in range(rank):
                            val += U_mat[u_idx, c] * V_mat[v_idx, c] * W_mat[w_idx, c]
                        if abs(val - 1.0) > 1e-9:
                            return False
            return True
        
        if _verify(U_arr, V_arr, W_arr, n, m, p):
            r = total_rank
            
            # Build the constraint matrix: for each (i,j,k), the equation is
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # Build A where A[(i,j,k), c] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            num_entries = n * m * p
            A = np.zeros((num_entries, r))
            entry_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            A[entry_idx, c] = U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                        entry_idx += 1
            
            # Greedy elimination: repeatedly find the column with the smallest
            # residual when projected onto the span of others, and remove it.
            # Continue until no more columns can be removed.
            
            current_cols = list(range(r))
            current_A = A[:, current_cols].copy()
            current_U = U_arr[:, current_cols].copy()
            current_V = V_arr[:, current_cols].copy()
            current_W = W_arr[:, current_cols].copy()
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                curr_rank = current_A.shape[1]
                if curr_rank <= 1:
                    break
                    
                # For each column, compute the residual of projecting it onto the span of others
                best_col_to_remove = -1
                best_residual = float('inf')
                
                for t in range(curr_rank):
                    if time.time() > deadline - 5:
                        break
                    others = [c for c in range(curr_rank) if c != t]
                    A_without_t = current_A[:, others]
                    
                    # Check if current_A[:, t] is in the column space of A_without_t
                    # Use least squares
                    x, residuals, rank, sv = np.linalg.lstsq(A_without_t, current_A[:, t], rcond=1e-10)
                    residual = np.linalg.norm(A_without_t @ x - current_A[:, t])
                    
                    if residual < 1e-8 and residual < best_residual:
                        best_residual = residual
                        best_col_to_remove = t
                
                if best_col_to_remove >= 0:
                    # Remove the best column
                    new_cols = [c for c in range(curr_rank) if c != best_col_to_remove]
                    current_A = current_A[:, new_cols]
                    current_U = current_U[:, new_cols]
                    current_V = current_V[:, new_cols]
                    current_W = current_W[:, new_cols]
                    improved = True
                    print(f"Removed column, rank now {current_A.shape[1]}")
            
            # Final verification
            if _verify(current_U, current_V, current_W, n, m, p):
                if current_A.shape[1] < best_rank:
                    best_rank = current_A.shape[1]
                    best_U, best_V, best_W = current_U, current_V, current_W
                    print(f"Final reduced rank: {best_rank}")
    
    # Convert back to lists if needed
    if isinstance(best_U, np.ndarray):
        U_out = [[int(best_U[i][j]) if abs(best_U[i][j] - round(best_U[i][j])) < 1e-9 else best_U[i][j] for j in range(best_rank)] for i in range(n*m)]
        V_out = [[int(best_V[i][j]) if abs(best_V[i][j] - round(best_V[i][j])) < 1e-9 else best_V[i][j] for j in range(best_rank)] for i in range(m*p)]
        W_out = [[int(best_W[i][j]) if abs(best_W[i][j] - round(best_W[i][j])) < 1e-9 else best_W[i][j] for j in range(best_rank)] for i in range(p*n)]
    else:
        U_out = [[int(best_U[i][j]) if abs(best_U[i][j] - round(best_U[i][j])) < 1e-9 else best_U[i][j] for j in range(best_rank)] for i in range(n*m)]
        V_out = [[int(best_V[i][j]) if abs(best_V[i][j] - round(best_V[i][j])) < 1e-9 else best_V[i][j] for j in range(best_rank)] for i in range(m*p)]
        W_out = [[int(best_W[i][j]) if abs(best_W[i][j] - round(best_W[i][j])) < 1e-9 else best_W[i][j] for j in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
