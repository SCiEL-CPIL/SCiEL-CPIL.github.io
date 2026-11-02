import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 2.0

def _remaining():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)

def construct():
    n, m, p = 2, 4, 5
    
    # Base rank-36 decomposition using Strassen for 2x2x2 blocks and schoolbook for 2x2x1
    U = [[0] * 36 for _ in range(n * m)]
    V = [[0] * 36 for _ in range(m * p)]
    W = [[0] * 36 for _ in range(p * n)]
    
    def add_strassen(col_offset, j_offset, k_offset):
        U[0 * m + j_offset + 0][col_offset + 0] = 1
        U[1 * m + j_offset + 1][col_offset + 0] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 0] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 0] = 1
        W[(k_offset + 0) * n + 0][col_offset + 0] = 1
        W[(k_offset + 1) * n + 1][col_offset + 0] = 1

        U[1 * m + j_offset + 0][col_offset + 1] = 1
        U[1 * m + j_offset + 1][col_offset + 1] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 1] = 1
        W[(k_offset + 0) * n + 1][col_offset + 1] = 1
        W[(k_offset + 1) * n + 1][col_offset + 1] = -1

        U[0 * m + j_offset + 0][col_offset + 2] = 1
        V[(j_offset + 0) * p + k_offset + 1][col_offset + 2] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 2] = -1
        W[(k_offset + 1) * n + 0][col_offset + 2] = 1
        W[(k_offset + 1) * n + 1][col_offset + 2] = 1

        U[1 * m + j_offset + 1][col_offset + 3] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 3] = -1
        V[(j_offset + 1) * p + k_offset + 0][col_offset + 3] = 1
        W[(k_offset + 0) * n + 0][col_offset + 3] = 1
        W[(k_offset + 0) * n + 1][col_offset + 3] = 1

        U[0 * m + j_offset + 0][col_offset + 4] = 1
        U[0 * m + j_offset + 1][col_offset + 4] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 4] = 1
        W[(k_offset + 0) * n + 0][col_offset + 4] = -1
        W[(k_offset + 1) * n + 0][col_offset + 4] = 1

        U[0 * m + j_offset + 0][col_offset + 5] = -1
        U[1 * m + j_offset + 0][col_offset + 5] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 5] = 1
        V[(j_offset + 0) * p + k_offset + 1][col_offset + 5] = 1
        W[(k_offset + 1) * n + 1][col_offset + 5] = 1

        U[0 * m + j_offset + 1][col_offset + 6] = 1
        U[1 * m + j_offset + 1][col_offset + 6] = -1
        V[(j_offset + 1) * p + k_offset + 0][col_offset + 6] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 6] = 1
        W[(k_offset + 0) * n + 0][col_offset + 6] = 1

    def add_schoolbook_2x2x1(col_offset, j_offset, k_offset):
        U[0 * m + j_offset + 0][col_offset + 0] = 1
        V[(j_offset + 0) * p + k_offset][col_offset + 0] = 1
        W[k_offset * n + 0][col_offset + 0] = 1
        
        U[0 * m + j_offset + 1][col_offset + 1] = 1
        V[(j_offset + 1) * p + k_offset][col_offset + 1] = 1
        W[k_offset * n + 0][col_offset + 1] = 1
        
        U[1 * m + j_offset + 0][col_offset + 2] = 1
        V[(j_offset + 0) * p + k_offset][col_offset + 2] = 1
        W[k_offset * n + 1][col_offset + 2] = 1
        
        U[1 * m + j_offset + 1][col_offset + 3] = 1
        V[(j_offset + 1) * p + k_offset][col_offset + 3] = 1
        W[k_offset * n + 1][col_offset + 3] = 1

    col = 0
    add_strassen(col, 0, 0); col += 7
    add_strassen(col, 0, 2); col += 7
    add_schoolbook_2x2x1(col, 0, 4); col += 4
    
    add_strassen(col, 2, 0); col += 7
    add_strassen(col, 2, 2); col += 7
    add_schoolbook_2x2x1(col, 2, 4); col += 4
    
    rank = 36
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verification function
    def verify(U_mat, V_mat, W_mat, r):
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U_mat[i*m+j, r_idx] * V_mat[j*p+k, r_idx] * W_mat[k*n+i, r_idx]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U_np, V_np, W_np, rank):
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by merging columns
    # Strategy: try to express one column as a linear combination of others
    # This is equivalent to finding a linear dependency in the tensor
    
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    best_rank = rank
    
    # Try greedy elimination: for each column, see if it can be removed
    # by adjusting other columns
    for elim_col in range(rank - 1, -1, -1):
        if _remaining() < 5.0:
            break
            
        # Try to find coefficients c such that:
        # U[:, elim_col] = sum_{r != elim_col} c_r * U[:, r]
        # V[:, elim_col] = sum_{r != elim_col} c_r * V[:, r]  
        # W[:, elim_col] = sum_{r != elim_col} c_r * W[:, r]
        
        # This is a linear system. We can set up a least squares problem.
        # For each matrix, we need: U[:, elim_col] ≈ U[:, :elim_col] @ c + U[:, elim_col+1:] @ c
        # But we need the same c to work for all three matrices.
        
        # Build the system: for each (i, j, k), we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # If we remove column elim_col, we need to adjust other columns.
        
        # Simpler approach: try random perturbations and check if rank can be reduced
        # Or use the fact that if columns are linearly dependent, rank can be reduced.
        
        # Check if column elim_col is in the span of others for each matrix
        # For U: check if U[:, elim_col] is in span of U[:, r] for r != elim_col
        # For V: check if V[:, elim_col] is in span of V[:, r] for r != elim_col
        # For W: check if W[:, elim_col] is in span of W[:, r] for r != elim_col
        
        U_others = np.column_stack([U_np[:, r] for r in range(rank) if r != elim_col])
        V_others = np.column_stack([V_np[:, r] for r in range(rank) if r != elim_col])
        W_others = np.column_stack([W_np[:, r] for r in range(rank) if r != elim_col])
        
        # Check rank
        rank_U = np.linalg.matrix_rank(U_others)
        rank_V = np.linalg.matrix_rank(V_others)
        rank_W = np.linalg.matrix_rank(W_others)
        
        # If the column is in the span, we might be able to remove it
        # But we need to ensure the tensor decomposition still holds.
        
        # Try a simpler heuristic: if the column has very small norm, it might be redundant
        col_norm_U = np.linalg.norm(U_np[:, elim_col])
        col_norm_V = np.linalg.norm(V_np[:, elim_col])
        col_norm_W = np.linalg.norm(W_np[:, elim_col])
        
        if col_norm_U < 1e-10 and col_norm_V < 1e-10 and col_norm_W < 1e-10:
            # This column is effectively zero, remove it
            new_rank = rank - 1
            new_U = np.column_stack([U_np[:, r] for r in range(rank) if r != elim_col])
            new_V = np.column_stack([V_np[:, r] for r in range(rank) if r != elim_col])
            new_W = np.column_stack([W_np[:, r] for r in range(rank) if r != elim_col])
            
            if verify(new_U, new_V, new_W, new_rank):
                best_U = new_U
                best_V = new_V
                best_W = new_W
                best_rank = new_rank
                U_np = new_U
                V_np = new_V
                W_np = new_W
                rank = new_rank
    
    # Convert back to list of lists with integer/half-integer entries
    def to_half_int(val):
        # Round to nearest half-integer
        return round(val * 2) / 2.0
    
    U_out = [[to_half_int(best_U[i, r]) for r in range(best_rank)] for i in range(n * m)]
    V_out = [[to_half_int(best_V[i, r]) for r in range(best_rank)] for i in range(m * p)]
    W_out = [[to_half_int(best_W[i, r]) for r in range(best_rank)] for i in range(p * n)]
    
    # Final verification with the converted values
    U_final = np.array(U_out, dtype=float)
    V_final = np.array(V_out, dtype=float)
    W_final = np.array(W_out, dtype=float)
    
    if verify(U_final, V_final, W_final, best_rank):
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback to original
    return {"U": U, "V": V, "W": W}
