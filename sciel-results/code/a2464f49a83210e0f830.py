import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def verify_decomposition(U_mat, V_mat, W_mat, rank):
        """Verify that the decomposition correctly represents the multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i * m + j, :]
                    v_val = V_mat[j * p + k, :]
                    w_val = W_mat[k * n + i, :]
                    dot = np.sum(u_val * v_val * w_val)
                    if abs(dot - 1.0) > 1e-9:
                        return False
        return True
    
    # Start with schoolbook rank-72 decomposition
    total_rank = 72
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    best_rank = total_rank
    
    # Systematic rank reduction: try to remove columns by solving linear systems
    # For each column r, check if it can be expressed as a combination of others
    # such that the tensor constraint is still satisfied
    
    np.random.seed(42)
    
    # Try removing one column at a time, adjusting the remaining columns
    # This is equivalent to finding a rank-deficient representation
    
    def try_remove_column(U_mat, V_mat, W_mat, col_idx, rank):
        """Try to remove column col_idx by expressing it as a combination of others."""
        if rank <= 1:
            return None, None, None, False
        
        # Get the column to remove
        u_rem = U_mat[:, col_idx].copy()
        v_rem = V_mat[:, col_idx].copy()
        w_rem = W_mat[:, col_idx].copy()
        
        # Get all other columns
        keep_mask = np.ones(rank, dtype=bool)
        keep_mask[col_idx] = False
        other_cols = np.where(keep_mask)[0]
        
        if len(other_cols) == 0:
            return None, None, None, False
        
        # We need to find coefficients c_j such that for all (i,j,k):
        # u_rem[i*m+j] * v_rem[j*p+k] * w_rem[k*n+i] = sum_j c_j * U_mat[i*m+j, j] * V_mat[j*p+k, j] * W_mat[k*n+i, j]
        
        # This is a nonlinear system. Instead, let's try a different approach:
        # Check if the column is "redundant" by seeing if removing it and solving
        # for the tensor constraint works.
        
        # Alternative: use the fact that if we have a valid decomposition of rank r,
        # and one of the columns is zero, we can remove it. But in schoolbook, no column is zero.
        
        # Let's try a gradient-based approach: perturb the remaining columns slightly
        # to absorb the contribution of the removed column.
        
        # Actually, let's try a simpler heuristic: check if any two columns are
        # "similar" in the sense that their outer products are close.
        
        return None, None, None, False
    
    # Better approach: use random linear combinations more systematically
    # Try to replace groups of columns with fewer columns
    
    max_attempts = 20000
    attempts = 0
    
    while attempts < max_attempts and (time.time() - start_time) < deadline:
        attempts += 1
        
        if best_rank <= 50:
            break
        
        # Strategy 1: Try to merge two columns into one
        c1, c2 = np.random.choice(best_rank, size=2, replace=False)
        
        # Try several random combinations
        for trial in range(10):
            alpha = np.random.uniform(-3, 3)
            beta = np.random.uniform(-3, 3)
            
            new_U_col = alpha * best_U[:, c1] + beta * best_U[:, c2]
            new_V_col = alpha * best_V[:, c1] + beta * best_V[:, c2]
            new_W_col = alpha * best_W[:, c1] + beta * best_W[:, c2]
            
            keep_mask = np.ones(best_rank, dtype=bool)
            keep_mask[c1] = False
            keep_mask[c2] = False
            
            cand_U = np.column_stack([best_U[:, keep_mask], new_U_col])
            cand_V = np.column_stack([best_V[:, keep_mask], new_V_col])
            cand_W = np.column_stack([best_W[:, keep_mask], new_W_col])
            
            cand_rank = best_rank - 1
            
            if verify_decomposition(cand_U, cand_V, cand_W, cand_rank):
                best_U = cand_U
                best_V = cand_V
                best_W = cand_W
                best_rank = cand_rank
                break
        
        # Strategy 2: Try to remove a single column by adjusting others
        if best_rank > 1:
            col_to_remove = np.random.randint(best_rank)
            
            # Check if we can absorb this column's contribution
            # by slightly modifying other columns
            u_rem = best_U[:, col_to_remove]
            v_rem = best_V[:, col_to_remove]
            w_rem = best_W[:, col_to_remove]
            
            # For each (i,j,k), the contribution is u_rem[i*m+j] * v_rem[j*p+k] * w_rem[k*n+i]
            # We need to distribute this among the remaining columns
            
            # This is complex, so let's try a different approach:
            # If the column has small entries, maybe we can drop it
            col_norm = np.linalg.norm(u_rem) * np.linalg.norm(v_rem) * np.linalg.norm(w_rem)
            
            if col_norm < 1e-6:
                # Column is essentially zero, remove it
                keep_mask = np.ones(best_rank, dtype=bool)
                keep_mask[col_to_remove] = False
                cand_U = best_U[:, keep_mask]
                cand_V = best_V[:, keep_mask]
                cand_W = best_W[:, keep_mask]
                cand_rank = best_rank - 1
                
                if verify_decomposition(cand_U, cand_V, cand_W, cand_rank):
                    best_U = cand_U
                    best_V = cand_V
                    best_W = cand_W
                    best_rank = cand_rank
    
    # Final polish: try to find exact integer/half-integer solutions
    # by rounding and verifying
    
    # Round to nearest 0.5
    def round_to_half(x):
        return np.round(x * 2) / 2.0
    
    for _ in range(5):
        rounded_U = round_to_half(best_U)
        rounded_V = round_to_half(best_V)
        rounded_W = round_to_half(best_W)
        
        if verify_decomposition(rounded_U, rounded_V, rounded_W, best_rank):
            best_U = rounded_U
            best_V = rounded_V
            best_W = rounded_W
            break
    
    # Convert to list format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Final verification
    if not verify_decomposition(best_U, best_V, best_W, best_rank):
        # Fallback to schoolbook
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_list, "V": V_list, "W": W_list}
