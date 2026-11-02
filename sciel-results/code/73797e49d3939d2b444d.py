"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x5 matrix.
    Uses the schoolbook decomposition (rank 50) as the baseline, then attempts to reduce rank
    using a local search approach if time permits.
    """
    import time
    import random
    import numpy as np
    
    deadline = time.time() + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Convert to numpy for efficient manipulation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify the schoolbook decomposition
    def verify_decomposition(U, V, W):
        """Check if the decomposition exactly reproduces matrix multiplication."""
        n, m, p = DIMENSIONS
        # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
        # For each (i,j,k), we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # For other combinations, the sum should be 0.
        
        # Build the multiplication tensor
        # T[i, j, k] = 1 if we're multiplying A[i,j] * B[j,k]
        # The decomposition should satisfy: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k} ? No...
        
        # Actually, for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
        # C[i, k] = sum_j A[i, j] * B[j, k]
        # The tensor T[i, j, k] represents the trilinear form.
        # U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] summed over c should give 1 if this is a valid term.
        
        # Let's check by computing the full tensor reconstruction.
        U_mat = np.array(U)  # (n*m, rank)
        V_mat = np.array(V)  # (m*p, rank)
        W_mat = np.array(W)  # (p*n, rank)
        
        # Reshape to get the tensor
        # U reshaped: (n, m, rank)
        # V reshaped: (m, p, rank)
        # W reshaped: (p, n, rank)
        
        U_reshaped = U_mat.reshape(n, m, -1)
        V_reshaped = V_mat.reshape(m, p, -1)
        W_reshaped = W_mat.reshape(p, n, -1)
        
        # For each (i, j, k), compute sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This should be 1 for all (i,j,k) in the schoolbook case.
        
        # Efficient computation:
        # Result[i, j, k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
        
        result = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i, j, k] = np.dot(U_reshaped[i, j, :], np.multiply(V_reshaped[j, k, :], W_reshaped[k, i, :]))
        
        # In schoolbook, all entries should be 1
        if not np.allclose(result, 1.0, atol=1e-10):
            return False
        return True
    
    # Try to reduce rank by removing columns that are redundant
    # This is a heuristic approach; we try to zero out some columns and see if the decomposition still works.
    
    best_rank = rank
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    
    # Simple heuristic: try to remove each column and check if the decomposition still holds
    # This is expensive, so we only do a few iterations.
    
    current_rank = rank
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    
    # Try removing columns one by one in a greedy fashion
    removed = set()
    for attempt in range(min(10, current_rank)):  # Try at most 10 removals
        if time.time() > deadline - 20:
            break
        
        found_removal = False
        for c in range(current_rank):
            if c in removed:
                continue
            
            # Temporarily zero out column c
            old_U_c = current_U[:, c].copy()
            old_V_c = current_V[:, c].copy()
            old_W_c = current_W[:, c].copy()
            
            current_U[:, c] = 0
            current_V[:, c] = 0
            current_W[:, c] = 0
            
            # Check if decomposition still valid
            if verify_decomposition(current_U, current_V, current_W):
                removed.add(c)
                found_removal = True
                break
            else:
                current_U[:, c] = old_U_c
                current_V[:, c] = old_V_c
                current_W[:, c] = old_W_c
        
        if not found_removal:
            break
    
    # Count actual rank (non-zero columns)
    final_rank = 0
    for c in range(rank):
        if c not in removed:
            final_rank += 1
    
    # Convert back to list format
    U_final = np.zeros((n * m, final_rank), dtype=float)
    V_final = np.zeros((m * p, final_rank), dtype=float)
    W_final = np.zeros((p * n, final_rank), dtype=float)
    
    col_idx = 0
    for c in range(rank):
        if c not in removed:
            U_final[:, col_idx] = current_U[:, c]
            V_final[:, col_idx] = current_V[:, c]
            W_final[:, col_idx] = current_W[:, c]
            col_idx += 1
    
    # Round to integers if close
    U_final = np.round(U_final, 10)
    V_final = np.round(V_final, 10)
    W_final = np.round(W_final, 10)
    
    # Convert to Python lists
    U_out = U_final.tolist()
    V_out = V_final.tolist()
    W_out = W_final.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
