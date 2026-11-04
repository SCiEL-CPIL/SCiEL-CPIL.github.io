# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Start with schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # For matrix multiplication C[i,k] = sum_j A[i,j] B[j,k],
    # a rank-r decomposition means:
    # C[i,k] = sum_c (sum_{j'} U[i*m+j', c] A[i,j']) * (sum_{k'} V[j*p+k', c] B[j,k']) * W[k*n+i, c]
    # 
    # Wait, I think I'm overcomplicating this. Let me use the standard tensor form.
    # The tensor T[i,j,k] for matrix multiplication satisfies:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # A rank-r decomposition is:
    # C[i,k] = sum_c M_c * W_{i,k}^{(c)}
    # where M_c = L_c(A) * R_c(B)
    # L_c(A) = sum_{i',j'} U_{i',j'}^{(c)} A[i',j']
    # R_c(B) = sum_{j',k'} V_{j',k'}^{(c)} B[j',k']
    #
    # In vectorized form:
    # U[i*m + j, c] = U_{i,j}^{(c)}
    # V[j*p + k, c] = V_{j,k}^{(c)}
    # W[k*n + i, c] = W_{i,k}^{(c)}
    #
    # The validity condition is that for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # (This ensures that each term A[i,j]*B[j,k] contributes exactly 1 to C[i,k])
    
    # Let's verify this condition for the schoolbook decomposition
    # and then try to reduce rank.
    
    def is_valid(U_mat, V_mat, W_mat, n, m, p):
        """Check if the decomposition is valid."""
        # For each (i, j, k), check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U_mat[i*m + j, c] * V_mat[j*p + k, c] * W_mat[k*n + i, c]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    # Verify schoolbook
    assert is_valid(U, V, W, n, m, p)
    
    # Try to reduce rank by finding and removing linearly dependent columns
    # This is a heuristic - not guaranteed optimal
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Greedy column removal
    for _ in range(20):  # Try up to 20 removals
        if time.time() > deadline - 5:
            break
        
        removed_any = False
        for col_idx in range(current_rank):
            if time.time() > deadline - 5:
                break
            
            mask = [i for i in range(current_rank) if i != col_idx]
            new_U = current_U[:, mask]
            new_V = current_V[:, mask]
            new_W = current_W[:, mask]
            
            if is_valid(new_U, new_V, new_W, n, m, p):
                current_U, current_V, current_W = new_U, new_V, new_W
                current_rank -= 1
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U, best_V, best_W = new_U.copy(), new_V.copy(), new_W.copy()
                removed_any = True
                break
        
        if not removed_any:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
