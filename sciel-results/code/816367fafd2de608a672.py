"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0
    
    # Start with schoolbook
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to reduce rank by finding linear dependencies
    # Build the tensor T[i,j,k, c] where T[i,j,k,c] = U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We want to find a smaller set of columns that spans the same tensor
    
    # Use a greedy rank-reduction approach
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    # Try iterative removal: for each column, check if removing it still gives exact decomposition
    # This is expensive, so do it with a time limit
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank_school
    removed = [False] * rank_school
    
    for attempt in range(rank_school):
        if time.time() > deadline - 10:
            break
        if current_rank <= 20:  # Stop if we get very low
            break
            
        # Try to remove one column at a time
        found_removal = False
        for c in range(rank_school):
            if removed[c]:
                continue
            if time.time() > deadline - 10:
                break
            
            # Check if column c is redundant
            # Temporarily remove it
            removed[c] = True
            
            # Verify the decomposition is still exact
            # The tensor is T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
            # We need to check if this sum is unchanged
            # This is expensive, so use a random test
            np.random.seed(42)
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute with current decomposition (excluding removed columns)
            C_test = np.zeros((n, p))
            for c in range(rank_school):
                if not removed[c]:
                    uc = current_U[:, c]
                    vc = current_V[:, c]
                    wc = current_W[:, c]
                    # uc is (n*m,), reshape to (n,m)
                    U_c = uc.reshape(n, m)
                    V_c = vc.reshape(m, p)
                    W_c = wc.reshape(p, n)
                    # The contribution is: sum over i,j,k of A[i,j]*B[j,k]*U_c[i,j]*V_c[j,k]*W_c[k,i]
                    # = (A * U_c) @ (B * V_c) then multiplied by W_c in a specific way
                    # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
                    # With rank-1 term: C[i,k] += A[i,j]*U_c[i,j] * B[j,k]*V_c[j,k] * W_c[k,i]... no
                    # The tensor decomposition: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
                    # C[i,k] = sum_j A[i,j]*B[j,k]*T[i,j,k] = sum_j sum_c A[i,j]*B[j,k]*U[i,j,c]*V[j,k,c]*W[k,i,c]
                    # = sum_c (sum_i A[i,j]*U[i,j,c] for j) ... this is not a simple matrix product
                    
                    # Let me compute directly:
                    # C_contrib[i,k] = sum_j A[i,j]*B[j,k]*U_c[i,j]*V_c[j,k]*W_c[k,i]
                    # = W_c[k,i] * sum_j (A[i,j]*U_c[i,j]) * (B[j,k]*V_c[j,k])
                    # This is not separable in i,k simply.
                    
                    # Direct computation:
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                C_test[i, k] += A[i, j] * B[j, k] * U_c[i, j] * V_c[j, k] * W_c[k, i]
            
            if np.allclose(C_test, C_expected, atol=1e-9):
                found_removal = True
                current_rank -= 1
                break
            else:
                removed[c] = False
        
        if not found_removal:
            break
    
    # Extract non-removed columns
    keep_cols = [c for c in range(rank_school) if not removed[c]]
    final_U = current_U[:, keep_cols]
    final_V = current_V[:, keep_cols]
    final_W = current_W[:, keep_cols]
    
    # Convert to lists
    U_list = final_U.tolist()
    V_list = final_V.tolist()
    W_list = final_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

