<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Initialize with schoolbook
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
    
    # Verify schoolbook
    def verify(U_mat, V_mat, W_mat, n, m, p):
        # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == delta_{i,k} for all i,j,k
        # Actually, the tensor T[i,j,k] should equal delta_{i,k} * 1 for the (i,j) element
        # The multiplication tensor has T_{i,j,k} = 1 if the (i,j) entry of A and (j,k) entry of B contribute
        # For matrix multiplication C = A*B, C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i} (trace convention)
        # So T[i,j,k] = 1 for all i,j,k where the indices match the multiplication
        # Actually T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # The tensor decomposition: sum_c U[i,j,c] V[j,k,c] W[k,i,c] should give 1 for valid (i,j,k)
        # In matrix multiplication, every (i,j,k) triple contributes to some C[i,k], so T[i,j,k]=1 for all
        # Wait, no. The tensor for matrix multiplication is T[i,j,k] = 1 for all i,j,k in valid ranges
        # Let me think again: C_{i,k} = sum_j A_{i,j} B_{j,k}
        # As a trilinear form: sum_{i,j,k} C_{k,i} A_{i,j} B_{j,k} = sum_{i,k} C_{k,i} (A B)_{i,k}
        # This equals the trace of C^T A B, which is a scalar
        # The tensor T has T[i,j,k] = 1 for all i,j,k because each term A_{i,j} B_{j,k} C_{k,i} appears
        # So we need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k
        
        # Build the tensor
        T = np.ones((n, m, p))
        # Compute from decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a lower rank decomposition using random search with refinement
    # For 4x4x7, known lower bounds and upper bounds
    # Schoolbook is 112. Let's try to find something better.
    
    best_rank = rank_schoolbook
    best_decomp = (U.copy(), V.copy(), W.copy())
    
    # Try ranks from lower to higher
    # Lower bound for 4x4x7: max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # But actual lower bound is higher. Let's try a few ranks.
    
    for target_rank in range(28, rank_schoolbook):
        if time.time() > deadline:
            break
        
        # Random initialization
        np.random.seed(42 + target_rank)
        U_r = np.random.randn(n * m, target_rank) * 0.5
        V_r = np.random.randn(m * p, target_rank) * 0.5
        W_r = np.random.randn(p * n, target_rank) * 0.5
        
        # Gradient-free optimization: try to minimize error
        # Error = sum_{i,j,k} (sum_c U[i,j,c]V[j,k,c]W[k,i,c] - 1)^2
        
        def compute_error(U_mat, V_mat, W_mat):
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                        err += (val - 1.0) ** 2
            return err
        
        # Simple local search: perturb one matrix at a time
        current_error = compute_error(U_r, V_r, W_r)
        
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            # Try random perturbations
            improved = False
            for _ in range(10):
                if time.time() > deadline:
                    break
                
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    row = np.random.randint(n * m)
                    col = np.random.randint(target_rank)
                    old_val = U_r[row, col]
                    U_r[row, col] += np.random.randn() * 0.1
                    new_error = compute_error(U_r, V_r, W_r)
                    if new_error < current_error:
                        current_error = new_error
                        improved = True
                    else:
                        U_r[row, col] = old_val
                elif mat_idx == 1:
                    row = np.random.randint(m * p)
                    col = np.random.randint(target_rank)
                    old_val = V_r[row, col]
                    V_r[row, col] += np.random.randn() * 0.1
                    new_error = compute_error(U_r, V_r, W_r)
                    if new_error < current_error:
                        current_error = new_error
                        improved = True
                    else:
                        V_r[row, col] = old_val
                else:
                    row = np.random.randint(p * n)
                    col = np.random.randint(target_rank)
                    old_val = W_r[row, col]
                    W_r[row, col] += np.random.randn() * 0.1
                    new_error = compute_error(U_r, V_r, W_r)
                    if new_error < current_error:
                        current_error = new_error
                        improved = True
                    else:
                        W_r[row, col] = old_val
            
            if not improved:
                break
        
        # Check if we found an exact solution
        if current_error < 1e-8:
            if verify(U_r, V_r, W_r, n, m, p):
                best_rank = target_rank
                best_decomp = (U_r.copy(), V_r.copy(), W_r.copy())
                break
    
    U_final, V_final, W_final = best_decomp
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
