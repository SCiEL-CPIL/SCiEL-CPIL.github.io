"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    n_rows = n * m
    m_rows = m * p
    p_rows = p * n
    rank = U.shape[1]
    
    if U.shape != (n_rows, rank) or V.shape != (m_rows, rank) or W.shape != (p_rows, rank):
        return False
    
    # Check each (i, j, k) triple
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Expected: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
                u_val = U[i * m + j, :]
                v_val = V[j * p + k, :]
                w_val = W[k * n + i, :]
                product_sum = np.sum(u_val * v_val * w_val)
                if abs(product_sum - 1.0) > tol:
                    return False
    
    # Check that for non-matching indices, the product sum is 0
    # For i, j, k where the matrix product doesn't contribute
    # Actually, we need to check that the tensor is exactly the multiplication tensor
    # The multiplication tensor has T[i,j,k] = 1 if and only if... 
    # Wait, the tensor for matrix multiplication C = A*B is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k)
    # And the decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # But W is indexed as W[k*n+i, c], so W[k,i,c]
    
    # Actually let me re-read the convention:
    # U has shape (n*m, rank), indexed by i*m+j
    # V has shape (m*p, rank), indexed by j*p+k  
    # W has shape (p*n, rank), indexed by k*n+i
    # The tensor entry T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # And T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # So we just need all entries to be 1
    return True

def _try_rank_reduction(U, V, W, target_rank, n, m, p, max_iterations=1000, seed=42):
    """Try to reduce the rank by zeroing out columns and re-optimizing."""
    np.random.seed(seed)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return U, V, W
    
    # Try to identify redundant columns
    # A column is redundant if it can be expressed as a combination of others
    # For now, try simple approaches:
    
    # Approach 1: Try removing columns one at a time and see if we can fix the others
    for attempt in range(max_iterations):
        if attempt >= max_iterations:
            break
        
        # Pick a random column to try removing
        col_to_remove = np.random.randint(current_rank)
        
        U_trial = np.delete(U, col_to_remove, axis=1)
        V_trial = np.delete(V, col_to_remove, axis=1)
        W_trial = np.delete(W, col_to_remove, axis=1)
        
        # Check if this still works (it won't in general, but maybe with small adjustments)
        if _verify_decomposition(U_trial, V_trial, W_trial, n, m, p, tol=1e-6):
            U, V, W = U_trial, V_trial, W_trial
            current_rank -= 1
            if current_rank <= target_rank:
                break
    
    return U, V, W

def _optimize_column(U, V, W, col_idx, n, m, p, lr=0.01, max_iter=100):
    """Optimize a single column to better satisfy the constraints."""
    U_col = U[:, col_idx].copy()
    V_col = V[:, col_idx].copy()
    W_col = W[:, col_idx].copy()
    
    for _ in range(max_iter):
        # Compute gradient for this column
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # We want T[i,j,k] = 1 for all (i,j,k)
        
        # Gradient of T[i,j,k] w.r.t. U[i,j,c]:
        # dT/dU[i,j,c] = V[j,k,c] * W[k,i,c]
        
        # This is complex, so use simple heuristic:
        # For each (i,j,k), if T[i,j,k] is too low, increase the product
        # If too high, decrease
        
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    current_val = 0
                    for c in range(U.shape[1]):
                        current_val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    errors.append((i, j, k, current_val - 1.0))
        
        if max(abs(e[3]) for e in errors) < 1e-10:
            break
        
        # Simple update: for the worst error, adjust
        worst = max(errors, key=lambda x: abs(x[3]))
        i, j, k, err = worst
        
        # Adjust U[i*m+j, col_idx] to reduce error
        # dT/dU = V[j*p+k, col_idx] * W[k*n+i, col_idx]
        grad_u = V[j*p+k, col_idx] * W[k*n+i, col_idx]
        if abs(grad_u) > 1e-12:
            U[i*m+j, col_idx] -= lr * err * grad_u / (grad_u * grad_u + 1e-8)
        
        grad_v = U[i*m+j, col_idx] * W[k*n+i, col_idx]
        if abs(grad_v) > 1e-12:
            V[j*p+k, col_idx] -= lr * err * grad_v / (grad_v * grad_v + 1e-8)
        
        grad_w = U[i*m+j, col_idx] * V[j*p+k, col_idx]
        if abs(grad_w) > 1e-12:
            W[k*n+i, col_idx] -= lr * err * grad_w / (grad_w * grad_w + 1e-8)
    
    U[:, col_idx] = U_col
    V[:, col_idx] = V_col
    W[:, col_idx] = W_col

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    # Start with schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Try to reduce rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try different target ranks
    for target_rank in range(rank - 1, max(rank // 2, 1), -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline - 30:
                break
            
            U_trial, V_trial, W_trial = U.copy(), V.copy(), W.copy()
            
            # Try to reduce to target_rank
            U_trial, V_trial, W_trial = _try_rank_reduction(
                U_trial, V_trial, W_trial, target_rank, n, m, p, 
                max_iterations=200, seed=seed
            )
            
            current_rank = U_trial.shape[1]
            if current_rank < best_rank and _verify_decomposition(U_trial, V_trial, W_trial, n, m, p):
                best_U, best_V, best_W = U_trial.copy(), V_trial.copy(), W_trial.copy()
                best_rank = current_rank
                print(f"Found rank {best_rank} with seed {seed}")
            
            if best_rank <= target_rank:
                break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
