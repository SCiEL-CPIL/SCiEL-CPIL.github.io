import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 5]

def check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Check if the tensor decomposition is valid.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Target: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == (1 if i==k else 0)
    """
    rank = U.shape[1]
    # Precompute W indexed by (k, i) for faster access in loops? 
    # W is (p*n, rank). Index is k*n + i.
    # U is (n*m, rank). Index is i*m + j.
    # V is (m*p, rank). Index is j*p + k.
    
    # For small dimensions, a direct loop is fine and avoids complex reshaping logic errors.
    # However, we can vectorize the inner product over r for each (i,j,k).
    
    # Let's try a vectorized approach for the check to be fast.
    # We want T[i,j,k] for all i,j,k.
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Reshape U to (n, m, rank)
    U_reshaped = U.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V_reshaped = V.reshape(m, p, rank)
    # Reshape W to (p, n, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # We need to sum over r: U[i,j,r] * V[j,k,r] * W[k,i,r]
    # This is a bit awkward with standard einsum because indices are mixed.
    # Let's stick to a loop over i,j,k but use dot products for r.
    # Or even better, since n,m,p are small (3,3,5), the total number of (i,j,k) is 45.
    # Rank is likely ~20-45. 45 * 45 = 2025 multiplications. Very fast.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Extract vectors for this (i,j,k)
                u_vec = U[i * m + j, :]
                v_vec = V[j * p + k, :]
                w_vec = W[k * n + i, :]
                
                # Compute sum of element-wise products
                val = np.dot(u_vec, v_vec * w_vec)
                
                target = 1.0 if i == k else 0.0
                if abs(val - target) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Initialize with Schoolbook Decomposition (Rank 45)
    rank_seed = n * m * p
    U = np.zeros((n * m, rank_seed))
    V = np.zeros((m * p, rank_seed))
    W = np.zeros((p * n, rank_seed))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    best_rank = rank_seed
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # 2. Greedy Removal Strategy
    # We iteratively try to remove one term (column) from the decomposition.
    # If the decomposition remains valid after removal, we keep it removed.
    # This is a form of greedy sparsification.
    
    current_rank = rank_seed
    
    while current_rank > 0 and time.time() < deadline:
        removed_any = False
        # Try removing each column r
        # We iterate from 0 to current_rank-1
        # If we remove column r, we shift columns r+1...end to the left.
        
        # To avoid modifying the array while iterating in a way that skips indices,
        # we can check all columns, find the first removable one, remove it, and restart the loop.
        # Or, we can iterate and mark for removal.
        
        # Efficient approach: Iterate r from 0 to current_rank-1.
        # If removable, delete it, decrement current_rank, and break to restart the outer loop?
        # No, if we remove one, the indices change. It's safer to find ONE removable column,
        # remove it, and then restart the search from the beginning (or continue, but restart is simpler for correctness).
        
        for r in range(current_rank):
            if time.time() > deadline:
                break
            
            # Temporarily remove column r
            # We can just check if the sum without this column matches the target.
            # Let S = sum_{r'} U[:,r'] V[:,r'] W[:,r']
            # We want S - U[:,r] V[:,r] W[:,r] to be 0 where target is 0, and 1 where target is 1.
            # Actually, it's easier to just construct the modified matrices and check.
            # But constructing new arrays is expensive.
            
            # Alternative: Compute the residual of the current decomposition.
            # Residual[i,j,k] = Target[i,j,k] - Sum_{r'} ...
            # If we remove term r, the new residual is Residual[i,j,k] + U[i,j,r] V[j,k,r] W[k,i,r].
            # We want the new residual to be 0.
            # So we need Residual[i,j,k] + U[i,j,r] V[j,k,r] W[k,i,r] == 0 for all i,j,k.
            
            # Let's compute the full residual once per iteration? No, that's O(Rank * N*M*P).
            # Let's just do the check on the fly.
            
            # To speed up the check, we can precompute the current tensor T_current.
            # But T_current changes only if we modify U,V,W.
            # Since we are just *testing* removal, we can compute the contribution of term r and see if it equals the current "error" (which is 0 if current is valid).
            # Wait, the current decomposition IS valid. So Sum_{all r'} = Target.
            # We want Sum_{r' != r} = Target.
            # This implies Sum_{all r'} - Term_r = Target.
            # Since Sum_{all r'} = Target, this implies Term_r = 0.
            
            # So, a term r can be removed IF AND ONLY IF its contribution is identically zero.
            # Contribution_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # We need this to be 0 for all i,j,k.
            
            # This is a much simpler check!
            # We just need to check if the tensor defined by the outer product of the r-th columns is zero.
            
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            
            # Check if u_col * v_col * w_col is zero everywhere.
            # Since we are dealing with floats, check if max abs value is close to 0.
            # Actually, if any of the vectors is zero, the product is zero.
            # But they might not be zero individually, but their product might be.
            # E.g., U[i] is non-zero only for i=0, V[j] non-zero only for j=1, W[k] non-zero only for k=2.
            # The product is non-zero only at (i=0, j=1, k=2).
            
            # So we must check the product.
            
            # Efficient check:
            # Iterate i, j, k?
            # Or, reshape and use einsum?
            # u_col is (n*m,). v_col is (m*p,). w_col is (p*n,).
            # We want to know if sum_{i,j,k} |u[i*m+j] v[j*p+k] w[k*n+i]| > 0.
            
            # Let's just loop. It's small.
            is_zero = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        prod = u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
                        if abs(prod) > 1e-12:
                            is_zero = False
                            break
                    if not is_zero:
                        break
                if not is_zero:
                    break
            
            if is_zero:
                # Remove column r
                U = np.delete(U, r, axis=1)
                V = np.delete(V, r, axis=1)
                W = np.delete(W, r, axis=1)
                current_rank -= 1
                removed_any = True
                # Update best
                best_rank = current_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                # Restart the loop to check the new configuration
                break
        
        if not removed_any:
            # No more removable terms
            break
            
    # 3. Additional Optimization: Randomized Search for Lower Rank
    # The greedy removal might get stuck.
    # We can try to randomly perturb the remaining terms and see if we can drop rank further?
    # Or, we can try to find a lower rank decomposition from scratch using random initialization and optimization?
    # Given the 120s limit and the complexity of tensor rank minimization,
    # and the fact that we already have a valid decomposition,
    # let's try a few random restarts for a slightly lower rank if we have time.
    
    # However, the greedy removal from schoolbook is often quite effective for these small cases.
    # Let's add a simple "shuffle and remove" heuristic.
    # Shuffle the columns, then apply greedy removal.
    
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Start from best known
            U_curr = best_U.copy()
            V_curr = best_V.copy()
            W_curr = best_W.copy()
            curr_rank = best_rank
            
            # Shuffle columns
            perm = np.random.permutation(curr_rank)
            U_curr = U_curr[:, perm]
            V_curr = V_curr[:, perm]
            W_curr = W_curr[:, perm]
            
            # Apply greedy removal
            while curr_rank > 0 and time.time() < deadline:
                removed_any = False
                for r in range(curr_rank):
                    if time.time() > deadline:
                        break
                    
                    u_col = U_curr[:, r]
                    v_col = V_curr[:, r]
                    w_col = W_curr[:, r]
                    
                    is_zero = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                prod = u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
                                if abs(prod) > 1e-12:
                                    is_zero = False
                                    break
                            if not is_zero:
                                break
                        if not is_zero:
                            break
                    
                    if is_zero:
                        U_curr = np.delete(U_curr, r, axis=1)
                        V_curr = np.delete(V_curr, r, axis=1)
                        W_curr = np.delete(W_curr, r, axis=1)
                        curr_rank -= 1
                        removed_any = True
                        break
                
                if not removed_any:
                    break
            
            if curr_rank < best_rank:
                best_rank = curr_rank
                best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                
            # Early exit if we hit a theoretical lower bound?
            # Lower bound is max(n*m, m*p, p*n) = 15.
            if best_rank <= 15:
                break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
