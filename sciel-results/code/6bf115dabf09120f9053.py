import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    max_time = 140.0  # Leave 10s margin
    
    # Target Tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if it's a valid multiplication term?
    # The tensor for matrix multiplication M_{n,m,p} is defined as:
    # T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
    # Actually, the tensor T such that C = A x B is T[i, j, k] = delta_{j, j} ?
    # No. The bilinear map is (A, B) -> C.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # So the tensor T has indices i, j, k.
    # T[i, j, k] = 1 if the index j in A and j in B match.
    # In the standard basis, the tensor is the identity in the middle dimension?
    # T[i, j, k] = 1 if j is the same for A and B?
    # Yes. T[i, j, k] = 1 for all i, k, and j=1..m.
    # Wait, for a fixed j, A[i,j] and B[j,k] multiply.
    # So T[i, j, k] = 1 for all i in 0..n-1, k in 0..p-1, and for each j in 0..m-1.
    # So T is a tensor where T[i, j, k] = 1.
    # Let's verify.
    # Sum_c U[i,c] V[j,c] W[k,c] = C[i,k].
    # C[i,k] = sum_j A[i,j] B[j,k].
    # So we need sum_c U[i,c] V[j,c] W[k,c] to be 1 if we set A[i,j]=1, B[j,k]=1?
    # No, the decomposition is for the *operation*.
    # The tensor T is defined such that the operation is contraction with T.
    # T[i, j, k] = 1 if the indices match.
    # Specifically, T[i, j, k] = 1 for all i, k, and j.
    # So T is an all-ones tensor?
    # Let's check the seed code.
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # This corresponds to a specific mapping.
    # The tensor T in the standard basis (i, j, k) where i is row of A, j is col of A / row of B, k is col of B.
    # The term is A[i,j] B[j,k].
    # So T[i, j, k] = 1.
    # Yes, T is the all-ones tensor of shape (n, m, p).
    
    T = np.ones((n, m, p))
    
    # Schoolbook Rank 42
    rank_seed = n * m * p
    U_seed = np.zeros((n*m, rank_seed))
    V_seed = np.zeros((m*p, rank_seed))
    W_seed = np.zeros((p*n, rank_seed))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i*m + j, col] = 1
                V_seed[j*p + k, col] = 1
                W_seed[k*n + i, col] = 1
                col += 1
                
    def to_dict(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def is_valid(U, V, W, T):
        # Check if sum_c U[:,c] V[:,c] W[:,c] == T
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        n, m, p = T.shape
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        # Compute T_reconstructed
        # T_rec[i, j, k] = sum_c U_r[i, j, c] * V_r[j, k, c] * W_r[k, i, c]
        # Note: W index is k*n+i. So W_r[k, i, c] is correct.
        
        # Efficient computation:
        # For each c, add outer product of U_r[:,:,c], V_r[:,:,c], W_r[:,:,c]
        # This is slow in Python loops.
        # Use einsum?
        # T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec, T, atol=1e-9)

    # Check seed
    if is_valid(U_seed, V_seed, W_seed, T):
        best_result = to_dict(U_seed, V_seed, W_seed)
        best_rank = rank_seed
    else:
        # This should not happen
        best_result = to_dict(U_seed, V_seed, W_seed)
        best_rank = rank_seed

    # Try to find lower rank
    # We will try ranks from best_rank - 1 down to 1
    # But given the difficulty, we will only try best_rank - 1 and maybe best_rank - 2
    # with a time limit.
    
    for target_rank in [best_rank - 1, best_rank - 2]:
        if time.time() - start_time > max_time - 10:
            break
            
        # Try a few random restarts
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() - start_time > max_time - 5:
                break
                
            # Initialize randomly
            # Scale to keep values reasonable
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Objective function
            def objective(params):
                # params is flat array of size (n*m + m*p + p*n) * target_rank
                size_U = n*m * target_rank
                size_V = m*p * target_rank
                size_W = p*n * target_rank
                
                U_flat = params[:size_U]
                V_flat = params[size_U:size_U+size_V]
                W_flat = params[size_U+size_V:]
                
                U_curr = U_flat.reshape(n*m, target_rank)
                V_curr = V_flat.reshape(m*p, target_rank)
                W_curr = W_flat.reshape(p*n, target_rank)
                
                U_r = U_curr.reshape(n, m, target_rank)
                V_r = V_curr.reshape(m, p, target_rank)
                W_r = W_curr.reshape(p, n, target_rank)
                
                T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                error = np.sum((T_rec - T)**2)
                return error
            
            # Use least_squares
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                # Method 'trf' is robust
                result = opt.minimize(objective, x0, method='L-BFGS-B', 
                                      options={'maxiter': 1000, 'ftol': 1e-10})
                
                if result.success or result.fun < 1e-8:
                    # Extract solution
                    size_U = n*m * target_rank
                    size_V = m*p * target_rank
                    size_W = p*n * target_rank
                    
                    U_sol = result.x[:size_U].reshape(n*m, target_rank)
                    V_sol = result.x[size_U:size_U+size_V].reshape(m*p, target_rank)
                    W_sol = result.x[size_U+size_V:].reshape(p*n, target_rank)
                    
                    # Try to round to half-integers
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    if is_valid(U_rounded, V_rounded, W_rounded, T):
                        best_result = to_dict(U_rounded, V_rounded, W_rounded)
                        best_rank = target_rank
                        break # Found a better rank, stop searching for this rank
            except Exception:
                pass

    return best_result
