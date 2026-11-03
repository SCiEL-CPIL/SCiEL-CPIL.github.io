"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import scipy
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import solve

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    n, m, p = DIMENSIONS
    num_terms = n * m * p
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # 1. Initialize with schoolbook decomposition
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    U = np.zeros((num_u, num_terms), dtype=float)
    V = np.zeros((num_v, num_terms), dtype=float)
    W = np.zeros((num_w, num_terms), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    current_rank = num_terms
    
    # Helper to check if current decomposition is valid
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want T to be 1 if i,j,k are "connected" in the multiplication sense?
    # Wait, the tensor for matrix multiplication C = A * B is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is defined such that T[i,j,k] = 1 if we are computing C[i,k] from A[i,j] and B[j,k].
    # Specifically, the tensor M_{ijk} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # In standard indexing: A is n x m, B is m x p, C is n x p.
    # C_{ik} = sum_{j=0}^{m-1} A_{ij} B_{jk}.
    # So the tensor entry T[i, j, k] is 1 for all valid i, j, k.
    # Wait, the task says "exactly decompose the matrix multiplication tensor".
    # The tensor is T where T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p).
    # Actually, usually the tensor is defined as T_{ijk} = delta_{ik}? No.
    # Let's look at the schoolbook code:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # Product sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # For a specific (i,j,k), there is exactly one c where U[i*m+j,c]=1 and V[j*p+k,c]=1.
    # For that c, W[k*n+i,c] is 1.
    # So the reconstructed tensor T'[i,j,k] = 1 for all i,j,k.
    # This matches the standard matrix multiplication tensor where every input pair (A_ij, B_jk) contributes to the output C_ik.
    # So the target tensor T is all ones of shape (n*m, m*p, p*n) flattened appropriately?
    # No, the tensor is T_{ijk} = 1 for all i,j,k.
    
    def is_valid(U, V, W, rank):
        # Reconstruct tensor and check if it's all 1s
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can compute this efficiently by iterating over c
        T = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            u_col = U[:, c]
            v_col = V[:, c]
            w_col = W[:, c]
            # u_col has non-zeros at indices i*m+j
            # v_col has non-zeros at indices j*p+k
            # w_col has non-zeros at indices k*n+i
            
            # Extract non-zero indices
            u_idx = np.nonzero(u_col)[0]
            v_idx = np.nonzero(v_col)[0]
            w_idx = np.nonzero(w_col)[0]
            
            for ui in u_idx:
                i = ui // m
                j = ui % m
                u_val = u_col[ui]
                
                for vi in v_idx:
                    # Check if j matches
                    v_j = vi // p
                    k = vi % p
                    if v_j != j:
                        continue
                    v_val = v_col[vi]
                    
                    for wi in w_idx:
                        w_k = wi // n
                        w_i = wi % n
                        if w_k != k or w_i != i:
                            continue
                        w_val = w_col[wi]
                        
                        T[i, j, k] += u_val * v_val * w_val
                        
        # Check if T is all ones
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    # Local Search Strategy
    # Try to remove rank term c by solving for updates to other terms.
    # This is a non-convex problem, but we can try a greedy approach:
    # Assume we remove term r. We have a deficit D = - (U[:,r] \otimes V[:,r] \otimes W[:,r]).
    # We need to find small changes deltaU, deltaV, deltaW to the remaining terms such that
    # sum_{c != r} delta(T_c) = D.
    # This is linear if we linearize around the current point? No, it's cubic.
    # However, for small perturbations, we can use a Newton-like step or simply solve a linear system
    # if we fix two of the three matrices and solve for the third?
    # Let's try a simpler heuristic: 
    # If we remove a term, can we adjust the values of the *other* terms to compensate?
    # This is generally hard.
    
    # Alternative Strategy: 
    # Use the known Strassen decomposition for 2x2x2 blocks.
    # The dimensions are 2x4x7.
    # We can decompose A (2x4) into two 2x2 blocks: A_L, A_R.
    # B (4x7) into two 2x7 blocks: B_L, B_R.
    # C = A_L B_L + A_R B_R.
    # This doesn't directly help with rank reduction unless we mix them.
    
    # Let's try a different approach: 
    # The rank of matrix multiplication is at least max(n*m, m*p, p*n) = max(8, 28, 14) = 28.
    # Schoolbook is 56.
    # Strassen for 2x2x2 has rank 7.
    # If we could map the 2x4x7 problem to multiple 2x2x2 problems...
    # A is 2x4. B is 4x7.
    # Let's split the middle dimension m=4 into two parts of size 2.
    # A = [A1 A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x7.
    # C = A1 B1 + A2 B2.
    # This is a sum of two matrix multiplications.
    # Rank of sum of two tensors is at most sum of ranks.
    # Rank of 2x2x7 multiplication?
    # We can split the output dimension p=7? No, 7 is prime.
    # We can split the input dimension?
    
    # Let's stick to the local search but make it smarter.
    # Actually, there is a known result for small matrices.
    # For 2x4x7, is there a decomposition better than 56?
    # The lower bound is 28.
    # Strassen-like algorithms might get us to ~ 2 * Rank(2x2x7).
    # Rank(2x2x7) is at least max(4, 14, 14) = 14.
    # Schoolbook for 2x2x7 is 28.
    # If we can do 2x2x7 in rank 14 (which is the lower bound, so likely not possible, usually higher), 
    # then 2 * 14 = 28? No, that's just the lower bound.
    
    # Let's try to construct a specific decomposition using block structures.
    # We can use the fact that 4 = 2+2.
    # C = A1 B1 + A2 B2.
    # If we use Strassen for 2x2x2, we need to handle the 7 dimension.
    # This is getting complex.
    
    # Let's go back to the local search idea but with a simpler objective:
    # Just try to find *any* decomposition with rank < 56.
    # We can use a randomized local search:
    # 1. Start with schoolbook.
    # 2. Pick a random term to remove.
    # 3. Solve for the best update to the remaining terms to minimize the error.
    #    This is a non-linear least squares problem.
    #    We can use scipy.optimize.least_squares.
    
    def try_reduce_rank(U, V, W, rank, term_to_remove, time_limit):
        """
        Attempts to remove 'term_to_remove' by adjusting the other rank-1 terms.
        Returns (success, new_U, new_V, new_W, new_rank)
        """
        start_t = time.time()
        
        # Flatten U, V, W
        u_flat = U.flatten()
        v_flat = V.flatten()
        w_flat = W.flatten()
        
        # Indices of the term to remove
        u_idx = slice(term_to_remove * 1, term_to_remove * 1 + 1) # This is wrong, U is 2D
        # U shape: (n*m, rank). Term c is column c.
        # We keep all columns except c.
        cols_keep = [c for c in range(rank) if c != term_to_remove]
        
        U_keep = U[:, cols_keep]
        V_keep = V[:, cols_keep]
        W_keep = W[:, cols_keep]
        
        new_rank = rank - 1
        
        # Define the residual function
        # We want T_calc = T_target (all ones)
        # T_calc[i,j,k] = sum_{c in keep} U_keep[i*m+j, c] * V_keep[j*p+k, c] * W_keep[k*n+i, c]
        
        def residuals(params):
            if time.time() > time_limit:
                raise TimeoutError()
            
            # Unpack params
            # params contains new values for U_keep, V_keep, W_keep
            n_u = U_keep.size
            n_v = V_keep.size
            n_w = W_keep.size
            
            u_new = params[:n_u].reshape(U_keep.shape)
            v_new = params[n_u:n_u+n_v].reshape(V_keep.shape)
            w_new = params[n_u+n_v:].reshape(W_keep.shape)
            
            # Compute tensor
            T_calc = np.zeros((n, m, p), dtype=float)
            for c in range(new_rank):
                u_col = u_new[:, c]
                v_col = v_new[:, c]
                w_col = w_new[:, c]
                
                u_idx = np.nonzero(u_col)[0]
                v_idx = np.nonzero(v_col)[0]
                w_idx = np.nonzero(w_col)[0]
                
                for ui in u_idx:
                    i = ui // m
                    j = ui % m
                    u_val = u_col[ui]
                    
                    for vi in v_idx:
                        v_j = vi // p
                        k = vi % p
                        if v_j != j:
                            continue
                        v_val = v_col[vi]
                        
                        for wi in w_idx:
                            w_k = wi // n
                            w_i = wi % n
                            if w_k != k or w_i != i:
                                continue
                            w_val = w_col[wi]
                            
                            T_calc[i, j, k] += u_val * v_val * w_val
                            
            # Residual is T_calc - 1
            return (T_calc - 1.0).flatten()
            
        # Initial guess: the current values
        x0 = np.concatenate([U_keep.flatten(), V_keep.flatten(), W_keep.flatten()])
        
        try:
            result = scipy.optimize.least_squares(residuals, x0, method='lm', xtol=1e-12, ftol=1e-12)
            if result.success and np.max(np.abs(result.fun)) < 1e-9:
                # Success
                u_new = result.x[:U_keep.size].reshape(U_keep.shape)
                v_new = result.x[U_keep.size:U_keep.size+V_keep.size].reshape(V_keep.shape)
                w_new = result.x[U_keep.size+V_keep.size:].reshape(W_keep.shape)
                
                # Reconstruct full matrices with zero column for removed term
                # Actually, we just return the reduced matrices
                return True, u_new, v_new, w_new, new_rank
        except Exception as e:
            pass
            
        return False, U, V, W, rank

    # Main Loop
    # We will try to remove terms one by one.
    # To speed up, we can try removing terms in a specific order or randomly.
    # Let's try removing terms from the end.
    
    # Note: scipy.optimize.least_squares might be slow. 
    # Given the 120s limit, we can only try a few reductions.
    
    # Let's try a few random reductions
    attempts = 0
    max_attempts = 50 # Limit attempts to avoid timeout
    
    while current_rank > 28 and attempts < max_attempts:
        if time.time() > deadline - 10:
            break
            
        # Pick a random term to remove
        term_to_remove = np.random.randint(0, current_rank)
        
        success, U_new, V_new, W_new, new_rank = try_reduce_rank(U, V, W, current_rank, term_to_remove, deadline - 15)
        
        if success:
            U, V, W = U_new, V_new, W_new
            current_rank = new_rank
            # Reset attempts counter? Or keep it? 
            # If we succeed, we might be able to remove more.
            attempts = 0
        else:
            attempts += 1
            
    # Final check
    if not is_valid(U, V, W, current_rank):
        # Fallback to schoolbook if something went wrong
        U = np.zeros((num_u, num_terms), dtype=float)
        V = np.zeros((num_v, num_terms), dtype=float)
        W = np.zeros((num_w, num_terms), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        current_rank = num_terms

    # Convert to lists of lists as required by interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
