"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]
import numpy as np
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Schoolbook decomposition
    rank_school = n * m * p  # 45
    U_fb = np.zeros((n*m, rank_school), dtype=np.float64)
    V_fb = np.zeros((m*p, rank_school), dtype=np.float64)
    W_fb = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1
    
    def verify_fast(U, V, W):
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(result, np.ones((n,m,p)), atol=1e-8, rtol=1e-8)
    
    # Try structured search with small integer/half-integer values
    # Domain: -1, -0.5, 0, 0.5, 1, 1.5, 2
    domain = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Try to find rank-14 decomposition using backtracking
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    
    # For small ranks, we can try systematic search
    # But the search space is huge, so let's use a smarter approach
    
    # First, try a known structure: for 3x3x3, Strassen's algorithm gives rank 7
    # For 3x3x5, we can try to extend this
    
    # Let's try a different approach: use the fact that the tensor is all-ones
    # and try to find a low-rank decomposition
    
    # For rank r, we have r variables in each of U, V, W
    # The constraint is that the sum over c of U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    
    # Let's try a gradient-based approach with random restarts
    # But first, let's try a simple heuristic: 
    # If we set all U, V, W to have the same structure, we might find a solution
    
    # Actually, let's try a more systematic approach for small ranks
    # For rank 14, we can try to use the structure of the all-ones tensor
    
    # The all-ones tensor T[i,j,k] = 1 can be written as:
    # T[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k]
    # where u_c, v_c, w_c are vectors
    
    # For the matrix multiplication tensor, we have:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's try to find a solution by assuming a specific structure
    # For example, if we set U[i*m+j, c] = a_c[i] * b_c[j]
    # V[j*p+k, c] = b_c[j] * c_c[k]
    # W[k*n+i, c] = c_c[k] * a_c[i]
    # Then T[i,j,k] = sum_c a_c[i]^2 * b_c[j]^2 * c_c[k]^2
    # This is a sum of rank-1 tensors, but with a specific structure
    
    # Actually, this approach is too restrictive. Let's try a different approach.
    
    # Let's use a randomized local search approach
    # Start with a random initialization and try to reduce the error
    
    def local_search(U_init, V_init, W_init, max_iter=1000):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        r = U.shape[1]
        
        # Compute the error
        def compute_error(U, V, W):
            U_r = U.reshape(n, m, r)
            V_r = V.reshape(m, p, r)
            W_r = W.reshape(p, n, r)
            result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            return np.sum((result - 1.0)**2)
        
        current_error = compute_error(U, V, W)
        best_error = current_error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try to improve one element at a time
            # Randomly select which matrix and which element to change
            mat_choice = np.random.randint(3)
            if mat_choice == 0:
                row = np.random.randint(n*m)
                col = np.random.randint(r)
                old_val = U[row, col]
                # Try small perturbations
                for delta in [-0.5, -0.25, 0.25, 0.5, 1.0, -1.0]:
                    U[row, col] = old_val + delta
                    new_error = compute_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        if new_error < best_error:
                            best_error = new_error
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    else:
                        U[row, col] = old_val
            elif mat_choice == 1:
                row = np.random.randint(m*p)
                col = np.random.randint(r)
                old_val = V[row, col]
                for delta in [-0.5, -0.25, 0.25, 0.5, 1.0, -1.0]:
                    V[row, col] = old_val + delta
                    new_error = compute_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        if new_error < best_error:
                            best_error = new_error
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    else:
                        V[row, col] = old_val
            else:
                row = np.random.randint(p*n)
                col = np.random.randint(r)
                old_val = W[row, col]
                for delta in [-0.5, -0.25, 0.25, 0.5, 1.0, -1.0]:
                    W[row, col] = old_val + delta
                    new_error = compute_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        if new_error < best_error:
                            best_error = new_error
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    else:
                        W[row, col] = old_val
        
        return best_U, best_V, best_W, best_error
    
    # Try different ranks with local search
    best_rank = rank_school
    best_result = None
    
    # Try ranks from 44 down to 10
    for rank in range(44, 9, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Use small integers for initialization
            U_try = np.random.randint(-1, 2, size=(n*m, rank)).astype(np.float64)
            V_try = np.random.randint(-1, 2, size=(m*p, rank)).astype(np.float64)
            W_try = np.random.randint(-1, 2, size=(p*n, rank)).astype(np.float64)
            
            # Run local search
            U_best, V_best, W_best, error = local_search(U_try, V_try, W_try, max_iter=200)
            
            if error < 1e-6:
                # Verify the solution
                if verify_fast(U_best, V_best, W_best):
                    best_rank = rank
                    best_result = {
                        "U": U_best.tolist(),
                        "V": V_best.tolist(),
                        "W": W_best.tolist()
                    }
                    break
        
        if best_result is not None:
            continue
    
    if best_result is not None:
        return best_result
    
    # Return schoolbook
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }
# EVOLVE-BLOCK-END
