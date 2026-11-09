<<<<<<< SEARCH
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    start_time = time.time()
    deadline = start_time + 140

    # Search for lower rank
    # We try ranks from 32 to 55
    for R in range(32, rank_sb):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for trial in range(10):
            if time.time() > deadline:
                break
            
            U = np.random.randint(-1, 2, (n * m, R))
            V = np.random.randint(-1, 2, (m * p, R))
            W = np.random.randint(-1, 2, (p * n, R))
            
            # Local search
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Check if exact
                # Compute the tensor product sum
                # T_ij,k = sum_c U_ic V_ck W_ki
                # We need T_ij,k = delta_jk (for the inner product)
                # Actually, the condition is:
                # sum_c W_{k,i,c} U_{i,j,c} V_{j,k,c} = delta_{j,j'}
                
                # Let's compute the error
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for j2 in range(m):
                                val = 0
                                for c in range(R):
                                    val += W[k*n + i, c] * U[i*m + j, c] * V[j2*p + k, c]
                                target = 1 if j == j2 else 0
                                error += (val - target) ** 2
                
                if error == 0:
                    best_rank = R
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    return {"U": best_U, "V": best_V, "W": best_W}
                
                # Perturb
                # Pick a random element and flip it
                idx = np.random.randint(0, R)
                if np.random.rand() < 0.33:
                    r = np.random.randint(0, n*m)
                    U[r, idx] *= -1
                elif np.random.rand() < 0.66:
                    r = np.random.randint(0, m*p)
                    V[r, idx] *= -1
                else:
                    r = np.random.randint(0, p*n)
                    W[r, idx] *= -1

    return {"U": best_U, "V": best_V, "W": best_W}
=======
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    
    # Build standard basis solution
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Precompute indices for faster error computation
    # We need to check: sum_c U[i*m+j, c] * V[j2*p+k, c] * W[k*n+i, c] == 1 if j==j2 else 0
    # for all i in [0,n), j,j2 in [0,m), k in [0,p)
    
    # Vectorized error computation
    def compute_error(U, V, W, R):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # For each c, we have vectors U[:,c], V[:,c], W[:,c]
        # The contribution to the error from term c is:
        # For each (i, j, j2, k): (U[i*m+j,c]*V[j2*p+k,c]*W[k*n+i,c])
        # We need sum over c of this to be delta(j,j2)
        
        # Let's compute the full tensor T[i,j,j2,k] = sum_c U[i*m+j,c]*V[j2*p+k,c]*W[k*n+i,c]
        # and check if T[i,j,j2,k] == 1 if j==j2 else 0
        
        # Using numpy for speed
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)
        
        # T[i,j,j2,k] = sum_c U[i,j,c] * V[j2,k,c] * W[k,i,c]
        # This is a 4D tensor
        # We can compute this as: for each c, outer product of U[:,j,c], V[:,k,c], W[:,i,c]
        
        # More efficiently: 
        # T[i,j,j2,k] = sum_c U[i,j,c] * V[j2,k,c] * W[k,i,c]
        
        # Let's use einsum
        # U[i,j,c], V[j2,k,c], W[k,i,c] -> T[i,j,j2,k]
        # np.einsum('ijc,jkc,kic->ijjk', U_reshaped, V_reshaped, W_reshaped)
        
        T = np.einsum('ijc,jkc,kic->ijjk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        
        # Target: 1 if j==j2 else 0
        # Target tensor has shape (n, m, m, p)
        target = np.zeros((n, m, m, p))
        for j in range(m):
            target[:, j, j, :] = 1
        
        error = np.sum((T - target) ** 2)
        return error
    
    # Search for lower rank
    for R in range(32, rank_sb):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline:
                break
            
            # Initialize with random values in {-1, 0, 1}
            U = np.random.randint(-1, 2, (n * m, R))
            V = np.random.randint(-1, 2, (m * p, R))
            W = np.random.randint(-1, 2, (p * n, R))
            
            # Local search with simulated annealing-like approach
            best_local_error = compute_error(U, V, W, R)
            best_local_U = U.copy()
            best_local_V = V.copy()
            best_local_W = W.copy()
            
            temperature = 1.0
            cooling_rate = 0.95
            
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Generate neighbor by flipping a random entry
                U_new = U.copy()
                V_new = V.copy()
                W_new = W.copy()
                
                # Choose which matrix to modify
                mat_choice = np.random.randint(0, 3)
                if mat_choice == 0:
                    row = np.random.randint(0, n * m)
                    col = np.random.randint(0, R)
                    U_new[row, col] = -U_new[row, col]
                elif mat_choice == 1:
                    row = np.random.randint(0, m * p)
                    col = np.random.randint(0, R)
                    V_new[row, col] = -V_new[row, col]
                else:
                    row = np.random.randint(0, p * n)
                    col = np.random.randint(0, R)
                    W_new[row, col] = -W_new[row, col]
                
                error_new = compute_error(U_new, V_new, W_new, R)
                
                # Accept if better, or with probability based on temperature
                if error_new < best_local_error:
                    U, V, W = U_new, V_new, W_new
                    best_local_error = error_new
                    best_local_U = U.copy()
                    best_local_V = V.copy()
                    best_local_W = W.copy()
                    
                    if error_new == 0:
                        best_rank = R
                        best_U = U.tolist()
                        best_V = V.tolist()
                        best_W = W.tolist()
                        return {"U": best_U, "V": best_V, "W": best_W}
                else:
                    # Accept with probability exp(-(error_new - best_local_error) / temperature)
                    delta = error_new - best_local_error
                    if delta < 0:
                        U, V, W = U_new, V_new, W_new
                        best_local_error = error_new
                        best_local_U = U.copy()
                        best_local_V = V.copy()
                        best_local_W = W.copy()
                    elif np.random.rand() < np.exp(-delta / temperature):
                        U, V, W = U_new, V_new, W_new
                
                temperature *= cooling_rate
            
            # Keep the best found in this trial
            if best_local_error < compute_error(U, V, W, R):
                U, V, W = best_local_U, best_local_V, best_local_W

    return {"U": best_U, "V": best_V, "W": best_W}
>>>>>>> REPLACE
