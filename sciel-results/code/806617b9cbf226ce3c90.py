import numpy as np
import time
import random

def construct():
    n, m, p = 3, 3, 3
    deadline = time.time() + 110  # Safety margin
    
    # Indices for the 27 non-zero entries of the tensor T
    # T[i, j, k] = 1 if it contributes to C[i, k]
    # Equation index e corresponds to (i, j, k)
    # Variable index for W is (c, k*n + i)
    # Coefficient is U[i*m+j, c] * V[j*p+k, c]
    
    eq_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_indices.append((i, j, k))
                
    num_eqs = len(eq_indices)
    
    best_rank = 27
    best_U, best_V, best_W = None, None, None
    
    # Seed with naive solution
    U_naive = np.zeros((n*m, 27), dtype=int)
    V_naive = np.zeros((m*p, 27), dtype=int)
    W_naive = np.zeros((p*n, 27), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i*m+j, col] = 1
                V_naive[j*p+k, col] = 1
                W_naive[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_naive, V_naive, W_naive
    best_rank = 27
    
    # Search for lower rank
    # Known minimum is 23. We try 23 first, then 24, 25, 26.
    # Rank 22 is theoretically impossible for 3x3 (best known is 23).
    
    # Define search parameters
    # We will iterate through ranks 23, 24, 25, 26
    # For each rank, we try multiple random configurations
    
    for rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
            
        # Number of trials: allocate more time to rank 23
        if rank == 23:
            num_trials = 20000
        else:
            num_trials = 5000
            
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            # Strategy: Vary sparsity and value range
            # Use {-1, 0, 1} with varying density
            density = random.uniform(0.1, 0.6)
            max_val = random.choice([1, 2, 3]) # Integers
            
            U = np.zeros((n*m, rank), dtype=int)
            V = np.zeros((m*p, rank), dtype=int)
            
            for c in range(rank):
                for idx in range(n*m):
                    if random.random() < density:
                        U[idx, c] = random.choice([-max_val, max_val])
                for idx in range(m*p):
                    if random.random() < density:
                        V[idx, c] = random.choice([-max_val, max_val])
                        
            # Construct the linear system matrix M
            # Size: num_eqs x (p*n * rank)
            M = np.zeros((num_eqs, p*n * rank))
            
            for e_idx, (i, j, k) in enumerate(eq_indices):
                w_idx_base = k * n + i
                for c in range(rank):
                    coeff = U[i*m+j, c] * V[j*p+k, c]
                    if coeff != 0:
                        M[e_idx, c * p*n + w_idx_base] = coeff
            
            # Solve M * W_vec = 1
            # We want W_vec to be half-integers or integers.
            # Let Y = 2 * W_vec. Then M * Y = 2 * ones.
            # Y should be integer.
            try:
                # Check if M has full row rank? 
                # If not, lstsq will find a solution, but it might not be unique.
                # We just need *a* solution that is integer.
                
                Y_vec, residuals, rank_m, s = np.linalg.lstsq(M, 2 * np.ones(num_eqs), rcond=None)
                
                # Check if Y_vec is close to integers
                Y_rounded = np.round(Y_vec)
                error = np.max(np.abs(Y_vec - Y_rounded))
                
                if error < 1e-9: # Tighter tolerance
                    # Valid solution found for Y
                    # W = Y / 2
                    W_solution = Y_rounded / 2.0
                    
                    # Verify the solution exactly
                    valid = True
                    for e_idx, (i, j, k) in enumerate(eq_indices):
                        w_idx = k * n + i
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W_solution[c, w_idx]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                            
                    if valid:
                        best_rank = rank
                        best_U = U
                        best_V = V
                        best_W = W_solution
                        break # Found best for this rank, break trial loop
            except Exception:
                continue
        
        if best_rank == rank:
            break # Found a solution for this rank, no need to check higher ranks
            
    # Convert to lists for return
    def to_list(mat):
        return mat.tolist()
        
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
