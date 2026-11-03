import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for 120s limit (or 150s if that's the hard limit, but 120s is stated in prompt as "finish within")
    
    # Target tensor T of shape (n, m, p)
    # Identity: sum_{r=1}^R U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    
    # Schoolbook decomposition (Rank 36)
    def get_schoolbook():
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

    # Verification function using exact integer arithmetic
    def verify(U, V, W):
        # U, V, W are lists of lists of ints or floats (we assume ints if from search)
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(R):
                        total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if total != 1:
                        return False
        return True

    # Search for lower rank
    # We try ranks from 20 up to 35
    # For each rank, we try random integer U, V in {-1, 0, 1} and solve for W
    
    # Precompute indices for speed
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
                
    num_eqs = len(indices) # 36

    # Try a few known small ranks first? 
    # It is unknown if 3x3x4 has rank < 36. 
    # Lower bound is max(n*m, m*p, p*n) = max(9, 12, 12) = 12.
    # Also lower bound from Strassen-like bounds.
    # Let's search from 20 to 35.
    
    best_result = None
    
    for R in range(20, 36):
        if time.time() > deadline:
            break
            
        # Number of attempts per rank
        # We have ~115 seconds total. 
        # Let's allocate time proportionally or just run until deadline.
        # Random search is stochastic.
        
        attempts = 0
        max_attempts = 1000 # Cap attempts per rank to avoid stalling on one rank if it's hopeless, but we check time anyway
        
        while attempts < max_attempts and (time.time() < deadline):
            attempts += 1
            
            # Generate random U, V with entries in {-1, 0, 1}
            # Using numpy for speed
            U_np = np.random.randint(-1, 2, size=(n * m, R))
            V_np = np.random.randint(-1, 2, size=(m * p, R))
            
            # Construct linear system for W
            # Variables: W[k*n+i, r] for k in 0..p-1, i in 0..n-1, r in 0..R-1
            # Total variables: p * n * R
            # Equations: n * m * p = 36
            # Eq (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            num_vars = p * n * R
            A_mat = np.zeros((num_eqs, num_vars))
            b_vec = np.ones(num_eqs)
            
            eq_idx = 0
            for i, j, k in indices:
                u_idx = i
                v_idx = j
                w_base = k
                for r in range(R):
                    var_idx = w_base * R + r
                    A_mat[eq_idx, var_idx] = U_np[u_idx, r] * V_np[v_idx, r]
                eq_idx += 1
            
            # Solve A_mat @ W_flat = b_vec
            # We want an integer solution.
            # Since the system is underdetermined (usually), we look for a solution in the null space.
            # Simple approach: Find *any* real solution, round to integers, and verify.
            # If R is small relative to num_vars, there's a high chance a small integer solution exists if the coefficients are small.
            
            try:
                # Use least squares to find a minimum norm solution
                W_flat, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Check if the solution is close to integers
                W_int = np.round(W_flat).astype(int)
                
                # Verify if W_int is an exact solution
                # A_mat @ W_int should equal b_vec
                if np.array_equal(A_mat @ W_int, b_vec):
                    # Reshape W
                    W_sol = W_int.reshape((p * n, R))
                    
                    # Convert to lists of ints for the return interface
                    U_list = U_np.tolist()
                    V_list = V_np.tolist()
                    W_list = W_sol.tolist()
                    
                    # Final verification with exact integer arithmetic to be 100% sure
                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception:
                pass

    # Fallback to schoolbook
    return get_schoolbook()
