import time
import numpy as np
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s limit to be safe within 150s
    
    # Schoolbook decomposition (Rank 105)
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
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Helper to verify a solution
    def verify(U, V, W, R):
        """Verify that U, V, W correctly decompose matrix multiplication"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(R):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True

    # Strategy 1: Fix U to identity-like structure, search for V and W
    # If U[i*m+j, r] = delta_{r, i*m+j} for r < n*m, then we need V and W
    # such that for each (i,j,k): V[j*p+k, i*m+j] * W[k*n+i, i*m+j] = 1
    
    # This means V[j*p+k, i*m+j] = 1 and W[k*n+i, i*m+j] = 1 for all valid (i,j,k)
    # But this only works if the mapping is consistent.
    
    # Better approach: Use structured U with block patterns
    
    # Try ranks from lower bound up
    lower_bound = max(n * m, m * p, p * n)  # 35
    
    # Pre-compute equation indices for faster access
    # Equation e corresponds to (i, j, k) where e = i * (m*p) + j*p + k
    
    for R in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Number of trials based on rank
        if R < 40:
            trials = 50
        elif R < 50:
            trials = 30
        else:
            trials = 15
            
        for t in range(trials):
            if time.time() > deadline - 5:
                    break
                    
            # Generate random U and V with entries in {-1, 0, 1}
            # Use different random seeds for diversity
            np.random.seed(t * 1000 + R)
            U_rand = np.random.randint(-1, 2, size=(n * m, R))
            V_rand = np.random.randint(-1, 2, size=(m * p, R))
            
            # Ensure each row of U and V has at least one non-zero
            for i in range(n * m):
                if np.all(U_rand[i] == 0):
                    U_rand[i, np.random.randint(R)] = 1
            for j in range(m * p):
                if np.all(V_rand[j] == 0):
                    V_rand[j, np.random.randint(R)] = 1
            
            # Construct linear system A w = b
            # A is (n*m*p) x (p*n*R)
            # b is (n*m*p)
            b = np.ones(n * m * p)
            A = np.zeros((n * m * p, p * n * R))
            
            # Fill A more efficiently
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * (m * p) + j * p + k
                        u_row = U_rand[i * m + j]
                        v_row = V_rand[j * p + k]
                        for r in range(R):
                            coeff = u_row[r] * v_row[r]
                            if coeff != 0:
                                unk_idx = (k * n + i) * R + r
                                A[eq_idx, unk_idx] = coeff
            
            # Solve A w = b
            try:
                w_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if solution is valid
                if len(residuals) > 0:
                    if residuals[0] > 1e-8:
                        continue
                else:
                    if not np.allclose(A @ w_sol, b, atol=1e-8):
                        continue

                # Check if w_sol entries are integers or half-integers
                w_rounded = np.round(w_sol * 2) / 2
                if not np.allclose(w_sol, w_rounded, atol=1e-8):
                    continue
                
                # Construct W matrix
                W_list = [[0] * R for _ in range(p * n)]
                for idx in range(len(w_sol)):
                    k_val = idx // (n * R)
                    rem = idx % (n * R)
                    i_val = rem // R
                    r_val = rem % R
                    W_list[k_val * n + i_val][r_val] = w_rounded[idx]
                
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                
                # Verify before returning
                if verify(U_list, V_list, W_list, R):
                    return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception:
                continue
                
    # Strategy 2: Try with U having more structure
    for R in range(lower_bound, min(best_rank, 80)):
        if time.time() > deadline - 5:
            break
        
        trials = 20
        for t in range(trials):
            if time.time() > deadline - 2:
                break
                
            # Use a structured U: first n*m columns are identity, rest random
            U_struct = np.zeros((n * m, R))
            for idx in range(min(n * m, R)):
                U_struct[idx, idx] = 1
            
            # Add random components to remaining columns
            if R > n * m:
                U_struct[:, n*m:] = np.random.randint(-1, 2, size=(n * m, R - n * m))
            
            V_rand = np.random.randint(-1, 2, size=(m * p, R))
            
            # Ensure rows are non-zero
            for j in range(m * p):
                if np.all(V_rand[j] == 0):
                    V_rand[j, np.random.randint(R)] = 1
            
            b = np.ones(n * m * p)
            A = np.zeros((n * m * p, p * n * R))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * (m * p) + j * p + k
                        u_row = U_struct[i * m + j]
                        v_row = V_rand[j * p + k]
                        for r in range(R):
                            coeff = u_row[r] * v_row[r]
                            if coeff != 0:
                                unk_idx = (k * n + i) * R + r
                                A[eq_idx, unk_idx] = coeff
            
            try:
                w_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                
                if len(residuals) > 0:
                    if residuals[0] > 1e-8:
                        continue
                else:
                    if not np.allclose(A @ w_sol, b, atol=1e-8):
                        continue

                w_rounded = np.round(w_sol * 2) / 2
                if not np.allclose(w_sol, w_rounded, atol=1e-8):
                    continue
                
                W_list = [[0] * R for _ in range(p * n)]
                for idx in range(len(w_sol)):
                    k_val = idx // (n * R)
                    rem = idx % (n * R)
                    i_val = rem // R
                    r_val = rem % R
                    W_list[k_val * n + i_val][r_val] = w_rounded[idx]
                
                U_list = U_struct.tolist()
                V_list = V_rand.tolist()
                
                if verify(U_list, V_list, W_list, R):
                    return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception:
                continue
                
    return best_solution
# EVOLVE-BLOCK-END
