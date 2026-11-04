import numpy as np
import time
import random

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # 1. Establish the baseline: Schoolbook Rank 27
    best_rank = 27
    best_U = [[0] * 27 for _ in range(n * m)]
    best_V = [[0] * 27 for _ in range(m * p)]
    best_W = [[0] * 27 for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1

    # 2. Define the target tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T_target = np.zeros((n * m, m * p, p * n))
    eq_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                eq_indices.append((a, b, c))
                
    num_eqs = len(eq_indices)
    
    # 3. Search for lower rank decompositions
    # We try ranks 23, 24, 25, 26
    # The linear system approach:
    # sum_r U[a, r] * V[b, r] * W[c, r] = 1
    # For fixed U, V, this is a linear system in W.
    # Variables: W[c, r] for c in 0..8, r in 0..rank-1.
    # Equations: For each (a, b, c) in eq_indices, sum_r U[a, r] V[b, r] W[c, r] = 1.
    
    # Note: The system is underdetermined if rank is low, but we want a specific sparse/structured solution.
    # Actually, for a fixed rank R, we have 9*R unknowns in W.
    # We have 27 equations.
    # If R >= 3, 9*R >= 27, so a solution likely exists.
    # The constraint is that U and V must be such that the resulting W has "nice" numbers (integers/half-integers).
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """
        U, V, W are numpy arrays of shape (9, R)
        """
        # Reconstruct T
        # T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(T_recon, T_target, atol=tol):
            return False
        return True

    def check_half_integer(mat, tol=1e-6):
        """Check if mat entries are close to half-integers (x.0 or x.5)"""
        doubled = mat * 2
        rounded = np.round(doubled)
        if np.max(np.abs(doubled - rounded)) < tol:
            return rounded / 2.0
        return None

    # Search parameters
    # We limit the search to avoid timeout.
    # Rank 23 is the target.
    
    for target_rank in [23, 24, 25, 26]:
        if time.time() > deadline - 10:
            break
            
        # Number of trials per rank
        # We can afford a few hundred trials of lstsq.
        max_trials = 200 if target_rank == 23 else 500
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
                
            # Random sparse initialization for U and V
            # Density should be moderate.
            # Entries in {-1, 0, 1}
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            
            # Strategy: Random sparsity
            for c in range(target_rank):
                # Ensure each column has some non-zeros
                # Randomly select indices
                for idx in range(n * m):
                    if random.random() < 0.4:
                        U[idx, c] = random.choice([-1, 1])
                for idx in range(m * p):
                    if random.random() < 0.4:
                        V[idx, c] = random.choice([-1, 1])
                        
            # Construct the linear system matrix M
            # Size: num_eqs (27) x (p*n * rank) (9 * rank)
            # Row e corresponds to (a, b, c)
            # Col (c_idx, r_idx) corresponds to W[c_idx, r_idx]
            # Coefficient for W[c_idx, r_idx] in equation (a, b, c) is:
            # U[a, r_idx] * V[b, r_idx] if c_idx == c, else 0.
            
            M = np.zeros((num_eqs, p * n * target_rank))
            
            for e_idx, (a, b, c) in enumerate(eq_indices):
                for r in range(target_rank):
                    coeff = U[a, r] * V[b, r]
                    if coeff != 0:
                        # Column index: c * target_rank + r
                        M[e_idx, c * target_rank + r] = coeff
                        
            # Solve M * W_vec = 1
            try:
                W_vec, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(num_eqs), rcond=None)
                
                # Check if solution is valid for the linear system
                # (lstsq minimizes error, check if error is small)
                if residuals and residuals[0] > 1e-8:
                    continue
                    
                W_sol = W_vec.reshape(target_rank, p * n)
                
                # Check for half-integer structure
                W_snapped = check_half_integer(W_sol)
                if W_snapped is None:
                    continue
                
                # Final verification with snapped values
                if verify_decomposition(U, V, W_snapped, tol=1e-9):
                    # Success!
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W_snapped
                    break # Found best for this rank, try lower? No, we are iterating 23->26.
                    # If we found 23, we can break out of the rank loop entirely.
            except Exception:
                continue
        
        if best_rank == target_rank:
            break # Found the target rank, no need to check higher (worse) ranks
            
    # Convert best result to lists
    def to_list(mat):
        if isinstance(mat, np.ndarray):
            return mat.tolist()
        return mat

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }

def construct_seed():
    n, m, p = 3, 3, 3
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
