import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s limit
    
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
    
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks just below schoolbook first (most likely to succeed if at all)
    # Focus on rank 104, 103, 102, etc.
    ranks_to_try = list(range(best_rank - 1, max(lower_bound, best_rank - 5) - 1, -1))
    
    for R in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        found = False
        num_attempts = 3  # Limit attempts per rank to stay within time
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            # Strategy: Start from schoolbook and try to merge two terms
            # This is a local search approach: take two columns of the schoolbook
            # decomposition and see if they can be replaced by a single column
            # with rational/half-integer coefficients.
            
            # For rank R = 104, we need to merge exactly 2 of the 105 schoolbook terms.
            # Let's try all pairs of terms and see if we can merge them.
            
            if R == rank_sb - 1:
                # Try to merge pairs of schoolbook terms
                # Term (i1,j1,k1) and (i2,j2,k2) can be merged if there exists
                # a single term with coefficients a,b,c such that:
                # a * b * c = 1 for the (i1,j1,k1) component
                # a * b * c = 1 for the (i2,j2,k2) component
                # and 0 for all other components
                
                # Actually, this is more complex. Let's use a different approach:
                # Randomized search with structured initialization.
                
                # Initialize with schoolbook-like structure but with some randomness
                np.random.seed(42 + R + attempt * 1000)
                
                # Create U, V, W with rank R
                # Start with first R-1 columns as schoolbook, and the last column
                # as a random combination
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                
                # Fill first R-1 columns with schoolbook terms (skip last term)
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < R - 1:
                                U[i * m + j][col] = 1
                                V[j * p + k][col] = 1
                                W[k * n + i][col] = 1
                                col += 1
                
                # The last column needs to account for the missing term
                # Let's try random half-integer values for the last column
                # and see if we can satisfy the constraints
                
                # This approach is unlikely to work directly. Let's try a
                # gradient-based optimization instead.
                
                # Alternative: Use the fact that for small reductions, we can
                # try to find a low-rank approximation that is exact.
                
                # Let's try a different strategy: use the structure of the
                # tensor to guide a local search.
                
                # For now, this is a placeholder for a more sophisticated
                # search. The schoolbook solution is guaranteed to work.
                continue
            
            # For other ranks, try random initialization with iterative refinement
            np.random.seed(42 + R + attempt * 1000)
            
            U = np.random.uniform(-0.5, 0.5, (n * m, R))
            V = np.random.uniform(-0.5, 0.5, (m * p, R))
            W = np.random.uniform(-0.5, 0.5, (p * n, R))
            
            # Iterative refinement: alternate between updating U, V, W
            # For each (i,j,k), we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            
            # This is a non-convex problem. Let's try a simple coordinate descent.
            # Fix V and W, solve for U. Fix U and W, solve for V. Fix U and V, solve for W.
            
            # For a fixed V and W, the constraint for U[i*m+j] is:
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is a linear system in U[i*m+j, :].
            
            # Let's implement this for a few iterations.
            
            converged = False
            max_iters = 50
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # For each k, we have:
                        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                        # Let A_k[r] = V[j*p+k,r] * W[k*n+i,r]
                        # Then sum_r U[i*m+j,r] * A_k[r] = 1 for all k
                        # This is: A @ U[i*m+j] = 1 (vector of ones of length p)
                        
                        A = np.zeros((p, R))
                        for k in range(p):
                            A[k] = V[j * p + k] * W[k * n + i]
                        
                        # Solve A @ x = 1
                        try:
                            if np.linalg.matrix_rank(A) >= min(p, R):
                                x, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(p), rcond=None)
                                U[i * m + j] = x
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # For each i, we have:
                        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                        # Let B_i[r] = U[i*m+j,r] * W[k*n+i,r]
                        # Then sum_r V[j*p+k,r] * B_i[r] = 1 for all i
                        # This is: B @ V[j*p+k] = 1 (vector of ones of length n)
                        
                        B = np.zeros((n, R))
                        for i in range(n):
                            B[i] = U[i * m + j] * W[k * n + i]
                        
                        try:
                            if np.linalg.matrix_rank(B) >= min(n, R):
                                x, residuals, rank_B, s = np.linalg.lstsq(B, np.ones(n), rcond=None)
                                V[j * p + k] = x
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # For each j, we have:
                        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                        # Let C_j[r] = U[i*m+j,r] * V[j*p+k,r]
                        # Then sum_r W[k*n+i,r] * C_j[r] = 1 for all j
                        # This is: C @ W[k*n+i] = 1 (vector of ones of length m)
                        
                        C = np.zeros((m, R))
                        for j in range(m):
                            C[j] = U[i * m + j] * V[j * p + k]
                        
                        try:
                            if np.linalg.matrix_rank(C) >= min(m, R):
                                x, residuals, rank_C, s = np.linalg.lstsq(C, np.ones(m), rcond=None)
                                W[k * n + i] = x
                        except:
                            pass
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-9:
                    converged = True
                    break
            
            if converged:
                # Convert to half-integer representation if possible
                # For now, just check if we can round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify the rounded solution
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_rounded[i * m + j] * V_rounded[j * p + k] * W_rounded[k * n + i])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to list of lists
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    
                    best_solution = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = R
                    found = True
                    break
        
        if found:
            break  # Found a better solution, stop searching
    
    return best_solution
# EVOLVE-BLOCK-END
