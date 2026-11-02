"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Leave margin for evaluator overhead
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    
    # Try to find lower rank solutions
    # The condition: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    def verify(U, V, W, tol=1e-6):
        """Check if decomposition is valid."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1) > tol:
                        return False
        return True
    
    # Try ranks from lower to higher
    min_possible = max(n*m, m*p, p*n)  # Lower bound
    max_rank = n * m * p
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = max_rank
    
    # Try to find a solution with rank < max_rank
    for rank in range(min_possible, max_rank):
        if time.time() > deadline - 30:
            break
        
        # Try random initializations with gradient-based approach
        found = False
        n_tries = 50
        
        for trial in range(n_tries):
            if time.time() > deadline - 30:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Simple iterative refinement
            for it in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Compute residuals
                # For each (i,j,k): residual = 1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                # Gradient w.r.t. U[i*m+j,c]: -residual * V[j*p+k,c] * W[k*n+i,c]
                # This is complex, so let's use a simpler approach:
                # Fix two matrices, solve for the third linearly
                
                # Fix U and V, solve for W
                # For each (i,k): sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * V[j*p+k,c]) = 1
                # Let M[i,k,c] = sum_j U[i*m+j,c] * V[j*p+k,c]
                # Then: sum_c W[k*n+i,c] * M[i,k,c] = 1 for all (i,k)
                
                M = np.zeros((n, p, rank))
                for i in range(n):
                    for k in range(p):
                        for c in range(rank):
                            M[i, k, c] = np.sum(U[i*m:(i+1)*m, c] * V[0:m*p: p * 1, c][:m] if False else 
                                                np.dot(U[i*m:(i+1)*m, c], V[0:m, k] if False else 
                                                np.sum([U[i*m+j, c] * V[j*p+k, c] for j in range(m)])))
                
                # M[i,k,c] = sum_j U[i*m+j,c] * V[j*p+k,c]
                # We need: for each (i,k): sum_c W[k*n+i, c] * M[i,k,c] = 1
                # This is a linear system for W
                # W has shape (p*n, rank), and we have n*p equations
                
                # Build the linear system: A * w = b where w is W flattened
                # For equation (i,k): sum_c W[k*n+i, c] * M[i,k,c] = 1
                # This is: dot(W[k*n+i, :], M[i,k, :]) = 1
                
                # We have n*p equations and p*n*rank unknowns (but each equation only involves rank unknowns)
                # Actually, each equation (i,k) involves only the row W[k*n+i, :] of W
                # So we can solve each row independently if the system is consistent
                # But we have n*p equations and each involves one row of W, so we need:
                # For each (i,k): W[k*n+i, :] @ M[i,k, :] = 1
                # This is underdetermined if rank > 1 (which it is)
                # We can use least squares or pick a particular solution
                
                # Let's use: W[k*n+i, c] = M[i,k,c] / (sum_c' M[i,k,c']^2) if sum != 0
                # This gives a particular solution
                
                W_new = np.zeros((p * n, rank))
                for i in range(n):
                    for k in range(p):
                        M_ik = M[i, k, :]
                        norm_sq = np.dot(M_ik, M_ik)
                        if norm_sq > 1e-12:
                            W_new[k * n + i, :] = M_ik / norm_sq
                        else:
                            # M is zero, can't satisfy the equation
                            W_new[k * n + i, :] = 0
                
                W = W_new
                
                # Check if this is valid
                if verify(U, V, W, tol=1e-4):
                    found = True
                    break
                
                # Now fix V and W, solve for U similarly
                # For each (i,j): sum_c U[i*m+j,c] * (sum_k V[j*p+k,c] * W[k*n+i,c]) = 1
                # Let N[i,j,c] = sum_k V[j*p+k,c] * W[k*n+i,c]
                N = np.zeros((n, m, rank))
                for i in range(n):
                    for j in range(m):
                        for c in range(rank):
                            N[i, j, c] = np.sum([V[j*p+k, c] * W[k*n+i, c] for k in range(p)])
                
                U_new = np.zeros((n * m, rank))
                for i in range(n):
                    for j in range(m):
                        N_ij = N[i, j, :]
                        norm_sq = np.dot(N_ij, N_ij)
                        if norm_sq > 1e-12:
                            U_new[i * m + j, :] = N_ij / norm_sq
                        else:
                            U_new[i * m + j, :] = 0
                
                U = U_new
                
                # Check
                if verify(U, V, W, tol=1e-4):
                    found = True
                    break
                
                # Fix U and W, solve for V
                # For each (j,k): sum_c V[j*p+k,c] * (sum_i U[i*m+j,c] * W[k*n+i,c]) = 1
                # Let P[j,k,c] = sum_i U[i*m+j,c] * W[k*n+i,c]
                P = np.zeros((m, p, rank))
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            P[j, k, c] = np.sum([U[i*m+j, c] * W[k*n+i, c] for i in range(n)])
                
                V_new = np.zeros((m * p, rank))
                for j in range(m):
                    for k in range(p):
                        P_jk = P[j, k, :]
                        norm_sq = np.dot(P_jk, P_jk)
                        if norm_sq > 1e-12:
                            V_new[j * p + k, :] = P_jk / norm_sq
                        else:
                            V_new[j * p + k, :] = 0
                
                V = V_new
                
                # Check
                if verify(U, V, W, tol=1e-4):
                    found = True
                    break
        
        if found:
            # Verify with tighter tolerance
            if verify(U, V, W, tol=1e-8):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                # Try to find even lower rank
                continue
            else:
                # Try to polish
                # Use the current solution as initialization and do more iterations
                for polish_it in range(500):
                    if time.time() > deadline - 30:
                        break
                    
                    # Same alternating scheme
                    M = np.zeros((n, p, rank))
                    for i in range(n):
                        for k in range(p):
                            for c in range(rank):
                                M[i, k, c] = np.sum([U[i*m+j, c] * V[j*p+k, c] for j in range(m)])
                    
                    W_new = np.zeros((p * n, rank))
                    for i in range(n):
                        for k in range(p):
                            M_ik = M[i, k, :]
                            norm_sq = np.dot(M_ik, M_ik)
                            if norm_sq > 1e-12:
                                W_new[k * n + i, :] = M_ik / norm_sq
                    W = W_new
                    
                    if verify(U, V, W, tol=1e-8):
                        break
                    
                    N = np.zeros((n, m, rank))
                    for i in range(n):
                        for j in range(m):
                            for c in range(rank):
                                N[i, j, c] = np.sum([V[j*p+k, c] * W[k*n+i, c] for k in range(p)])
                    
                    U_new = np.zeros((n * m, rank))
                    for i in range(n):
                        for j in range(m):
                            N_ij = N[i, j, :]
                            norm_sq = np.dot(N_ij, N_ij)
                            if norm_sq > 1e-12:
                                U_new[i * m + j, :] = N_ij / norm_sq
                    U = U_new
                    
                    if verify(U, V, W, tol=1e-8):
                        break
                    
                    P = np.zeros((m, p, rank))
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank):
                                P[j, k, c] = np.sum([U[i*m+j, c] * W[k*n+i, c] for i in range(n)])
                    
                    V_new = np.zeros((m * p, rank))
                    for j in range(m):
                        for k in range(p):
                            P_jk = P[j, k, :]
                            norm_sq = np.dot(P_jk, P_jk)
                            if norm_sq > 1e-12:
                                V_new[j * p + k, :] = P_jk / norm_sq
                    V = V_new
                    
                    if verify(U, V, W, tol=1e-8):
                        break
                
                if verify(U, V, W, tol=1e-8):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    continue
        
        # If we couldn't find a solution for this rank, try next
        # But we should also try more sophisticated methods
    
    # Return the best solution found
    # Convert to lists for the expected format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
