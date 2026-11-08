"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    tensor = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute lower bounds from matrix flattening
    T0 = tensor.reshape(n*m, -1)
    rank0 = np.linalg.matrix_rank(T0)
    T1 = tensor.transpose(1,0,2).reshape(m*p, -1)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = tensor.transpose(2,0,1).reshape(p*n, -1)
    rank2 = np.linalg.matrix_rank(T2)
    lower_bound = max(rank0, rank1, rank2)
    
    # Trivial solution with rank = n*m*p
    rank_trivial = n * m * p
    U_trivial = np.zeros((n*m, rank_trivial))
    V_trivial = np.zeros((m*p, rank_trivial))
    W_trivial = np.zeros((p*n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, col] = 1
                V_trivial[j*p+k, col] = 1
                W_trivial[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
    best_rank = rank_trivial
    
    # If lower bound equals trivial rank, we're done
    if lower_bound >= rank_trivial:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try to find lower rank decompositions using vectorized ALS
    for trial_rank in range(lower_bound, rank_trivial):
        if time.time() > deadline - 40:
            break
        
        found = False
        for init in range(10):
            if time.time() > deadline - 40:
                break
            
            np.random.seed(42 + init)
            U_curr = np.random.randn(n*m, trial_rank)
            V_curr = np.random.randn(m*p, trial_rank)
            W_curr = np.random.randn(p*n, trial_rank)
            
            # Alternating least squares
            for iteration in range(200):
                if time.time() > deadline - 40:
                    break
                
                # Precompute Gram matrices for each factor update
                # G_U = (V^T V) o (W^T W)
                # G_V = (U^T U) o (W^T W)
                # G_W = (U^T U) o (V^T V)
                
                VtV = V_curr.T @ V_curr
                WtW = W_curr.T @ W_curr
                UtU = U_curr.T @ U_curr
                
                G_U = VtV * WtW
                G_V = UtU * WtW
                G_W = UtU * VtV
                
                # Compute RHS vectors using einsum
                # rhs_U[a, r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r]
                rhs_U = np.einsum('abc,br,cr->ar', tensor, V_curr, W_curr)
                
                # rhs_V[b, r] = sum_{a,c} T[a,b,c] U[a,r] W[c,r]
                rhs_V = np.einsum('abc,ar,cr->br', tensor, U_curr, W_curr)
                
                # rhs_W[c, r] = sum_{a,b} T[a,b,c] U[a,r] V[b,r]
                rhs_W = np.einsum('abc,ar,br->cr', tensor, U_curr, V_curr)
                
                # Solve linear systems
                # Add small regularization to avoid singularity
                reg = 1e-8 * np.eye(trial_rank)
                
                try:
                    U_curr = np.linalg.solve(G_U + reg, rhs_U.T).T
                except np.linalg.LinAlgError:
                    U_curr = np.linalg.lstsq(G_U + reg, rhs_U.T, rcond=None)[0].T
                
                try:
                    V_curr = np.linalg.solve(G_V + reg, rhs_V.T).T
                except np.linalg.LinAlgError:
                    V_curr = np.linalg.lstsq(G_V + reg, rhs_V.T, rcond=None)[0].T
                
                try:
                    W_curr = np.linalg.solve(G_W + reg, rhs_W.T).T
                except np.linalg.LinAlgError:
                    W_curr = np.linalg.lstsq(G_W + reg, rhs_W.T, rcond=None)[0].T
                
                # Check convergence
                recon = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
                error = np.linalg.norm(recon - tensor)
                if error < 1e-8:
                    break
            
            # Check if decomposition is valid
            recon = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
            error = np.linalg.norm(recon - tensor)
            
            if error < 1e-6:
                # Try to round to integers/half-integers
                for factor in [1, 2, 4]:
                    U_r = np.round(U_curr * factor) / factor
                    V_r = np.round(V_curr * factor) / factor
                    W_r = np.round(W_curr * factor) / factor
                    
                    recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    error_r = np.linalg.norm(recon_r - tensor)
                    
                    if error_r < 1e-6:
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = trial_rank
                        found = True
                        break
                if found:
                    break
        
        if found:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
