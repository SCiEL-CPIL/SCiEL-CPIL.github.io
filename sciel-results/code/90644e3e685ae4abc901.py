"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

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
    
    # Strategy: Fix U to be identity-like (sparse) to decouple the problem.
    # We search for decompositions where U is sparse (mostly 0s and 1s),
    # which allows us to solve for V and W more efficiently or verify them.
    
    # First, try a structured search with fixed sparse U patterns.
    # We iterate over trial ranks and try to find a sparse U that allows
    # a low-rank decomposition.
    
    found_better = False
    
    for trial_rank in range(lower_bound, rank_trivial):
        if time.time() > deadline - 30:
            break
        
        # Try random sparse U matrices with entries in {-1, 0, 1}
        # The sparsity pattern can help decouple the tensor equation.
        num_attempts = 50
        for attempt in range(num_attempts):
            if time.time() > deadline - 30:
                break
            
            # Generate a random sparse U
            # We want U to be "almost" identity or permutation-like to simplify the system
            # Let's try U with exactly one non-zero per row, or a few non-zeros.
            # Actually, let's just use random integers from {-1, 0, 1} with some sparsity.
            
            np.random.seed(12345 + trial_rank * 100 + attempt)
            U_fixed = np.zeros((n*m, trial_rank))
            # For each row in U, pick 1 or 2 random columns to be non-zero
            for a in range(n*m):
                num_nz = np.random.randint(1, 3)  # 1 or 2 non-zeros
                cols = np.random.choice(trial_rank, size=num_nz, replace=False)
                for c in cols:
                    U_fixed[a, c] = np.random.choice([-1, 1])
            
            # Now we need to find V and W such that:
            # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
            # This is a system of linear equations in the entries of V and W?
            # No, it's still bilinear in V and W for each r.
            # But if U is fixed, we can view this as:
            # For each a, the matrix T_a (shape m*p) must be decomposed as:
            # T_a[b,c] = sum_r U[a,r] V[b,r] W[c,r]
            
            # If U is sparse, many a will have simple forms.
            # Let's use a least-squares approach for V and W given U_fixed.
            # We can flatten V and W? No, that's huge.
            
            # Alternative: Use ALS starting from this U_fixed.
            # But we want to exploit the structure.
            
            # Let's try to solve for V and W using a different approach.
            # If we fix V, we can solve for W. If we fix W, we can solve for V.
            # This is ALS again, but with a good initial U.
            
            V_curr = np.random.randn(m*p, trial_rank)
            W_curr = np.random.randn(p*n, trial_rank)
            
            # Run a few ALS iterations
            converged = False
            for iteration in range(100):
                if time.time() > deadline - 30:
                    break
                
                # Update W: For each c, W[:,c] = argmin ||T[:, :, c] - U o V[:,:,c]||
                # T[:, :, c] is a vector of length n*m.
                # We want to minimize sum_{a,b} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
                # This is linear in W[c,r].
                # The system is: G_W W_c = rhs_W_c
                # Where G_W[r1, r2] = sum_{a,b} U[a,r1] U[a,r2] V[b,r1] V[b,r2] ? No.
                # Let's derive properly.
                # Error E = ||T - U o V o W||^2
                # dE/dW[c,r] = -2 sum_{a,b} T[a,b,c] U[a,r] V[b,r] + 2 sum_{a,b} (sum_{r'} U[a,r'] V[b,r'] W[c,r']) U[a,r] V[b,r]
                # = -2 (rhs_W) + 2 (G_W W)_r
                # G_W[r1, r2] = sum_{a,b} U[a,r1] V[b,r1] U[a,r2] V[b,r2]
                # = (sum_a U[a,r1] U[a,r2]) * (sum_b V[b,r1] V[b,r2])
                # = (U^T U)[r1, r2] * (V^T V)[r1, r2]
                
                UtU = U_fixed.T @ U_fixed
                VtV = V_curr.T @ V_curr
                G_W = UtU * VtV
                
                # rhs_W[c, r] = sum_{a,b} T[a,b,c] U[a,r] V[b,r]
                # = sum_b V[b,r] (sum_a T[a,b,c] U[a,r])
                # Let M1[b, r] = sum_a T[a,b,c] U[a,r] ? No, c is fixed.
                # Let's compute rhs_W for all c at once.
                # rhs_W[c, r] = sum_{a,b} T[a,b,c] U[a,r] V[b,r]
                # = np.einsum('abc,ar,br->cr', tensor, U_fixed, V_curr)
                
                rhs_W = np.einsum('abc,ar,br->cr', tensor, U_fixed, V_curr)
                
                # Solve G_W W_c = rhs_W_c for each c
                reg = 1e-8 * np.eye(trial_rank)
                try:
                    W_curr = np.linalg.solve(G_W + reg, rhs_W.T).T
                except np.linalg.LinAlgError:
                    W_curr = np.linalg.lstsq(G_W + reg, rhs_W.T, rcond=None)[0].T
                
                # Update V
                VtV = V_curr.T @ V_curr
                WtW = W_curr.T @ W_curr
                G_V = UtU * WtW
                
                rhs_V = np.einsum('abc,ar,cr->br', tensor, U_fixed, W_curr)
                
                try:
                    V_curr = np.linalg.solve(G_V + reg, rhs_V.T).T
                except np.linalg.LinAlgError:
                    V_curr = np.linalg.lstsq(G_V + reg, rhs_V.T, rcond=None)[0].T
                
                # Check convergence
                recon = np.einsum('ar,br,cr->abc', U_fixed, V_curr, W_curr)
                error = np.linalg.norm(recon - tensor)
                if error < 1e-8:
                    converged = True
                    break
            
            if converged:
                # Check if V and W can be rounded to integers/half-integers
                for factor in [1, 2, 4]:
                    V_r = np.round(V_curr * factor) / factor
                    W_r = np.round(W_curr * factor) / factor
                    
                    recon_r = np.einsum('ar,br,cr->abc', U_fixed, V_r, W_r)
                    error_r = np.linalg.norm(recon_r - tensor)
                    
                    if error_r < 1e-6:
                        best_U, best_V, best_W = U_fixed.copy(), V_r.copy(), W_r.copy()
                        best_rank = trial_rank
                        found_better = True
                        break
                
                if found_better:
                    break
        
        if found_better:
            break
    
    # If no better solution found, return the trivial one
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
