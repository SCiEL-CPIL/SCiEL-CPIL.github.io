"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 40
    
    # Build the schoolbook decomposition
    U = np.zeros((n * m, rank_schoolbook), dtype=object)
    V = np.zeros((m * p, rank_schoolbook), dtype=object)
    W = np.zeros((p * n, rank_schoolbook), dtype=object)
    
    for c in range(rank_schoolbook):
        i = c // (m * p)
        j = (c // p) % m
        k = c % p
        U[i * m + j, c] = 1
        V[j * p + k, c] = 1
        W[k * n + i, c] = 1
    
    # The tensor T[i,j,k] = 1 for all i,j,k (matrix multiplication)
    # We want to find a rank-r decomposition:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Strategy: Try to find linear dependencies among the 40 schoolbook terms.
    # Each term c corresponds to (i_c, j_c, k_c).
    # A rank-r decomposition exists if we can find r vectors u_c, v_c, w_c
    # such that for all (i,j,k): sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1
    #
    # Actually, let's think about this differently.
    # The matrix multiplication tensor M_{n,m,p} has entries T[i,j,k] = 1.
    # A rank-r decomposition gives: T[i,j,k] = sum_{c=1}^{r} U_{i,j}^{(c)} * V_{j,k}^{(c)} * W_{k,i}^{(c)}
    #
    # For n=2, m=4, p=5, the schoolbook rank is 40.
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # Also rank >= (n+m+p-1) for some cases, but that's not tight here.
    #
    # Let's try a random search for rank < 40.
    # We'll use the fact that we can try to find r < 40 such that
    # there exist U, V, W with the product matching T.
    
    # First, let's check if rank 20 is achievable (the theoretical lower bound)
    # This is unlikely for small dimensions, but let's try.
    
    # Approach: Use random initialization and gradient-free search
    # For a given rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try ranks from the lower bound up to schoolbook rank
    best_result = None
    best_rank = rank_schoolbook
    
    # Store the schoolbook solution as fallback
    U_fb = U.tolist()
    V_fb = V.tolist()
    W_fb = W.tolist()
    
    # Try to find lower rank decompositions
    # For each candidate rank, do a random search
    
    def verify(U_mat, V_mat, W_mat, n, m, p):
        """Check if U, V, W give a valid decomposition."""
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    # Try random searches for ranks below 40
    # We'll use a simple approach: initialize randomly and try to minimize error
    
    rng = np.random.RandomState(42)
    
    for target_rank in range(20, rank_schoolbook):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        n_trials = 50
        found = False
        
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                break
            
            U_r = rng.randn(n*m, target_rank)
            V_r = rng.randn(m*p, target_rank)
            W_r = rng.randn(p*n, target_rank)
            
            # Simple iterative refinement: adjust one factor at a time
            # For each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is a system of n*m*p = 40 equations with r unknowns per factor
            
            # Actually, let's use a different approach.
            # Fix U and V, then solve for W.
            # For each (k,i): W[k*n+i, c] appears in equations for all j.
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
            # This is a linear system in W[k*n+i, :] for each (k,i).
            
            # Let's try: fix U and V randomly, solve for W, then check.
            # For each (k,i), we have m equations (one for each j):
            # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
            # This is a system: A_{(k,i)} @ W[k*n+i, :] = 1_m
            # where A_{(k,i)}[j, c] = U[i*m+j,c] * V[j*p+k,c]
            
            for _ in range(20):  # iterations
                # Solve for W given U, V
                W_new = np.zeros((p*n, target_rank))
                feasible = True
                for k in range(p):
                    for i in range(n):
                        # Build A[j, c] = U[i*m+j, c] * V[j*p+k, c] for j=0..m-1
                        A_ki = np.zeros((m, target_rank))
                        for j in range(m):
                            A_ki[j, :] = U_r[i*m+j, :] * V_r[j*p+k, :]
                        # Solve A_ki @ w = ones(m)
                        try:
                            w_sol, residuals, rank_a, sv = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                            W_new[k*n+i, :] = w_sol
                            if residuals[0] > 1e-8:
                                feasible = False
                        except:
                            feasible = False
                            break
                    if not feasible:
                        break
                
                if not feasible:
                    # Perturb U and V slightly
                    U_r += 0.1 * rng.randn(n*m, target_rank)
                    V_r += 0.1 * rng.randn(m*p, target_rank)
                    continue
                
                # Check if this solution works
                if verify(U_r, V_r, W_new, n, m, p):
                    found = True
                    break
            
            if found:
                break
        
        if found:
            best_rank = target_rank
            best_result = (U_r.tolist(), V_r.tolist(), W_new.tolist())
            break
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    else:
        # Return schoolbook decomposition
        return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
