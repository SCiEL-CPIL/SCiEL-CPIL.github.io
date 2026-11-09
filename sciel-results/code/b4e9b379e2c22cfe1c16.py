import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    rank_trivial = n * m * p  # 50
    
    def build_trivial():
        U = np.zeros((n * m, rank_trivial))
        V = np.zeros((m * p, rank_trivial))
        W = np.zeros((p * n, rank_trivial))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U0, V0, W0 = build_trivial()
    
    def verify(U, V, W, tol=1e-6):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to reduce rank by merging terms
    # Strategy: try to combine pairs of terms that share structure
    
    best_rank = rank_trivial
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Try structured rank reductions
    # For each j, try to use shared terms across i=0 and i=1
    
    # Attempt: for each j, use p terms that each handle both i=0 and i=1
    # This would give rank = m * p = 25, but let's see if it's feasible
    
    # More realistic: try ranks from 49 down to 30
    for target_rank in range(rank_trivial - 1, 25, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try several random perturbations of the trivial decomposition
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Start from trivial and try to merge
            U = U0.copy()
            V = V0.copy()
            W = W0.copy()
            
            # Try to eliminate one term by redistributing
            # Pick a random term to eliminate
            elim_col = np.random.randint(0, rank_trivial)
            
            # Get the (i,j,k) for this column
            i_elim = elim_col // (m * p)
            j_elim = (elim_col // p) % m
            k_elim = elim_col % p
            
            # Try to absorb this term into other terms
            # We need to add corrections to other columns
            # For simplicity, try to split the contribution across random other columns
            
            # Alternative: use numerical optimization
            # Set up as: minimize ||tensor - decomposed||^2
            # with rank = target_rank
            
            if attempt == 0:
                # Try a structured approach: for each j, use terms that cover all k for both i
                # This gives rank = m * p = 25
                if target_rank >= 25:
                    r = target_rank
                    U_new = np.zeros((n*m, r))
                    V_new = np.zeros((m*p, r))
                    W_new = np.zeros((p*n, r))
                    
                    # For each j, allocate p terms
                    # Term (j, k) handles: for i=0, contribute to (0,j,k); for i=1, contribute to (1,j,k)
                    # But we need: U[0*m+j, r] * V[j*p+k, r] * W[k*n+0, r] = 1
                    # and: U[1*m+j, r] * V[j*p+k, r] * W[k*n+1, r] = 1
                    # If we set V[j*p+k, r] = 1, then U[0*m+j, r]*W[k*n, r] = 1 and U[1*m+j, r]*W[k*n+1, r] = 1
                    # This works with r = m*p = 25 terms
                    for j in range(m):
                        for k in range(p):
                            idx = j * p + k
                            U_new[0 * m + j, idx] = 1
                            U_new[1 * m + j, idx] = 1
                            V_new[j * p + k, idx] = 1
                            W_new[k * n + 0, idx] = 1
                            W_new[k * n + 1, idx] = 1
                    
                    if verify(U_new, V_new, W_new):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_new, V_new, W_new
                        found = True
                        break
                    
                    # Try with half-integers
                    U_new = np.zeros((n*m, r))
                    V_new = np.zeros((m*p, r))
                    W_new = np.zeros((p*n, r))
                    for j in range(m):
                        for k in range(p):
                            idx = j * p + k
                            U_new[0 * m + j, idx] = 0.5
                            U_new[1 * m + j, idx] = 0.5
                            V_new[j * p + k, idx] = 1
                            W_new[k * n + 0, idx] = 2
                            W_new[k * n + 1, idx] = 2
                    
                    if verify(U_new, V_new, W_new):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_new, V_new, W_new
                        found = True
                        break
        
        if found:
            break
    
    if best_rank < rank_trivial:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

