"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin
    
    # Build the multiplication tensor
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook rank
    rank_school = n * m * p
    
    # Try to find lower rank decompositions
    # For rank r, we need U (n*m x r), V (m*p x r), W (p*n x r)
    # such that T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
    best_rank = rank_school
    best_U = None
    best_V = None
    best_W = None
    
    # Start with schoolbook
    U = np.zeros((n*m, rank_school))
    V = np.zeros((m*p, rank_school))
    W = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank by solving linear systems
    # For a given rank r, we can try to find U, V, W
    # This is a hard problem, but for small cases we can try
    
    # Try ranks from lower bound up
    # Lower bound: max(n*p, m*p, n*m)
    lower_bound = max(n * p, m * p, n * m)
    
    # For each candidate rank, try to find a decomposition
    # Use the fact that if we fix U and V, W is determined (if possible)
    # Or use tensor rank algorithms
    
    # Let's try a greedy / iterative approach
    # For rank r, set up the system:
    # For each (a,b,c) with T[a,b,c]=1: sum_r U[a,r]*V[b,r]*W[c,r] = 1
    # For each (a,b,c) with T[a,b,c]=0: sum_r U[a,r]*V[b,r]*W[c,r] = 0
    
    # This is a nonlinear system. Let's try to use the structure.
    
    # Alternative: try specific patterns
    # For 2x3x3, try rank 12
    
    # Let me try to construct a rank 12 decomposition manually
    # by exploiting symmetries
    
    # Actually, let me try a numerical search
    # For small ranks, we can try to solve using least squares
    
    for r in range(lower_bound, rank_school):
        if time.time() > deadline - 10:
            break
        
        # Try to find a rank-r decomposition
        # Use random initialization and iterative refinement
        success = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U_try = np.random.randn(n*m, r)
            V_try = np.random.randn(m*p, r)
            W_try = np.random.randn(p*n, r)
            
            # Iterate: fix two, solve for third
            for it in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Fix V and W, solve for U
                # T[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
                # For each a: sum_{b,c} T[a,b,c] * something...
                # Actually, this is a linear system in U[a,:]
                
                # Better approach: for each r, we have a rank-1 tensor
                # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                
                # Let's use the unfolding approach
                # Fix V and W, then for each a:
                # T[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
                # This is a linear system: T[a,:,:].reshape(-1) = U[a,:] @ M
                # where M[r, (b,c)] = V[b,r] * W[c,r]
                
                M = np.zeros((r, m*p * p*n))
                for rr in range(r):
                    for b in range(m*p):
                        for c in range(p*n):
                            M[rr, b * p*n + c] = V_try[b, rr] * W_try[c, rr]
                
                # For each a, solve U[a,:] from T[a,:,:].reshape(-1) = U[a,:] @ M
                T_unfold = T.reshape(n*m, m*p * p*n)
                try:
                    U_try = np.linalg.lstsq(M.T, T_unfold.T, rcond=None)[0].T
                except:
                    pass
                
                # Fix U and W, solve for V
                M2 = np.zeros((r, n*m * p*n))
                for rr in range(r):
                    for a in range(n*m):
                        for c in range(p*n):
                            M2[rr, a * p*n + c] = U_try[a, rr] * W_try[c, rr]
                
                T_unfold2 = T.reshape(m*p, n*m * p*n)
                T_unfold2 = T_unfold2.T  # Now (n*m*p*n, m*p) -> reshape properly
                # Actually T[a,b,c], fix a,c, solve for b
                T_reshape = T.reshape(p*n, m*p, n*m)  # T[c,b,a]
                T_reshape = T_reshape.transpose(2, 0, 1)  # T[a,c,b]
                T_reshape = T_reshape.reshape(n*m * p*n, m*p)
                
                try:
                    V_try = np.linalg.lstsq(M2.T, T_reshape.T, rcond=None)[0].T
                except:
                    pass
                
                # Fix U and V, solve for W
                M3 = np.zeros((r, n*m * m*p))
                for rr in range(r):
                    for a in range(n*m):
                        for b in range(m*p):
                            M3[rr, a * m*p + b] = U_try[a, rr] * V_try[b, rr]
                
                T_reshape3 = T.reshape(p*n, n*m * m*p)
                try:
                    W_try = np.linalg.lstsq(M3.T, T_reshape3.T, rcond=None)[0].T
                except:
                    pass
                
                # Check convergence
                recon = np.zeros((n*m, m*p, p*n))
                for rr in range(r):
                    recon += U_try[:, rr] * V_try[:, rr] * W_try[:, rr]
                err = np.linalg.norm(recon - T)
                if err < 1e-6:
                    success = True
                    break
            
            if success:
                break
        
        if success:
            best_rank = r
            best_U, best_V, best_W = U_try, V_try, W_try
            break
    
    # Convert to lists with integer/half-integer entries if possible
    # Round to nearest 0.5
    U_out = np.round(best_U * 2) / 2
    V_out = np.round(best_V * 2) / 2
    W_out = np.round(best_W * 2) / 2
    
    # Verify
    recon = np.zeros((n*m, m*p, p*n))
    for rr in range(best_rank):
        recon += U_out[:, rr] * V_out[:, rr] * W_out[:, rr]
    if np.linalg.norm(recon - T) > 1e-6:
        # Rounding broke it, use unrounded
        U_out, V_out, W_out = best_U, best_V, best_W
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
