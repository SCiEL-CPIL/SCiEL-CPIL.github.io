"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s to stay safely under 150s
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition of rank n*m*p
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Build the target tensor T
    # T[u, v, w] = 1 if u = i*m+j, v = j*p+k, w = k*n+i for some i,j,k
    # Otherwise T[u,v,w] = 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    
    # Try to find a lower rank decomposition
    # We'll use a numerical approach: parameterize U, V, W and minimize ||T - sum_r U[:,r] o V[:,r] o W[:,r]||^2
    
    def cp_decompose(rank, U0, V0, W0, max_iter=1000, tol=1e-10):
        """Try to find a CP decomposition of rank 'rank' using iterative refinement."""
        # Initialize
        U = U0.copy().astype(float)
        V = V0.copy().astype(float)
        W = W0.copy().astype(float)
        
        # Reshape for computation
        T_flat = T.reshape(n*m * m*p * p*n)
        
        # Precompute the "flattened" tensor structure
        # For each rank-1 term r, the contribution to T_flat is:
        # sum_{u,v,w} (U[u,r]*V[v,r]*W[w,r])^2 * (T_flat - ...)^2
        # This is complex, so let's use a simpler approach
        
        # Instead, let's use a greedy approach: try to fit one term at a time
        # Or use ALS (Alternating Least Squares)
        
        # ALS for CP decomposition
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U
            for r in range(rank):
                # Fix V[:,r] and W[:,r], solve for U[:,r]
                # T ≈ sum_s U[:,s] o V[:,s] o W[:,s]
                # For fixed r, we want U[:,r] to minimize:
                # ||T - U[:,r] o V[:,r] o W[:,r] - sum_{s≠r} U[:,s] o V[:,s] o W[:,s]||^2
                
                # This is equivalent to:
                # (V[:,r] * W[:,r])^T * T * (V[:,r] * W[:,r]) ... this is getting complex
                
                # Let's use a different approach: flatten and use linear algebra
                pass
        
        # This is getting too complex. Let's use a simpler heuristic.
        return U, V, W
    
    # Alternative approach: try specific structured decompositions
    # Since n=2 is small, maybe we can find a decomposition with rank < 36
    
    # Let's try rank 30, 32, 34, etc. and see if we can find a solution
    
    # For now, let's just return the schoolbook result if we can't find better quickly
    # But let's try a few random restarts with numerical optimization
    
    # Actually, let's try a more systematic approach:
    # The tensor has 36 nonzero entries. The rank is at least the size of the largest
    # set of entries that are "independent" in some sense.
    
    # A lower bound on the rank is given by the "slice rank" or other tensor rank bounds.
    # For now, let's just return schoolbook.
    
    if time.time() > deadline - 5:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try a few random initializations with a simple optimization
    # We'll use a penalty method: minimize ||T - CP(U,V,W)||^2 + penalty * ||deviation from half-integers||^2
    
    for target_rank in [35, 34, 33, 32, 30, 28, 24, 20]:
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U_init = np.random.randn(n*m, target_rank) * 0.1
            V_init = np.random.randn(m*p, target_rank) * 0.1
            W_init = np.random.randn(p*n, target_rank) * 0.1
            
            # Flatten the parameters
            def params_to_tensors(params):
                U = params[:n*m*target_rank].reshape(n*m, target_rank)
                V = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                return U, V, W
            
            def objective(params):
                U, V, W = params_to_tensors(params)
                # Compute CP decomposition
                T_approx = np.zeros((n*m, m*p, p*n))
                for r in range(target_rank):
                    T_approx += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
                diff = T - T_approx
                return np.sum(diff**2)
            
            initial_params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(objective, initial_params, method='L-BFGS-B', maxiter=100, 
                                  options={'maxls': 50, 'gtol': 1e-6})
                
                if result.success and result.fun < 1e-8:
                    # Check if we can round to half-integers
                    U, V, W = params_to_tensors(result.x)
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify the rounded version
                    T_approx = np.zeros((n*m, m*p, p*n))
                    for r in range(target_rank):
                        T_approx += np.einsum('u,v,w->uvw', U_rounded[:,r], V_rounded[:,r], W_rounded[:,r])
                    
                    if np.allclose(T_approx, T, atol=1e-6):
                        # Found a valid decomposition!
                        best_rank = target_rank
                        best_U = U_rounded.tolist()
                        best_V = V_rounded.tolist()
                        best_W = W_rounded.tolist()
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
