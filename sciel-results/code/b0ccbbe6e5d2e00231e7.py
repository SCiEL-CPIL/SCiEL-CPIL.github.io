"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Try to find minimal rank decomposition
    # The multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Rank-r decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    def try_rank(r, max_iters=5000, seed=42):
        """Try to find rank-r decomposition using iterative optimization."""
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random values
        U = rng.randn(n * m, r) * 0.5
        V = rng.randn(m * p, r) * 0.5
        W = rng.randn(p * n, r) * 0.5
        
        # We want: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is a non-convex optimization problem.
        
        # Use a simple gradient-free approach: alternating minimization
        # For fixed U,V, solve for W: W[k*n+i,c] = sum over (i,j,k) ... this is coupled.
        
        # Alternative: treat as a system and use iterative refinement
        # Let's use a different approach: random restarts with local search
        
        best_loss = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(5):
            if time.time() > deadline:
                break
                
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            W = rng.randn(p * n, r)
            
            # Scale to make initial loss reasonable
            U /= np.sqrt(np.mean(U**2) + 1e-10)
            V /= np.sqrt(np.mean(V**2) + 1e-10)
            W /= np.sqrt(np.mean(W**2) + 1e-10)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Compute loss: sum over all (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
                loss = 0.0
                # For efficiency, compute in batches
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            loss += (val - 1.0) ** 2
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-10:
                    break
                
                # Simple gradient step (approximate)
                # This is getting complex; let's use a different strategy
                
                # Instead, let's try to solve it as a linear system for one factor at a time
                # Fix U and V, solve for W:
                # For each (k,i,c): W[k*n+i,c] appears in equations for all j
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
                # This is still coupled across c.
                
                # Let's just do random perturbation (simulated annealing style)
                lr = 0.1 / (1 + it/100)
                U += lr * rng.randn(n*m, r)
                V += lr * rng.randn(m*p, r)
                W += lr * rng.randn(p*n, r)
                
                # Clip to prevent explosion
                U = np.clip(U, -10, 10)
                V = np.clip(V, -10, 10)
                W = np.clip(W, -10, 10)
        
        if best_loss < 1e-8:
            return best_U, best_V, best_W
        return None
    
    # Try ranks from low to high
    for r in range(1, 61):
        if time.time() > deadline:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(U[i, c]) for c in range(r)] for i in range(n*m)]
            V_list = [[float(V[i, c]) for c in range(r)] for i in range(m*p)]
            W_list = [[float(W[i, c]) for c in range(r)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
