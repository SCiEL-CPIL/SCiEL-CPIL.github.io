"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    best_rank = rank_trivial
    
    # Try to find lower rank decompositions
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    # We need rank-r decomposition: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    # Strategy: try ranks from lower to upper, use optimization to find valid decomposition
    # For each rank r, we have r*9 + r*21 + r*21 = r*51 variables
    # Constraints: for each (a,b,c) where a in [0,nm), b in [0,mp), c in [0,pn):
    #   sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    # where T[a,b,c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i, else 0
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try various ranks
    for r in range(15, min(60, rank_trivial + 1)):
        if time.time() > deadline - 30:
            break
        
        # Try to find a rank-r decomposition using iterative optimization
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Use gradient-free approach: try to minimize error
        # Error = sum_{a,b,c} (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
        
        best_err = float('inf')
        for trial in range(5):
            if time.time() > deadline - 30:
                break
            
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Iterative refinement: fix two matrices, solve for third
            for iteration in range(100):
                if time.time() > deadline - 30:
                    break
                
                # Fix V and W, solve for U: for each a, U[a,:] should satisfy
                # sum_r U[a,r] * (V[:,r] ⊙ W[:,r])[a-related] ... this is complex
                # Alternative: use ALS (Alternating Least Squares)
                
                # For each r, we can't easily separate. Let's use a different approach.
                # Actually, let's try a simpler heuristic: use the structure of the problem.
                
                break  # ALS is complex, try different approach
            
            # Try a structured initialization based on the tensor structure
            # Set U, V, W to exploit the fact that T is sparse
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Assign each nonzero of T to one of the r terms
            nonzeros = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        nonzeros.append((i*m+j, j*p+k, k*n+i))
            
            # Distribute nonzeros among r terms
            for idx, (a, b, c) in enumerate(nonzeros):
                r_idx = idx % r
                U[a, r_idx] += 1.0
                V[b, r_idx] += 1.0
                W[c, r_idx] += 1.0
            
            # Check if this works
            err = 0.0
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        computed = np.sum(U[a, :] * V[b, :] * W[c, :])
                        err += (computed - T[a, b, c]) ** 2
            
            if err < 1e-6 and r < best_rank:
                best_rank = r
                best_result = {
                    "U": [[float(U[i, j]) for j in range(r)] for i in range(n*m)],
                    "V": [[float(V[i, j]) for j in range(r)] for i in range(m*p)],
                    "W": [[float(W[i, j]) for j in range(r)] for i in range(p*n)]
                }
        
        if best_rank <= r:
            # Found a decomposition at this rank or lower, stop
            break
    
    return best_result
# EVOLVE-BLOCK-END
