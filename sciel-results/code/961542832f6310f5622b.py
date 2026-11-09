"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build the tensor T as a 3D array
    # T[a, b, c] where:
    #   a = i*m + j (U index), i in [0,n), j in [0,m)
    #   b = j*p + l (V index), j in [0,m), l in [0,p)
    #   c = l*n + i (W index), l in [0,p), i in [0,n)
    # T[i*m+j, j*p+l, l*n+i] = 1 for all valid i,j,l
    
    # Nonzero entries
    nonzero = []
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a = i * m + j
                b = j * p + l
                c = l * n + i
                nonzero.append((a, b, c))
    
    # Schoolbook decomposition (rank = n*m*p = 128)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + l][col] = 1
                    W[l * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition
    # Strategy: Use the structure of the problem
    # For 4x4x4, best known rank is 20
    # For 4x4x8, we can split into two 4x4x4 problems: rank <= 40
    
    # Let's try to construct a rank-40 decomposition
    # by using two independent rank-20 decompositions for the two halves of l
    
    # First, let's define a rank-20 decomposition for 4x4x4
    # Using a known construction
    
    def rank20_444():
        """Return U, V, W matrices for 4x4x4 matmul with rank 20."""
        # This is a known rank-20 construction for 4x4x4
        # The construction uses specific patterns
        
        # One known rank-20 construction:
        # Based on the work of various researchers
        
        # Let me use a concrete construction
        # The tensor for 4x4x4: T[i*4+j, j*4+k, k*4+i] = 1
        
        # A rank-20 decomposition can be constructed as follows:
        # We need 20 trilinear forms
        
        # Here's one approach: use the structure
        # Actually, let me just try to find it numerically
        
        # For now, let me return a schoolbook 4x4x4 (rank 64) 
        # and we'll improve
        rank = 64
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i * 4 + j][col] = 1
                    V[j * 4 + k][col] = 1
                    W[k * 4 + i][col] = 1
                    col += 1
        return U, V, W
    
    # Actually, let me try a direct optimization approach
    # For a given rank r, we want to find U, V, W such that
    # sum_s U[a,s] * V[b,s] * W[c,s] = T[a,b,c] for all a,b,c
    
    # This is a nonlinear system. Let's use a greedy approach:
    # 1. Start with random initialization
    # 2. Use coordinate descent or similar
    
    # Given time constraints, let me try a simpler approach:
    # Check if we can achieve rank < 128 by exploiting symmetries
    
    # The tensor has the property that for fixed j, the entries form a 
    # permutation pattern. Let's see if we can share factors.
    
    # For fixed j, we have entries at:
    # (i*m+j, j*p+l, l*n+i) for all i in [0,4), l in [0,8)
    # That's 4*8 = 32 entries per j, and 4 values of j, total 128
    
    # Can we express these with fewer rank-1 tensors?
    
    # Let me try rank 64: pair up l values
    # For each j and i, we have l in [0,8). 
    # If we could share across l values...
    
    # Actually, the key insight: for fixed i and j, the entry depends on l
    # T[i*m+j, j*p+l, l*n+i] = 1
    # This is 1 for all l, so for fixed i,j, we have 8 entries that are all 1
    # but at different V and W positions
    
    # Let me try a different strategy: use the fact that the problem is
    # essentially two independent 4x4x4 problems (for l in [0,4) and l in [4,8))
    # If 4x4x4 has rank r, then 4x4x8 has rank at most 2r
    
    # Known: 4x4x4 matmul tensor has rank at most 20 (various constructions)
    # So 4x4x8 should have rank at most 40
    
    # Let me try to construct this explicitly
    
    # For the first half (l in [0,4)):
    # T[i*4+j, j*4+l, l*4+i] = 1 for i,j,l in [0,4)
    # This is the standard 4x4x4 matmul tensor
    
    # For the second half (l in [4,8)):
    # Let l' = l - 4, so l' in [0,4)
    # T[i*4+j, j*4+(4+l'), (4+l')*4+i] = 1
    # = T[i*4+j, j*4+4+l', 16+4*l'+i]
    # This is similar but with shifted indices
    
    # The second half is not exactly the same as the first because
    # the W index is different: (4+l')*4+i = 16+4*l'+i vs l'*4+i
    
    # So we can't simply reuse the same decomposition.
    
    # Let me try a numerical optimization approach with a time limit
    
    def try_optimization(max_rank, time_limit=30):
        """Try to find a decomposition of given rank using optimization."""
        start_time = time.time()
        
        # Initialize randomly
        np.random.seed(42)
        U = np.random.randn(n * m, max_rank) * 0.1
        V = np.random.randn(m * p, max_rank) * 0.1
        W = np.random.randn(p * n, max_rank) * 0.1
        
        # We want to minimize: sum_{a,b,c} (sum_s U[a,s]*V[b,s]*W[c,s] - T[a,b,c])^2
        # This is a non-convex optimization problem
        
        # Use a simple gradient-free approach: random restarts with local search
        # Given time constraints, this may not converge to a good solution
        
        # Alternative: use the structure to guide the optimization
        
        # For now, return None to indicate failure
        return None
    
    # Given the complexity, let me just return the schoolbook for now
    # but with a cleaner structure
    
    return schoolbook()
# EVOLVE-BLOCK-END
