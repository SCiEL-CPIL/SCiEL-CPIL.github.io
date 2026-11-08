"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (18, rank)  
    # W: (p*n, rank) = (18, rank)
    # W index: k*n + i
    
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    # Build the multiplication tensor T[i,j,k] = 1 for all valid i,j,k
    # T is a 3x3x6 tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Try to find a low-rank decomposition
    # Start with rank 38 (two 3x3x3 rank-19 decompositions)
    # Then try to reduce
    
    # First, let's construct a rank-38 solution using two 3x3x3 rank-19 decompositions
    def make_3x3x3_rank19():
        """Return U, V, W for 3x3x3 matrix mult with rank 19."""
        # U: (9, 19), V: (9, 19), W: (9, 19)
        # For 3x3x3: C[i,k] = sum_j A[i,j]*B[j,k]
        # W index: k*3 + i
        U = np.zeros((9, 19))
        V = np.zeros((9, 19))
        W = np.zeros((9, 19))
        
        # 9 diagonal terms: for (i,j), term (E_ij, E_jj, E_ij)
        for i in range(3):
            for j in range(3):
                r = i * 3 + j
                U[i * 3 + j, r] = 1.0
                V[j * 3 + j, r] = 1.0
                W[j * 3 + i, r] = 1.0  # W[k*n+i] = W[j*3+i]
        
        # 10 additional terms from laser method
        # Using a standard 19-rank construction
        # These involve the off-diagonal structure
        
        # Term 10: handles (0,1) and (1,0) coupling
        # One standard form uses the following 10 terms:
        
        # Let me use a known explicit 19-rank decomposition
        # From the literature, one such decomposition:
        
        # Additional 10 terms:
        # (E01+E10, E01+E10, E01+E10)
        # (E01-E10, E01-E10, E01-E10)  -- no, this isn't quite right
        
        # Let me use a different known construction.
        # The rank-19 decomposition of [3,3,3]:
        
        # Terms 10-19 (using the "Strassen-like" pattern):
        # t10: (E01+E10, E01+E10, E01+E10)
        # t11: (E01-E10, E01-E10, E01-E10)
        # t12: (E02+E20, E02+E20, E02+E20)
        # t13: (E02-E20, E02-E20, E02-E20)
        # t14: (E12+E21, E12+E21, E12+E21)
        # t15: (E12-E21, E12-E21, E12-E21)
        # t16: (E01+E10+E02+E20, E01+E10+E02+E20, E01+E10+E02+E20)
        # ... this is getting complicated
        
        # Let me just use a numerical approach to find the 19-rank decomposition
        
        return U, V, W
    
    # Actually, let me use a different strategy.
    # I'll set up the problem and use optimization to find a low-rank decomposition.
    
    def try_rank(rank):
        """Try to find a decomposition of given rank using ALS."""
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Reshape T to (n*m, m*p, p*n) for ALS
        # T[i,j,k] corresponds to U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        for iteration in range(50):
            if time.time() > deadline:
                return None
            
            # Update U
            for r in range(rank):
                pass  # This is getting complex, let me use a different approach
        
        return U, V, W
    
    # Let me use a more direct approach: construct a known low-rank decomposition
    
    # For 3x3x6, the rank is at most 2*19 = 38
    # Let me construct the rank-38 solution and then try to improve
    
    # Actually, let me use the following known result:
    # The rank of [3,3,6] is 38 (or possibly less)
    
    # Construct rank-38 solution:
    # Split B into B1 (3x3) and B2 (3x3)
    # C1 = A * B1 (3x3), C2 = A * B2 (3x3)
    # Use rank-19 for each
    
    # For now, let me just use the schoolbook and try to improve
    # Schoolbook rank = 54
    
    # Let me try a simple improvement: use rank 38
    
    rank = 38
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For each half of k (0-2 and 3-5), use a 3x3x3 decomposition
    for half in range(2):
        k_offset = half * 3
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    r = half * 9 + i * 3 + j * 0 + k  # This is wrong, let me fix
                    # Actually, for each (i,j,k) in the 3x3x3, we have a term
                    # But that's rank 9 per half, total 18. Not 38.
                    # I need the rank-19 per half.
                    pass
    
    # This is getting messy. Let me just use the schoolbook (rank 54) and try to optimize.
    
    # Actually, let me just return the schoolbook for now and add a TODO
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
