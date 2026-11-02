"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p  # 60

    # We'll try to find a lower rank decomposition.
    # The tensor T[i,j,k] = e_{i,j} * e_{j,k} * e_{k,i} (with trace convention W[k*n+i])
    # Actually, the matrix multiplication tensor is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] is 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The decomposition is: T = sum_c U[:,c] * V[:,c] * W[:,c]
    # where U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # T[i,j,k] = 1 for all valid i,j,k (it's the identity tensor in the sense that each
    # product term contributes to exactly one output)
    
    # Actually let me think about this more carefully.
    # The matrix multiplication tensor M_{i,j,k} where:
    # - i indexes rows of A (0..n-1)
    # - j indexes columns of A / rows of B (0..m-1)  
    # - k indexes columns of B (0..p-1)
    # M_{i,j,k} = 1 (it's a 0-1 tensor)
    # 
    # A decomposition of rank r means:
    # M_{i,j,k} = sum_{c=0}^{r-1} U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    #
    # The schoolbook decomposition has r = n*m*p with each term being a single 1.

    # Strategy: Try to find rank < 60 decompositions.
    # Known bounds: For 3x4x5, the lower bound is max(n*p, m*p, n*m) = max(15, 20, 12) = 20
    # Upper bound from schoolbook: 60
    # 
    # Let's try a systematic search for low-rank decompositions.
    # 
    # One approach: Start with the schoolbook and try to merge terms.
    # Another: Use the fact that we can have entries that are 0, 1, -1, or fractions.

    # Let's try a different approach: search for rank-48, rank-45, etc.
    # by trying to find structured decompositions.

    # First, let's try the known result: for m=4, we can split into two 2x2 blocks.
    # The 2x2x2 matrix multiplication tensor has rank 7 (Strassen).
    # If we partition the 4 middle indices into two groups of 2:
    # Group 1: j=0,1
    # Group 2: j=2,3
    # 
    # For each group, we have a 3x2x5 sub-problem... that's not quite right.
    # 
    # Let me think differently. The middle dimension is m=4.
    # We can write:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #        = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    #
    # The first sum involves A[:,0:2] (3x2) times B[0:2,:] (2x5)
    # The second sum involves A[:,2:4] (3x2) times B[2:4,:] (2x5)
    #
    # Each 3x2x5 multiplication has schoolbook rank 3*2*5 = 30.
    # Total: 60. No improvement.
    
    # Let's try a more sophisticated approach.
    # We'll search for decompositions with rank < 60 using random initialization
    # and gradient-free optimization (since the problem is non-convex).

    # Actually, let me try a constructive approach.
    # 
    # Consider the tensor as a 3-mode tensor of size (n*m) x (m*p) x (p*n) = 12 x 20 x 15.
    # We want to find rank-r decomposition.
    
    # Let's try rank values from low to high and see if we can find a valid decomposition.
    # For each rank r, we'll try random initializations.

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        # Check that for all i, j, k:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_fast(U, V, W, rank):
        """Fast verification using numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each c, we have a rank-1 tensor U[:,c] * V[:,c] * W[:,c]
        # We need the sum to equal the target tensor.
        # Target: T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
        
        # Reshape and check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.dot(U_arr[idx_u], V_arr[idx_v]) * W_arr[idx_w]  # This is wrong
                    # Actually it's: sum_c U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                    val = np.sum(U_arr[idx_u, :] * V_arr[idx_v, :] * W_arr[idx_w, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try to find decompositions with rank < 60
    # Start with some promising ranks
    for target_rank in [48, 45, 42, 40, 38, 36, 35, 34, 33, 32, 30, 28, 26, 24, 22, 20]:
        if time.time() > deadline - 10:
            break
        
        # Try several random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values in {-1, 0, 1}
            random.seed(attempt * 1000 + target_rank)
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify_fast(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}

    # If we didn't find a better decomposition, return the schoolbook
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
