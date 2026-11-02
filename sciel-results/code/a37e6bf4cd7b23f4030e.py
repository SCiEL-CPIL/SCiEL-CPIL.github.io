"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    # Schoolbook decomposition as fallback (rank = n*m*p = 84)
    def schoolbook():
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

    def verify_fast(U, V, W):
        """Vectorized verification using numpy."""
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        # Check all entries equal 1
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def verify_integer(U, V, W):
        """Check if entries are integer or half-integer."""
        for mat in [U, V, W]:
            for row in mat:
                for x in row:
                    # Check if x is integer or half-integer
                    doubled = 2 * x
                    if abs(doubled - round(doubled)) > 1e-9:
                        return False
        return True

    best_result = schoolbook()
    best_rank = n * m * p

    import numpy as np

    # Strategy: Build structured decompositions using algebraic identities
    # Key insight: We can group terms to reduce rank below n*m*p
    
    # Approach 1: Group by one dimension
    # For each (i,k), we need sum_j A[i,j]*B[j,k] = 1
    # We can try to share computation across i or k
    
    # Approach 2: Use the fact that for fixed j, the contribution is
    # A[i,j] * B[j,k] which is a rank-1 tensor in (i,k) space
    # So we need to represent the all-ones matrix as sum of rank-1 matrices
    
    # For n x p output, we need to write J (all ones) = sum_c u_c * v_c^T
    # where u_c is n-vector and v_c is p-vector
    # The rank of J is 1, but we're constrained by the j-index coupling
    
    # Better approach: Use structured random search with gradient-based refinement
    
    # Value pool for half-integer entries - keep it small for speed
    value_pool = [0, 1, -1, 0.5, -0.5]
    
    # Try ranks from lower bound up to current best
    for target_rank in range(28, min(best_rank, 45)):
        if time.time() > deadline:
            break
        
        # Limit attempts per rank to stay within time budget
        max_attempts = 50 if target_rank < 35 else 20
        
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Use structured initialization based on attempt number
            if attempt < 10:
                # Binary {-1, 0, 1}
                pool = [0, 1, -1]
            elif attempt < 20:
                # Include halves
                pool = [0, 1, -1, 0.5, -0.5]
            else:
                # Sparse binary
                pool = [0, 1, -1]
            
            # Generate random matrices
            U = [[random.choice(pool) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(pool) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(pool) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Quick sparsity check
            if all(all(x == 0 for x in row) for row in U):
                continue
            if all(all(x == 0 for x in row) for row in V):
                continue
            if all(all(x == 0 for x in row) for row in W):
                continue
            
            if verify_fast(U, V, W):
                if verify_integer(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    # If random search didn't find improvement, try structured construction
    # using known algebraic techniques for matrix multiplication
    
    # For 3x4x7, try a decomposition based on grouping
    # We can write the multiplication as a sum of products where
    # we share some factors
    
    # Construct a decomposition with rank = m * max(n,p) = 4 * 7 = 28
    # This uses the identity: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # Group by j: for each j, we have a rank-1 tensor in (i,k)
    # But we need to couple with the j-index
    
    # Actually, the schoolbook already gives rank 84 = n*m*p
    # To do better, we need to find linear dependencies
    
    # Try a more clever construction:
    # Use the fact that we can combine terms across i or k
    
    # For now, return the best result found
    return best_result
# EVOLVE-BLOCK-END
