"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import random

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check that the tensor coefficient is correctly represented
                # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _construct_rank30():
    """Try to construct a rank-30 decomposition for 2x3x10."""
    n, m, p = 2, 3, 10
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For 2x3x10, we can use the following structure:
    # The key insight is that for 2 rows, we can share computations.
    # Let's try: for each j (0,1,2) and k (0..9), we need terms.
    # Total independent terms: 2*3*10 = 60 for schoolbook.
    # We want to reduce to 30.
    
    # One approach: use the fact that C[0,k] - C[1,k] only involves 
    # (A[0,j] - A[1,j]) * B[j,k], so we can combine.
    
    # Let's try a specific construction:
    # For j=0,1,2 and k=0..9:
    #   Term 1: U[0*3+0][c]=1, V[0*10+k][c]=1, W[k*2+0][c]=1  -> contributes to C[0,k] via A[0,0]*B[0,k]
    #   Term 2: U[0*3+1][c]=1, V[1*10+k][c]=1, W[k*2+0][c]=1  -> contributes to C[0,k] via A[0,1]*B[1,k]
    #   Term 3: U[0*3+2][c]=1, V[2*10+k][c]=1, W[k*2+0][c]=1  -> contributes to C[0,k] via A[0,2]*B[2,k]
    #   Term 4: U[1*3+0][c]=1, V[0*10+k][c]=1, W[k*2+1][c]=1  -> contributes to C[1,k] via A[1,0]*B[0,k]
    #   Term 5: U[1*3+1][c]=1, V[1*10+k][c]=1, W[k*2+1][c]=1  -> contributes to C[1,k] via A[1,1]*B[1,k]
    #   Term 6: U[1*3+2][c]=1, V[2*10+k][c]=1, W[k*2+1][c]=1  -> contributes to C[1,k] via A[1,2]*B[2,k]
    
    # That's 60 terms. To reduce, we need to find linear dependencies.
    
    # Alternative: use Strassen-like for the 2x3 part.
    # For 2x3 * 3x1, we can do it in rank 5 (known result).
    # Then for 3x10, we can share across the 10 outputs.
    
    # Actually, let's just return schoolbook for now and try optimization.
    # This is a placeholder - we'll use search below.
    return None

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Start with schoolbook as fallback
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
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find better decompositions
    # For 2x3x10, try ranks from 30 upward
    random.seed(42)
    
    # Try a systematic approach: use the structure of the problem
    # The tensor T_{i,j,k} = delta_{i,j,k} (all ones for valid indices)
    # We want to factor this as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # One known technique: for matrices with one small dimension, we can use
    # the "border rank" approach.
    
    # Let's try rank = 2*m*p = 60 (same as schoolbook, no gain)
    # Or rank = n*m*p - something
    
    # Try to find a rank-45 decomposition by combining terms
    # For each k, we have a 2x3x1 sub-tensor. The rank of 2x3x1 is 5.
    # So for 10 values of k, we might get 10*5 = 50, but with sharing maybe less.
    
    # Let me try a direct construction for rank 40:
    # Use the fact that we can combine terms across k values.
    
    # Actually, let's just try random search for lower ranks
    for target_rank in range(30, min(60, best_rank + 1)):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition of this rank
        # This is a hard problem, so we'll use a heuristic
        # Initialize with schoolbook and try to merge terms
        
        U = [row[:] for row in U_sb]
        V = [row[:] for row in V_sb]
        W = [row[:] for row in W_sb]
        
        # Truncate to target_rank
        U = [row[:target_rank] for row in U]
        V = [row[:target_rank] for row in V]
        W = [row[:target_rank] for row in W]
        
        # This won't work directly, need a different approach
        # Let's skip random search and just return schoolbook for now
        # unless we find a known construction
        
    # Return the best result found
    return best_result
# EVOLVE-BLOCK-END
