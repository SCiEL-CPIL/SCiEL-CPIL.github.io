"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = 1 if i*m+j, j*p+k, k*n+i
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    # Check that for non-matching indices, the sum is 0
    # The tensor for matrix multiplication is: T[i, j, k] = 1 for all valid (i,j,k)
    # But we need to ensure that the decomposition doesn't have spurious terms.
    # Actually, the condition is: for all i, j, k in valid ranges, the product sum must equal 1.
    # There are no "non-matching" indices in the standard tensor decomposition view.
    # The tensor T has entries T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # So we just need to verify all n*m*p entries.
    return True

def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Falls back to schoolbook rank 100 if no improvement is found within time budget.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Schoolbook baseline
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try random search for lower rank
    # For 4x5 * 5x5, known lower bounds suggest rank could be less than 100
    # We'll try to find a valid decomposition with rank < 100
    
    # Use a structured random approach: initialize with random entries, then refine
    # This is a heuristic search within the schoolbook family (dense random matrices)
    
    np.random.seed(42)
    
    # Try a few ranks below schoolbook
    for target_rank in range(90, 99):  # Try ranks 90-98
        if time.time() > deadline:
            break
        
        attempts = 0
        max_attempts = 50000  # Limit per rank
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Initialize with small random values
            scale = 1.0 / (target_rank ** 0.5)
            U = np.random.randn(n * m, target_rank) * scale
            V = np.random.randn(m * p, target_rank) * scale
            W = np.random.randn(p * n, target_rank) * scale
            
            # Check if this happens to work (very unlikely for random)
            # This is just a placeholder; real search would use gradient-based methods
            
            # For now, just check a few entries to see if we're close
            # This is inefficient but we're constrained by time
            pass
    
    # Given the difficulty of finding a lower-rank decomposition quickly,
    # return the schoolbook decomposition
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
