"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Check if U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    # Build the multiplication tensor from U, V, W
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] should be 1 if this is a valid decomposition
    # Actually, the decomposition means: C[i,k] = sum_j A[i,j]*B[j,k]
    # which corresponds to tensor T[i,j,k] = delta (the multiplication tensor)
    # T[i,j,k] = 1 for all i,j,k in the standard basis sense
    
    # The tensor is: for each (i,j,k), the coefficient is sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # This should equal 1 for the multiplication tensor (since A[i,j]*B[j,k] contributes to C[i,k])
    
    # Actually let me think more carefully. The multiplication tensor T has:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    # Start with schoolbook
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try structured decompositions with reduced rank
    # Strategy: use patterns where some entries are shared
    
    # Approach 1: Try rank = n*m + m*p + n*p - n - m - p + 1 style patterns
    # This is related to the "border rank" and structured decompositions
    
    # Let's try a grid search over possible structured decompositions
    # A common approach: fix the sparsity pattern of U, V, W and solve for values
    
    # For now, try random search with verification
    random.seed(42)
    
    for attempt in range(10000):
        if time.time() > deadline:
            break
        
        # Try various ranks
        # Known lower bounds: max(n*m, m*p, n*p) = max(20, 25, 20) = 25
        # Upper bound from schoolbook: 100
        # Let's try ranks from 25 to 50
        
        target_rank = 25 + (attempt % 26)
        
        # Random initialization with sparsity
        U = [[0.0] * target_rank for _ in range(n * m)]
        V = [[0.0] * target_rank for _ in range(m * p)]
        W = [[0.0] * target_rank for _ in range(p * n)]
        
        # Initialize with sparse random values
        for i in range(n * m):
            for c in range(target_rank):
                if random.random() < 0.3:
                    U[i][c] = random.choice([-1, 0, 1])
        
        for j in range(m * p):
            for c in range(target_rank):
                if random.random() < 0.3:
                    V[j][c] = random.choice([-1, 0, 1])
        
        for k in range(p * n):
            for c in range(target_rank):
                if random.random() < 0.3:
                    W[k][c] = random.choice([-1, 0, 1])
        
        # Verify
        if _verify(U, V, W, n, m, p):
            if target_rank < best_rank:
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                if best_rank <= 25:
                    break
    
    # Convert to lists of numbers (keep as is since we're using floats)
    return best
# EVOLVE-BLOCK-END
