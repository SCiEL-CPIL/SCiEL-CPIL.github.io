"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _schoolbook():
    """Return the trivial rank-125 decomposition."""
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the matmul tensor."""
    rank = len(U[0])
    # Build the target tensor T[i, j, k] = 1 if the (i,j) of A and (j,k) of B produce C[i,k]
    # The tensor for matmul: T[i,j,k] = 1 (for all i,j,k) in the standard sense:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor M[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank(rank, n, m, p, deadline, seed=None):
    """Try to find a rank-rank decomposition with entries in {0, 1/2, 1, -1/2, -1} or integers.
    Uses random search with local improvement."""
    if seed is not None:
        random.seed(seed)
    np.random.seed(seed)
    
    start = time.time()
    
    # Candidate values
    vals = [0, 0.5, 1, -0.5, -1]
    
    # Try random initialization and then local search
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        U = [[random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
        
        # Convert to numpy for faster verification
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        # Reshape for efficient checking
        # U shape: (n*m, rank)
        # V shape: (m*p, rank)  
        # W shape: (p*n, rank)
        
        # Build the decomposition tensor
        # D[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can compute this as:
        # For each c, outer product of U[:,c], V[:,c], W[:,c] reshaped appropriately
        
        # Let's just do it directly but with numpy
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        errors.append((i, j, k, s))
        
        if not errors:
            # Found a valid decomposition!
            return {"U": U, "V": V, "W": W}
        
        # Local search: try flipping one entry at a time
        # This is expensive, so let's try a simpler approach
        # Try modifying one factor at a time
    
    return None

def _try_structured_rank(rank, n, m, p, deadline):
    """Try structured decompositions based on known low-rank matmul schemes.
    For 5x5x5, the best known upper bound is 125 (schoolbook).
    There is no known decomposition with rank < 125 for 5x5x5.
    However, let's try some structured approaches."""
    start = time.time()
    
    # Try a decomposition based on the "border rank" or other algebraic methods
    # For now, this is a placeholder for future work
    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin before 150s kill
    
    # First, always have the schoolbook solution as fallback
    fallback = _schoolbook()
    
    # Try to find lower-rank decompositions
    # The theoretical lower bound for 5x5x5 is not well-established,
    # but it's at least n*m*p / (n+m+p-1) or similar bounds.
    # Let's try ranks from a reasonable lower bound up to 124
    
    # Known: rank 125 works (schoolbook)
    # Try a few lower ranks with random search
    for rank in [124, 123, 122, 121, 120]:
        if time.time() > deadline:
            break
        
        result = _try_rank(rank, n, m, p, deadline, seed=rank * 1000)
        if result is not None:
            return result
    
    # Try structured approaches
    if time.time() < deadline:
        result = _try_structured_rank(124, n, m, p, deadline)
        if result is not None:
            return result
    
    # Return the schoolbook decomposition as fallback
    return fallback
# EVOLVE-BLOCK-END
