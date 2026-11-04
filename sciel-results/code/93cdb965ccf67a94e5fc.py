<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        
        # M3
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M6
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # Part 1: 2x4x4
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # Part 2: 2x4x2
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # Part 3: 2x4x1
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import random
import itertools

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _greedy_search(n, m, p, max_rank=49, time_limit=120):
    """Try to find a lower-rank decomposition using greedy search."""
    start_time = time.time()
    
    # Build the target tensor
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i,j,k)] = 1.0
    
    # Start with rank-1 terms
    U = []
    V = []
    W = []
    
    # Greedily add rank-1 terms to cover the tensor
    remaining = set(T.keys())
    
    for rank in range(1, max_rank + 1):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Find the best rank-1 term that covers the most remaining entries
        best_term = None
        best_count = 0
        
        # Sample some candidate rank-1 terms
        candidates = []
        
        # Try all possible (i,j) for U, (j,k) for V, (k,i) for W
        # This is too many, so we'll be smart about it
        
        # For each remaining (i,j,k), we can try to cover it with a rank-1 term
        # where U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
        # But this might conflict with other terms
        
        # Instead, let's try a different approach: start from the known rank-50
        # decomposition and try to merge or eliminate terms
        
        if rank >= 45:
            # Use the constructive approach as a starting point
            U, V, W = _constructive_decomposition(n, m, p)
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W, len(U[0])
    
    return None

def _constructive_decomposition(n, m, p):
    """Construct the rank-50 decomposition."""
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        
        # M3
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M6
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # Part 1: 2x4x4
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # Part 2: 2x4x2
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # Part 3: 2x4x1
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 135  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # First, get the constructive rank-50 decomposition
    U, V, W = _constructive_decomposition(n, m, p)
    current_rank = len(U[0])
    
    # Try to reduce rank by merging or eliminating terms
    # Check if any two rank-1 terms can be combined
    for attempt in range(200):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try to find redundant terms
        # If two terms have the same U, V, W patterns, they can be merged
        # Or if a term is zero, remove it
        
        # Check for zero terms
        new_U = []
        new_V = []
        new_W = []
        for r in range(current_rank):
            u_norm = sum(abs(x) for x in U[r])
            v_norm = sum(abs(x) for x in V[r])
            w_norm = sum(abs(x) for x in W[r])
            if u_norm > 0 and v_norm > 0 and w_norm > 0:
                new_U.append(U[r])
                new_V.append(V[r])
                new_W.append(W[r])
        
        if len(new_U) < current_rank:
            U, V, W = new_U, new_V, new_W
            current_rank = len(U)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I made a mistake in the verification function - I'm indexing U, V, W incorrectly. Let me fix that and also reconsider the approach.

The issue is that the tensor decomposition requires:
T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]

Let me rewrite this more carefully:

