"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook gives rank 20. Try to find lower rank.
    # The multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # We need: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Try ranks from 15 upward, search for valid decomposition
    deadline = time.time() + 110  # safety margin before 120s limit
    
    for rank in range(15, 21):
        if time.time() > deadline:
            break
        result = _search_rank(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback: schoolbook rank 20
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _search_rank(n, m, p, rank, deadline):
    """Search for a rank-`rank` decomposition with small integer/half-integer entries."""
    # The tensor has n*m*p = 20 entries that must be matched.
    # Each rank-1 term contributes U[idx1]*V[idx2]*W[idx3] to specific positions.
    # 
    # Strategy: Fix U and V to have simple structure, solve for W, or use random search.
    # For small ranks, try structured patterns.
    
    # Known: for 2x2x2, Strassen gives rank 7.
    # For 2x2x5, we can potentially combine.
    
    # Let's try a direct approach: random initialization + local search
    # with the constraint that entries are multiples of 0.5.
    
    import random
    random.seed(rank * 12345 + 67890)
    
    # We have rank variables for each of U (4*rank), V (10*rank), W (10*rank)
    # Total: 24*rank variables. We need 20 equations (one per tensor entry).
    # The tensor entries T[i,j,k] are: for each (i,j,k), the coefficient is 1
    # if it's a valid multiplication term, i.e., always 1 for the 20 terms.
    
    # Actually the tensor has exactly 20 non-zero entries, each = 1.
    # T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    
    # So we need: for each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # This is a system of 20 polynomial equations in 24*rank variables.
    
    # Try random search with small values
    max_iters = 50000
    start = time.time()
    
    for attempt in range(max_iters):
        if time.time() > deadline:
            return None
        
        # Random initialization with values from {-1, 0, 0.5, 1}
        U = [[random.choice([-1, -0.5, 0, 0, 0, 0.5, 1]) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.choice([-1, -0.5, 0, 0, 0, 0.5, 1]) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.choice([-1, -0.5, 0, 0, 0, 0.5, 1]) for _ in range(rank)] for _ in range(p*n)]
        
        # Check if this works
        if _check_decomp(n, m, p, U, V, W):
            return {"U": U, "V": V, "W": W}
        
        # Local search: perturb one entry
        if attempt % 10 == 0:
            # Perturb a random entry
            which = random.choice(['U', 'V', 'W'])
            if which == 'U':
                idx = random.randrange(n*m)
            elif which == 'V':
                idx = random.randrange(m*p)
            else:
                idx = random.randrange(p*n)
            col = random.randrange(rank)
            mat = {'U': U, 'V': V, 'W': W}[which]
            old = mat[idx][col]
            mat[idx][col] = random.choice([-1, -0.5, 0, 0.5, 1])
            if not _check_decomp(n, m, p, U, V, W):
                mat[idx][col] = old
    
    return None


def _check_decomp(n, m, p, U, V, W):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
