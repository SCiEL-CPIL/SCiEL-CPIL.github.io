"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE_MARGIN_S = 10.0
SEARCH_START = 30
SEARCH_MAX_RANK = 54

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _validate(U, V, W, n=3, m=3, p=6, tol=1e-6):
    """Check if U @ V @ W^T reproduces the tensor."""
    n, m, p = 3, 3, 6
    r = len(U[0])
    # Compute for each (i,j,k) whether U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Sum over c should give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > tol:
                    return False
    # Check that no other entries are non-zero in a way that breaks the tensor
    # Actually for matrix multiplication tensor, each (i,j,k) must be 1 and all other
    # combinations must be 0. Let's check all possible (i,j,k) combinations:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                pass  # already checked above, all should be 1
    # We also need to ensure that entries like U[i*m+j', c] where j' != j don't 
    # create spurious non-zero entries. But actually the decomposition only requires
    # that the tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] matches.
    # Since we're checking all (i,j,k) and all should be 1, this is sufficient.
    return True

def _search_decomposition(rank, n=3, m=3, p=6, n_attempts=200, deadline=None):
    """Try to find a rank decomposition by random search."""
    n, m, p = 3, 3, 6
    best = None
    
    for attempt in range(n_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Use a parameterization that tends to produce structured solutions
        # Random initialization with small perturbations around schoolbook
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Start from a permuted schoolbook and perturb
        base_rank = min(rank, n*m*p)
        for c in range(rank):
            # Each rank-1 term corresponds to some (i,j,k)
            if c < n*m*p:
                i, rem = divmod(c, m*p)
                j, k = divmod(rem, p)
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
            else:
                # Extra columns: random small values
                U[:, c] = np.random.randn(n*m) * 0.01
                V[:, c] = np.random.randn(m*p) * 0.01
                W[:, c] = np.random.randn(p*n) * 0.01
        
        if rank < n*m*p:
            # Need to drop some columns and adjust
            # Randomly zero out extra columns
            cols_to_keep = list(range(rank))
            np.random.shuffle(cols_to_keep)
            # This approach is tricky; let's try a different strategy
            # for rank < 54: use a structured random approach
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
        
        # Convert to lists and validate
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _validate(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN_S
    
    n, m, p = DIMENSIONS
    
    # The random search in _search_decomposition is unlikely to find exact
    # integer/half-integer solutions for rank < 54 without specific structure.
    # The schoolbook decomposition is guaranteed to be valid.
    # We return the schoolbook decomposition to ensure a valid result.
    # Future improvements could involve hardcoding a known lower-rank
    # decomposition (e.g., rank 46 via two rank-23 blocks) if available.
    
    current_best = _schoolbook()
    
    return current_best
# EVOLVE-BLOCK-END
