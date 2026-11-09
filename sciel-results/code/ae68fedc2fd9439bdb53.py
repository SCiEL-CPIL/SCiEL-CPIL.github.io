"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for c in range(rank):
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
    # Verify: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # And for all other (i, j, k) combinations... wait, that's not right.
    # The tensor T[i,j,k] = 1 if this is a valid multiplication term, 0 otherwise.
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k in range(n)
    # No wait. The matrix multiplication tensor is defined as:
    # T(A,B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor has entries T[i,j,k] where the "output" index is (i,k) and "input" indices are (j) for A and (j) for B.
    # In the standard tensor decomposition formulation:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T has indices (i, j, k) with T[i,j,k] = 1 (it's the all-ones tensor in this context? No...)
    # Actually, the matrix multiplication tensor is T[i,j,k] = delta_{i,k}... no.
    # Let me reconsider. The standard formulation:
    # We have three sets of indices: i (row of C, row of A), j (col of A, row of B), k (col of C, col of B)
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T has entries T[i,j,k] = 1 for all i,j,k. Wait, no.
    # T is a 3-tensor where T[i,j,k] is the coefficient relating A_{i,j}, B_{j,k} to C_{i,k}.
    # So T[i,j,k] = 1 if we're computing C_{i,k} from A_{i,j} and B_{j,k}, which is always the case.
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # And the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    # This is expensive to check directly for large rank. Let's use a smarter approach.
    # Actually, for the schoolbook decomposition, this is trivially true.
    # For verification, let's just check a random subset or all if small enough.
    
    # For n=m=p=5, there are 125 (i,j,k) triples. With rank up to 125, that's 125*125 = 15625 operations.
    # That's fast enough.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_random_search(n, m, p, max_rank, num_trials, deadline):
    """Try random integer/half-integer decompositions."""
    import random
    import numpy as np
    
    best_result = None
    best_rank = max_rank + 1
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Try ranks from max_rank down to some minimum
        # Start with a reasonable range
        for rank in range(max_rank, 100, -1):
            if time.time() > deadline:
                break
            
            # Generate random sparse matrices
            # Each entry is 0 with high probability, otherwise ±1 or ±0.5
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            
            # Make them sparse
            density = 0.3
            for i in range(n * m):
                for c in range(rank):
                    if random.random() < density:
                        val = random.choice([1, -1, 0.5, -0.5])
                        U[i][c] = val
            for j in range(m * p):
                for c in range(rank):
                    if random.random() < density:
                        val = random.choice([1, -1, 0.5, -0.5])
                        V[j][c] = val
            for k in range(p * n):
                for c in range(rank):
                    if random.random() < density:
                        val = random.choice([1, -1, 0.5, -0.5])
                        W[k][c] = val
            
            if _verify(U, V, W, n, m, p):
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U, V, W)
                    return best_result
    return best_result

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    deadline = time.time() + 140  # Leave 10s margin
    
    # Try to find better decompositions
    result = _try_random_search(n, m, p, best_rank - 1, 1000, deadline)
    if result is not None:
        U, V, W = result
        best_U, best_V, best_W = U, V, W
        best_rank = len(U[0])
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
