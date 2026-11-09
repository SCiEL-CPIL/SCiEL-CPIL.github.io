import time
import random
from itertools import product
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline guard
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    
    # Build the schoolbook decomposition as baseline
    def get_schoolbook():
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
        return U, V, W
    
    U_school, V_school, W_school = get_schoolbook()
    
    # Verify function
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    if not verify(U_school, V_school, W_school, n, m, p):
        raise ValueError("Schoolbook verification failed")
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower-rank decompositions using structured ansätze
    # Strategy: Try tensors with entries from a small set {-1, 0, 0.5, 1}
    # and search for valid decompositions at various ranks
    
    random.seed(42)
    
    # Try ranks from 20 to 44
    # For each rank, try random structured tensors
    
    # Use a smarter approach: exploit the tensor structure
    # The tensor has a specific pattern: T[i,j,k] = 1 for all valid (i,j,k)
    
    # Try to use the fact that we can combine terms
    # For example, if we have terms (i1,j,k1) and (i2,j,k2) with the same j,
    # we might be able to combine them if i1=i2 or k1=k2
    
    # Let's try a systematic search with limited randomness
    
    # Try rank 44 first (just remove one term and see if we can compensate)
    # This is unlikely to work directly, but let's try structured modifications
    
    # Better approach: Try to find a decomposition using bilinear identities
    # For 3x3x3, we know rank 7 is achievable (Strassen)
    # For 3x3x5, we can try to extend this
    
    # Let's try to construct a rank-r decomposition where r < 45
    # by using the following idea:
    # Group the k-index into blocks and use structured decompositions
    
    # For now, try random search with small entry sets
    entry_set = [-1, 0, 0.5, 1]
    
    # Try several ranks
    for target_rank in [40, 35, 30, 25, 20]:
        if time.time() > deadline - 10:
            break
        
        # Try random structured tensors
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            U = [[random.choice(entry_set) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(entry_set) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(entry_set) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
    
    # Try a more structured approach: use the known Strassen decomposition for 3x3x3
    # and extend it to 3x3x5
    
    # Strassen 3x3x3 decomposition (rank 7)
    # This is complex, so let's try a simpler structured approach
    
    # Try to use the fact that for each i, the sum over j and k has structure
    # Specifically, (AB)[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can try to use fewer than m terms per (i,k) by using linear combinations
    # For example, if we define C[i] = A[i,:] (row i of A) and D[k] = B[:,k] (column k of B)
    # then (AB)[i,k] = C[i] · D[k]
    
    # This is a dot product, which can be computed with m multiplications
    # But we can potentially use fewer if we exploit structure in C or D
    
    # For now, return the best found
    return {"U": best_U, "V": best_V, "W": best_W}
