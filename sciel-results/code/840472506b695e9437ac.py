"""Matrix multiplication 2x5x6 - rank-minimal exact tensor decomposition."""
import time
import random
import itertools

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n=2, m=5, p=6):
    """Return the standard rank-nmp decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W, "rank": rank}

def _verify_decomposition(U, V, W, n=2, m=5, p=6):
    """Check if U,V,W correctly decompose the matrix multiplication tensor.
    For each output element (k,i) in C = A*B where A is n x m, B is m x p, C is n x p:
    Wait, the convention is W[k*n+i, r]. Let me re-read.
    
    The task says: output factor follows the official trace convention W[k*n+i,r].
    A is 2x5 (n x m), B is 5x6 (m x p), C is 2x6 (n x p).
    
    So C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,j,k] = 1 (the multiplication tensor).
    
    Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, let me check the convention. W[k*n+i, r] means W is indexed by (k,i) pairs.
    But C has dimensions n x p, so C[i,k] for i in range(n), k in range(p).
    
    So the W index k*n+i corresponds to output position (i,k) in C.
    
    Let me verify: for each (i,j,k), we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    And for all other triples, the sum should be 0.
    
    Actually, the tensor for matrix multiplication has T[i,j,k] = 1 iff the (i,k) element
    of C depends on A[i,j] and B[j,k]. Since C[i,k] = sum_j A[i,j]B[j,k],
    the tensor entry T[i,j,k] is 1 for all valid (i,j,k) and 0 otherwise.
    But actually all (i,j,k) with i<n, j<m, k<p are valid, and T[i,j,k]=1 for all of them.
    
    Wait no. The tensor is T[i,j,k] = 1 if and only if i,j,k form a valid multiplication step.
    For standard matrix mult, every (i,j,k) contributes to exactly one output C[i,k].
    So T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    
    The decomposition requires:
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    And we need this to hold for ALL (i,j,k) in the range.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _try_rank_reduction(n=2, m=5, p=6, max_rank=59, time_limit=140):
    """Attempt to find a decomposition with rank < n*m*p using random search."""
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    full_rank = n * m * p
    
    # We'll try to find a rank-r decomposition by random sampling
    # Strategy: start with schoolbook, then try to reduce rank by
    # finding linear dependencies and eliminating terms
    
    # For small dimensions, we can try exact enumeration of small structures
    # Let's try rank = full_rank - 1, full_rank - 2, etc.
    
    for target_rank in range(full_rank - 1, max(max_rank, 0), -1):
        if time.time() - start_time > time_limit:
            break
        
        attempts = 0
        max_attempts = 200
        
        while attempts < max_attempts:
            if time.time() - start_time > time_limit:
                break
            attempts += 1
            
            # Generate random decomposition
            # U: nm x target_rank, V: mp x target_rank, W: pn x target_rank
            # We want: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            
            # Random initialization
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(pn)]
            
            # Check if it works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(target_rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U, "V": V, "W": W, "rank": target_rank}
    
    return None

def construct():
    """Construct the tensor decomposition for 2x5x6 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' keys containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    # First, try to find a reduced-rank decomposition
    try:
        result = _try_rank_reduction(n, m, p, max_rank=n*m*p - 1, time_limit=110)
        if result is not None:
            return {"U": result["U"], "V": result["V"], "W": result["W"]}
    except Exception:
        pass
    
    # Fall back to schoolbook decomposition (guaranteed valid, rank = n*m*p = 60)
    sb = _schoolbook_decomposition(n, m, p)
    return {"U": sb["U"], "V": sb["V"], "W": sb["W"]}
# EVOLVE-BLOCK-END
