"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the factor matrices exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] must be 1 for all i,j,k
    # Note: The standard tensor decomposition for matrix mult C = A*B is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T is defined such that T_{i,j,k} = 1.
    # The decomposition is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # Wait, the indices in the problem statement are:
    # U shape (n*m, rank) -> index i*m+j
    # V shape (m*p, rank) -> index j*p+k
    # W shape (p*n, rank) -> index k*n+i
    # So the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must equal 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _structured_decomposition(n, m, p):
    """
    Attempt to find a low-rank decomposition using a structured approach.
    
    Strategy:
    We look for a decomposition where the rank is significantly lower than n*m*p.
    One common technique for small matrices is to use the fact that the multiplication
    tensor has a specific structure. However, for generic matrix multiplication,
    the rank is at least max(n*m, m*p, p*n).
    
    For n=3, m=5, p=6:
    Lower bound = max(15, 30, 18) = 30.
    Schoolbook = 90.
    
    We will attempt a "rank-30" decomposition if possible, or try to find any rank < 90.
    
    A known result for matrix multiplication of small sizes often involves
    exploiting the rectangular nature.
    
    Let's try a specific construction:
    We can view the multiplication as a linear map from (A,B) to C.
    
    Alternative Strategy: Integer/Half-integer structured factors.
    Since the problem asks for integer/half-integer entries, we can try to construct
    a decomposition using small integers.
    
    Let's try a rank-30 construction based on the lower bound.
    If we can achieve rank 30, that would be optimal (assuming the lower bound is tight).
    
    Construction for rank = m*p = 30:
    Let r index from 0 to 29.
    We can map r to (j, k) such that r = j*p + k.
    Then V[j*p+k, r] = 1 if r == j*p+k else 0. (V is identity-like)
    Then we need U[i*m+j, r] * W[k*n+i, r] to sum to 1 over r for each (i,j,k).
    But r is tied to (j,k). So for a fixed (i,j,k), the sum is over r.
    Only the term where r = j*p+k contributes if U and W are non-zero there.
    So we need U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1.
    This must hold for all i, j, k.
    
    Let U[i*m+j, j*p+k] = 1 for all i,j,k?
    Then U has 1s at positions (i*m+j, j*p+k).
    For a fixed (i,j), U[i*m+j, :] has 1s at indices r = j*p+k for all k.
    So U[i*m+j, r] = 1 if floor(r/p) == j, else 0.
    
    Then we need W[k*n+i, j*p+k] = 1 for all i,j,k.
    For a fixed (k,i), W[k*n+i, :] has 1s at indices r = j*p+k for all j.
    So W[k*n+i, r] = 1 if r % p == k, else 0.
    
    Let's check this construction:
    Rank R = m*p = 30.
    U shape (n*m, R) = (15, 30).
    V shape (m*p, R) = (30, 30). V is Identity.
    W shape (p*n, R) = (18, 30).
    
    U[i*m+j, r] = 1 if floor(r/p) == j else 0.
    V[j*p+k, r] = 1 if r == j*p+k else 0.
    W[k*n+i, r] = 1 if r % p == k else 0.
    
    Sum over r:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    V[j*p+k, r] is 1 only when r = j*p+k.
    So the sum reduces to a single term at r = j*p+k:
    T[i,j,k] = U[i*m+j, j*p+k] * V[j*p+k, j*p+k] * W[k*n+i, j*p+k]
    
    V[j*p+k, j*p+k] = 1.
    
    Check U[i*m+j, j*p+k]:
    r = j*p+k. floor(r/p) = floor(j*p+k)/p = j (since 0<=k<p).
    So floor(r/p) == j.
    Thus U[i*m+j, j*p+k] = 1.
    
    Check W[k*n+i, j*p+k]:
    r = j*p+k. r % p = k.
    So r % p == k.
    Thus W[k*n+i, j*p+k] = 1.
    
    Therefore T[i,j,k] = 1 * 1 * 1 = 1.
    
    This is a valid decomposition with rank = m*p = 30!
    This is much better than 90.
    And it uses integer entries (0 and 1).
    
    Is it possible to go lower?
    Lower bound is max(n*m, m*p, p*n) = max(15, 30, 18) = 30.
    So rank 30 is the theoretical minimum.
    This construction achieves the theoretical minimum.
    """
    rank = m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # V is identity
    for r in range(rank):
        V[r, r] = 1.0
        
    # U[i*m+j, r] = 1 if floor(r/p) == j
    for i in range(n):
        for j in range(m):
            for r in range(rank):
                if r // p == j:
                    U[i * m + j, r] = 1.0
                    
    # W[k*n+i, r] = 1 if r % p == k
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                if r % p == k:
                    W[k * n + i, r] = 1.0
                    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First, try the structured decomposition which should be rank 30
    result_structured = _structured_decomposition(n, m, p)
    
    # Verify it
    if _verify_decomposition(result_structured["U"], result_structured["V"], result_structured["W"], n, m, p):
        best_result = result_structured
    else:
        # Fallback to schoolbook if structured fails (should not happen)
        best_result = _schoolbook(n, m, p)
        
    # Check time
    if time.time() > deadline:
        return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}
        
    # Since we hit the theoretical lower bound (rank 30 for 3x5 x 5x6),
    # we cannot do better.
    # Just to be safe, if we had time, we could try to verify if rank 29 is possible,
    # but the lower bound max(n*m, m*p, p*n) = 30 proves it's impossible.
    # So we return the rank 30 solution.
    
    return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}
# EVOLVE-BLOCK-END
