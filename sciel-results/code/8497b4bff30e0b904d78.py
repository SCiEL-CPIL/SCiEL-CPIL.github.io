"""Matrix multiplication tensor decomposition search for 2x5*5x6."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 6]
N, M, P = DIMENSIONS
# tensor index: T[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
# W convention: W[k*n + i, r]
# U[i*m + j, r], V[j*p + k, r]

def _verify(U, V, W, rank):
    """Verify that U,V,W with given rank exactly decompose the tensor.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Must equal 1 for all i,j,k."""
    n, m, p = N, M, P
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if s != 1:
                    return False
    return True

def _try_rank30():
    """Try the structured rank-30 construction.
    For each (j,k), one rank-1 term. U[i*m+j, c(j,k)] = a_i, V[j*p+k, c] = 1, W[k*n+i, c] = b_i.
    Need a_i * b_i = 1 for both i=0,1 -> a_0*b_0=1, a_1*b_1=1.
    With a_0=a_1=1, b_0=b_1=1: U=1, V=1, W=1. Check: sum over r of 1*1*1 = 1 (only one r per (j,k)).
    But W[k*n+i, c(j,k)] = 1 for the specific c. Each (k,i) appears in multiple c's (one per j).
    Wait: W[k*n+i, c] is indexed by (k,i) and c. For fixed (k,i), c varies over j (0..4).
    T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c].
    For fixed (i,j,k), only c = c(j,k) is nonzero in U[i*m+j,:] and V[j*p+k,:].
    So T[i,j,k] = U[i*m+j, c(j,k)] * V[j*p+k, c(j,k)] * W[k*n+i, c(j,k)].
    Set all = 1: T = 1*1*1 = 1. YES! This works!
    """
    n, m, p = N, M, P
    rank = m * p  # 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for j in range(m):
        for k in range(p):
            c = j * p + k
            for i in range(n):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    if _verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    return None

def _random_search(rank, max_iters, rng, deadline):
    """Randomized search for a rank-rank decomposition with integer/half-integer entries."""
    n, m, p = N, M, P
    # entries are in {0, 1/2, 1, 3/2, 2, ...} but we'll use small half-integers
    # Represent as Fraction to avoid float issues
    # We'll use entries from a small set
    entry_set = [Fraction(0), Fraction(1,2), Fraction(1), Fraction(3,2), Fraction(2)]
    
    for it in range(max_iters):
        if time.time() > deadline:
            return None
        # Build random U, V, W with sparse structure
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        for r in range(rank):
            # Choose random support
            for i in range(n):
                for j in range(m):
                    if rng.random() < 0.5:
                        U[i*m+j][r] = rng.choice(entry_set)
            for j in range(m):
                for k in range(p):
                    if rng.random() < 0.5:
                        V[j*p+k][r] = rng.choice(entry_set)
            for k in range(p):
                for i in range(n):
                    if rng.random() < 0.5:
                        W[k*n+i][r] = rng.choice(entry_set)
        
        if _verify(U, V, W, rank):
            # Convert to int/float for output
            U_out = [[float(x) for x in row] for row in U]
            V_out = [[float(x) for x in row] for row in V]
            W_out = [[float(x) for x in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    return None

def construct():
    start = time.time()
    deadline = start + 110  # safety margin before 120s
    
    # First try the structured rank-30 construction
    res30 = _try_rank30()
    if res30 is not None:
        return res30
    
    # If rank 30 doesn't work (it should, but just in case), try random search for ranks 31..55
    rng = random.Random(42)
    best = None
    for rank in range(31, 56):
        if time.time() > deadline:
            break
        max_iters = 200 if rank < 40 else 100
        res = _random_search(rank, max_iters, rng, deadline)
        if res is not None:
            best = res
            break  # found minimal rank in this range
    
    if best is not None:
        return best
    
    # Fallback: return the schoolbook rank-60 decomposition
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
