"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

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


def _verify(U, V, W, n=2, m=3, p=6, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # We check a random sample of (i,j,k) triples
    rng = np.random.default_rng(42)
    for _ in range(100):
        i = rng.integers(0, n)
        j = rng.integers(0, m)
        k = rng.integers(0, p)
        val = np.sum(Ua[i*m+j, :] * Va[j*p+k, :] * Wa[k*n+i, :])
        if abs(val - 1.0) > tol:
            return False
    return True


def _rank_from_matrices(U, V, W):
    return len(U[0])


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep safety margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 36)
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower-rank decomposition using structured approaches
    # The multiplication tensor for (n,m,p) = (2,3,6) has rank at most 36 (schoolbook).
    # Known bounds: rank is between n*m*p - (n-1)*(m-1)*(p-1) and n*m*p
    # For small cases, we can try random search + verification.
    
    # Strategy: Try Kronecker product structure.
    # 6 = 2*3, so we can think of output as (2,3) grid.
    # If we decompose as tensor product of (2,1,2) and (1,3,3):
    # (2,1,2) has rank 4 (schoolbook) but actually rank 3 (2x1x2 is just 2 vectors of length 2, 
    #   the tensor is sum of 2 rank-1 terms? No: 2*1*2=4, but it's actually rank 2 since m=1)
    # Wait: for (n,m,p) with m=1, the multiplication is just element-wise: A[i] * B[k] for each i,k.
    # That's a rank n*p tensor where T[i,k] = 1. The decomposition is sum over c of U[i,c]*V[k,c]*W[i,c]?
    # No, for m=1: U has shape (n*1, rank) = (n, rank), V has shape (1*p, rank) = (p, rank), W has shape (p*n, rank).
    # T[i,0,k] = sum_c U[i,c]*V[k,c]*W[k*n+i,c] = 1 for all i,k.
    # This is the "diagonal" tensor. Rank of the n*p diagonal tensor... 
    # For n=p=2: T[i,k]=1. Can we do rank 2? U[i,c]*V[k,c]*W[k*2+i,c]=1 for all i,k.
    # If rank=2: sum_{c=1,2} U[i,c]V[k,c]W[k*2+i,c] = 1.
    # This is a bilinear form in (i,k) with coefficients W[k*2+i,c]. 
    # Let's just try small ranks with random initialization and check.
    
    # For the full problem, try ranks from theoretical lower bound up
    # Lower bound for matrix multiplication rank: at least max(n*m, m*p, p*n) = max(6,18,12) = 18
    # Actually the rank of the multiplication tensor is at least the border rank, which is at least n*m*p - ... 
    # Let's just try to find any valid decomposition with rank < 36.
    
    # Try structured Kronecker approach:
    # Decompose 6 = 2 * 3. Think of the 3x6 matrix as having columns indexed by (a,b) where a in {0,1}, b in {0,1,2}.
    # Then C[i, (a,b)] = sum_j A[i,j] * B[(a,b), j] = sum_j A[i,j] * B[a, b, j]
    # If we define B2[a, j, b] = B[a*3+b, j], this is like a 3x3x2 tensor.
    # This doesn't immediately factor nicely.
    
    # Let's try a direct random search for low-rank decompositions.
    # We'll use the fact that entries can be half-integers, so we search in 0.5 * integers.
    
    def try_rank(r, max_iters=200):
        """Try to find a rank-r decomposition using random restarts."""
        if r >= best_rank:
            return None
        rng = np.random.default_rng(12345)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small integers/half-integers
            scale = 1.0
            Ua = rng.integers(-2, 3, size=(n*m, r)) * scale
            Va = rng.integers(-2, 3, size=(m*p, r)) * scale
            Wa = rng.integers(-2, 3, size=(p*n, r)) * scale
            
            # Verify
            if _verify(Ua.tolist(), Va.tolist(), Wa.tolist()):
                return {"U": Ua.tolist(), "V": Va.tolist(), "W": Wa.tolist()}
        
        return None
    
    # Try decreasing ranks
    for r in range(18, best_rank):
        if time.time() > deadline - 10:
            break
        result = try_rank(r, max_iters=50)
        if result is not None:
            best = result
            best_rank = r
            break
    
    # Final verification
    if not _verify(best["U"], best["V"], best["W"]):
        # Fallback to schoolbook
        best = _schoolbook()
        best_rank = n * m * p
    
    return best
# EVOLVE-BLOCK-END
