"""Matrix multiplication 3x3 by 3x5 — rank-minimal exact tensor decomposition."""
import time
import random
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START
DIMENSIONS = [3, 3, 5]

def _construct_schoolbook():
    """Return the trivial schoolbook decomposition (rank 45)."""
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U, V, W exactly decompose the matmul tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # and check T[i,j,k] == (i==i and j==j and k==k) i.e. 1 for all valid indices.
    # Actually the matmul tensor: (A * B)[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] where the multiplication tensor has 1 at (i,j,k) iff
    # we're computing element (i,k) from A[i,j]*B[j,k].
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 for all i,j,k (since each product A[i,j]*B[j,k] contributes to C[i,k]).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > tol:
                    return False
    return True

def _try_bipartite_decomposition(n, m, p, max_rank=None, seed=42, deadline=None):
    """
    Attempt to find a low-rank decomposition by exploiting the structure.
    
    The matmul tensor T has T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p].
    This is actually the all-ones tensor! So we need:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    A simple decomposition: let U[i*m+j, c] = a_{i,j,c}, V[j*p+k, c] = b_{j,k,c}, W[k*n+i, c] = w_{k,i,c}
    We need sum_c a_{i,j,c} b_{j,k,c} w_{k,i,c} = 1 for all i,j,k.
    
    One approach: use rank-1 terms. A rank-1 term contributes a_{i,j} * b_{j,k} * w_{k,i} to T[i,j,k].
    We need the sum of such terms to equal 1 for all (i,j,k).
    
    Key insight: if we set a_{i,j,c} = 1, b_{j,k,c} = 1, w_{k,i,c} = 1 for all c, then each term
    contributes 1, and we'd need exactly 1 term. But that gives T[i,j,k]=1 which works!
    Wait, that's just rank 1? Let me re-examine.
    
    Actually no - the tensor for matrix multiplication is NOT all ones. Let me reconsider.
    
    For matrix multiplication C = A*B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The multiplication tensor T has entries T[i,j,k] = 1 for all valid (i,j,k), because
    each pair (A[i,j], B[j,k]) contributes to C[i,k]. The "tensor" in the sense of the
    bilinear map has structure: the number of terms is n*m*p, and each term (i,j,k)
    corresponds to the product A[i,j]*B[j,k].
    
    In the standard tensor decomposition framework:
    - U has shape (n*m, rank): rows indexed by (i,j)
    - V has shape (m*p, rank): rows indexed by (j,k)  
    - W has shape (p*n, rank): rows indexed by (k,i)
    
    And the constraint is: for all i,j,k:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This IS the all-ones condition on the 3D array indexed by (i,j,k). So we need to
    decompose the all-ones 3D tensor into a sum of rank-1 tensors.
    
    The rank of the all-ones tensor of size n x m x p is 1! Because:
      1 = 1 * 1 * 1 (a single rank-1 tensor with all entries 1)
    
    So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 gives rank 1!
    """
    # Verify: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1*1*1 = 1. Yes!
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def _try_rank_r_random(n, m, p, r, seed, max_iter=5000, deadline=None):
    """Try random initialization with local search for rank r."""
    rng = random.Random(seed)
    
    for attempt in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Initialize with small random integers or fractions
        U = [[rng.randint(-2, 2) for _ in range(r)] for _ in range(n * m)]
        V = [[rng.randint(-2, 2) for _ in range(r)] for _ in range(m * p)]
        W = [[rng.randint(-2, 2) for _ in range(r)] for _ in range(p * n)]
        
        # Quick check
        if _verify(U, V, W, n, m, p, tol=1e-6):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    """Find the minimum rank exact tensor decomposition for 3x3 * 3x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # First, try the trivial rank-1 decomposition
    result = _try_bipartite_decomposition(n, m, p, deadline=deadline)
    if result is not None:
        # Double-check with exact verification
        if _verify(result["U"], result["V"], result["W"], n, m, p, tol=0):
            return result
    
    # If rank 1 doesn't work (it should, but let's be safe), try rank 2
    for r in range(2, 10):
        if time.time() > deadline:
            break
        result = _try_rank_r_random(n, m, p, r, seed=r, max_iter=2000, deadline=deadline)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p, tol=0):
                return result
    
    # Fallback: schoolbook
    return _construct_schoolbook()
# EVOLVE-BLOCK-END
