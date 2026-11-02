"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists for the schoolbook decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    should equal 1 if i,j,k are valid indices (they always are) and 0 otherwise.
    Actually for matrix multiplication: (AB)[i,k] = sum_j A[i,j] B[j,k]
    The tensor is T[i,j,k] = delta (i.e., the multiplication tensor has T[i,j,k]=1 for all i,j,k
    in the standard indexing where we're computing the full product).
    
    Wait, let me reconsider. The multiplication tensor for C = AB where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor T has indices (i, j, k) where T[i,j,k] = 1 for all valid i, j, k.
    This is because the multiplication map is: for each (i,k), C[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor is the sum over j of A[i,j] * B[j,k] * C[i,k]... 
    
    Actually, the standard tensor decomposition of matrix multiplication:
    The tensor T is defined such that T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    This is because the multiplication bilinear form is:
    sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k] = 0 for all A,B with AB=C
    
    Hmm, let me think more carefully. The standard way:
    We want to decompose the multiplication tensor. The tensor M has:
    M[i, j, k] = 1 for all i, j, k (in the appropriate ranges)
    
    No wait. Let's use the standard convention from the task:
    W[k*n+i, r] - this tells us the output factor W has row index k*n+i.
    
    The decomposition says:
    C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  ... summed over j?
    
    Actually, the standard tensor decomposition for matrix multiplication:
    (AB)[i,k] = sum_j A[i,j] B[j,k]
    
    In tensor form, the multiplication tensor T has T[i,j,k] = 1 for all valid i,j,k.
    A rank-r decomposition is: T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    Wait, the indices need to be consistent. Let me re-read the task:
    "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    "The output factor follows the official trace convention W[k*n+i,r]"
    
    So:
    - U has rows indexed by (i, j) where i is row of A (0..n-1) and j is column of A / row of B (0..m-1)
    - V has rows indexed by (j, k) where j is row of B (0..m-1) and k is column of B (0..p-1)
    - W has rows indexed by (k, i) where k is column of B / column of C (0..p-1) and i is row of A / row of C (0..n-1)
    
    The decomposition is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k
    
    This is the standard multiplication tensor decomposition.
    """
    rank = len(U[0]) if U else 0
    # Convert to numpy for efficient verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # T[i,j,k] should equal 1 for all i, j, k
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j], V_np[j*p+k])
                # Wait, that's not right. We need the triple product.
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # This is element-wise product then sum
                prod = U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i]
                val = np.sum(prod)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_r(r, n, m, p, deadline):
    """
    Try to find a rank-r decomposition. This is a hard problem, so we'll use
    a heuristic approach: start from schoolbook and try to reduce rank.
    For now, we'll just return None unless we have a known decomposition.
    """
    # For 2x4x6, the schoolbook rank is 48.
    # Known bounds: the rank is at most n*m*p = 48.
    # There might be better decompositions, but finding them is non-trivial.
    
    # One approach: use the fact that we can sometimes combine terms.
    # For example, if we have terms that share structure, we might be able to merge them.
    
    # A simpler heuristic: try to find a low-rank approximation and then check if it's exact.
    # But this is very hard in general.
    
    # For now, let's just return the schoolbook decomposition if r >= n*m*p
    if r >= n * m * p:
        return _schoolbook(n, m, p)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = len(U[0])
    
    # Try to find a better decomposition
    # For now, we'll just return the schoolbook decomposition
    # since finding a better one is non-trivial and we need to be safe about time.
    
    # Check if we have time to do anything else
    if time.time() < deadline:
        # We could try various heuristics here, but for safety, let's just return schoolbook
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
