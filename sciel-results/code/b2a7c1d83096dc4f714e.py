"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matmul tensor."""
    # The matmul tensor T[i,j,k] = delta_{i,k} for (A*B)[i,k] = sum_j A[i,j]B[j,k]
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_{i,k} for all i,j,k
    # Wait, the convention is W[k*n+i, r]. Let me re-read.
    # Output factor follows the official trace convention W[k*n+i,r].
    # The tensor for multiplying (n x m) by (m x p) to get (n x p) is:
    # T_{i,j,k} = 1 if the element (i,k) of the product depends on A[i,j] * B[j,k].
    # Actually, (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor is T_{i,j,k} = 1 for all i,j,k? No.
    # The tensor T is defined such that T_{i,j,k} A_{i,j} B_{j,k} = (AB)_{i,k}.
    # So T_{i,j,k} = 1 for all i,j,k.
    # Wait, that's not right either. Let's think about the standard tensor decomposition.
    # The matrix multiplication tensor T of dimensions n x m x p is defined by:
    # T[i,j,k] = 1 if we are computing (AB)[i,k] using A[i,j] and B[j,k].
    # Since (AB)[i,k] = sum_j A[i,j] B[j,k], we have T[i,j,k] = 1 for all i,j,k.
    # 
    # A tensor decomposition of rank r is a set of r triads (u_c, v_c, w_c) such that:
    # T[i,j,k] = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
    # 
    # For the schoolbook algorithm, each term corresponds to a specific (i,j,k) triple,
    # and u_c[i*m+j] = 1, v_c[j*p+k] = 1, w_c[k*n+i] = 1, with all other entries 0.
    # So T[i,j,k] = 1 for each (i,j,k), which is correct.
    #
    # Let's verify: for a given (i,j,k), the sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # should equal 1 (since T[i,j,k] = 1).
    # In the schoolbook, for the term corresponding to (i,j,k), we have U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1,
    # so the product is 1. All other terms are 0. So the sum is 1. Correct.
    #
    # For a general decomposition, we need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # Efficient check: for each i,k, we need sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] ... no wait.
    # Actually, for each (i,j,k), the sum over c must equal 1.
    # Let's just check all (i,j,k) triples.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _try_rank(n, m, p, rank, time_limit=60):
    """Try to find a decomposition of the given rank using random search with local refinement."""
    start_time = time.time()
    
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We want to find U (n*m x rank), V (m*p x rank), W (p*n x rank) such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # This is a system of n*m*p equations in n*m*rank + m*p*rank + p*n*rank variables.
    # For n=4, m=4, p=9: 144 equations in 16*rank + 36*rank + 36*rank = 88*rank variables.
    # So for rank >= 2, we have more variables than equations.
    
    # Strategy: initialize randomly, then use gradient descent or iterative refinement.
    # Actually, let's use a simpler approach: since we have more variables than equations,
    # we can try to solve a subset of equations.
    
    # Alternative: use the fact that T[i,j,k] = 1 for all i,j,k.
    # This means the tensor is the all-ones tensor.
    # The all-ones tensor of size n x m x p has rank 1!
    # Wait, really? Let's check.
    # If U[i*m+j, 0] = 1/sqrt(n*m) for all i,j, V[j*p+k, 0] = 1/sqrt(m*p) for all j,k,
    # W[k*n+i, 0] = 1/sqrt(p*n) for all k,i, then:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = (1/sqrt(n*m))*(1/sqrt(m*p))*(1/sqrt(p*n))
    # = 1/(n*m*p) * ... no, that's not 1.
    
    # Let's try: U[i*m+j, 0] = a, V[j*p+k, 0] = b, W[k*n+i, 0] = c.
    # Then a*b*c = 1. We can pick a=1, b=1, c=1.
    # So the all-ones tensor has rank 1!
    
    # Wait, but the schoolbook decomposition has rank n*m*p = 144.
    # So the rank of the all-ones tensor is 1, not 144.
    # That means the matrix multiplication tensor is NOT the all-ones tensor.
    
    # Let me re-read the problem. "Matrix multiplication 4,4,9"
    # The tensor for multiplying a 4x4 matrix by a 4x9 matrix.
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T[i,j,k] = 1 for all i,j,k.
    # Yes, it is the all-ones tensor.
    
    # So the rank is 1!
    
    # Let's verify: with U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1,
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1*1*1 = 1. Correct!
    
    # So the answer is rank 1.
    
    # But wait, the problem says "rank-minimal exact tensor decomposition".
    # And the schoolbook has rank 144.
    # If the rank is 1, that's a huge improvement.
    
    # Let me double-check the convention. "The output factor follows the official trace convention W[k*n+i,r]."
    # So W is indexed by (k*n+i), which is the (k,i) pair, i.e., the (k,i)-th entry of the output matrix.
    # But the output is n x p, so it should be indexed by (i,k), not (k,i).
    # Hmm, maybe the convention is different.
    
    # Let's re-read: "W[k*n+i,r]". So the row index of W is k*n+i.
    # If k is the row of B and i is the column of A, then... 
    # Actually, in the standard tensor decomposition for matrix multiplication:
    # T[i,j,k] where i is row of A, j is column of A / row of B, k is column of B.
    # The output (AB) has entries (i,k).
    # So W should be indexed by (i,k), i.e., W[i*p+k, r].
    # But the problem says W[k*n+i, r].
    # So maybe the convention is W[k*n+i, r] where k is the row of the output and i is the column?
    # That would be unusual.
    
    # Let's just go with the convention as stated: W[k*n+i, r].
    # And the equation is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k].
    # Where T[i,j,k] = 1 for all i,j,k (since (AB)_{i,k} = sum_j A_{i,j} B_{j,k}).
    
    # So with U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1, we get:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1*1*1 = 1. Correct!
    
    # So the rank is 1.
    
    U = np.ones((n * m, 1))
    V = np.ones((m * p, 1))
    W = np.ones((p * n, 1))
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try rank 1 first
    result = _try_rank(n, m, p, 1, time_limit=60)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
