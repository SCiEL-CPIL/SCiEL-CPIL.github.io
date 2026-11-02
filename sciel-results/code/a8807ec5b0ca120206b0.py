"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill

    # Schoolbook decomposition of rank n*m*p
    def schoolbook():
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

    best = schoolbook()
    best_rank = n * m * p

    # Nonzero entries of the tensor
    nz = [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]

    # Theoretical lower bound
    lb = max(n*m, m*p, p*n)

    # Strategy: Use structured decompositions based on known algebraic techniques
    # For rectangular matrix multiplication, we can use "laser method" style constructions
    # or block-based approaches to reduce rank below schoolbook
    
    # Approach 1: Try a structured decomposition using the fact that
    # we can group terms to share computation
    
    # For 3x4*4x5, one approach is to use a "border rank" style construction
    # Let's try to find a rank-r decomposition by solving a system of equations
    
    # The key equation: for each (i,j,k) in [0,n)x[0,m)x[0,p):
    #   sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    # For all other (a,b,c) not in the tensor:
    #   sum_t U[a, t] * V[b, t] * W[c, t] = 0
    
    # Let's try ranks from a reasonable lower bound up to schoolbook
    # For 3x4x5, known best upper bounds are around 40-50
    
    # Try to find exact decompositions using rational arithmetic
    # We'll use a greedy/structured approach
    
    # First, try a simple structured approach:
    # Partition the computation into blocks and use shared factors
    
    # Method: Try rank r = n*p + m*p - p = p*(n+m-1) for some constructions
    # Or use the "Strassen-like" approach for rectangular matrices
    
    # Let's try several candidate ranks and use exact linear algebra
    
    # For a given rank r, we can try to find U, V, W with entries in {0, 1/2, 1}
    # or small rationals by solving a constraint satisfaction problem
    
    # Approach: Use the fact that the tensor has a specific structure
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's try to construct a decomposition using "separation" of indices
    
    # For n=3, m=4, p=5:
    # Try rank = 40 (which is n*p + m*p - p = 15 + 20 - 5 = 30, or other formulas)
    
    # Actually, let's use a more systematic approach:
    # Try to find a decomposition where each "slice" shares structure
    
    # Method: Fix W to be identity-like, then solve for U and V
    # If W[c, t] = delta_{c, t} for t < p*n, then:
    #   U[i*m+j, t] * V[j*p+k, t] = T[i*m+j, j*p+k, t]
    # This means for each t, U[:,t] * V[:,t]^T = T[:,:,t]
    # T[:,:,t] is a (n*m) x (m*p) matrix with entries:
    #   T[:,:,t][i*m+j, j*p+k] = 1 if k*n+i == t else 0
    
    # So T[:,:,t] has at most m non-zero entries (one for each j, where k = t//n, i = t%n)
    # Actually, for fixed t = k*n + i, the non-zero entries are at positions
    # (i*m+j, j*p+k) for j in range(m)
    # These are m entries, all equal to 1
    
    # Can we factor T[:,:,t] as a rank-1 matrix u_t * v_t^T?
    # T[:,:,t] has entries: T[i*m+j, j*p+k] = 1 for j=0..m-1, where t=k*n+i
    # This is NOT rank-1 in general (unless m=1)
    
    # So this approach doesn't directly work. Let's try a different structure.
    
    # Alternative: Use a "diagonal" W matrix
    # Let W be diagonal with entries w_0, w_1, ..., w_{p*n-1}
    # Then sum_t U[a,t]*V[b,t]*W[c,t] = U[a,c]*V[b,c]*W[c,c] (if W is diagonal)
    # This gives: U[a,c]*V[b,c]*W[c,c] = T[a,b,c]
    # For T[a,b,c] = 1: U[a,c]*V[b,c]*W[c,c] = 1
    # For T[a,b,c] = 0: U[a,c]*V[b,c]*W[c,c] = 0
    
    # This means for each c, either W[c,c] = 0 (and then U[a,c]=V[b,c]=0 for all a,b
    # where T[a,b,c]=0, but U[a,c]*V[b,c]=0 for all a,b where T=0),
    # or W[c,c] != 0 and then U[a,c]*V[b,c] = 1/W[c,c] when T[a,b,c]=1,
    # and U[a,c]*V[b,c] = 0 when T[a,b,c]=0.
    
    # For fixed c = k*n + i, T[a,b,c] = 1 iff a = i*m+j and b = j*p+k for some j
    # So for fixed c, the set of (a,b) with T[a,b,c]=1 is:
    #   {(i*m+j, j*p+k) : j in range(m)}
    
    # We need U[a,c]*V[b,c] to be 1/W[c,c] for these (a,b) pairs,
    # and 0 for all other (a,b).
    
    # This is possible if and only if the set of (a,b) pairs forms a "rectangle"
    # in the sense that a depends only on i and b depends only on k, with j being free.
    # But here a = i*m+j depends on j, and b = j*p+k also depends on j.
    
    # So U[a,c] * V[b,c] must be constant for all j, and zero otherwise.
    # Let a_j = i*m+j, b_j = j*p+k.
    # We need U[a_j, c] * V[b_j, c] = 1/W[c,c] for all j.
    # And U[a, c] * V[b, c] = 0 for all other (a,b).
    
    # One way: Set U[a_j, c] = 1/sqrt(m) * alpha_j and V[b_j, c] = 1/sqrt(m) * beta_j
    # with alpha_j * beta_j = m/W[c,c] for all j.
    # But we also need U[a, c] * V[b, c] = 0 for (a,b) not in the set.
    
    # This is tricky. Let's try a simpler approach: set U[a_j, c] = 1 and V[b_j, c] = 1/W[c,c]
    # for all j. But then for a = i*m+j1 and b = j2*p+k with j1 != j2,
    # U[a, c] * V[b, c] = 1 * 1/W[c,c] != 0, but T[a,b,c] = 0.
    
    # So this doesn't work. The diagonal W approach is too restrictive.
    
    # Let's try a different approach: use a "block diagonal" structure
    
    # For now, let's just return the schoolbook result, which is guaranteed to work.
    # Future work: implement more sophisticated decompositions.
    
    return best
# EVOLVE-BLOCK-END
