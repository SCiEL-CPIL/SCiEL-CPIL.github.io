"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import random
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill

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

    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix mult, T[i,j,k] should be 1 if this is the (i,j,k) entry
        # Actually the tensor for matmul is: T[i, j, k] where the multiplication
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T_{i,j,k} = 1 for all i,j,k (it's the identity tensor structure)
        # More precisely: the tensor has T[i, j, k] = 1 for all valid i,j,k
        # because C[i,k] = sum_j A[i,j] B[j,k] means the tensor coefficient is 1
        
        # Reconstruct and check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != 1:
                        return False
        return True

    def find_rank1_component(U, V, W):
        """
        Find a rank-1 component (u, v, w) such that:
        U' = U - u*w^T has lower rank contribution, etc.
        Actually, we want to find u, v, w such that the residual
        tensor T - u⊗v⊗w is still decomposable.
        
        Strategy: For each pair of "support patterns" in V and W,
        try to find a consistent u.
        """
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        # For a rank-1 component, u has shape (n*m,), v has shape (m*p,), w has shape (p*n,)
        # The contribution to tensor entry (i,j,k) is u[i*m+j] * v[j*p+k] * w[k*n+i]
        # We need: for all (i,j,k): sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] 
        #           = sum of rank-1 components
        
        # Peeling: find one rank-1 component that matches a "corner" of the tensor
        # Try all possible support patterns for v and w
        
        best_component = None
        best_residual_rank = rank + 1
        
        # For each possible support of v (subset of m*p indices) and w (subset of p*n indices)
        # that are "compatible" (they share the same k-index pattern), try to solve for u
        
        # A rank-1 component has v supported on indices {j*p+k : j in J, k in K}
        # and w supported on indices {k*n+i : k in K, i in I}
        # where K is the same set of k-values
        
        # For small problems, enumerate all possible (I, J, K) triples
        # I subset of {0,...,n-1}, J subset of {0,...,m-1}, K subset of {0,...,p-1}
        
        all_i = list(range(n))
        all_j = list(range(m))
        all_k = list(range(p))
        
        # Enumerate non-empty subsets
        def subsets(lst):
            from itertools import combinations
            for r in range(1, len(lst)+1):
                for c in combinations(lst, r):
                    yield c
        
        # To limit search, try small subsets first
        # Priority: single-element subsets, then pairs, etc.
        i_subs = list(subsets(all_i))
        j_subs = list(subsets(all_j))
        k_subs = list(subsets(all_k))
        
        # Sort by size for early termination
        i_subs.sort(key=len)
        j_subs.sort(key=len)
        k_subs.sort(key=len)
        
        found = False
        for I in i_subs:
            if found:
                break
            for J in j_subs:
                if found:
                    break
                for K in k_subs:
                    if found:
                        break
                    
                    # Build support sets
                    v_support = set()
                    for j in J:
                        for k in K:
                            v_support.add(j * p + k)
                    
                    w_support = set()
                    for k in K:
                        for i in I:
                            w_support.add(k * n + i)
                    
                    u_support = set()
                    for i in I:
                        for j in J:
                            u_support.add(i * m + j)
                    
                    # For entries (i,j,k) where i in I, j in J, k in K:
                    # The tensor value is 1. The rank-1 component contributes
                    # u[i*m+j] * v[j*p+k] * w[k*n+i]
                    # If we assume this is the ONLY component contributing to these entries
                    # (or we're peeling it off), we can try to find u,v,w.
                    
                    # Simplest: assume u[i*m+j] = 1 for all i in I, j in J
                    # Then v[j*p+k] * w[k*n+i] should give consistent values
                    
                    # Let's try: set u[i*m+j] = 1 for i in I, j in J
                    # Then for each (j,k) in J×K and (k,i) in K×I:
                    # v[j*p+k] * w[k*n+i] = 1 (if these are the only contributions)
                    
                    # This gives v[j*p+k] * w[k*n+i] = 1
                    # So w[k*n+i] = 1 / v[j*p+k]
                    # For consistency: if k is in K and i1, i2 in I, j1, j2 in J:
                    # v[j1*p+k] * w[k*n+i1] = 1 and v[j2*p+k] * w[k*n+i2] = 1
                    # This is always satisfiable by choosing v and w appropriately.
                    
                    # Let's pick v[j*p+k] = 1 for all j in J, k in K
                    # Then w[k*n+i] = 1 for all k in K, i in I
                    
                    # Check if this rank-1 component is "valid" in the sense that
                    # subtracting it doesn't create negative or fractional issues
                    # Actually, let's just check if the residual tensor is still
                    # a valid matmul tensor (all entries are 0 or 1, and the
                    # "missing" entries form a valid sub-problem)
                    
                    # Build the rank-1 component
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    
                    for i in I:
                        for j in J:
                            u[i * m + j] = 1
                    for j in J:
                        for k in K:
                            v[j * p + k] = 1
                    for k in K:
                        for i in I:
                            w[k * n + i] = 1
                    
                    # Check: does this component "fit" in the current tensor?
                    # For each (i,j,k) in the full grid:
                    #   if i in I, j in J, k in K: contribution is 1 (good, matches tensor)
                    #   otherwise: contribution is 0 (good, doesn't affect other entries)
                    # So this is always a valid "partial" component.
                    
                    # The question is: can the remaining tensor be decomposed?
                    # The remaining tensor has 1s at (i,j,k) where NOT (i in I AND j in J AND k in K)
                    # This is equivalent to the original tensor minus the sub-box I×J×K.
                    
                    # This sub-box removal is valid if the remaining structure
                    # can still be decomposed. For matmul, removing a sub-box
                    # corresponds to a smaller matmul problem with some dimensions
                    # partially covered.
                    
                    # For now, accept this component and recurse.
                    # The key insight: the rank of the full problem = 1 + rank(remaining)
                    # if the sub-box can be cleanly separated.
                    
                    # Actually, this approach is flawed because the rank-1 component
                    # must exactly match a subset of the tensor, and the remaining
                    # tensor must still be a valid matmul tensor.
                    
                    # Let me reconsider. The tensor for matmul n×m × m×p has
                    # T[i,j,k] = 1 for all i,j,k. A rank-1 decomposition term
                    # contributes u[i*m+j]*v[j*p+k]*w[k*n+i].
                    
                    # If we set u[i*m+j]=1 for i in I, j in J, and 0 otherwise,
                    # v[j*p+k]=1 for j in J, k in K, and 0 otherwise,
                    # w[k*n+i]=1 for k in K, i in I, and 0 otherwise,
                    # then the contribution is 1 exactly when i in I, j in J, k in K.
                    
                    # So the remaining tensor has 1s everywhere except the box I×J×K.
                    # This is NOT a standard matmul tensor, so it's not clear if it
                    # can be decomposed with fewer terms.
                    
                    # Better approach: use the fact that matmul tensor has a specific
                    # algebraic structure. The rank is known to be n*m*p for most
                    # small cases, but let's try to verify by attempting decomposition.
                    
                    break  # Don't break out of all loops, just this k_subs iteration
        
        # The above approach is getting complicated. Let's use a different strategy:
        # Try to find a decomposition by solving a system of equations.
        
        # For rank r, we need to find U, V, W of rank r such that
        # for all i,j,k: sum_{t=0}^{r-1} U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
        
        # This is a system of n*m*p equations in 3*r*(nm+mp+pn) unknowns.
        # For r < n*m*p, this is an underdetermined system.
        
        # Let's try r = n*m*p - 1, n*m*p - 2, etc. and see if we can find a solution.
        
        # Actually, for the 2×3×7 case, n*m*p = 42.
        # Known results: the rank of 2×3 matrix multiplication is 7 (not 6),
        # and for 3×3 it's 19-23 (exact value unknown).
        # For 2×3×7, the rank might be less than 42.
        
        # Let's try a different approach: use the fact that the tensor is
        # a "matrix multiplication tensor" and use known bounds.
        
        # For now, return the schoolbook solution as a fallback.
        return schoolbook()

    return schoolbook()
# EVOLVE-BLOCK-END
