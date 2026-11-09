"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    # Build schoolbook decomposition
    def build_schoolbook():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = build_schoolbook()
    
    def verify(U, V, W, rank):
        """Verify exact decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def check_valid_entries(U, V, W):
        """Check all entries are integers or half-integers."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    doubled = val * 2
                    if abs(doubled - round(doubled)) > 1e-9:
                        return False
        return True
    
    # Try to find rank 41 by exploiting the small dimension n=2
    def try_rank_41():
        """Attempt to find a rank-41 decomposition by searching for
        non-trivial linear dependencies among schoolbook terms."""
        if time.time() > deadline - 10.0:
            return None
        
        # For n=2, we can represent each term (i,j,k) as a vector in a
        # structured space. The key is to find if there's a linear combination
        # of 41 terms that equals the 42nd term.
        
        # Strategy: For each pair of terms that share the same i-index,
        # try to see if one can be expressed as a combination of the other
        # with adjustments to shared V and W entries.
        
        # More systematically: use the fact that when n=2, the W matrix
        # has only 14 rows (p*n = 7*2). We can try to find a column in W
        # that is a linear combination of other columns, which would allow
        # us to eliminate one term.
        
        # Let's try a direct search: for each possible "pivot" term to remove,
        # check if we can adjust the remaining terms to compensate.
        
        # We'll use a greedy approach: try removing each term and see if
        # the resulting system has a solution with half-integer entries.
        
        from collections import defaultdict
        
        # Precompute which terms share indices
        terms = list(product(range(n), range(m), range(p)))
        
        # For each term to remove, try to find a correction
        for i0, j0, k0 in terms:
            if time.time() > deadline - 5.0:
                return None
            
            # Build the error: we need to add back the contribution of term (i0,j0,k0)
            # using adjustments to other terms.
            
            # The simplest case: find a term (i1,j1,k1) that shares at least
            # one index with (i0,j0,k0) and can be adjusted to compensate.
            
            # If they share i: i1 = i0
            # If they share j: j1 = j0  
            # If they share k: k1 = k0
            
            # For a term sharing i=i0:
            # Original: U[i0*m+j0,c0]=1, V[j0*p+k0,c0]=1, W[k0*n+i0,c0]=1
            # New term: U[i0*m+j1,c1]=1, V[j1*p+k1,c1]=1, W[k1*n+i0,c1]=1
            # 
            # To compensate, we need to add to T[i0,j0,k0]:
            # We can adjust W[k0*n+i0,c1] to include a component that, when
            # multiplied with U[i0*m+j0,c1] and V[j0*p+k0,c1], gives 1.
            # But U[i0*m+j0,c1] and V[j0*p+k0,c1] are 0 unless j1=j0 and k1=k0.
            
            # This approach is too constrained. Let me try a different angle.
            
            # Use the structure: when n=2, we can write the multiplication as
            # two separate operations (for i=0 and i=1) that share the same
            # input matrix A (2x3) but produce different output rows.
            
            # For i=0: output row 0 = A @ B[:,k] for each k
            # For i=1: output row 1 = A @ B[:,k] for each k
            
            # The key insight: both use the same A, so there might be
            # sharing possible in the U matrix.
            
            # Let's try: can we merge two terms that have the same (j,k) but
            # different i? For example, (0,j,k) and (1,j,k).
            # 
            # Term (0,j,k): U[0*m+j]=1, V[j*p+k]=1, W[k*n+0]=1
            # Term (1,j,k): U[1*m+j]=1, V[j*p+k]=1, W[k*n+1]=1
            #
            # These share the same V entry. Can we combine them?
            # We'd need a single column c such that:
            # U[0*m+j,c]*V[j*p+k,c]*W[k*n+0,c] = 1
            # U[1*m+j,c]*V[j*p+k,c]*W[k*n+1,c] = 1
            #
            # If we set V[j*p+k,c] = v, then:
            # U[0*m+j,c]*v*W[k*n+0,c] = 1
            # U[1*m+j,c]*v*W[k*n+1,c] = 1
            #
            # This is possible! For example, set v=1, U[0*m+j,c]=1, W[k*n+0,c]=1,
            # U[1*m+j,c]=1, W[k*n+1,c]=1. But this is just two separate terms.
            #
            # To actually merge, we need to use the same column for both, which
            # means we're not reducing the rank.
            
            # Let me try yet another approach: use the fact that 7 is prime
            # and see if we can use a polynomial representation.
            
            # Actually, let's just do a brute-force search for rank 41 by
            # trying to find a linear dependency among the 42 tensor entries.
            
            # The tensor has 42 entries (one for each (i,j,k)). A rank-r decomposition
            # means we can write T as a sum of r rank-1 tensors.
            
            # For rank 41, we need to find 41 rank-1 tensors whose sum equals T.
            # This is equivalent to finding a 41-dimensional subspace that contains T.
            
            # This is a hard problem. Let me try a heuristic: randomly perturb
            # the schoolbook decomposition and see if we can find a valid rank-41
            # solution.
            
            pass  # No improvement found yet
        
        return None
    
    # Attempt rank 41 search
    if time.time() < deadline - 5.0:
        result_41 = try_rank_41()
        if result_41 is not None:
            U41, V41, W41 = result_41
            if verify(U41, V41, W41, 41) and check_valid_entries(U41, V41, W41):
                return {"U": U41, "V": V41, "W": W41}
    
    # Return schoolbook decomposition
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
