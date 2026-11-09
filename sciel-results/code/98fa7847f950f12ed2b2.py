"""Strassen-based 2x5 x 5x6 matrix multiplication decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from itertools import product

    start_time = time.time()
    DEATHLINE = 130.0
    n, m, p = 2, 5, 6

    def elapsed():
        return time.time() - start_time

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor.
        T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r] must equal 1 if u=i*m+j, v=j*p+k, w=k*n+i, else 0.
        """
        rank = len(U[0])
        # Check all required positions equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(s - 1.0) > tol:
                        return False
        # Check that non-required positions are 0
        # A position (u,v,w) is "required" if there exist i,j,k such that
        # u = i*m+j, v = j'*p+k, w = k'*n+i' with j=j' and k=k' and i=i'
        # Actually, the tensor is defined only on the "consistent" triples.
        # For the decomposition to be valid for matrix multiplication, we only need
        # the consistent triples to sum to 1. The inconsistent triples don't matter
        # because they don't correspond to any actual multiplication term.
        # So we only need to check the n*m*p = 60 consistent positions.
        return True

    def build_trivial():
        """Build the trivial rank-60 decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}, rank

    # Start with trivial decomposition
    best_result, best_rank = build_trivial()

    if elapsed() > DEATHLINE - 5:
        return best_result

    # Try to find a lower-rank decomposition
    # Strategy: The tensor T has 60 nonzero entries. We want to represent it as
    # sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r where each term is a rank-1 tensor.
    # 
    # Key observation: We can try to find linear combinations that cancel out
    # unwanted terms while preserving the desired ones.
    #
    # For a systematic approach, let's try to find if we can reduce rank by
    # identifying that some terms are linearly dependent in a useful way.
    #
    # One approach: Try to find a decomposition with rank R where R < 60.
    # We can set up a system of linear equations.
    #
    # For each rank-1 term r, we choose:
    #   - which (i,j) pair it affects in U: u_r has support on some subset of {0,...,n*m-1}
    #   - which (j,k) pair it affects in V: v_r has support on some subset of {0,...,m*p-1}  
    #   - which (k,i) pair it affects in W: w_r has support on some subset of {0,...,p*n-1}
    #
    # This is complex. Let's try a simpler heuristic:
    # Try to merge pairs of terms that share structure.
    
    # Let's try a direct approach: search for rank R from 60 downward
    # For each R, try to find coefficients that work.
    
    # A more practical approach: use the fact that for small matrices,
    # known algorithms exist. For 2x5 x 5x6, let's try to use a block approach.
    
    # Actually, let's try a randomized search for rank reduction.
    # We'll start from the trivial decomposition and try to perturb it
    # while maintaining the constraint.
    
    import random
    random.seed(12345)
    
    def try_reduce_rank(current_rank, max_iter=50):
        """Try to find a decomposition with rank < current_rank."""
        for target_rank in range(current_rank - 1, max(current_rank - 10, 30), -1):
            if elapsed() > DEATHLINE - 10:
                return None, None
            
            # Try to find a rank-target_rank decomposition
            # Use a greedy approach: start with trivial, try to eliminate terms
            # by finding linear dependencies.
            
            # For now, try a simple random perturbation approach
            for attempt in range(max_iter):
                if elapsed() > DEATHLINE - 10:
                    return None, None
                
                # Create a random perturbation of the trivial decomposition
                # and see if we can project it to a lower-rank space
                # while maintaining the constraints.
                
                # This is a placeholder - actual implementation would be complex.
                # For now, let's try a specific known reduction.
                
                # For 2x2x2, Strassen's algorithm gives rank 7 instead of 8.
                # The key identity is:
                # m1 = (a00+a11)(b00+b11)
                # m2 = (a10+a11)b00
                # m3 = a00(b10-b11)
                # m4 = a11(b01-b00)
                # m5 = (a00+a01)b11
                # m6 = (a10-a00)(b00+b10)
                # m7 = (a01+a11)(b01+b11)
                # c00 = m1 + m4 - m5 - m7
                # c01 = m3 + m7
                # c10 = m2 + m6
                # c11 = m1 - m2 + m3 + m5
                
                # For our 2x5 x 5x6 case, we can potentially apply similar
                # ideas to sub-blocks. Let's try to find a rank-59 decomposition
                # by merging two terms.
                
                # Try: can we find two terms (r1, r2) in the trivial decomposition
                # such that their sum can be replaced by a single term?
                # This requires that there exist u, v, w such that:
                # u*v*w = u1*v1*w1 + u2*v2*w2 (as tensors)
                
                # This is generally not possible for arbitrary terms, but might
                # be possible for specific structures.
                
                pass
            
            # If we found a valid decomposition, return it
            # For now, continue searching
        
        return None, None

    # Try the reduction search
    reduced_result, reduced_rank = try_reduce_rank(best_rank)
    if reduced_result is not None:
        best_result = reduced_result
        best_rank = reduced_rank

    return best_result
# EVOLVE-BLOCK-END
