"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k
    # Build the set of non-zero entries
    nonzeros = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzeros.add((a, b, c))
    
    # Greedy hypergraph matching: iteratively select rank-1 terms
    # A rank-1 term is defined by (u, v, w) where u is support on A-dim, v on B-dim, w on C-dim
    # For simplicity, we'll use "coordinate" rank-1 terms but try to combine
    
    # Actually, let's try a different approach: use the fact that we can group terms
    # The tensor T = sum over (i,j,k) of e_i^A e_j^B e_k^C where the indices are linked
    
    # Let me try a simple improvement: for each fixed i, the sub-tensor over j,k has rank min(m,p) potentially
    # But this is getting complex. Let me just implement the greedy heuristic properly.
    
    remaining = set(nonzeros)
    terms = []  # list of (u_dict, v_dict, w_dict)
    
    while remaining:
        # Find the rank-1 term that covers the most remaining entries
        # A rank-1 term is determined by choosing a subset of A-indices, B-indices, C-indices
        # such that the product covers entries in remaining
        
        # For efficiency, let's consider "simple" rank-1 terms where each vector has small support
        # Best simple term: pick a single (a,b,c) - that's rank 1 covering 1 entry
        
        # To do better, we need terms where u, v, w each have support > 1
        # and the Cartesian product u_support × v_support × w_support intersects remaining well
        
        # Greedy: for each candidate (a,b) pair, find best c, etc.
        # This is expensive. Let's limit to small supports.
        
        best_term = None
        best_count = 0
        
        # Try all pairs of (a,b) and see which c values they can cover
        # A rank-1 term with u having support {a1,a2}, v having support {b1,b2}, w having support {c1,c2}
        # covers (ai, bj, ck) for all i,j,k in {1,2}
        
        # For simplicity, try rank-1 terms with support size 1 in each dim (covers 1 entry)
        # and support size 2 in some dims
        
        # This is O(n^6) which is 16^6 = too large. Let me be smarter.
        
        # For each (a,b), the set of c such that (a,b,c) in remaining
        # If we fix a and b, we can only cover entries with that specific a and b
        
        # A better rank-1 term: fix a, vary b and c? No, rank-1 means outer product.
        
        # Let me just use the schoolbook for now but check if there's a known better bound.
        # For 4x4 * 4x9, using Strassen: 4x4 = 2x2 blocks, each 2x2*2x9
        # 2x2*2x9: can we do better than 12?
        
        # Actually, let me just return schoolbook as a fallback and note that finding
        # rank < 144 requires sophisticated methods.
        
        # For this implementation, let me try a simple grouping:
        # For each i, the terms with that i form a 4x9 subproblem (m x p)
        # The sub-tensor for fixed i has entries (j,k) -> (j, k) essentially
        # This is the identity-like tensor of rank min(m,p) if we could, but it's actually rank m*p
        
        # I'll just do schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][column] = 1
                    V[b][column] = 1
                    W[c][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
