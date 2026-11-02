"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    # Attempt rank-12 "middle product" decomposition
    # Each rank term r has U[i,j,r], V[j,k,r], W[k,i,r]
    # We need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    
    rank = 12
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Define terms as (u_entries, v_entries, w_entries)
    # u_entries: list of (i, j, val)
    # v_entries: list of (j, k, val)
    # w_entries: list of (k, i, val)
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
    
    # Strategy: Use 6 Strassen-like patterns, each covering both k-blocks
    # in a coordinated way. For each pattern, we design V and W to cover
    # k=0,1,2,3 simultaneously.
    
    # Term 0: (a00+a11) * (b00+b11+b02+b12 pattern)
    # u: (0,0):1, (1,1):1
    # v: (0,0):1, (1,1):1, (0,2):1, (1,3):1
    # w: (0,0):1, (1,1):1, (2,0):1, (3,1):1
    add_term(
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1), (0,2,1), (1,3,1)],
        [(0,0,1), (1,1,1), (2,0,1), (3,1,1)]
    )
    
    # Term 1: (a10+a11) * b0k pattern
    # u: (1,0):1, (1,1):1
    # v: (0,0):1, (0,2):1
    # w: (0,1):1, (1,1):-1, (2,1):1, (3,1):-1
    add_term(
        [(1,0,1), (1,1,1)],
        [(0,0,1), (0,2,1)],
        [(0,1,1), (1,1,-1), (2,1,1), (3,1,-1)]
    )
    
    # Term 2: a00 * (b0k - b1k) pattern
    # u: (0,0):1
    # v: (0,1):1, (1,1):-1, (0,3):1, (1,3):-1
    # w: (1,0):1, (1,1):1, (3,0):1, (3,1):1
    add_term(
        [(0,0,1)],
        [(0,1,1), (1,1,-1), (0,3,1), (1,3,-1)],
        [(1,0,1), (1,1,1), (3,0,1), (3,1,1)]
    )
    
    # Term 3: a11 * (b1k - b0k) pattern
    # u: (1,1):1
    # v: (1,0):1, (0,0):-1, (1,2):1, (0,2):-1
    # w: (0,0):1, (0,1):1, (2,0):1, (2,1):1
    add_term(
        [(1,1,1)],
        [(1,0,1), (0,0,-1), (1,2,1), (0,2,-1)],
        [(0,0,1), (0,1,1), (2,0,1), (2,1,1)]
    )
    
    # Term 4: (a00+a01) * b1k pattern
    # u: (0,0):1, (0,1):1
    # v: (1,1):1, (1,3):1
    # w: (1,0):1, (0,0):-1, (3,0):1, (2,0):-1
    add_term(
        [(0,0,1), (0,1,1)],
        [(1,1,1), (1,3,1)],
        [(1,0,1), (0,0,-1), (3,0,1), (2,0,-1)]
    )
    
    # Term 5: (a10-a00) * (b0k+b0k+1) pattern
    # u: (1,0):1, (0,0):-1
    # v: (0,0):1, (0,1):1, (0,2):1, (0,3):1
    # w: (1,1):1, (3,1):1
    add_term(
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (0,1,1), (0,2,1), (0,3,1)],
        [(1,1,1), (3,1,1)]
    )
    
    # Term 6: (a01-a11) * (b1k+b1k+1) pattern
    # u: (0,1):1, (1,1):-1
    # v: (1,0):1, (1,1):1, (1,2):1, (1,3):1
    # w: (0,0):1, (2,0):1
    add_term(
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (1,1,1), (1,2,1), (1,3,1)],
        [(0,0,1), (2,0,1)]
    )
    
    # That's 7 terms. We need more to cover all 8 outputs.
    # Let me add 5 more terms to reach rank 12, designed to fix gaps.
    
    # Actually, let me verify the 7 terms above and see what's missing.
    # This is getting complex. Let me try a different systematic approach.
    
    # Reset and use a cleaner construction
    terms = []
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Use the proven 14-term Strassen-based solution but attempt to
    # merge two pairs of terms to reduce rank to 12.
    
    # Define the 7 Strassen patterns
    strassen_patterns = [
        # (u_entries, v_entries_k01, w_entries_k01)
        # v and w are for k in {0,1}; we'll shift for k in {2,3}
        ([(0,0,1), (1,1,1)],
         [(0,0,1), (1,1,1)],
         [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)],
         [(0,0,1)],
         [(0,1,1), (1,1,-1)]),
        ([(0,0,1)],
         [(0,1,1), (1,1,-1)],
         [(1,0,1), (1,1,1)]),
        ([(1,1,1)],
         [(1,0,1), (0,0,-1)],
         [(0,0,1), (0,1,1)]),
        ([(0,0,1), (0,1,1)],
         [(1,1,1)],
         [(1,0,1), (0,0,-1)]),
        ([(1,0,1), (0,0,-1)],
         [(0,0,1), (0,1,1)],
         [(1,1,1)]),
        ([(0,1,1), (1,1,-1)],
         [(1,0,1), (1,1,1)],
         [(0,0,1)]),
    ]
    
    # For patterns 0 and 3, try to merge them into a single term
    # that covers both k-blocks.
    # Pattern 0: u=(0,0):1,(1,1):1; v=(0,0):1,(1,1):1; w=(0,0):1,(1,1):1
    # Pattern 3: u=(1,1):1; v=(1,0):1,(0,0):-1; w=(0,0):1,(0,1):1
    
    # These have different u vectors, so can't simply merge.
    # Let's just use the 14-term solution as baseline and try small modifications.
    
    # Fill matrices with the 14-term solution
    for r in range(7):
        u_entries, v_entries, w_entries = strassen_patterns[r]
        
        # Block 1 (k=0,1)
        col1 = r
        for i, j, val in u_entries:
            u_idx = i * m + j
            U[u_idx][col1] += val
        for j, k, val in v_entries:
            v_idx = j * p + k
            V[v_idx][col1] += val
        for k, i, val in w_entries:
            w_idx = k * n + i
            W[w_idx][col1] += val
        
        # Block 2 (k=2,3)
        col2 = 7 + r
        for i, j, val in u_entries:
            u_idx = i * m + j
            U[u_idx][col2] += val
        for j, k, val in v_entries:
            k_new = k + 2
            v_idx = j * p + k_new
            V[v_idx][col2] += val
        for k, i, val in w_entries:
            k_new = k + 2
            w_idx = k_new * n + i
            W[w_idx][col2] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
