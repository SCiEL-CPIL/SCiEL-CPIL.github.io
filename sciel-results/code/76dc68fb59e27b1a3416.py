"""Program for matrix multiplication 4x4 * 4x9 using Strassen-based decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We'll build the decomposition using a block-based Strassen approach.
    # The 4x4 * 4x9 matrix multiplication can be viewed as:
    # - Split the 4x4 matrix into 2x2 blocks (4 blocks)
    # - Split the 4x9 matrix into 2x(4 or 5) blocks
    # 
    # A simpler approach: use the known rank-7 decomposition for 2x2 * 2x2,
    # and extend it to our case.
    #
    # Actually, let's use a direct approach:
    # The multiplication tensor for (n,m,p) has entries T[a,b,c] where
    # a = i*m + j, b = j*p + k, c = k*n + i
    #
    # For a rank-1 term, we need vectors u (size n*m), v (size m*p), w (size p*n)
    # such that T = sum_r u_r ⊗ v_r ⊗ w_r
    #
    # Let's try a structured decomposition based on grouping.
    #
    # Key insight: For fixed i, the sub-tensor over (j,k) is essentially the
    # "identity" tensor mapping (j,k) to (j,k) with indices shifted.
    # This has rank m*p = 36 for each i, giving 4*36 = 144 total (schoolbook).
    #
    # To do better, we can use the fact that the structure allows sharing
    # between different i values.
    #
    # Let's implement a greedy hypergraph matching approach:
    # 1. Build the set of non-zero tensor entries
    # 2. Iteratively find rank-1 terms that cover many remaining entries
    # 3. Subtract and repeat
    
    # Build non-zero entries
    nonzeros = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzeros.add((a, b, c))
    
    remaining = set(nonzeros)
    terms = []  # Each term is (u_dict, v_dict, w_dict)
    
    # Precompute for each (a,b) pair, the set of c values in nonzeros
    ab_to_cs = {}
    for (a, b, c) in nonzeros:
        key = (a, b)
        if key not in ab_to_cs:
            ab_to_cs[key] = set()
        ab_to_cs[key].add(c)
    
    # Precompute for each (a,c) pair, the set of b values
    ac_to_bs = {}
    for (a, b, c) in nonzeros:
        key = (a, c)
        if key not in ac_to_bs:
            ac_to_bs[key] = set()
        ac_to_bs[key].add(b)
    
    # Precompute for each (b,c) pair, the set of a values
    bc_to_as = {}
    for (a, b, c) in nonzeros:
        key = (b, c)
        if key not in bc_to_as:
            bc_to_as[key] = set()
        bc_to_as[key].add(a)
    
    while remaining:
        best_term = None
        best_coverage = 0
        
        # Strategy: try rank-1 terms with small supports
        # For each pair of "center" indices, try to build a term
        
        # Try terms where u has support of size 1, v has support of size s, w has support of size t
        # The term covers entries (a, b, c) where a is in u_support, b in v_support, c in w_support
        # AND (a,b,c) is in remaining
        
        # More systematically: for each candidate (a0, b0, c0) in remaining,
        # try to extend it to a larger rank-1 term
        
        # Heuristic: for each a, find the best (v_support, w_support) that covers
        # many entries with that a
        
        for a in range(n * m):
            # Find all (b, c) such that (a, b, c) in remaining
            candidates = []
            for (a2, b, c) in remaining:
                if a2 == a:
                    candidates.append((b, c))
            
            if not candidates:
                continue
            
            # Try to find a good v_support and w_support
            # Start with single (b,c) pairs and try to merge
            # For now, try: fix a, pick a set of b's and c's
            
            # Simple: for each b, find all c with that b
            b_to_cs = {}
            for (b, c) in candidates:
                if b not in b_to_cs:
                    b_to_cs[b] = set()
                b_to_cs[b].add(c)
            
            # Try different combinations of b's
            b_list = list(b_to_cs.keys())
            if not b_list:
                continue
            
            # Try single b
            for b in b_list:
                c_set = b_to_cs[b]
                coverage = len(c_set)
                if coverage > best_coverage:
                    # Check if this is valid: u={a}, v={b}, w=c_set
                    # All (a, b, c) for c in c_set must be in remaining
                    valid = all((a, b, c) in remaining for c in c_set)
                    if valid:
                        best_coverage = coverage
                        best_term = ({a}, {b}, c_set)
            
            # Try pairs of b's
            for i1 in range(len(b_list)):
                for i2 in range(i1 + 1, len(b_list)):
                    b1, b2 = b_list[i1], b_list[i2]
                    c1_set, c2_set = b_to_cs[b1], b_to_cs[b2]
                    # For a rank-1 term, w_support must be the same for all b's
                    # So we need c_set that works for both b1 and b2
                    # The entries covered would be (a, b1, c) for c in c_set AND (a, b2, c) for c in c_set
                    # So c_set must be subset of c1_set AND c2_set
                    c_common = c1_set & c2_set
                    if c_common:
                        coverage = 2 * len(c_common)
                        if coverage > best_coverage:
                            valid = all((a, b1, c) in remaining for c in c_common) and \
                                    all((a, b2, c) in remaining for c in c_common)
                            if valid:
                                best_coverage = coverage
                                best_term = ({a}, {b1, b2}, c_common)
        
        # Also try terms where u has support > 1
        # For each pair of a's, find common (b,c) structure
        a_list = list(range(n * m))
        for i1 in range(len(a_list)):
            for i2 in range(i1 + 1, len(a_list)):
                a1, a2 = a_list[i1], a_list[i2]
                # Find (b,c) that appear for both a1 and a2
                ac1 = {}
                ac2 = {}
                for (a, b, c) in remaining:
                    if a == a1:
                        if b not in ac1:
                            ac1[b] = set()
                        ac1[b].add(c)
                    elif a == a2:
                        if b not in ac2:
                            ac2[b] = set()
                        ac2[b].add(c)
                
                common_bs = set(ac1.keys()) & set(ac2.keys())
                for b in common_bs:
                    c_common = ac1[b] & ac2[b]
                    if c_common:
                        coverage = 2 * len(c_common)
                        if coverage > best_coverage:
                            valid = all((a1, b, c) in remaining for c in c_common) and \
                                    all((a2, b, c) in remaining for c in c_common)
                            if valid:
                                best_coverage = coverage
                                best_term = ({a1, a2}, {b}, c_common)
        
        # If no good term found, pick a single entry
        if best_term is None:
            # Pick any remaining entry
            a, b, c = next(iter(remaining))
            best_term = ({a}, {b}, {c})
            best_coverage = 1
        
        u_support, v_support, w_support = best_term
        
        # Create the term
        u_dict = {a: 1 for a in u_support}
        v_dict = {b: 1 for b in v_support}
        w_dict = {c: 1 for c in w_support}
        
        terms.append((u_dict, v_dict, w_dict))
        
        # Remove covered entries from remaining
        for a in u_support:
            for b in v_support:
                for c in w_support:
                    if (a, b, c) in remaining:
                        remaining.discard((a, b, c))
        
        if not remaining:
            break
    
    # Convert terms to matrix form
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for a, val in u_dict.items():
            U[a][r] = val
        for b, val in v_dict.items():
            V[b][r] = val
        for c, val in w_dict.items():
            W[c][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
