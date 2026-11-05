def construct():
    # Precompute all non-empty subsets
    all_subsets = []
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        residue = sum(pow(x, -1, 101) for x in terms) % 101
        size = len(terms)
        all_subsets.append((residue, size, terms))
    
    # Find minimum k such that all 101 residues are covered
    for k in range(1, 11):
        covered = set()
        for residue, size, terms in all_subsets:
            if size <= k:
                covered.add(residue)
        if len(covered) == 101:
            break
    
    # Greedy set cover with load balancing
    # We want to assign a subset to each residue.
    # We process residues in a fixed order.
    # For each residue, we choose the subset that minimizes (size, max_usage_in_subset).
    # To be more effective, we can track usage counts.
    
    subsets_by_residue = {}
    for residue, size, terms in all_subsets:
        if size <= k:
            if residue not in subsets_by_residue:
                subsets_by_residue[residue] = []
            subsets_by_residue[residue].append((size, terms))
            
    # Initialize usage counts
    usage = [0] * 11 # indices 1..10
    
    # Sort residues by number of available options (hardest first) to avoid getting stuck?
    # Or just process 0..100.
    # Since we know a solution exists for k, any order should work if we just pick valid ones.
    # But to balance, we should pick the one that uses least-used denominators.
    
    best_assignment = {}
    for r in range(101):
        if r in subsets_by_residue:
            candidates = subsets_by_residue[r]
            # Sort candidates by size, then by "cost" (sum of current usage of its terms)
            # We want to minimize usage concentration.
            def cost(terms):
                return sum(usage[t] for t in terms)
            
            # We prefer smaller size. If sizes are equal, we prefer lower cost.
            # If we strictly want to minimize max length, we must pick size <= k.
            # All candidates have size <= k.
            # We should prioritize size first to keep rows small.
            candidates.sort(key=lambda x: (x[0], cost(x[1])))
            
            best_size, best_terms = candidates[0]
            best_assignment[r] = best_terms
            
            # Update usage
            for t in best_terms:
                usage[t] += 1
        else:
            best_assignment[r] = []
            
    return [best_assignment[r] for r in range(101)]
