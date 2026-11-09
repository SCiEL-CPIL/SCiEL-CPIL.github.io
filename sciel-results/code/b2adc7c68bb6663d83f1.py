def construct():
    # Precompute all subsets and their sums mod 101
    # subsets_by_size[k] = list of (mask, terms, residue)
    subsets_by_size = [[] for _ in range(11)]
    
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x-1)) & 1]
        size = len(terms)
        residue = sum(pow(x, -1, 101) for x in terms) % 101
        subsets_by_size[size].append((mask, terms, residue))
    
    # Find minimum k such that all residues 0..100 are covered by subsets of size <= k
    for k in range(1, 11):
        covered = set()
        for size in range(1, k+1):
            for mask, terms, residue in subsets_by_size[size]:
                covered.add(residue)
        if len(covered) == 101:
            break
    
    # Now assign: for each residue, pick a subset of size <= k
    # To minimize max, we prefer smaller subsets
    assignment = {}
    for size in range(1, k+1):
        for mask, terms, residue in subsets_by_size[size]:
            if residue not in assignment:
                assignment[residue] = terms
    
    # Build result
    result = []
    for r in range(101):
        result.append(assignment[r])
    
    return result
