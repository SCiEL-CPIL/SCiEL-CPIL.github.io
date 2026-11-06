def construct():
    P = 101
    inv = [0] + [pow(x, -1, P) for x in range(1, 11)]

    # Enumerate all non-empty subsets, group by size
    subsets_by_size = [[] for _ in range(11)]
    
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
                s = (s + inv[x]) % P
        subsets_by_size[len(terms)].append((s, terms))
    
    # Find minimum L such that all 101 residues are covered by subsets of size <= L
    assignment = {}
    for L in range(1, 11):
        for residue, terms in subsets_by_size[L]:
            if residue not in assignment:
                assignment[residue] = terms
        if len(assignment) == P:
            return [assignment[r] for r in range(P)]
    
    # Should never reach here (L=10 covers all)
    return [assignment[r] if r in assignment else [1] for r in range(P)]
