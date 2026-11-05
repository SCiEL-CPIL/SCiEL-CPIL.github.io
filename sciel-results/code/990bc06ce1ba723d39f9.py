import time

def construct():
    P = 101
    DENOMS = list(range(1, 11))
    INV = {x: pow(x, -1, P) for x in DENOMS}
    
    # Precompute all non-empty subsets with their sum mod P and size
    subsets_by_residue = {r: [] for r in range(P)}
    
    for mask in range(1, 1 << 10):
        terms = [DENOMS[i] for i in range(10) if (mask >> i) & 1]
        s = sum(INV[x] for x in terms) % P
        size = len(terms)
        subsets_by_residue[s].append((size, terms))
    
    # Find minimum k such that every residue has at least one subset of size <= k
    for k in range(1, 11):
        all_covered = True
        for r in range(P):
            if not any(sz <= k for sz, _ in subsets_by_residue[r]):
                all_covered = False
                break
        if all_covered:
            break
    
    # Assign to each residue the smallest-size subset (any one of min size)
    result = []
    for r in range(P):
        # Find min size subset for this residue
        min_size = min(sz for sz, _ in subsets_by_residue[r])
        # Pick one with min size
        chosen = None
        for sz, terms in subsets_by_residue[r]:
            if sz == min_size:
                chosen = terms
                break
        result.append(chosen)
    
    return result
