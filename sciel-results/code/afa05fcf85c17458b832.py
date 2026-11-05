import time

def construct():
    start = time.time()
    deadline = 55.0  # Safety margin
    P = 101
    DENOMS = list(range(1, 11))
    n = len(DENOMS)
    
    # Precompute inverses
    inv = [0] * (n + 1)
    for x in DENOMS:
        inv[x] = pow(x, -1, P)

    # For each residue, store the best (smallest) subset found so far
    # best_for_residue[r] = (size, tuple_of_denominators)
    best_for_residue = [(10**9, ())] * P
    
    # Iterate over all non-empty subsets
    for mask in range(1, 1 << n):
        if time.time() - start > deadline:
            break
        
        # Quick size check: if current mask has more bits than the current best for its sum, skip detailed check?
        # Actually, we can just compute. 1023 iterations is very fast.
        terms = []
        s = 0
        for i in range(n):
            if (mask >> i) & 1:
                x = DENOMS[i]
                terms.append(x)
                s += inv[x]
        s %= P
        sz = len(terms)
        
        # Update if this subset is smaller than the current best for residue s
        if sz < best_for_residue[s][0]:
            best_for_residue[s] = (sz, tuple(terms))
        elif sz == best_for_residue[s][0]:
            # Tie-breaker: lexicographic order (optional, but keeps determinism)
            if tuple(terms) < best_for_residue[s][1]:
                best_for_residue[s] = (sz, tuple(terms))

    # Construct the result
    result = []
    for r in range(P):
        sz, terms = best_for_residue[r]
        if sz < 10**9:
            result.append(list(terms))
        else:
            # Fallback (should not trigger)
            result.append([1])

    return result
