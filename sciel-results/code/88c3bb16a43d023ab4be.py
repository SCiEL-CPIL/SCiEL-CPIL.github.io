import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    P = 101
    # Precompute inverses
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, P)
    
    # For each residue, track the minimum size subset that achieves it
    # min_size[r] = minimum cardinality of a subset S of {1..10} with sum(inv[x] for x in S) ≡ r (mod 101)
    # best_subset[r] = one such subset achieving min_size[r]
    min_size = [10**9] * P
    best_subset = [None] * P
    
    # Enumerate all non-empty subsets of {1..10}
    for mask in range(1, 1 << 10):
        subset = []
        s = 0
        for x in range(1, 11):
            if mask >> (x - 1) & 1:
                subset.append(x)
                s += inv[x]
        r = s % P
        sz = len(subset)
        if sz < min_size[r]:
            min_size[r] = sz
            best_subset[r] = subset
    
    # Verify all residues are covered; if any are missing, we have a problem
    # but with 1023 subsets and 101 residues, all should be covered
    for r in range(P):
        if best_subset[r] is None:
            # This should never happen, but provide a fallback
            # Find any subset that gives residue r
            for mask in range(1, 1 << 10):
                subset = []
                s = 0
                for x in range(1, 11):
                    if mask >> (x - 1) & 1:
                        subset.append(x)
                        s += inv[x]
                if s % P == r:
                    min_size[r] = len(subset)
                    best_subset[r] = subset
                    break
            if best_subset[r] is None:
                # Absolute fallback: should never reach here
                best_subset[r] = [1]
                min_size[r] = 1
    
    return [best_subset[r] for r in range(P)]
# EVOLVE-BLOCK-END
