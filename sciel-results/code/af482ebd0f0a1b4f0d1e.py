import itertools
import time

def construct():
    """
    Returns a list of 101 lists. Each list contains distinct denominators in 1..10
    such that the sum of their modular inverses modulo 101 equals the index.
    Minimizes the largest row length.
    """
    MOD = 101
    MAX_DENOM = 10
    denominators = list(range(1, MAX_DENOM + 1))
    
    # Precompute modular inverses
    inverses = {d: pow(d, -1, MOD) for d in denominators}
    
    # best[r] = (length, list_of_denoms)
    # We want to find the shortest subset for each residue r.
    # This minimizes the maximum row length and the sum of row lengths.
    
    best = [None] * MOD
    
    # Iterate by length to ensure we find the shortest subsets first
    for length in range(1, MAX_DENOM + 1):
        for combo in itertools.combinations(denominators, length):
            s = sum(inverses[d] for d in combo) % MOD
            if best[s] is None:
                best[s] = (length, list(combo))
        
        # Optimization: If all residues are covered, we can stop early
        # but we must ensure we've checked all lengths up to the current one.
        # Since we iterate by length, the first time we cover a residue is its min length.
        # We can stop if all are covered.
        if all(b is not None for b in best):
            break
    
    # Build results
    results = []
    for r in range(MOD):
        if best[r] is not None:
            results.append(best[r][1])
        else:
            # Fallback (should not happen for p=101, denom=10)
            # Find any subset
            found = False
            for length in range(1, MAX_DENOM + 1):
                for combo in itertools.combinations(denominators, length):
                    s = sum(inverses[d] for d in combo) % MOD
                    if s == r:
                        results.append(list(combo))
                        found = True
                        break
                if found:
                    break
            if not found:
                results.append([1])
                
    return results
