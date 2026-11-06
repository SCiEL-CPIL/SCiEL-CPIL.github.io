import itertools
import time

def construct():
    MOD = 101
    MAX_DENOM = 10
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    denominators = list(range(1, MAX_DENOM + 1))
    
    # Precompute modular inverses
    inverses = {d: pow(d, -1, MOD) for d in denominators}
    
    # best_subset[r] will store the shortest list of denominators summing to r.
    best_subset = [None] * MOD
    
    # Iterate through all non-empty subsets by length to find minimum length for each residue
    for length in range(1, MAX_DENOM + 1):
        for combo in itertools.combinations(denominators, length):
            s = sum(inverses[d] for d in combo) % MOD
            
            # If we haven't found a subset for this residue, assign it.
            # Since we iterate by increasing length, the first one found is the shortest.
            if best_subset[s] is None:
                best_subset[s] = list(combo)
        
        # If we found a subset for all residues, we can stop early
        if all(x is not None for x in best_subset):
            break
        
        # Check deadline
        if time.time() > deadline:
            break

    # Construct the final result
    results = [None] * MOD
    max_len = 0
    for r in range(MOD):
        if best_subset[r] is not None:
            results[r] = best_subset[r]
            if len(results[r]) > max_len:
                max_len = len(results[r])
        else:
            # Fallback: find ANY valid subset for this residue
            for length in range(1, MAX_DENOM + 1):
                for combo in itertools.combinations(denominators, length):
                    s = sum(inverses[d] for d in combo) % MOD
                    if s == r:
                        results[r] = list(combo)
                        if len(results[r]) > max_len:
                            max_len = len(results[r])
                        break
                else:
                    continue
                break
            else:
                results[r] = [1]
                max_len = max(max_len, 1)
    
    return results
