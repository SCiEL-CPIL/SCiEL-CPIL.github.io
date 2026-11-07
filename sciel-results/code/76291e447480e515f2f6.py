"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _max_distinct_subset_sums(exponents):
    """
    Find the largest subset of exponents such that all subset sums are distinct.
    """
    best_size = 0
    best_subset = []
    # Sort exponents to make it deterministic
    exps = sorted(exponents)
    n = len(exps)
    # Try all subsets
    for k in range(n, 0, -1):
        found = False
        for subset in itertools.combinations(exps, k):
            # Check if all subset sums are distinct
            sums = set()
            valid = True
            # Generate all subset sums of the chosen subset
            for r in range(len(subset) + 1):
                for s in itertools.combinations(subset, r):
                    val = sum(s)
                    if val in sums:
                        valid = False
                        break
                    sums.add(val)
                if not valid:
                    break
            if valid:
                best_size = k
                best_subset = list(subset)
                found = True
                break
        if found:
            break
    return best_subset

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    result = []
    
    for p in primes:
        # Find powers p^e <= 40
        powers = []
        e = 1
        val = p
        while val <= 40:
            powers.append((e, val))
            e += 1
            val = p ** e
        
        if not powers:
            continue
            
        exponents = [e for e, v in powers]
        best_exps = _max_distinct_subset_sums(exponents)
        
        # Map exponents back to values
        exp_to_val = {e: v for e, v in powers}
        for e in best_exps:
            result.append(exp_to_val[e])
            
        # Time check
        if time.time() > deadline:
            break

    # Sort the result for consistency (though order doesn't matter for validity)
    return sorted(result)
# EVOLVE-BLOCK-END
