import time
import itertools

def construct():
    start_time = time.time()
    LIMIT = 55.0  # Safety margin under 60s
    
    p = 101
    denominators = list(range(1, 11))
    n = 10
    
    # Precompute modular inverses
    inverses = [0] * (n + 1)
    for d in denominators:
        inverses[d] = pow(d, -1, p)
    
    # We want to assign to each residue r in [0, 100] a non-empty subset of {1..10}
    # such that sum of inverses mod 101 == r.
    # Since denominators can recur across rows, each row is independent.
    # To minimize the largest row length, we should assign the smallest possible
    # subset to each residue.
    
    # Initialize assignments with None
    assignments = [None] * p
    covered_count = 0
    
    # Iterate through all non-empty subsets by increasing size
    # This ensures that the first subset found for a residue is the smallest one.
    for size in range(1, n + 1):
        if time.time() - start_time > LIMIT:
            break
            
        for combo in itertools.combinations(denominators, size):
            s = 0
            for d in combo:
                s += inverses[d]
            residue = s % p
            
            if assignments[residue] is None:
                assignments[residue] = combo
                covered_count += 1
                
                # If all residues are covered, we can stop early
                if covered_count == p:
                    break
        
        if covered_count == p:
            break
            
    # If some residues are still None (should not happen with 1023 subsets for 101 residues),
    # fallback to any valid subset (e.g., the full set) to ensure validity.
    # The full set {1..10} has a specific residue, so this might not cover all,
    # but given the density of subsets, coverage is guaranteed well before size 10.
    # However, to be absolutely safe against any logical edge case or time limit break:
    if covered_count < p:
        # Fallback: try to fill remaining with largest subsets or any valid subset
        # This path is unlikely to be reached.
        for r in range(p):
            if assignments[r] is None:
                # Find any subset that sums to r
                for size in range(n, 0, -1):
                    found = False
                    for combo in itertools.combinations(denominators, size):
                        s = sum(inverses[d] for d in combo) % p
                        if s == r:
                            assignments[r] = combo
                            covered_count += 1
                            found = True
                            break
                    if found:
                        break
                if assignments[r] is None:
                    # Absolute fallback: use first denominator (incorrect residue, but valid structure)
                    # This would fail the check, so we should avoid this.
                    # Given 1023 subsets and 101 residues, collisions are abundant.
                    # The only way this fails is if we ran out of time or logic error.
                    pass

    # Construct the final list
    result = []
    for r in range(p):
        if assignments[r] is not None:
            result.append(list(assignments[r]))
        else:
            # Should not happen
            result.append([1]) 

    # Sanity check
    for r in range(p):
        combo = result[r]
        assert len(combo) > 0
        assert len(set(combo)) == len(combo)
        s = sum(inverses[d] for d in combo)
        assert s % p == r, f"Residue mismatch at r={r}: got {s % p}"
        assert all(1 <= d <= 10 for d in combo)
        
    return result
