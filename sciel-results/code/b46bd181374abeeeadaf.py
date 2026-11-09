import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    p = 101
    denominators = list(range(1, 11))
    n = 10
    
    # Precompute modular inverses
    inverses = [0] * (n + 1)
    for x in denominators:
        inverses[x] = pow(x, -1, p)
    
    # Generate all non-empty subsets and their corresponding residues
    # We want to process them in increasing order of size (number of denominators)
    # This ensures that for each residue, we assign the smallest possible subset.
    
    # List of (size, residue, tuple_of_denominators)
    all_subsets = []
    
    for size in range(1, n + 1):
        for combo in itertools.combinations(denominators, size):
            s = 0
            for d in combo:
                s += inverses[d]
            residue = s % p
            all_subsets.append((size, residue, combo))
            
    # Sort by size, then by residue for determinism (though not strictly necessary for correctness)
    # Since we generate by size, they are already grouped by size.
    # Within the same size, order doesn't affect the "max size" metric, 
    # but processing smaller sizes first is crucial.
    
    # assignments: residue -> tuple of denominators
    assignments = {}
    
    for size, residue, combo in all_subsets:
        if residue not in assignments:
            assignments[residue] = combo
            
        # If all residues are covered, we can stop immediately
        if len(assignments) == p:
            break
            
        # Time guard (should never trigger for this small problem size)
        if time.time() > deadline:
            break
            
    # Construct the result
    result = []
    for r in range(p):
        if r in assignments:
            result.append(list(assignments[r]))
        else:
            # Fallback: use all denominators (size 10)
            # This should not happen as 1023 subsets > 101 residues
            result.append(denominators[:])
            
    return result
