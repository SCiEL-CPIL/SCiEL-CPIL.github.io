import time
from collections import defaultdict

def construct():
    t0 = time.time()
    deadline = 80.0

    P = 101
    D = list(range(1, 11))

    # Precompute modular inverses
    inv = {x: pow(x, -1, P) for x in D}

    # Map residue to list of (length, sum_denoms, terms)
    # We want to find the best subset for each residue.
    # "Best" is defined as: Minimize length, then minimize sum of denominators.
    residue_options = defaultdict(list)
    
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        sum_denoms = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
                s = (s + inv[x]) % P
                sum_denoms += x
        length = len(terms)
        # Store (length, sum_denoms, terms)
        residue_options[s].append((length, sum_denoms, terms))
        
    # Check time
    if time.time() > deadline:
        # Fallback, though this is very fast
        pass

    # Select the best option for each residue
    result = []
    for r in range(P):
        options = residue_options[r]
        if not options:
            # This should not happen if a solution exists
            # Fallback to a dummy valid subset (e.g., [1] if inv[1]==r, else error)
            # But we know a solution exists.
            result.append([1]) 
            continue
            
        # Sort by length, then by sum of denominators
        # We only need the best one
        best_option = min(options, key=lambda x: (x[0], x[1]))
        result.append(best_option[2])
        
    return result
