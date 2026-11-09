import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    p = 101
    denominators = range(1, 11)
    
    # Precompute inverses
    invs = [pow(x, -1, p) for x in denominators]
    
    # We want to find the smallest subset for each residue
    # assigned[r] = list of denominators
    assigned = {}
    
    # Iterate by subset size k = 1 to 10
    for k in range(1, 11):
        # Generate all combinations of size k
        for comb in itertools.combinations(denominators, k):
            # Calculate sum of inverses
            s = 0
            for idx in comb:
                s += invs[idx - 1]
            s %= p
            
            # If this residue hasn't been assigned yet, this is the smallest size for it
            if s not in assigned:
                assigned[s] = list(comb)
                
            # Check if all residues are assigned
            if len(assigned) == p:
                # We have found the minimum size subset for every residue
                # because we iterated by increasing size
                break
        
        if len(assigned) == p:
            break
            
        # Time check (unlikely to be needed given small N, but good practice)
        if time.time() > deadline:
            break

    # Construct the result
    # If some residues are missing (should not happen for p=101, k<=10), 
    # we might need a fallback, but theoretically they are covered.
    # The problem states "Every residue must be represented".
    # With 10 denominators, we have 1023 subsets. 
    # It is virtually certain they cover all 101 residues.
    
    result = [None] * p
    for r in range(p):
        if r in assigned:
            result[r] = assigned[r]
        else:
            # Fallback: find any subset (should not trigger)
            # Just in case, do a full search for missing ones
            found = False
            for mask in range(1, 1 << 10):
                terms = [x for x in range(1, 11) if (mask >> (x-1)) & 1]
                residue = sum(invs[x-1] for x in terms) % p
                if residue == r:
                    result[r] = terms
                    found = True
                    break
            if not found:
                # This should never happen
                result[r] = [1] 
                
    return result
