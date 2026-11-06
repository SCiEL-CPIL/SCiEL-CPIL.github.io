import time
import random
from itertools import combinations

def construct():
    # Precompute all 4-subsets of [1, 16] as bitmasks
    all_sets = []
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                for l in range(k+1, 16):
                    mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                    all_sets.append(mask)
    
    # Map mask to index for quick lookup?
    # Or just use a set of masks for F.
    
    def is_valid_addition(C, pairs):
        for A, B, K, S in pairs:
            if (C & K) == K and (C & S) == 0:
                return False
        return True

    def greedy_init():
        random.seed(42)
        candidates = all_sets[:]
        random.shuffle(candidates)
        
        F = []
        F_set = set()
        pairs = []
        
        for C in candidates:
            if is_valid_addition(C, pairs):
                F.append(C)
                F_set.add(C)
                # Update pairs
                for A in F[:-1]:
                    K = A & C
                    S = A ^ C
                    pairs.append((A, C, K, S))
        
        return F, F_set, pairs

    # Initialize
    F, F_set, pairs = greedy_init()
    
    deadline = time.time() + 55.0
    
    # Local Search
    while time.time() < deadline:
        # Try to add a random set
        C = random.choice(all_sets)
        if C in F_set:
            continue
            
        if is_valid_addition(C, pairs):
            # Add C
            F.append(C)
            F_set.add(C)
            for A in F[:-1]:
                K = A & C
                S = A ^ C
                pairs.append((A, C, K, S))
        else:
            # Try to swap
            # Find a conflicting pair
            conflict_A = None
            conflict_B = None
            for A, B, K, S in pairs:
                if (C & K) == K and (C & S) == 0:
                    conflict_A = A
                    conflict_B = B
                    break
            
            if conflict_A is not None:
                # Try swapping C with conflict_A
                # Check if F \ {conflict_A} U {C} is valid
                # This is equivalent to checking if C is valid against all pairs NOT involving conflict_A
                # AND checking if the new pairs involving C are valid?
                # Actually, if we remove A, the pairs involving A are gone.
                # We need to check C against remaining pairs.
                
                valid_swap = True
                for A2, B2, K2, S2 in pairs:
                    if A2 == conflict_A or B2 == conflict_A:
                        continue
                    if (C & K2) == K2 and (C & S2) == 0:
                        valid_swap = False
                        break
                
                if valid_swap:
                    # Perform swap
                    # Remove conflict_A from F, F_set
                    # Add C to F, F_set
                    # Update pairs:
                    # Remove all pairs involving conflict_A
                    # Add pairs (C, X) for all X in F \ {conflict_A}
                    
                    F.remove(conflict_A)
                    F_set.remove(conflict_A)
                    F.append(C)
                    F_set.add(C)
                    
                    new_pairs = []
                    for A2, B2, K2, S2 in pairs:
                        if A2 != conflict_A and B2 != conflict_A:
                            new_pairs.append((A2, B2, K2, S2))
                    
                    for A2 in F:
                        if A2 != C:
                            K2 = A2 & C
                            S2 = A2 ^ C
                            new_pairs.append((A2, C, K2, S2))
                    
                    pairs = new_pairs

    # Convert bitmasks back to sorted lists of 1-based indices
    result = []
    for mask in F:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
        
    return result
