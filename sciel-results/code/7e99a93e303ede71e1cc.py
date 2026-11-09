import time
from math import gcd
from itertools import combinations

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    # Precompute GCDs
    gcds = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g

    # Identify bad triples and conflicts
    # conflicts[i] will be a set of encoded pairs (a, b) such that {a, b, i} is a bad triple.
    # Encode pair (a, b) with a < b as a * (N + 1) + b
    def encode(a, b):
        if a > b:
            a, b = b, a
        return a * (N + 1) + b

    conflicts = [set() for _ in range(N + 1)]
    
    # Iterate through all triples
    # To optimize, we can just check all triples
    # C(128, 3) is ~340k, which is fine.
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            for k in range(j + 1, N + 1):
                if gcds[i][j] == gcds[j][k] == gcds[k][i]:
                    # This triple is bad
                    p_ik = encode(i, k)
                    p_ij = encode(i, j)
                    p_jk = encode(j, k)
                    
                    conflicts[i].add(p_jk)
                    conflicts[j].add(p_ik)
                    conflicts[k].add(p_ij)

    # Calculate degree for each number (number of conflicts)
    degrees = [len(conflicts[i]) for i in range(N + 1)]
    
    # Sort indices by degree ascending (least conflicting first)
    indices = list(range(1, N + 1))
    indices.sort(key=lambda x: degrees[x])
    
    selected = []
    selected_pairs = set()
    
    def can_add(x):
        # Check if any pair in selected_pairs is in conflicts[x]
        # conflicts[x] is a set of encoded pairs
        # We need to check intersection
        # Since selected_pairs can be large, and conflicts[x] is fixed,
        # it's faster to iterate over the smaller set?
        # Actually, we can just check if any p in conflicts[x] is in selected_pairs.
        for p in conflicts[x]:
            if p in selected_pairs:
                return False
        return True

    for x in indices:
        if time.time() > deadline:
            break
        if can_add(x):
            # Add x to selected
            selected.append(x)
            # Update selected_pairs
            for y in selected[:-1]:
                selected_pairs.add(encode(x, y))

    # Try to improve: 
    # 1. Try to add any remaining numbers in different orders? 
    #    The greedy order is fixed. We can try a few random restarts or shuffles if time permits.
    #    But deterministic is safer for reproducibility if not specified.
    #    However, the task says "Maximize cardinality".
    #    Let's try a simple local search: try to swap out a selected number for an unselected one if it increases size? No, size is fixed in swap.
    #    Try to add unselected numbers if we remove one selected number?
    
    # Let's try a second pass: 
    # Try to add numbers that were skipped, maybe by removing one conflicting selected number?
    # This is complex. 
    # Given the 50s budget, we can do multiple random restarts of the greedy.
    
    best_set = set(selected)
    best_size = len(selected)
    
    # Random restarts
    import random
    random.seed(42)
    
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Do a few random restarts
        for _ in range(10): # Max 10 restarts
            if time.time() > deadline:
                break
            
            # Shuffle indices slightly or use random order
            # A common trick: sort by degree + random noise
            noisy_indices = list(range(1, N + 1))
            random.shuffle(noisy_indices)
            # Sort by degree but with some randomness?
            # Or just use random order?
            # Let's try sorting by degree but shuffling ties or adding small noise.
            # Actually, just shuffling the order of processing often helps escape local optima of greedy.
            # But the "degree" heuristic is strong.
            # Let's try: Sort by (degree, random_value)
            
            # Re-initialize
            sel = []
            sel_pairs = set()
            
            # Use a mixed strategy: 50% degree order, 50% random?
            # Or just try random order a few times.
            
            # Let's try random order
            for x in noisy_indices:
                if time.time() > deadline:
                    break
                # Check can_add
                valid = True
                for p in conflicts[x]:
                    if p in sel_pairs:
                        valid = False
                        break
                if valid:
                    sel.append(x)
                    for y in sel[:-1]:
                        sel_pairs.add(encode(x, y))
            
            if len(sel) > best_size:
                best_size = len(sel)
                best_set = set(sel)
                
    return list(best_set)
