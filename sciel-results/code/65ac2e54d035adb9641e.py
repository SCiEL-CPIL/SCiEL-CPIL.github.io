def construct():
    import math
    import time
    import random

    n = 128
    elements = list(range(1, n + 1))
    
    def get_lcm(a, b):
        return a * b // math.gcd(a, b)

    # Precompute bad triples
    # bad_mask[a][b] is a bitmask of c such that (a,b,c) is a bad triple
    bad_mask = [[0] * (n + 1) for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L = get_lcm(a, b)
            mask = 0
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if get_lcm(a, c) == L and get_lcm(b, c) == L:
                    mask |= (1 << (c - 1))
            bad_mask[a][b] = mask
            bad_mask[b][a] = mask

    def popcount(x):
        return bin(x).count('1')

    def get_conflicts(S_mask):
        """
        Returns a list of bad triples (as tuples of 1-indexed values) in S_mask.
        Also returns a dictionary of element -> count of conflicts it's involved in.
        """
        conflicts = []
        conflict_count = {}
        for a in range(1, n + 1):
            if not (S_mask & (1 << (a - 1))):
                continue
            for b in range(a + 1, n + 1):
                if not (S_mask & (1 << (b - 1))):
                    continue
                common = S_mask & bad_mask[a][b]
                # common contains c's that form a bad triple with a and b
                c_mask = common
                while c_mask:
                    # Get lowest set bit
                    lsb = c_mask & -c_mask
                    c_idx = lsb.bit_length() - 1
                    c_val = c_idx + 1
                    if c_val > b: # Ensure we count each triple once, with a < b < c
                        conflicts.append((a, b, c_val))
                        conflict_count[a] = conflict_count.get(a, 0) + 1
                        conflict_count[b] = conflict_count.get(b, 0) + 1
                        conflict_count[c_val] = conflict_count.get(c_val, 0) + 1
                    c_mask ^= lsb
        return conflicts, conflict_count

    def is_valid(S_mask):
        # Fast check: just see if any conflict exists
        for a in range(1, n + 1):
            if not (S_mask & (1 << (a - 1))):
                continue
            for b in range(a + 1, n + 1):
                if not (S_mask & (1 << (b - 1))):
                    continue
                if S_mask & bad_mask[a][b]:
                    return False
        return True

    def local_search(S_mask):
        """
        Iteratively repair S_mask until it is valid, then try to add elements.
        Returns the improved mask.
        """
        curr_mask = S_mask
        improved = True
        while improved:
            improved = False
            # Phase 1: Repair (remove elements to make valid)
            while True:
                conflicts, conflict_count = get_conflicts(curr_mask)
                if not conflicts:
                    break
                
                # Pick a conflict to resolve
                # Strategy: Resolve the conflict involving the element with highest degree
                # Or just pick the first conflict and remove the highest degree element from it
                # For efficiency, let's pick a random conflict if many, or the first one
                # We'll pick the conflict where the max degree element is highest?
                # Simpler: Pick first conflict, remove element with highest degree in that triple.
                
                # Actually, to be robust, let's find the element with the global highest degree among all elements in conflicts
                # and remove it. This breaks the most conflicts at once.
                if conflict_count:
                    # Find element with max conflict count
                    max_deg_elem = max(conflict_count, key=lambda k: conflict_count[k])
                    # Remove it
                    curr_mask &= ~(1 << (max_deg_elem - 1))
                    improved = True
                else:
                    break

            # Phase 2: Add (try to add elements to increase size)
            # We iterate through all elements not in set
            # To make it deterministic but effective, we can shuffle or sort by global degree
            # Let's sort by global degree (number of bad triples involving x in [1,128])
            # Precompute global degrees
            # But we can just try adding any element that fits.
            
            added = False
            # Iterate in a fixed order, e.g., 1..128, or random
            # Random is better for ILS
            candidates = [x for x in elements if not (curr_mask & (1 << (x - 1)))]
            random.shuffle(candidates)
            
            for x in candidates:
                new_mask = curr_mask | (1 << (x - 1))
                if is_valid(new_mask):
                    curr_mask = new_mask
                    added = True
                    # Continue trying to add more? 
                    # If we add x, it might block others. 
                    # Greedy add: just keep going.
            
            if added:
                improved = True
                # If we added something, we might need to re-check? 
                # No, is_valid ensures it's still valid.
                # But adding x might make it "harder" to add others? 
                # The loop above tries all candidates. If we add x, we continue to next candidate.
                # This is a greedy set cover style.
        
        return curr_mask

    def greedy_init():
        S_mask = 0
        # Sort by global degree? 
        # Elements with fewer constraints are "safer" to include? 
        # Or elements with more constraints are "rarer" and might be better to exclude?
        # Let's just random shuffle for diversity in ILS
        elems = elements[:]
        random.shuffle(elems)
        for x in elems:
            new_mask = S_mask | (1 << (x - 1))
            if is_valid(new_mask):
                S_mask = new_mask
        return S_mask

    rng = random.Random(42)
    start_time = time.time()
    max_time = 55.0
    
    best_mask = 0
    best_size = 0
    
    # Initial run
    curr_mask = greedy_init()
    curr_mask = local_search(curr_mask)
    curr_size = popcount(curr_mask)
    
    if curr_size > best_size:
        best_size = curr_size
        best_mask = curr_mask

    # Iterated Local Search
    iterations = 0
    while time.time() - start_time < max_time:
        iterations += 1
        
        # Perturbation: Randomly remove 2-5 elements and re-run local search
        # Or swap 1 element
        curr_mask = best_mask
        perturb_strength = rng.randint(2, 5)
        
        # Remove random elements
        in_set = [x for x in elements if (curr_mask & (1 << (x - 1)))]
        if len(in_set) > perturb_strength:
            to_remove = rng.sample(in_set, perturb_strength)
            for x in to_remove:
                curr_mask &= ~(1 << (x - 1))
        
        # Re-run local search
        new_mask = local_search(curr_mask)
        new_size = popcount(new_mask)
        
        if new_size > best_size:
            best_size = new_size
            best_mask = new_mask
        elif new_size == best_size and rng.random() < 0.1:
            # Accept equal size with small probability to explore
            best_mask = new_mask

    result = []
    for i in range(1, n + 1):
        if best_mask & (1 << (i - 1)):
            result.append(i)
            
    if not result:
        result = [1]
        
    return result
