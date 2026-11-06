import time
import random

def construct():
    n = 512
    start_time = time.time()
    time_limit = 58.0  # Leave 2s buffer
    
    # Precompute all 4-APs in [1, n]
    # A 4-AP is (a, a+d, a+2d, a+3d) with d >= 1
    # For each element x, precompute the list of 4-APs that contain x
    # This allows fast checking: when adding x, we just check if any of the other 3 elements in those APs are already in S
    
    # For each x, store the list of (y1, y2, y3) such that {x, y1, y2, y3} is a 4-AP
    # We need to check if all of y1, y2, y3 are in S
    
    # Actually, let's precompute conflicts differently:
    # For each pair (x, y) with x < y, compute d = y - x
    # Then check if x-3d, x-2d, x-d, x, x+d, x+2d, x+3d form APs
    
    # Better approach: For each x, precompute a bitmask of which other elements are in the same 4-AP as x
    # But that's complex. Let's use a different strategy.
    
    # Precompute for each x, the list of tuples (y1, y2, y3) such that adding x creates a 4-AP with y1, y2, y3 in S
    # We'll store this as a list of frozensets or just check on the fly with precomputed data
    
    # Let's precompute for each x, the set of "conflict triples" - sets of 3 elements such that if all 3 are in S, adding x creates a 4-AP
    
    # Actually, the most efficient way: For each x, precompute the list of (a, b, c) where {a, b, c, x} is a 4-AP
    # Then creates_4ap(S, x) is True iff there exists (a, b, c) in the precomputed list for x such that a in S and b in S and c in S
    
    # Precompute conflict_triples[x] = list of (a, b, c) tuples
    conflict_triples = [None] * (n + 1)
    for x in range(1, n + 1):
        triples = []
        # Case 1: x is first term: x, x+d, x+2d, x+3d
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            triples.append((x + d, x + 2 * d, x + 3 * d))
        
        # Case 2: x is second term: x-d, x, x+d, x+2d
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            triples.append((x - d, x + d, x + 2 * d))
        
        # Case 3: x is third term: x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            triples.append((x - 2 * d, x - d, x + d))
        
        # Case 4: x is fourth term: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            triples.append((x - 3 * d, x - 2 * d, x - d))
        
        conflict_triples[x] = triples
    
    # Helper: Check if adding x to set S creates a 4-AP using precomputed triples
    def creates_4ap(S, x):
        for (a, b, c) in conflict_triples[x]:
            if a in S and b in S and c in S:
                return True
        return False
    
    # Helper: Greedily fill the set given a starting set
    def greedy_fill(S, order=None):
        if order is None:
            order = list(range(1, n + 1))
        S = set(S)
        for x in order:
            if x not in S:
                if not creates_4ap(S, x):
                    S.add(x)
        return S

    # Initial Greedy Construction
    # We try a few different greedy orders to find a good starting point
    best_set = set()
    best_size = 0
    
    random.seed(42)
    
    # Standard order
    current = greedy_fill(set(), list(range(1, n + 1)))
    if len(current) > best_size:
        best_set = set(current)
        best_size = len(current)
        
    # Reverse order
    current = greedy_fill(set(), list(range(n, 0, -1)))
    if len(current) > best_size:
        best_set = set(current)
        best_size = len(current)
        
    # Random order - generate many diverse seeds
    num_random_tries = 100
    for _ in range(num_random_tries):
        if time.time() - start_time > time_limit * 0.3: # 30% time for init
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        current = greedy_fill(set(), order)
        if len(current) > best_size:
            best_set = set(current)
            best_size = len(current)

    # Local Search
    # Randomized Hill Climbing
    current_set = set(best_set)
    
    while time.time() - start_time < time_limit:
        # Try to add
        added = False
        candidates = [i for i in range(1, n + 1) if i not in current_set]
        
        if not candidates:
            break
        
        # Try a few random candidates
        random.shuffle(candidates)
        for x in candidates[:100]: # Check up to 100 random candidates
            if not creates_4ap(current_set, x):
                current_set.add(x)
                added = True
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
                break
        
        if not added:
            # Stuck. Perform a kick.
            # Remove 1-3 random elements
            k = random.randint(1, 3)
            if len(current_set) > k:
                to_remove = random.sample(list(current_set), k)
                for x in to_remove:
                    current_set.remove(x)
                
                # Now try to add as many as possible
                # Greedy fill from current state
                candidates = [i for i in range(1, n + 1) if i not in current_set]
                random.shuffle(candidates)
                for x in candidates:
                    if not creates_4ap(current_set, x):
                        current_set.add(x)
                
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)

    return sorted(best_set)
