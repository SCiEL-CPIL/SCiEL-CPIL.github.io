def construct():
    import time
    import random
    import numpy as np

    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill

    N = 17
    MIN_D = 6
    SIZE = 1 << N

    # Precompute weights for all 17-bit vectors
    # Using a lookup table for speed
    weights = np.fromiter((i.bit_count() for i in range(SIZE)), dtype=np.int8, count=SIZE)

    def try_greedy_linear(order, time_limit):
        """
        Try to build a linear code using candidates in the given order.
        Returns the list of codewords.
        """
        current_code = [0]
        basis = []
        target_basis_size = 8

        for g in order:
            if time.time() > time_limit:
                break
            if len(basis) >= target_basis_size:
                break

            # Check if g can be added
            valid = True
            # Optimization: iterate through current_code
            # current_code size is 2^len(basis)
            for c in current_code:
                if weights[g ^ c] < MIN_D:
                    valid = False
                    break

            if valid:
                basis.append(g)
                # Expand code
                # New code = current_code U {c ^ g for c in current_code}
                new_words = [c ^ g for c in current_code]
                current_code.extend(new_words)

        return current_code

    # Generate candidate order: sorted by weight descending
    # This is a good heuristic for finding high-distance codes
    indices = list(range(1, SIZE))
    indices.sort(key=lambda x: -weights[x])

    # Phase 1: Find a 256-word linear code
    # Try deterministic high-weight order first
    code = try_greedy_linear(indices, deadline - 30)
    best_code = code
    best_size = len(code)

    # If we didn't find 256, try a few random restarts with different shuffles
    if best_size < 256:
        for _ in range(5):
            if time.time() > deadline - 30:
                break
            random.shuffle(indices)
            code = try_greedy_linear(indices, deadline - 30)
            if len(code) > best_size:
                best_code = code
                best_size = len(code)

    # Phase 2: Non-linear extension
    # We have a linear code C of size best_size (ideally 256).
    # We want to add words w not in C such that d(w, c) >= 6 for all c in C.
    # Then, among the added words, we need to ensure pairwise distance >= 6.
    
    # Strategy:
    # 1. Find all "valid" vectors w (not in C) such that min_{c in C} d(w, c) >= 6.
    # 2. Select a maximal subset of these valid vectors with pairwise distance >= 6.
    
    C = best_code
    C_set = set(C)
    
    # If we have a 256-word code, the number of valid vectors might be large.
    # Let's find them efficiently.
    # For each w not in C, compute min_{c in C} weights[w ^ c].
    
    # Optimization:
    # We can scan all w from 0 to SIZE-1.
    # Skip if w in C.
    # Compute dists = weights[w ^ C_np]
    # If min(dists) >= 6, add w to valid_candidates.
    
    C_np = np.array(C, dtype=np.int32)
    
    valid_candidates = []
    
    # We can vectorize the check for chunks of w?
    # Or just loop. 131k iterations * (check against 256 words)
    # 131k * 256 = 33M operations.
    # With numpy, weights[w ^ C_np] is a vectorized operation on an array of 256 elements.
    # This should be fast enough.
    
    # To improve the selection in Phase 2, we want a good ordering for the greedy selection of valid_candidates.
    # High weight candidates are often "more independent".
    
    # Let's collect valid candidates first.
    # Time budget for this phase: 30s
    # Time budget for selection: remaining time
    
    t_start_phase2 = time.time()
    time_limit_phase2 = t_start_phase2 + 20
    
    for w in range(SIZE):
        if time.time() > time_limit_phase2:
            break
        if w in C_set:
            continue
        
        # Check distance to C
        # dists = weights[w ^ C_np]
        # min_d = dists.min()
        
        # Vectorized
        dists = weights[w ^ C_np]
        if dists.min() >= MIN_D:
            valid_candidates.append(w)

    # Now we have valid_candidates.
    # We need to select a subset S of valid_candidates such that:
    # 1. d(u, v) >= 6 for all u, v in S.
    # 2. |S| is maximized.
    
    # Greedy selection:
    # Order valid_candidates in some way.
    # Iterate through ordered list, add to S if compatible with current S.
    
    # What ordering?
    # Random? Weight?
    # Let's try multiple random shuffles of valid_candidates and keep the best S.
    
    best_extension = []
    time_limit_phase3 = deadline - 2
    
    # If valid_candidates is empty, we are done.
    if not valid_candidates:
        return best_code
    
    # Number of restarts depends on time available and size of valid_candidates
    # Let's do up to 20 restarts or until time is up.
    # Each restart: shuffle valid_candidates, greedy select.
    # Greedy selection cost: O(|V| * |S| * check_cost).
    # check_cost is O(1) if we use precomputed weights? No, we need weights[u ^ v].
    # weights is a lookup table, so weights[u ^ v] is O(1).
    
    # However, iterating |V| * |S| in Python might be slow if |V| is large.
    # |V| could be in the thousands or tens of thousands.
    # |S| will be small (maybe < 50).
    # 10k * 50 = 500k operations per restart.
    # 20 restarts = 10M operations.
    # Python loop overhead might make this slow.
    # Let's optimize the inner loop or use a different strategy.
    
    # Alternative:
    # Use a bitset or similar structure?
    # Or just try a few restarts and see.
    
    # Let's try 5 restarts first.
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > time_limit_phase3:
            break
            
        random.shuffle(valid_candidates)
        current_extension = []
        
        for v in valid_candidates:
            if time.time() > time_limit_phase3:
                break
                
            # Check compatibility with current_extension
            # current_extension is a list of ints
            valid = True
            for c in current_extension:
                if weights[v ^ c] < MIN_D:
                    valid = False
                    break
            
            if valid:
                current_extension.append(v)
                
        if len(current_extension) > len(best_extension):
            best_extension = current_extension
            
    # Combine
    final_code = best_code + best_extension
    
    return final_code
