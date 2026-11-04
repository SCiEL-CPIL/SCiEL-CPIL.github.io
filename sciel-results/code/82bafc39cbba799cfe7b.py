    def run_greedy(random_tie_break=False):
        remaining = set(range(num_blocks))
        selected = []
        while remaining:
            candidates = []
            min_conf = float('inf')
            for b in remaining:
                c = len(conflict[b] & remaining)
                if c < min_conf:
                    min_conf = c
                    candidates = [b]
                elif c == min_conf:
                    candidates.append(b)
            
            if random_tie_break:
                bi = random.choice(candidates)
            else:
                bi = min(candidates) # Deterministic min index
            
            selected.append(bi)
            to_remove = {bi} | (conflict[bi] & remaining)
            remaining -= to_remove
        return len(selected), selected

    best_count = 0
    best_packing = []
    
    # Run deterministic greedy
    c, p = run_greedy(False)
    if c > best_count:
        best_count = c
        best_packing = p
        
    # Run randomized greedy multiple times
    for _ in range(200):
        c, p = run_greedy(True)
        if c > best_count:
            best_count = c
            best_packing = p
