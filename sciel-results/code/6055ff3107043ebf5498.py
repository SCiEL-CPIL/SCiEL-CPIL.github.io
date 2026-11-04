# ... (existing code) ...

def _random_strassen_search(n, m, p, time_budget=10):
    """
    Randomly search for non-overlapping Strassen block placements.
    """
    best_rank = n * m * p
    best_result = None
    
    # Generate all possible 2x2x2 block positions
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_blocks.append((i0, j0, k0))
                
    import random
    random.seed(42)
    
    # Try random subsets of blocks
    for _ in range(1000):
        if _time_left() < 5:
            break
        # Pick a random number of blocks
        num_blocks = random.randint(1, len(all_blocks))
        # Pick random non-overlapping blocks
        blocks = []
        used = set()
        candidates = all_blocks.copy()
        random.shuffle(candidates)
        for b in candidates:
            if len(blocks) >= num_blocks:
                break
            # Check overlap
            overlap = False
            for db in blocks:
                # Check if b and db overlap
                # b: (i0, j0, k0), db: (i0', j0', k0')
                # Overlap if ranges intersect
                if not (b[0]+1 < db[0] or db[0]+1 < b[0] or b[1]+1 < db[1] or db[1]+1 < b[1] or b[2]+1 < db[2] or db[2]+1 < b[2]):
                    overlap = True
                    break
            if not overlap:
                blocks.append(b)
        
        if not blocks:
            continue
            
        try:
            U, V, W = _build_strassen_embedding(n, m, p, blocks)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
            
    return best_result, best_rank

def construct():
    # ... (existing code) ...
    # Add random search
    if _time_left() > 10:
        result, rank = _random_strassen_search(n, m, p)
        if result and rank < best_rank:
            best_rank = rank
            best_result = result
    # ...
