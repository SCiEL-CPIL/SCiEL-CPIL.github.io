def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    
    # Precompute neighbors
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    best_size = total
    best_set = None
    
    # Precompute balls matrix once
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    # Generate Tetracode x Tetracode initial set
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
    
    init_words = []
    for u in tetracode:
        for v in tetracode:
            word = u + v
            # Convert to index
            idx = 0
            for i, digit in enumerate(word):
                idx += digit * weights[i]
            init_words.append(idx)
            
    init_set = set(init_words)

    for iteration in range(50):
        if time.time() > deadline - 5.0:
            break
            
        # 1. Greedy Construction starting from T x T
        selected = list(init_set)
        cover_count = np.zeros(total, dtype=np.int32)
        
        # Initialize cover count with init words
        for w in selected:
            cover_count[balls[w]] += 1
            
        uncovered_mask = (cover_count == 0)
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 10.0:
                break
                
            # Count coverage for each candidate
            # Optimization: Only consider candidates that cover at least one uncovered word?
            # balls[w] contains w and its neighbors.
            # counts[w] = number of uncovered words in balls[w]
            counts = uncovered_mask[balls].sum(axis=1)
            
            if np.max(counts) == 0:
                break
                
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best
            # To allow diversity, we can bias towards words that are "far" from existing ones?
            # Or just random.
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Update cover counts
            cover_count[balls[best_idx]] += 1
            
            # Update uncovered mask
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue # Invalid solution
            
        # 2. Local Search / Pruning
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 403:
                # Found optimal?
                pass

    # Convert best_set to output format
    if best_set is None:
        return [list(np.base_repr(w, 3).zfill(n)) for w in range(total)]
        
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
