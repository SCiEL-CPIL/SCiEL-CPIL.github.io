<<<<<<< SEARCH
    # Specific optimization for q=3, n=6, r=1
    
    total_words = q ** n
    all_words = [list(map(int, list(itertools.product(range(q), repeat=n))))]
    # Map word to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Pre-compute balls for all possible centers
    # ball[i] is a set of indices covered by center i
    balls = [set() for _ in range(total_words)]
    
    for i, w in enumerate(all_words):
        balls[i].add(i)
        for pos in range(n):
            for delta in [1, -1]:
                new_w = w.copy()
                new_w[pos] = (w[pos] + delta) % q
                j = word_to_idx[tuple(new_w)]
                balls[i].add(j)
    
    best_code = None
    best_size = total_words # Upper bound
    
    # Run multiple randomized greedy attempts
    # Each attempt: shuffle centers, pick best cover, remove covered, repeat.
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        uncovered = set(range(total_words))
        current_code = []
        
        # Create a list of indices to iterate over, shuffled to break ties randomly
        candidates = list(range(total_words))
        random.shuffle(candidates)
        
        while uncovered:
            # Find the candidate that covers the most uncovered words
            best_cover = -1
            best_center = None
            # Optimization: If we have few uncovered left, we might find them faster
            # But scanning all 729 candidates is fast enough (729 * 13 ops)
            
            for c in candidates:
                # Count intersection with uncovered
                # Using set intersection is fast in Python
                count = len(balls[c] & uncovered)
                if count > best_cover:
                    best_cover = count
                    best_center = c
                    if best_cover == len(uncovered):
                        # Perfect cover for remaining
                        break
            
            if best_center is None or best_cover == 0:
                # Should not happen if balls cover everything
                break
                
            current_code.append(best_center)
            uncovered -= balls[best_center]
            
            if len(current_code) >= best_size:
                # No point continuing if already worse than best
                break
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code[:]
            
            # If we hit the theoretical lower bound (71) or known best (73), we can stop or keep trying
            if best_size <= 73:
                # We found a very good code. Let's do a few more to see if 71 is possible?
                # Lower bound is 71. Unlikely to find 71 with simple greedy.
                # But 73 is the target.
                pass

    if best_code is None:
        # Fallback to seed
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert indices back to word lists
    return [all_words[i] for i in best_code]
=======
    # Specific optimization for q=3, n=6, r=1
    
    total_words = q ** n
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Pre-compute balls for all possible centers
    balls = [set() for _ in range(total_words)]
    for i, w in enumerate(all_words):
        balls[i].add(i)
        for pos in range(n):
            for delta in [1, -1]:
                new_w = w.copy()
                new_w[pos] = (w[pos] + delta) % q
                j = word_to_idx[tuple(new_w)]
                balls[i].add(j)

    def get_tetracode_indices():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                w = h + list(tail)
                indices.append(word_to_idx[tuple(w)])
        return indices

    best_code = None
    best_size = total_words

    # Local Search: Start with Tetracode, try to remove redundant nodes
    # Then perturb/restart
    while time.time() < deadline:
        current_code = set(get_tetracode_indices())
        
        improved = True
        while improved:
            improved = False
            # Calculate coverage counts
            cover_count = [0] * total_words
            for c in current_code:
                for u in balls[c]:
                    cover_count[u] += 1
            
            # Try to remove any redundant codeword
            candidates_to_remove = list(current_code)
            random.shuffle(candidates_to_remove)
            
            for c in candidates_to_remove:
                if c not in current_code:
                    continue
                
                is_redundant = True
                for u in balls[c]:
                    if cover_count[u] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    current_code.remove(c)
                    for u in balls[c]:
                        cover_count[u] -= 1
                    improved = True
                    break # Restart the while loop to recalculate counts
            
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)
            
            if best_size <= 73:
                # We found a very good code. Keep trying to see if we can beat 73.
                pass

    if best_code is None:
        return [all_words[i] for i in get_tetracode_indices()]
    
    return [all_words[i] for i in best_code]
>>>>>>> REPLACE
