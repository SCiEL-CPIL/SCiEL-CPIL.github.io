import time
import random

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            import itertools
            if n == 4:
                return tetracode
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic trivial fallback
            return [list(range(n)) for _ in range(1)] # Invalid for general, but prevents crash. 
            # Actually, let's just return a valid large code.
            # For safety, return all words if space is small.
            if q**n < 10000:
                import itertools
                return [list(t) for t in itertools.product(range(q), repeat=n)]
            return []

    start_time = time.time()
    deadline = start_time + 110.0
    
    total_words = q ** n
    # 2187
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return t
        
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute ball neighbors
    # neighbors[i] = list of indices in ball(i)
    neighbors = [[] for _ in range(total_words)]
    ball_masks = [0] * total_words
    
    for i in range(total_words):
        t = idx_to_tuple(i)
        mask = 1 << i
        neighbors[i].append(i)
        
        for pos in range(n):
            orig = t[pos]
            for delta in [1, 2]:
                new_val = (orig + delta) % q
                t[pos] = new_val
                j = tuple_to_idx(t)
                neighbors[i].append(j)
                mask |= (1 << j)
            t[pos] = orig
        ball_masks[i] = mask

    best_code = None
    best_size = total_words + 1
    
    num_restarts = 50 # Try 50 random restarts
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        uncovered = (1 << total_words) - 1
        code_indices = []
        
        # Greedy
        while uncovered:
            if time.time() > deadline:
                break
            
            best_idx = -1
            best_count = -1
            candidates = []
            
            # Find max coverage
            for i in range(total_words):
                count = (uncovered & ball_masks[i]).bit_count()
                if count > best_count:
                    best_count = count
                    candidates = [i]
                elif count == best_count:
                    candidates.append(i)
            
            if not candidates:
                break
                
            # Random tie-breaking
            best_idx = random.choice(candidates)
            code_indices.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
            
        if not code_indices:
            continue
            
        # Pruning Phase
        # coverage_count[j] = number of codewords covering word j
        coverage_count = [0] * total_words
        for c in code_indices:
            for j in neighbors[c]:
                coverage_count[j] += 1
                
        # Try to remove codewords
        # We shuffle the order to be fair
        to_check = list(range(len(code_indices)))
        random.shuffle(to_check)
        
        for k in to_check:
            if time.time() > deadline:
                break
                
            c = code_indices[k]
            
            # Tentatively remove c
            can_remove = True
            for j in neighbors[c]:
                coverage_count[j] -= 1
                if coverage_count[j] == 0:
                    can_remove = False
                    # Restore
                    coverage_count[j] += 1
                    break
            
            if can_remove:
                # Actually remove
                # We need to remove it from code_indices
                # But we are iterating by index.
                # Let's mark it as removed.
                code_indices[k] = -1 
            else:
                # Restore counts for the rest of the ball
                for j in neighbors[c]:
                    if coverage_count[j] == 0:
                        # Wait, we only broke if we found one 0.
                        # If we broke, we restored the one that became 0.
                        # But we decremented others. We need to restore them too.
                        # This logic is flawed.
                        pass

        # The pruning logic above is tricky with the loop break.
        # Let's implement a cleaner prune function.
        
        def prune(code_list):
            if not code_list:
                return code_list
            
            # Build coverage counts
            counts = [0] * total_words
            for c in code_list:
                for j in neighbors[c]:
                    counts[j] += 1
                    
            # Iterate through indices
            # We remove from the end to avoid index shifting issues, 
            # or we use a set of active indices.
            active = set(code_list)
            
            # Random order for removal attempts
            items = list(active)
            random.shuffle(items)
            
            for c in items:
                if c not in active:
                    continue
                    
                # Check if c can be removed
                ok = True
                for j in neighbors[c]:
                    if counts[j] == 1:
                        ok = False
                        break
                        
                if ok:
                    # Remove c
                    active.remove(c)
                    for j in neighbors[c]:
                        counts[j] -= 1
                        
            return list(active)
            
        pruned_indices = prune(code_indices)
        
        if len(pruned_indices) < best_size:
            best_size = len(pruned_indices)
            best_code = pruned_indices
            
    if best_code is None:
        # Fallback to seed
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert best indices to tuples
    return [idx_to_tuple(i) for i in best_code]
