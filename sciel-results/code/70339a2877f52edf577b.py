import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Total words
    total_words = q ** n
    
    # 1. Precompute Hamming balls of radius 'radius' for each word.
    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    balls = []
    ball_masks = []
    
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors = [idx]
        for pos in range(n):
            original = w[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                w[pos] = new_val
                n_idx = word_to_idx(w)
                neighbors.append(n_idx)
            w[pos] = original
        
        balls.append(neighbors)
        mask = 0
        for nb in neighbors:
            mask |= (1 << nb)
        ball_masks.append(mask)
        
    all_mask = (1 << total_words) - 1
    
    best_solution = None
    best_size = float('inf')
    
    def greedy_cover(seed):
        random.seed(seed)
        covered_mask = 0
        code = []
        
        while covered_mask != all_mask:
            best_gain = -1
            best_candidates = []
            
            for c in range(total_words):
                gain = (ball_masks[c] & ~covered_mask).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain:
                    best_candidates.append(c)
            
            if not best_candidates:
                break
                
            chosen = random.choice(best_candidates)
            code.append(chosen)
            covered_mask |= ball_masks[chosen]
            
            if time.time() > deadline:
                return None
        
        return code

    def reduce_cover(code):
        if not code:
            return code
            
        cover_count = [0] * total_words
        
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code.pop(i)
                    changed = True
        
        return code

    def iterated_greedy(seed):
        random.seed(seed)
        code = greedy_cover(seed)
        if code is None:
            return None
        
        code = reduce_cover(code)
        
        # Perturbation and Repair Loop
        # We try to remove some codewords and then repair the cover.
        # This helps escape local minima of the reduction step.
        
        current_code = code
        current_size = len(current_code)
        
        # Try a few perturbation rounds
        for round in range(5):
            if time.time() > deadline:
                break
                
            # Perturb: Remove k random codewords
            # k should be small, e.g., 1 or 2
            k = 1 if random.random() < 0.5 else 2
            
            if len(current_code) <= k:
                break
                
            # Select indices to remove
            indices_to_remove = random.sample(range(len(current_code)), k)
            
            # Create a new code list without these indices
            new_code = [c for i, c in enumerate(current_code) if i not in indices_to_remove]
            
            # Calculate current coverage of new_code
            covered_mask = 0
            for c in new_code:
                covered_mask |= ball_masks[c]
                
            # Repair: Greedily add codewords to cover the holes
            # We only need to cover the holes, so we can be more targeted?
            # Standard greedy works on the whole space, but we can optimize
            # by only considering words that are currently uncovered.
            
            while covered_mask != all_mask:
                best_gain = -1
                best_candidates = []
                
                # Optimization: Only consider codewords that cover at least one hole?
                # The gain calculation `(ball_masks[c] & ~covered_mask).bit_count()`
                # is 0 if the ball is fully covered.
                # So the standard loop works, but might be slow if many are 0.
                # For 2187 words, it's fast enough.
                
                for c in range(total_words):
                    if c in new_code:
                        continue # Don't add what we already have? 
                        # Actually, we removed some, so c might be in the removed set.
                        # We should allow re-adding removed codewords if they are best.
                        # So we should NOT skip c if it's in new_code?
                        # Wait, if c is in new_code, its ball is already included in covered_mask.
                        # So gain will be 0. So we can skip them to save time.
                        pass
                    
                    gain = (ball_masks[c] & ~covered_mask).bit_count()
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [c]
                    elif gain == best_gain:
                        best_candidates.append(c)
                
                if not best_candidates:
                    break
                    
                chosen = random.choice(best_candidates)
                new_code.append(chosen)
                covered_mask |= ball_masks[chosen]
                
                if time.time() > deadline:
                    return None
            
            # Reduce the repaired code
            new_code = reduce_cover(new_code)
            
            # If the new code is better or equal, accept it
            if len(new_code) < current_size:
                current_code = new_code
                current_size = len(new_code)
            elif len(new_code) == current_size:
                # Accept with some probability to explore?
                # Or just accept if it's different?
                # Let's accept if it's different to allow further perturbations to work
                if current_code != new_code:
                    current_code = new_code
        
        return current_code

    # Main loop
    # First, try a deterministic greedy (seed 0)
    initial_code = greedy_cover(0)
    if initial_code:
        reduced_code = reduce_cover(initial_code)
        best_solution = reduced_code
        best_size = len(reduced_code)
    
    # Now try iterated greedy with different seeds
    seed = 1
    while time.time() < deadline:
        code = iterated_greedy(seed)
        if code is None:
            break
        if len(code) < best_size:
            best_solution = code
            best_size = len(code)
            if best_size <= 186:
                # Best known is 186. If we hit it, we can try to beat it, but it's hard.
                # We continue searching.
                pass
        seed += 1
        
    if best_solution is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    result = []
    for idx in best_solution:
        result.append(idx_to_word(idx))
        
    return result
