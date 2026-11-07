import time
import random
import itertools

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit
    
    n = 12
    m = 40
    
    def popcount(x):
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c
    
    def score_sets(sets):
        """Count comparable pairs where one is strict subset of other."""
        count = 0
        for i in range(m):
            for j in range(i + 1, m):
                if sets[i] & sets[j] == sets[i] and sets[i] != sets[j]:
                    count += 1
                elif sets[i] & sets[j] == sets[j] and sets[i] != sets[j]:
                    count += 1
        return count
    
    def score_sets_fast(sets):
        """Faster scoring using bitmasks."""
        count = 0
        masks = [0] * m
        for i, s in enumerate(sets):
            mask = 0
            for v in s:
                mask |= (1 << v)
            masks[i] = mask
        for i in range(m):
            for j in range(i + 1, m):
                a, b = masks[i], masks[j]
                if a != b and (a & b) == min(a, b, key=popcount) if popcount(a) < popcount(b) else (a & b) == b:
                    # a is subset of b
                    if (a & b) == a:
                        count += 1
                elif a != b and (a & b) == b:
                    count += 1
        return count
    
    def score_sets_bitmasks(masks):
        """Score using precomputed bitmasks."""
        count = 0
        for i in range(m):
            for j in range(i + 1, m):
                a, b = masks[i], masks[j]
                if a != b:
                    if (a & b) == a:
                        count += 1
                    elif (a & b) == b:
                        count += 1
        return count
    
    def masks_to_sets(masks):
        return [[v for v in range(n) if (mask >> v) & 1] for mask in masks]
    
    def sets_to_masks(sets):
        masks = []
        for s in sets:
            mask = 0
            for v in s:
                mask |= (1 << v)
            masks.append(mask)
        return masks
    
    # Precompute all subset masks for reference
    all_masks = list(range(1 << n))
    
    # Strategy: Build a dense structure
    # Start with a chain-like structure but with branching
    # Key idea: Use sets of sizes 0,1,2,3,... arranged so small sets are subsets of many large sets
    
    # Initial construction: greedy approach
    # Start with empty set, then add sets that maximize comparabilities
    
    def greedy_init():
        # Start with a base chain
        masks = []
        # Build a structure with multiple chains
        # Use sets: {}, {0}, {0,1}, {0,1,2}, ... but also branch
        # Better: use a "bushy" tree structure
        
        # Start with empty set
        masks.append(0)
        
        # Add singleton sets
        for i in range(n):
            masks.append(1 << i)
        
        # Add pairs that contain earlier elements
        used = set(masks)
        # Greedily add sets that maximize new comparabilities
        current_score = score_sets_bitmasks(masks)
        
        candidates = [mask for mask in range(1 << n) if mask not in used and mask != 0]
        random.shuffle(candidates)
        
        for cand in candidates:
            if len(masks) >= m:
                break
            if time.time() > deadline - 10:
                break
            # Check if adding this improves score
            new_masks = masks + [cand]
            new_score = score_sets_bitmasks(new_masks)
            if new_score > current_score:
                masks.append(cand)
                used.add(cand)
                current_score = new_score
            if len(masks) >= m:
                break
        
        # If we don't have enough, fill with random
        while len(masks) < m:
            cand = random.randint(1, (1 << n) - 1)
            if cand not in used:
                masks.append(cand)
                used.add(cand)
        
        return masks[:m]
    
    def local_search(masks, max_iters=5000):
        """Hill climbing with random restarts."""
        current = list(masks)
        current_score = score_sets_bitmasks(current)
        best = list(current)
        best_score = current_score
        
        used = set(current)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random index to replace
            idx = random.randint(0, m - 1)
            old_mask = current[idx]
            
            # Try random new masks
            improved = False
            for _ in range(50):
                new_mask = random.randint(0, (1 << n) - 1)
                if new_mask == old_mask or new_mask in used:
                    continue
                
                # Evaluate
                trial = current[:]
                trial[idx] = new_mask
                trial_score = score_sets_bitmasks(trial)
                
                if trial_score > current_score:
                    # Accept
                    used.discard(old_mask)
                    used.add(new_mask)
                    current[idx] = new_mask
                    current_score = trial_score
                    improved = True
                    
                    if current_score > best_score:
                        best = list(current)
                        best_score = current_score
                    break
            
            if not improved:
                # Try some structured moves: swap with similar size
                for _ in range(20):
                    new_mask = random.randint(0, (1 << n) - 1)
                    if new_mask == old_mask or new_mask in used:
                        continue
                    trial = current[:]
                    trial[idx] = new_mask
                    trial_score = score_sets_bitmasks(trial)
                    if trial_score > current_score:
                        used.discard(old_mask)
                        used.add(new_mask)
                        current[idx] = new_mask
                        current_score = trial_score
                        if current_score > best_score:
                            best = list(current)
                            best_score = current_score
                        break
        
        return best, best_score
    
    # Phase 1: Greedy initialization
    masks = greedy_init()
    best_masks = list(masks)
    best_score = score_sets_bitmasks(best_masks)
    
    # Phase 2: Local search with multiple restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Perturb the best solution
        perturbed = list(best_masks)
        for _ in range(random.randint(5, 15)):
            idx = random.randint(0, m - 1)
            perturbed[idx] = random.randint(0, (1 << n) - 1)
        
        # Ensure uniqueness
        seen = set()
        unique_perturbed = []
        for mask in perturbed:
            if mask not in seen:
                seen.add(mask)
                unique_perturbed.append(mask)
        while len(unique_perturbed) < m:
            new_mask = random.randint(0, (1 << n) - 1)
            if new_mask not in seen:
                seen.add(new_mask)
                unique_perturbed.append(new_mask)
        
        # Local search from perturbed state
        result, result_score = local_search(unique_perturbed, max_iters=2000)
        if result_score > best_score:
            best_masks = result
            best_score = result_score
    
    # Phase 3: Final polish - try targeted improvements
    if time.time() < deadline - 5:
        used = set(best_masks)
        for idx in range(m):
            if time.time() > deadline - 3:
                break
            old_mask = best_masks[idx]
            # Try all masks of similar size
            target_size = popcount(old_mask)
            for delta in [-1, 0, 1]:
                target = target_size + delta
                if target < 0 or target > n:
                    continue
                # Sample some candidates of this size
                for _ in range(30):
                    if target == 0:
                        new_mask = 0
                    else:
                        elems = random.sample(range(n), min(target, n))
                        new_mask = 0
                        for e in elems:
                            new_mask |= (1 << e)
                    
                    if new_mask == old_mask or new_mask in used:
                        continue
                    
                    trial = best_masks[:]
                    trial[idx] = new_mask
                    trial_score = score_sets_bitmasks(trial)
                    if trial_score > best_score:
                        used.discard(old_mask)
                        used.add(new_mask)
                        best_masks[idx] = new_mask
                        best_score = trial_score
    
    # Convert to lists of integers
    result = []
    for mask in best_masks:
        s = []
        for v in range(n):
            if (mask >> v) & 1:
                s.append(v)
        result.append(s)
    
    # Verify distinctness and correctness
    assert len(result) == m
    assert len(set(tuple(x) for x in result)) == m
    for s in result:
        assert all(isinstance(x, int) for x in s)
        assert len(set(s)) == len(s)
        assert all(0 <= x < n for x in s)
    
    return result
