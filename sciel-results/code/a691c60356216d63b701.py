import random
import time
import sys

def compute_alpha(triples):
    n = 16
    # is_ind[mask] is True if the set of points in mask is independent
    # We can optimize this by precomputing triple masks
    triple_masks = []
    for t in triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
    
    # For each mask, check if it contains any triple
    # We iterate masks from 0 to 2^16 - 1
    # To speed up, we can check if mask is independent by checking if (mask & tm) == tm for any tm
    # This is O(2^16 * M). M ~ 40. 65536 * 40 = 2.6M ops. Fast enough.
    
    max_alpha = 0
    # We can skip masks that are known to be non-independent?
    # Or just iterate all.
    for mask in range(1 << n):
        is_ind = True
        for tm in triple_masks:
            if (mask & tm) == tm:
                is_ind = False
                break
        if is_ind:
            pop = bin(mask).count('1')
            if pop > max_alpha:
                max_alpha = pop
                # Early exit if max possible is reached?
                # Max possible is 16.
                # If we find 16, we can stop? No, we want min alpha.
                # We want to find the MAX alpha.
                # So we must check all masks to be sure we found the MAX.
    return max_alpha

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin for 150s limit
    
    # Precompute all possible triples
    all_triples = [(i, j, k) for i in range(16) for j in range(i+1, 16) for k in range(j+1, 16)]
    triple_pairs = {t: [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])] for t in all_triples}
    
    def greedy_init():
        random.shuffle(all_triples)
        used_pairs = set()
        triples = []
        for t in all_triples:
            pairs = triple_pairs[t]
            if all(p not in used_pairs for p in pairs):
                triples.append(t)
                for p in pairs:
                    used_pairs.add(p)
        return triples, used_pairs

    best_triples = None
    best_alpha = 16
    best_count = 0
    
    # Initial run
    triples, used_pairs = greedy_init()
    alpha = compute_alpha(triples)
    count = len(triples)
    
    if alpha < best_alpha or (alpha == best_alpha and count > best_count):
        best_alpha = alpha
        best_count = count
        best_triples = triples[:]

    current_triples = triples[:]
    current_used_pairs = used_pairs
    current_alpha = alpha
    current_count = count
    
    # Local Search
    # We perform swaps to maintain count, and try to reduce alpha.
    # We also try to add triples if count < max_possible (39?)
    
    # Let's find the max possible count for 16 points.
    # It is known to be 39? Or 40?
    # Actually, if STS(16) doesn't exist, max is 39?
    # Let's just try to maximize count first, then minimize alpha.
    
    while time.time() < deadline:
        # 1. Try to add a triple if possible
        if current_count < 40: # Upper bound
            # Find a random triple that can be added
            added = False
            for _ in range(20): # Try 20 random candidates
                t = random.choice(all_triples)
                pairs = triple_pairs[t]
                if all(p not in current_used_pairs for p in pairs):
                    # Add it
                    current_triples.append(t)
                    for p in pairs:
                        current_used_pairs.add(p)
                    current_count += 1
                    added = True
                    # Recompute alpha
                    new_alpha = compute_alpha(current_triples)
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                    # Update best
                    if current_alpha < best_alpha or (current_alpha == best_alpha and current_count > best_count):
                        best_alpha = current_alpha
                        best_count = current_count
                        best_triples = current_triples[:]
                    break
            if added:
                continue
        
        # 2. Swap a triple
        if current_count == 0:
            continue
            
        idx = random.randint(0, current_count - 1)
        old_t = current_triples[idx]
        
        # Remove old_t
        old_pairs = triple_pairs[old_t]
        for p in old_pairs:
            current_used_pairs.discard(p)
        current_triples.pop(idx)
        current_count -= 1
        
        # Find a new triple to add
        new_t = None
        for _ in range(50):
            candidate = random.choice(all_triples)
            if candidate == old_t:
                continue
            pairs = triple_pairs[candidate]
            if all(p not in current_used_pairs for p in pairs):
                new_t = candidate
                break
        
        if new_t is None:
            # Revert
            current_triples.append(old_t)
            for p in old_pairs:
                current_used_pairs.add(p)
            current_count += 1
            continue
        
        # Add new_t
        new_pairs = triple_pairs[new_t]
        for p in new_pairs:
            current_used_pairs.add(p)
        current_triples.append(new_t)
        current_count += 1
        
        # Evaluate
        new_alpha = compute_alpha(current_triples)
        
        # Acceptance
        if new_alpha < current_alpha:
            current_alpha = new_alpha
            if current_alpha < best_alpha or (current_alpha == best_alpha and current_count > best_count):
                best_alpha = current_alpha
                best_count = current_count
                best_triples = current_triples[:]
        elif new_alpha == current_alpha:
            # Accept to explore, maybe it leads to a better state later
            # Or reject to stay at best alpha?
            # Let's accept with 50% probability to allow movement
            if random.random() < 0.5:
                current_alpha = new_alpha
            else:
                # Revert
                current_triples.pop()
                for p in new_pairs:
                    current_used_pairs.discard(p)
                current_count -= 1
                current_triples.append(old_t)
                for p in old_pairs:
                    current_used_pairs.add(p)
                current_count += 1
                current_alpha = compute_alpha(current_triples) # Restore alpha
        else:
            # Revert
            current_triples.pop()
            for p in new_pairs:
                current_used_pairs.discard(p)
            current_count -= 1
            current_triples.append(old_t)
            for p in old_pairs:
                current_used_pairs.add(p)
            current_count += 1
            # Alpha remains current_alpha (which is better than new_alpha)
            # No need to recompute if we assume current_alpha was correct for the reverted state
            # But we modified the list, so we should be sure.
            # Actually, we reverted to the state before the move, so alpha is the same as before the move.
            # So no change to current_alpha.
            
    if best_triples is None:
        # Fallback
        triples, _ = greedy_init()
        return [list(t) for t in triples]
        
    return [list(t) for t in best_triples]
