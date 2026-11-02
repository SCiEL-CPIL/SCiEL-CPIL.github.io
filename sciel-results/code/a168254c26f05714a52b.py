import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin from 150s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute balls
    # weights for positions
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    
    # Vectorized precomputation
    # balls[:, 0] is the word itself
    # balls[:, 1 + 2*pos : 2 + 2*pos+1] are the neighbors?
    # Actually, let's just fill it.
    # For each word w, ball is [w] + [w + (d - w_d) * weight for pos in range(n) for d in range(3) if d != w_d]
    
    # This is a bit tricky to vectorize fully without loops, but 6561 is small.
    # Let's use a loop for precomputation, it's fast enough once.
    
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
                
    # Precompute reverse mapping: for each word w, which codewords cover it?
    # Actually, we can just use balls. w is covered by c if w in balls[c].
    # Since balls is symmetric (if w in balls[c] then c in balls[w]), we can use balls[w] to find candidates.
    
    best_size = total
    best_set = None
    
    # Helper to convert index to list of digits
    def to_digits(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        return digits

    # Strategy: Multiple restarts of greedy + local search (pruning + swaps)
    
    while time.time() < deadline - 5.0:
        # 1. Randomized Greedy Construction
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask):
            # Count how many uncovered words each candidate covers
            # uncovered_mask[balls] shape (total, 17)
            # sum along axis 1
            counts = uncovered_mask[balls].sum(axis=1)
            
            # Find max count
            max_count = np.max(counts)
            
            # If max_count is 0, we are stuck (should not happen if space is connected)
            if max_count == 0:
                break
            
            # Find all candidates with max count
            candidates = np.where(counts == max_count)[0]
            
            # Pick one randomly
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Update cover counts
            cover_count[balls[best_idx]] += 1
            
            # Update uncovered mask
            # A word is uncovered if cover_count is 0
            # We can just recompute uncovered_mask = (cover_count == 0)
            # Or update it: for w in balls[best_idx], if cover_count[w] was 0, it is now 1, so uncovered.
            # But cover_count[w] could be >1 if it was already covered.
            # So: new_uncovered = (cover_count == 0) is safest.
            uncovered_mask = (cover_count == 0)
            
            # Check time
            if time.time() > deadline - 10.0:
                break
        
        if np.any(uncovered_mask):
            # If we broke early due to time, this set might be invalid.
            # We should only consider it if it's valid.
            continue
        
        # 2. Pruning Phase
        # Remove codewords that are redundant.
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                # Check if c is removable
                # min(cover_count[balls[c]]) > 1
                if np.min(cover_count[balls[c]]) > 1:
                    # Remove c
                    cover_count[balls[c]] -= 1
                    changed = True
                    # Do not add to new_selected
                else:
                    new_selected.append(c)
            selected = new_selected
            
        # 3. Local Search Phase (Swaps and 2-exchanges)
        # We try to reduce the size by finding a set of codewords to remove and replace with fewer codewords.
        # Specifically, we look for:
        # A) 1-out-1-in (Swap): Replace c with d. Might enable pruning.
        # B) 2-out-1-in (Merge): Replace {c1, c2} with d. Reduces size by 1.
        
        improved = True
        max_iterations = 100
        iter_count = 0
        
        while improved and iter_count < max_iterations and time.time() < deadline - 5.0:
            improved = False
            iter_count += 1
            
            # Shuffle to avoid bias
            c_list = list(selected)
            random.shuffle(c_list)
            
            # --- Attempt 2-out-1-in ---
            # This is computationally heavier. We try a limited number of pairs.
            if len(selected) > 2:
                # Sample a few pairs
                pairs_to_try = []
                for _ in range(50):
                    if len(c_list) < 2: break
                    i = random.randint(0, len(c_list) - 1)
                    j = random.randint(0, len(c_list) - 1)
                    if i != j:
                        pairs_to_try.append((c_list[i], c_list[j]))
                
                for c1, c2 in pairs_to_try:
                    if time.time() > deadline - 5.0:
                        break
                    if c1 not in selected or c2 not in selected:
                        continue
                    
                    # Union of coverage of c1 and c2
                    # We need to find d such that Cover(c1) U Cover(c2) \subseteq Cover(d) U Cover(Selected \ {c1, c2})
                    # This is equivalent to:
                    # For all w in Cover(c1) U Cover(c2):
                    #   if cover_count[w] == 1 (only covered by c1 or c2), then w must be in Cover(d)
                    #   if cover_count[w] > 1, it's fine.
                    
                    # Unique words to c1 or c2
                    unique_words = []
                    for w in balls[c1]:
                        if cover_count[w] == 1:
                            unique_words.append(w)
                    for w in balls[c2]:
                        if cover_count[w] == 1:
                            # Check if it was also in c1's unique? 
                            # cover_count is 1, so it's unique to the pair.
                            # If w in balls[c1] and balls[c2], cover_count[w] would be 2.
                            # So if cover_count[w]==1, it's unique to one of them.
                            unique_words.append(w)
                            
                    # Remove duplicates in unique_words
                    unique_words = list(set(unique_words))
                    
                    if not unique_words:
                        # Both can be removed?
                        # If no unique words, they are redundant. Pruning should have caught this.
                        # But maybe cover_count changed?
                        cover_count[balls[c1]] -= 1
                        cover_count[balls[c2]] -= 1
                        selected.remove(c1)
                        selected.remove(c2)
                        improved = True
                        break
                    
                    # Find candidates d that cover all unique_words
                    candidates = set(balls[unique_words[0]])
                    for w in unique_words[1:]:
                        candidates.intersection_update(balls[w])
                        
                    # d cannot be c1 or c2
                    candidates.discard(c1)
                    candidates.discard(c2)
                    
                    # d should not be in selected (if it is, we are just removing c1,c2 and relying on existing d? 
                    # No, if d is in selected, we are effectively just removing c1,c2 if d covers the uniques.
                    # But we are looking for a *new* d or an existing one?
                    # If d is in selected, we just remove c1,c2. This is a 2-out-0-in move, which is just pruning.
                    # If d is NOT in selected, we remove c1,c2 and add d. Size -1.
                    
                    # Let's allow d to be any word covering uniques.
                    # If d in selected, we just remove c1,c2.
                    # If d not in selected, we swap {c1,c2} -> {d}.
                    
                    if candidates:
                        # Pick random candidate
                        d = random.choice(list(candidates))
                        
                        # Check if this move is valid
                        # We need to ensure that removing c1,c2 and adding d (if new) keeps all words covered.
                        # We already checked unique words.
                        # What about words covered by c1 or c2 that are also covered by others?
                        # They remain covered by others.
                        # What about words covered by d?
                        # They become covered by d (count increases or stays same).
                        
                        # Perform move
                        selected.remove(c1)
                        selected.remove(c2)
                        cover_count[balls[c1]] -= 1
                        cover_count[balls[c2]] -= 1
                        
                        if d not in selected:
                            selected.append(d)
                            cover_count[balls[d]] += 1
                            
                        improved = True
                        break
            
            if improved:
                # Prune after successful move
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
                continue # Restart loop
            
            # --- Attempt 1-out-1-in (Swap) ---
            # Try to replace c with d to enable future pruning or direct improvement (if d covers uniques of c and c covers uniques of d? No, just coverage maintenance).
            # Actually, 1-swap doesn't reduce size directly. It changes the set.
            # It is useful if the new set allows a 2-out-1-in that wasn't possible before.
            
            for c in c_list:
                if time.time() > deadline - 5.0:
                    break
                if c not in selected:
                    continue
                
                # Check if c is removable
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    selected.remove(c)
                    improved = True
                    break
                
                # Find unique words of c
                unique_words = []
                for w in balls[c]:
                    if cover_count[w] == 1:
                        unique_words.append(w)
                
                if not unique_words:
                    # Redundant
                    cover_count[balls[c]] -= 1
                    selected.remove(c)
                    improved = True
                    break
                
                # Find candidates d covering unique_words
                candidates = set(balls[unique_words[0]])
                for w in unique_words[1:]:
                    candidates.intersection_update(balls[w])
                
                selected_set = set(selected)
                candidates -= selected_set
                
                if candidates:
                    d = random.choice(list(candidates))
                    
                    # Swap
                    selected.remove(c)
                    selected.append(d)
                    
                    cover_count[balls[c]] -= 1
                    cover_count[balls[d]] += 1
                    
                    improved = True
                    break
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
    # Convert best_set to list of digit lists
    if best_set is None:
        return [to_digits(w) for w in range(total)]
        
    result = []
    for w in best_set:
        result.append(to_digits(w))
        
    return result
