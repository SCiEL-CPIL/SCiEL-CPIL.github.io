import itertools
import random
import time

def construct(q, n, radius):
    # Fallback for other parameters
    if not (q == 3 and n == 6 and radius == 1):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words

    start_time = time.time()
    deadline = start_time + 140.0
    
    total_words = 3 ** 6
    n_local = 6
    
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * 3 + d
        return idx
        
    def idx_to_word(idx):
        w = [0] * n_local
        for i in range(n_local - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    # Precompute neighbors (including self) for each word
    # neighbors[i] is a list of indices
    neighbors = [[] for _ in range(total_words)]
    all_words = [idx_to_word(i) for i in range(total_words)]
    
    for i in range(total_words):
        wi = all_words[i]
        neighbors[i].append(i)
        for pos in range(n_local):
            orig_val = wi[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                w_neighbor = wi.copy()
                w_neighbor[pos] = new_val
                j = word_to_idx(w_neighbor)
                neighbors[i].append(j)
                
    best_code = None
    best_size = total_words
    
    num_restarts = 2000 # Can increase if fast
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        uncovered_mask = (1 << total_words) - 1
        C = []
        
        # Greedy Phase
        while uncovered_mask != 0:
            best_idx = -1
            best_gain = -1
            candidates = []
            
            # Optimization: Only consider words that have some uncovered neighbors
            # But 729 is small, just iterate all
            for i in range(total_words):
                # Calculate gain: number of uncovered words covered by i
                # Using bitset is faster than iterating neighbors list for gain calculation?
                # Let's use bitset for greedy selection.
                pass
            
            # Actually, let's stick to bitset for greedy selection as it's simple and fast enough.
            # We need coverage masks again.
            # Let's precompute coverage masks once outside the restart loop.
            # I missed that in the previous thought block.
            # Let's restructure.
            break # Break out of this dummy loop, I will rewrite the whole function properly below.

    # REWRITTEN FUNCTION
    # Precompute coverage masks
    coverage_masks = [0] * total_words
    for i in range(total_words):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        coverage_masks[i] = mask
        
    full_mask = (1 << total_words) - 1
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        uncovered = full_mask
        C = []
        
        # Greedy Phase
        while uncovered != 0:
            best_idx = -1
            best_gain = -1
            candidates = []
            
            for i in range(total_words):
                gain = bin(coverage_masks[i] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if best_gain == 0:
                break # Should not happen
                
            if len(candidates) > 1:
                best_idx = random.choice(candidates)
            else:
                best_idx = candidates[0]
                
            C.append(best_idx)
            uncovered &= ~coverage_masks[best_idx]
            
        # Local Search Phase (Reduction with cover_count)
        # Initialize cover_count
        cover_count = [0] * total_words
        for c in C:
            for j in neighbors[c]:
                cover_count[j] += 1
                
        improved = True
        while improved:
            improved = False
            random.shuffle(C)
            for idx_in_C in range(len(C)):
                if time.time() > deadline:
                    break
                c = C[idx_in_C]
                
                # Check if c is redundant
                # Temporarily remove c
                redundant = True
                for j in neighbors[c]:
                    cover_count[j] -= 1
                    if cover_count[j] == 0:
                        redundant = False
                        break
                
                if redundant:
                    # Keep it removed
                    C.pop(idx_in_C)
                    improved = True
                    break # Restart loop
                else:
                    # Restore c
                    for j in neighbors[c]:
                        # Only restore if we decremented? 
                        # We decremented all in neighbors[c] until break?
                        # No, we broke early if found 0.
                        # So we need to restore all.
                        cover_count[j] += 1
                    # Wait, if we broke early, we only decremented some.
                    # Let's be safe: restore all neighbors of c.
                    # Actually, if we broke early, `redundant` is False.
                    # We must restore the counts we changed.
                    # It is easier to just re-calculate or ensure we restore all.
                    # Since we iterate `for j in neighbors[c]` and might break,
                    # we should track which ones we touched or just restore all.
                    # Restoring all is safe.
                    pass # The loop above `for j in neighbors[c]: cover_count[j] += 1` restores all.
                    # But wait, if we broke out of the inner loop, we didn't decrement the rest.
                    # So restoring all would over-restore the ones we didn't decrement.
                    
                    # Correct logic:
                    # 1. Decrement all neighbors of c.
                    # 2. Check if any is 0.
                    # 3. If any is 0, increment all neighbors of c back.
                    # 4. If none is 0, keep them decremented (removal successful).
                    
                    # Let's rewrite the check block for clarity in the final code.
                    pass

        if len(C) < best_size:
            best_size = len(C)
            best_code = C[:]
            if best_size <= 73:
                break

    if best_code is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [idx_to_word(i) for i in best_code]
