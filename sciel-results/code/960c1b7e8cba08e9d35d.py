import time
import random
import numpy as np

def construct(q, n, radius):
    # Fallback for general cases
    if q != 3 or n != 8 or radius != 1:
        total_words = q ** n
        
        def word_to_idx(w):
            idx = 0
            for val in w:
                idx = idx * q + val
            return idx
            
        def idx_to_word(idx):
            w = [0] * n
            temp = idx
            for i in range(n - 1, -1, -1):
                w[i] = temp % q
                temp //= q
            return tuple(w)

        neighbors = [None] * total_words
        for idx in range(total_words):
            w = idx_to_word(idx)
            nbrs = [idx]
            for i in range(n):
                for val in range(q):
                    if val != w[i]:
                        nw = list(w)
                        nw[i] = val
                        nbrs.append(word_to_idx(tuple(nw)))
            neighbors[idx] = nbrs

        uncovered = [True] * total_words
        code = []
        
        while any(uncovered):
            best_idx = -1
            best_count = -1
            for idx in range(total_words):
                count = sum(1 for nbr in neighbors[idx] if uncovered[nbr])
                if count > best_count:
                    best_count = count
                    best_idx = idx
            
            if best_idx == -1:
                break
                
            code.append(best_idx)
            for nbr in neighbors[best_idx]:
                uncovered[nbr] = False
                
        return [list(idx_to_word(idx)) for idx in code]

    # Specific optimization for q=3, n=8, radius=1
    total_words = 3 ** 8  # 6561
    start_time = time.time()
    deadline = start_time + 115  # Safe margin

    # Precompute neighbors
    neighbors = [None] * total_words
    neighbors_np = [None] * total_words
    for idx in range(total_words):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        
        nbrs = [idx]
        for i in range(8):
            orig_val = w[i]
            for val in range(3):
                if val != orig_val:
                    nw = w[:]
                    nw[i] = val
                    new_idx = 0
                    for v in nw:
                        new_idx = new_idx * 3 + v
                    nbrs.append(new_idx)
        neighbors[idx] = nbrs
        neighbors_np[idx] = np.array(nbrs, dtype=np.int32)

    def idx_to_word(idx):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        return w

    # 1. Greedy Construction
    uncovered_mask = np.ones(total_words, dtype=bool)
    code_set = set()
    code_list = []
    
    # Precompute gains for initial greedy steps? 
    # Just do standard greedy with sampling
    while np.any(uncovered_mask) and time.time() < deadline - 40:
        best_idx = -1
        best_gain = -1
        
        # Sample candidates
        if len(code_set) > 50:
            candidates_pool = random.sample([i for i in range(total_words) if i not in code_set], min(200, total_words - len(code_set)))
        else:
            candidates_pool = list(range(total_words))
            
        for i in candidates_pool:
            gain = np.sum(uncovered_mask[neighbors_np[i]])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1 or best_gain == 0:
            # Fallback: full scan
            for i in range(total_words):
                if i in code_set: continue
                gain = np.sum(uncovered_mask[neighbors_np[i]])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1: break
            
        code_set.add(best_idx)
        code_list.append(best_idx)
        
        for nbr in neighbors[best_idx]:
            uncovered_mask[nbr] = False

    # Ensure validity
    while np.any(uncovered_mask):
         best_idx = -1
         best_gain = -1
         for i in range(total_words):
             if i in code_set: continue
             gain = np.sum(uncovered_mask[neighbors_np[i]])
             if gain > best_gain:
                 best_gain = gain
                 best_idx = i
         if best_idx == -1: break
         code_set.add(best_idx)
         code_list.append(best_idx)
         for nbr in neighbors[best_idx]:
             uncovered_mask[nbr] = False

    code_list = list(code_set)
    
    # 2. Local Search: ILS with Kick and Fill
    
    def update_counts_add(c):
        cover_count[neighbors_np[c]] += 1
        
    def update_counts_remove(c):
        cover_count[neighbors_np[c]] -= 1
        
    def is_redundant(c):
        return np.min(cover_count[neighbors_np[c]]) >= 2

    def remove_redundant_pass():
        changed = False
        # Iterate multiple times to catch chains
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            any_changed = False
            for c in list(code_list):
                if is_redundant(c):
                    code_list.remove(c)
                    code_set.discard(c)
                    update_counts_remove(c)
                    any_changed = True
                    changed = True
            if not any_changed:
                break
        return changed

    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code_list:
        update_counts_add(c)
    
    # Initial cleanup
    remove_redundant_pass()
    
    best_size = len(code_list)
    best_code = list(code_list)
    
    # ILS Loop
    while time.time() < deadline - 5:
        # Try to improve from current state
        # Kick: Remove a word, Find a replacement that covers holes
        # We accept moves that reduce size immediately.
        # We also accept moves that keep size same (neutral) to explore.
        
        c = random.choice(code_list)
        
        # Temporarily remove c
        update_counts_remove(c)
        code_set.discard(c)
        
        # Find holes
        holes_mask = cover_count == 0
        holes = np.where(holes_mask)[0]
        
        if len(holes) == 0:
            # c was redundant, but remove_redundant should have caught it.
            # Maybe it became redundant due to previous neutral moves?
            # Just leave it removed.
            remove_redundant_pass()
            if len(code_list) < best_size:
                best_size = len(code_list)
                best_code = list(code_list)
            continue
        
        # Find candidate c' that covers all holes
        # Holes are words that are now uncovered.
        # c' must be in neighbors[h] for all h in holes.
        
        candidates = set()
        # Use the first hole to generate candidates
        h0 = holes[0]
        for cand in neighbors[h0]:
            if cand in code_set:
                # If cand is already in code, adding it does nothing (or we can just check if it covers holes)
                # But we removed c. If cand is in code_set, it was there before.
                # If it covers holes, then c was redundant?
                # No, if cand is in code_set, it contributes to cover_count.
                # If holes exist, cover_count is 0. So no existing word covers them.
                # So cand cannot be in code_set.
                pass
            # Check if cand covers all holes
            # Optimization: Check against all holes
            valid = True
            for h in holes[1:]:
                if cand not in neighbors[h]:
                    valid = False
                    break
            if valid:
                candidates.add(cand)
        
        if not candidates:
            # Revert
            update_counts_add(c)
            code_set.add(c)
            continue
        
        # Pick a candidate
        # Heuristic: Pick one that maximizes overlap with existing code (creates redundancy)
        best_cand = None
        best_score = -1
        for cand in candidates:
            # Score: number of neighbors of cand that are already covered (cover_count >= 1)
            # This indicates how much "redundancy" it creates
            score = np.sum(cover_count[neighbors_np[cand]] >= 1)
            if score > best_score:
                best_score = score
                best_cand = cand
        
        # Add best_cand
        code_set.add(best_cand)
        code_list.append(best_cand)
        update_counts_add(best_cand)
        
        # Try to remove redundancies
        remove_redundant_pass()
        
        if len(code_list) < best_size:
            best_size = len(code_list)
            best_code = list(code_list)
            # Restart ILS from this new best state? 
            # Or continue?
            # Restarting helps escape local minima.
            # Let's restart with a fresh random seed for the next kick sequence?
            # Actually, just continuing is fine, but we might get stuck.
            # Let's force a restart of the "kick sequence" by resetting the code to best_code?
            # No, best_code IS the current code.
            # Let's just continue.
            
    # Final cleanup
    # Rebuild code_list from best_code to ensure consistency
    code_list = list(best_code)
    code_set = set(code_list)
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code_list:
        update_counts_add(c)
    remove_redundant_pass()
    
    return [idx_to_word(c) for c in code_list]
