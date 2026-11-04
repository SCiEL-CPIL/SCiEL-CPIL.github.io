import time
import random
import numpy as np
from collections import Counter

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback: simple greedy set cover
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
                if idx in code:
                    continue
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
    
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * 3 + val
        return idx
        
    def idx_to_word(idx):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        return tuple(w)

    # Precompute neighbors for all words
    neighbors = [None] * total_words
    for idx in range(total_words):
        w = idx_to_word(idx)
        nbrs = [idx]
        for i in range(8):
            for val in range(3):
                if val != w[i]:
                    nw = list(w)
                    nw[i] = val
                    nbrs.append(word_to_idx(tuple(nw)))
        neighbors[idx] = nbrs

    # Convert to sets for faster intersection operations later
    neighbor_sets = [set(nbrs) for nbrs in neighbors]

    # 1. Greedy Construction
    uncovered = np.ones(total_words, dtype=bool)
    code_list = []
    code_set = set()
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s for cleanup and return
    
    # Simple greedy: pick word covering most uncovered
    while np.any(uncovered) and time.time() < deadline:
        best_idx = -1
        best_gain = -1
        # Optimization: We can sample or just scan. Scanning 6561 is fast.
        for i in range(total_words):
            if i in code_set:
                continue
            gain = 0
            for nbr in neighbors[i]:
                if uncovered[nbr]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1 or best_gain == 0:
            break
            
        code_set.add(best_idx)
        code_list.append(best_idx)
        
        for nbr in neighbors[best_idx]:
            uncovered[nbr] = False

    # 2. Local Search / Pruning
    # Convert to list for easy manipulation
    code_list = list(code_set)
    
    # Compute cover counts
    cover_count = np.zeros(total_words, dtype=int)
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1

    # Phase 2a: Remove redundant words
    # A word c is redundant if for all nbr in neighbors[c], cover_count[nbr] > 1
    # We iterate until no more removals.
    
    def try_remove_redundant():
        removed = False
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(code_list) - 1, -1, -1):
                c = code_list[i]
                redundant = True
                for nbr in neighbors[c]:
                    if cover_count[nbr] <= 1:
                        redundant = False
                        break
                if redundant:
                    code_list.pop(i)
                    for nbr in neighbors[c]:
                        cover_count[nbr] -= 1
                    removed = True
                    changed = True
                    break # Restart scan to maintain consistency
        return removed

    try_remove_redundant()

    # Phase 2b: Improved Kick and Prune with better swap heuristics
    # Strategy: Try swaps that maximize the number of words that become redundantly covered,
    # enabling subsequent removals. Also try multi-step lookahead.
    
    swap_attempts = 0
    max_swap_attempts = 50000  # Increased to allow more search within time budget
    
    # Precompute for faster scoring: for each word, store its neighbors as a numpy array
    neighbor_arrays = [np.array(nbrs, dtype=np.int32) for nbrs in neighbors]
    
    def score_swap_candidate(cp, current_cover_count):
        """Score a swap candidate: count how many of its neighbors would have cover >= 2 after swap."""
        nbrs = neighbor_arrays[cp]
        # After adding cp, cover_count increases by 1 for each nbr
        # We want to maximize the number of nbrs that will have cover >= 2
        # i.e., current_cover_count[nbr] + 1 >= 2, which means current_cover_count[nbr] >= 1
        # But more importantly, we want to create opportunities for future removals.
        # A better heuristic: count how many nbrs have cover_count == 1 (will become 2, creating redundancy)
        # plus how many have cover_count >= 2 (already redundant, no change in redundancy status)
        counts = current_cover_count[nbrs]
        # Score: number of nbrs that will have cover >= 2 after adding cp
        return np.sum(counts >= 1)  # These will become >= 2
    
    while time.time() < deadline and swap_attempts < max_swap_attempts:
        # 1. Try to remove
        if try_remove_redundant():
            continue # Restart swap search after successful removal
        
        # 2. No removal possible. Try a swap.
        if len(code_list) == 0:
            break
            
        # Pick a random word to remove
        idx_c = random.randint(0, len(code_list) - 1)
        c = code_list[idx_c]
        
        # Temporarily remove c
        for nbr in neighbors[c]:
            cover_count[nbr] -= 1
            
        # Find uncovered words
        uncovered_now = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]
        
        if not uncovered_now:
            # Should have been removable, but maybe my redundancy check missed it?
            # Or cover_count logic error. Just remove it.
            code_list.pop(idx_c)
            continue
        
        # Find a replacement c' that covers all uncovered_now
        # c' must be in the intersection of neighbors[u] for u in uncovered_now.
        
        # Calculate intersection of neighbor sets of uncovered words
        # Start with set of neighbors of first uncovered word
        candidates = set(neighbors[uncovered_now[0]])
        for u in uncovered_now[1:]:
            candidates.intersection_update(neighbor_sets[u])
            if not candidates:
                break
        
        # Remove current code words from candidates
        candidates.difference_update(code_list)
        
        if candidates:
            # Pick the best candidate using improved scoring
            cand_list = list(candidates)
            
            # If too many candidates, sample
            if len(cand_list) > 100:
                cand_list = random.sample(cand_list, 100)
            
            best_cp = -1
            best_score = -1
            
            for cp in cand_list:
                score = score_swap_candidate(cp, cover_count)
                if score > best_score:
                    best_score = score
                    best_cp = cp
            
            if best_cp != -1:
                # Perform swap
                code_list[idx_c] = best_cp
                for nbr in neighbors[best_cp]:
                    cover_count[nbr] += 1
                swap_attempts += 1
            else:
                # Revert
                for nbr in neighbors[c]:
                    cover_count[nbr] += 1
                swap_attempts += 1
        else:
            # No candidate found, revert
            for nbr in neighbors[c]:
                cover_count[nbr] += 1
            swap_attempts += 1

    # Phase 2c: Final intensive polish - try all possible single removals and swaps
    # This is a more exhaustive search near the end
    polish_deadline = min(deadline, time.time() + 15)
    
    # Try removing each codeword in turn
    for i in range(len(code_list) - 1, -1, -1):
        if time.time() > polish_deadline:
            break
        c = code_list[i]
        # Check if c is redundant
        redundant = True
        for nbr in neighbors[c]:
            if cover_count[nbr] <= 1:
                redundant = False
                break
        if redundant:
            code_list.pop(i)
            for nbr in neighbors[c]:
                cover_count[nbr] -= 1
    
    # Try a few more swaps if time permits
    if time.time() < polish_deadline:
        for _ in range(500):
            if time.time() > polish_deadline:
                break
            if not code_list:
                break
            
            idx_c = random.randint(0, len(code_list) - 1)
            c = code_list[idx_c]
            
            for nbr in neighbors[c]:
                cover_count[nbr] -= 1
            
            uncovered_now = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]
            
            if not uncovered_now:
                code_list.pop(idx_c)
                continue
            
            candidates = set(neighbors[uncovered_now[0]])
            for u in uncovered_now[1:]:
                candidates.intersection_update(neighbor_sets[u])
                if not candidates:
                    break
            
            candidates.difference_update(code_list)
            
            if candidates:
                cand_list = list(candidates)
                if len(cand_list) > 50:
                    cand_list = random.sample(cand_list, 50)
                
                best_cp = -1
                best_score = -1
                for cp in cand_list:
                    score = score_swap_candidate(cp, cover_count)
                    if score > best_score:
                        best_score = score
                        best_cp = cp
                
                if best_cp != -1:
                    code_list[idx_c] = best_cp
                    for nbr in neighbors[best_cp]:
                        cover_count[nbr] += 1
                else:
                    for nbr in neighbors[c]:
                        cover_count[nbr] += 1
            else:
                for nbr in neighbors[c]:
                    cover_count[nbr] += 1

    # Final cleanup
    try_remove_redundant()

    return [list(idx_to_word(c)) for c in code_list]
