import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback: simple greedy for other parameters
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
        neighbors = {}
        for idx in range(total_words):
            w = idx_to_word(idx)
            nbrs = {idx}
            for i in range(n):
                for val in range(q):
                    if val != w[i]:
                        nw = list(w)
                        nw[i] = val
                        nbrs.add(word_to_idx(tuple(nw)))
            neighbors[idx] = list(nbrs)
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

    # Precompute neighbors as lists and sets
    neighbors = [None] * total_words
    neighbor_sets = [None] * total_words
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
        neighbor_sets[idx] = set(nbrs)

    # Precompute distance-2 balls (words within distance <= 2)
    # This is used to know which words' gains change when we add a codeword
    dist2_balls = [None] * total_words
    for c in range(total_words):
        ball = set(neighbors[c])
        for n1 in neighbors[c]:
            for n2 in neighbors[n1]:
                ball.add(n2)
        dist2_balls[c] = list(ball)

    # Precompute intersection sizes: inter_size[a][b] = |neighbors[a] ∩ neighbors[b]|
    # This is symmetric. We'll store as a 2D array or compute on the fly.
    # 6561x6561 is ~43M entries, too big for memory if int.
    # But we only need it for pairs within distance 2.
    # Let's just compute on the fly using sets, it's fast enough.
    
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s for cleanup

    # --- Phase 1: Greedy Construction ---
    uncovered = np.ones(total_words, dtype=bool)
    gains = np.full(total_words, 17, dtype=np.int32)  # Initial gain is 17 for all
    
    code_set = set()
    code_list = []
    
    while np.any(uncovered) and time.time() < deadline:
        # Find max gain
        best_idx = -1
        best_gain = -1
        # Scan all non-code words
        for i in range(total_words):
            if i in code_set:
                continue
            if gains[i] > best_gain:
                best_gain = gains[i]
                best_idx = i
        
        if best_idx == -1 or best_gain == 0:
            break
            
        code_set.add(best_idx)
        code_list.append(best_idx)
        
        # Update uncovered
        for nbr in neighbors[best_idx]:
            uncovered[nbr] = False
            
        # Update gains for all w in dist2_balls[best_idx]
        # gain[w] decreases by |neighbors[w] ∩ neighbors[best_idx]|
        # But only for the part that was just covered.
        # Actually, gain[w] = sum(uncovered[nbr] for nbr in neighbors[w])
        # When we cover neighbors[best_idx], for any w, the new gain is:
        # gain[w] -= |neighbors[w] ∩ neighbors[best_idx]|
        # This holds because we are setting uncovered to False for all in neighbors[best_idx].
        
        nb_set = neighbor_sets[best_idx]
        for w in dist2_balls[best_idx]:
            if w in code_set:
                continue
            # Calculate intersection size
            # neighbors[w] and nb_set
            # Since neighbors are small (17), we can iterate
            dec = 0
            for nbr_w in neighbors[w]:
                if nbr_w in nb_set:
                    dec += 1
            gains[w] -= dec

    # --- Phase 2: Pruning ---
    # Build cover count array
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1

    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        for i in range(len(code_list) - 1, -1, -1):
            c = code_list[i]
            # Check if c is redundant
            redundant = True
            for nbr in neighbors[c]:
                if cover_count[nbr] <= 1:
                    redundant = False
                    break
            if redundant:
                code_list.pop(i)
                code_set.remove(c)
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                removed_any = True
                break

    # --- Phase 3: Local Search (Kick and Prune) ---
    # We try to swap words to allow further pruning.
    # Strategy:
    # 1. Try to remove any redundant word.
    # 2. If no removal, pick a random word to "kick out" (remove temporarily).
    # 3. Identify words that become uncovered.
    # 4. Find a replacement word that covers all newly uncovered words.
    # 5. If found, swap. Then try to prune again.
    
    swap_attempts = 0
    max_swap_attempts = 50000
    
    while time.time() < deadline and swap_attempts < max_swap_attempts:
        # 1. Try to remove
        removed = False
        for i in range(len(code_list) - 1, -1, -1):
            c = code_list[i]
            redundant = True
            for nbr in neighbors[c]:
                if cover_count[nbr] <= 1:
                    redundant = False
                    break
            if redundant:
                code_list.pop(i)
                code_set.remove(c)
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                removed = True
                break
        
        if removed:
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
            # Should have been removable
            code_list.pop(idx_c)
            code_set.remove(c)
            continue
        
        # Find a replacement c' that covers all uncovered_now
        # c' must be in the intersection of neighbors[u] for all u in uncovered_now
        # And c' must not be in code_set
        
        candidates = neighbor_sets[uncovered_now[0]]
        for u in uncovered_now[1:]:
            candidates.intersection_update(neighbor_sets[u])
            if not candidates:
                break
        
        if candidates:
            # Remove current code words from candidates
            candidates.difference_update(code_set)
            
            if candidates:
                # Pick a candidate.
                # Heuristic: Pick the one that maximizes the minimum cover_count of its neighbors?
                # Or just random? Random is often good for escaping local minima.
                # Let's pick the one that adds the most redundancy (max min cover_count of neighbors)
                # to facilitate future pruning.
                best_cp = -1
                best_score = -1
                # To speed up, we can just pick random from candidates if set is large
                # But let's try a small heuristic.
                for cp in candidates:
                    # Score: sum of cover_count of neighbors? Or min?
                    # Min cover_count indicates how "safe" the replacement is.
                    # We want to create redundancy, so we want neighbors with high cover_count.
                    # Let's use max(cover_count[nbr] for nbr in neighbors[cp])
                    # Actually, we want to ensure that after adding cp, we can prune something else.
                    # So we want cp to share neighbors with existing code.
                    # Score = sum(1 for nbr in neighbors[cp] if cover_count[nbr] > 0)
                    score = 0
                    for nbr in neighbors[cp]:
                        if cover_count[nbr] > 0:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_cp = cp
                
                if best_cp != -1:
                    # Perform swap
                    code_list[idx_c] = best_cp
                    code_set.add(best_cp)
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
        else:
            # No candidate found, revert
            for nbr in neighbors[c]:
                cover_count[nbr] += 1
            swap_attempts += 1

    return [list(idx_to_word(c)) for c in code_list]
