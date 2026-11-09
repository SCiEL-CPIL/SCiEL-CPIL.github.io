import numpy as np
import time
import random

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback
        return []

    total_words = q ** n
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for covering (dist <= 1) and packing conflicts (dist <= 2)
    neighbors_cover = [] # dist <= 1
    neighbors_conflict = [] # dist <= 2
    
    for i in range(total_words):
        w = idx_to_word(i)
        cover = [i]
        conflict = [i]
        
        # dist 1
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig: continue
                w[pos] = val
                idx2 = word_to_idx(w)
                cover.append(idx2)
                conflict.append(idx2)
            w[pos] = orig
            
        # dist 2
        for p1 in range(n):
            for p2 in range(p1 + 1, n):
                o1 = w[p1]
                o2 = w[p2]
                for v1 in range(q):
                    if v1 == o1: continue
                    for v2 in range(q):
                        if v2 == o2: continue
                        w[p1] = v1
                        w[p2] = v2
                        idx2 = word_to_idx(w)
                        conflict.append(idx2)
                w[p1] = o1
                w[p2] = o2
                
        neighbors_cover.append(cover)
        neighbors_conflict.append(conflict)
        
    # Convert to sets for fast membership if needed, or lists for numpy indexing
    # For greedy covering, we need to sum uncovered[neighbors]. Lists are better for numpy fancy indexing.
    
    # 1. Generate 2-Packing (MIS on conflict graph)
    def generate_packing():
        available = set(range(total_words))
        packing = []
        while available:
            # Pick random vertex
            v = random.choice(list(available))
            packing.append(v)
            # Remove v and its conflicts
            to_remove = set(neighbors_conflict[v])
            available -= to_remove
        return packing

    best_packing = []
    # Run a few times to get a large packing
    for _ in range(20):
        p = generate_packing()
        if len(p) > len(best_packing):
            best_packing = p
            
    # 2. Initialize Covering Code with Packing
    code = set(best_packing)
    count = np.zeros(total_words, dtype=np.int32)
    for i in code:
        count[neighbors_cover[i]] += 1
        
    uncovered_mask = count == 0
    
    # 3. Greedy Completion
    while np.any(uncovered_mask):
        best_idx = -1
        best_cov = -1
        # Check all words? Or just uncovered?
        # A word that is already covered might cover more uncovered words?
        # Yes, if it's "central".
        # But usually, picking from uncovered is a good heuristic.
        # Let's check all words for speed vs quality. 2187 is small.
        for i in range(total_words):
            cov = np.sum(uncovered_mask[neighbors_cover[i]])
            if cov > best_cov:
                best_cov = cov
                best_idx = i
                
        if best_idx == -1:
            break
            
        code.add(best_idx)
        count[neighbors_cover[best_idx]] += 1
        uncovered_mask = count == 0

    # 4. Local Search
    start_time = time.time()
    time_limit = 115.0
    best_size = len(code)
    best_code = set(code)
    
    in_code = np.zeros(total_words, dtype=bool)
    for i in code:
        in_code[i] = True
        
    # Helper for greedy removal
    def try_remove(j):
        nonlocal count, in_code, code
        # Check if j is redundant
        # A word j is redundant if all its neighbors are covered by > 1 codeword
        # i.e., count[k] > 1 for all k in neighbors_cover[j]
        # But wait, if we remove j, count[k] decreases by 1 for k in neighbors_cover[j].
        # So we need count[k] > 1 for all k in neighbors_cover[j] BEFORE removal.
        if np.all(count[neighbors_cover[j]] > 1):
            code.remove(j)
            in_code[j] = False
            count[neighbors_cover[j]] -= 1
            return True
        return False

    while time.time() - start_time < time_limit:
        # 1. Try to remove redundant words
        changed = True
        while changed:
            changed = False
            for j in list(code):
                if try_remove(j):
                    changed = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = set(code)
                        
        # 2. Try 1-Swaps
        # Pick a codeword i, remove it, see if we can fix with one addition
        if code:
            i = random.choice(list(code))
            
            # Remove i
            code.remove(i)
            in_code[i] = False
            count[neighbors_cover[i]] -= 1
            
            # Check validity
            uncovered_idx = np.where(count == 0)[0]
            
            if len(uncovered_idx) == 0:
                # Success! We removed i and it's still valid.
                # This is a "deletion" not a "swap", but effectively a 1-step improvement.
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = set(code)
                # Try to remove more
                # (The loop above will handle this in next iteration, 
                # but we can do a quick pass here)
                continue
            
            # Need to fix uncovered_idx
            # Find k not in code such that k covers all uncovered_idx
            found_k = -1
            # Optimization: k must be in intersection of neighbors of all u in uncovered_idx
            # If uncovered_idx is large, intersection is likely empty.
            # Limit search: if len(uncovered_idx) > 5, skip?
            if len(uncovered_idx) < 10:
                # Candidate k must be neighbor to all u in uncovered_idx
                # Intersect neighbor sets
                # Since uncovered_idx is small, we can check candidates
                # Candidates: words not in code
                # To speed up, we can restrict to words that are "close" to the uncovered ones?
                # Just iterate all non-code words? 2187 is small.
                for k in range(total_words):
                    if in_code[k]:
                        continue
                    # Check if k covers all uncovered
                    # count[k] is not useful here, we need to check neighbors
                    # Actually, if we add k, new_count[u] = count[u] + (1 if u in neighbors_cover[k] else 0)
                    # We need new_count[u] > 0 for all u in uncovered_idx.
                    # Since count[u] == 0 for u in uncovered_idx, we need u in neighbors_cover[k].
                    
                    # Efficient check:
                    # Precomputed neighbor sets?
                    # We have neighbors_cover[k] as list.
                    # Check if all u in uncovered_idx are in neighbors_cover[k].
                    # Convert uncovered_idx to set for O(1) lookup?
                    # Or use numpy:
                    # np.all(np.isin(uncovered_idx, neighbors_cover[k]))
                    # This is fast enough for small uncovered_idx.
                    
                    if np.all(np.isin(uncovered_idx, neighbors_cover[k])):
                        found_k = k
                        break
            
            if found_k != -1:
                # Add k
                code.add(found_k)
                in_code[found_k] = True
                count[neighbors_cover[found_k]] += 1
                
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = set(code)
            else:
                # Failed to fix, revert
                code.add(i)
                in_code[i] = True
                count[neighbors_cover[i]] += 1

    final_indices = list(best_code)
    final_words = [idx_to_word(i) for i in final_indices]
    return final_words
