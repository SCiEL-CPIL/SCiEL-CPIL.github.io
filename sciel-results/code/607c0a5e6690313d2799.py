import time
import random
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback: simple greedy set cover for other parameters
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

        # Precompute neighbors
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
    
    # Precompute neighbors using numpy for speed
    # Each word covers itself + 16 neighbors = 17 words
    neighbors = []
    for idx in range(total_words):
        digits = np.zeros(8, dtype=np.int64)
        temp = idx
        for i in range(7, -1, -1):
            digits[i] = temp % 3
            temp //= 3
        
        nbrs = [idx]
        for i in range(8):
            for val in range(3):
                if val != digits[i]:
                    new_digits = digits.copy()
                    new_digits[i] = val
                    new_idx = 0
                    for v in new_digits:
                        new_idx = new_idx * 3 + v
                    nbrs.append(new_idx)
        neighbors.append(nbrs)
    
    neighbors_arr = np.array(neighbors, dtype=np.int64)
    
    def idx_to_word(idx):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        return w

    start_time = time.time()
    deadline = start_time + 115
    
    # Greedy construction
    uncovered_mask = np.ones(total_words, dtype=bool)
    code_list = []
    code_set = set()
    rng = np.random.default_rng(42)
    
    while np.any(uncovered_mask) and time.time() < deadline - 30:
        remaining = [i for i in range(total_words) if i not in code_set]
        if len(remaining) > 500:
            candidates = rng.choice(remaining, size=500, replace=False)
        else:
            candidates = remaining
        
        best_idx = -1
        best_gain = -1
        
        for i in candidates:
            nbrs = neighbors_arr[i]
            gain = np.sum(uncovered_mask[nbrs])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1 or best_gain == 0:
            break
            
        code_set.add(best_idx)
        code_list.append(best_idx)
        uncovered_mask[neighbors_arr[best_idx]] = False

    if np.any(uncovered_mask):
        while np.any(uncovered_mask) and time.time() < deadline - 5:
            best_idx = -1
            best_gain = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                nbrs = neighbors_arr[i]
                gain = np.sum(uncovered_mask[nbrs])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            code_set.add(best_idx)
            code_list.append(best_idx)
            uncovered_mask[neighbors_arr[best_idx]] = False

    code_list = list(code_set)
    
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code_list:
        cover_count[neighbors_arr[c]] += 1

    def try_remove_redundant():
        removed = False
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for i in range(len(code_list) - 1, -1, -1):
                c = code_list[i]
                nbrs = neighbors_arr[c]
                if np.all(cover_count[nbrs] > 1):
                    code_list.pop(i)
                    cover_count[nbrs] -= 1
                    code_set.discard(c)
                    removed = True
                    changed = True
                    break
        return removed

    try_remove_redundant()

    # Local search with improved swap scoring and aggressive kicks
    max_swaps = 15000
    swaps_done = 0
    no_improve_count = 0
    best_size = len(code_list)
    kick_threshold = 500
    
    while time.time() < deadline - 15 and swaps_done < max_swaps:
        if try_remove_redundant():
            no_improve_count = 0
            if len(code_list) < best_size:
                best_size = len(code_list)
            continue
        
        if len(code_list) == 0:
            break
            
        idx_c = rng.integers(0, len(code_list))
        c = code_list[idx_c]
        nbrs_c = neighbors_arr[c]
        cover_count[nbrs_c] -= 1
        
        uncovered_now = nbrs_c[cover_count[nbrs_c] == 0]
        
        if len(uncovered_now) == 0:
            code_list.pop(idx_c)
            code_set.discard(c)
            swaps_done += 1
            no_improve_count = 0
            continue
        
        candidates = set(neighbors_arr[uncovered_now[0]].tolist())
        for u in uncovered_now[1:]:
            candidates.intersection_update(neighbors_arr[u].tolist())
            if not candidates:
                break
        candidates.difference_update(code_set)
        
        if candidates:
            best_cp = -1
            best_score = -1
            
            cand_list = list(candidates)
            if len(cand_list) > 150:
                cand_list = rng.choice(cand_list, size=150, replace=False).tolist()
            
            for cp in cand_list:
                nbrs_cp = neighbors_arr[cp]
                # Improved score: maximize redundancy + minimize newly uncovered
                redundancy_gain = np.sum(cover_count[nbrs_cp] >= 2)
                # Penalize if this swap leaves any word with cover_count == 1
                new_min_cover = np.min(cover_count[nbrs_cp])
                score = redundancy_gain * 2 - (0 if new_min_cover >= 1 else 5)
                if score > best_score:
                    best_score = score
                    best_cp = cp
            
            if best_cp != -1:
                code_list[idx_c] = best_cp
                code_set.discard(c)
                code_set.add(best_cp)
                cover_count[neighbors_arr[best_cp]] += 1
                swaps_done += 1
                no_improve_count += 1
                
                if try_remove_redundant():
                    no_improve_count = 0
                    if len(code_list) < best_size:
                        best_size = len(code_list)
            else:
                cover_count[nbrs_c] += 1
                swaps_done += 1
                no_improve_count += 1
        else:
            cover_count[nbrs_c] += 1
            swaps_done += 1
            no_improve_count += 1
        
        # Aggressive kick: remove 2 words, find 1-2 replacements
        if no_improve_count > kick_threshold:
            if len(code_list) > 400:
                # Try removing 2 random words
                idx1 = rng.integers(0, len(code_list))
                idx2 = rng.integers(0, len(code_list) - 1)
                if idx2 >= idx1:
                    idx2 += 1
                
                c1 = code_list[idx1]
                c2 = code_list[idx2]
                
                cover_count[neighbors_arr[c1]] -= 1
                cover_count[neighbors_arr[c2]] -= 1
                
                uncovered_set = set(neighbors_arr[c1][cover_count[neighbors_arr[c1]] == 0].tolist())
                uncovered_set.update(neighbors_arr[c2][cover_count[neighbors_arr[c2]] == 0].tolist())
                uncovered_list = list(uncovered_set)
                
                if len(uncovered_list) <= 3:
                    # Find candidates covering all uncovered
                    if uncovered_list:
                        cands = set(neighbors_arr[uncovered_list[0]].tolist())
                        for u in uncovered_list[1:]:
                            cands.intersection_update(neighbors_arr[u].tolist())
                            if not cands:
                                break
                        cands.difference_update(code_set)
                        
                        if cands:
                            # Pick best candidate
                            best_cp = -1
                            best_sc = -1
                            for cp in list(cands)[:50]:
                                sc = np.sum(cover_count[neighbors_arr[cp]] >= 2)
                                if sc > best_sc:
                                    best_sc = sc
                                    best_cp = cp
                            
                            if best_cp != -1:
                                code_list[idx1] = best_cp
                                code_set.discard(c1)
                                code_set.discard(c2)
                                code_set.add(best_cp)
                                cover_count[neighbors_arr[best_cp]] += 1
                                del code_list[idx2]
                                try_remove_redundant()
                            else:
                                cover_count[neighbors_arr[c1]] += 1
                                cover_count[neighbors_arr[c2]] += 1
                        else:
                            cover_count[neighbors_arr[c1]] += 1
                            cover_count[neighbors_arr[c2]] += 1
                    else:
                        code_list.pop(idx1)
                        code_list.pop(idx2)
                        code_set.discard(c1)
                        code_set.discard(c2)
                        try_remove_redundant()
                else:
                    cover_count[neighbors_arr[c1]] += 1
                    cover_count[neighbors_arr[c2]] += 1
                
                no_improve_count = 0
                kick_threshold = int(kick_threshold * 1.5)

    # Final exhaustive polish
    try_remove_redundant()
    
    # One more pass with tighter time check
    while time.time() < deadline - 2:
        made_progress = False
        for i in range(len(code_list) - 1, -1, -1):
            c = code_list[i]
            nbrs = neighbors_arr[c]
            if np.all(cover_count[nbrs] > 1):
                code_list.pop(i)
                cover_count[nbrs] -= 1
                code_set.discard(c)
                made_progress = True
        if not made_progress:
            break

    return [idx_to_word(c) for c in code_list]
