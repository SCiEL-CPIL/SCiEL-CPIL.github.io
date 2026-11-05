import time
import random
import numpy as np
import itertools

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
    deadline = start_time + 115  # Safe margin before 120s kill (accounting for overhead)

    # Precompute neighbors as numpy arrays for fast indexing
    neighbors_np = [None] * total_words
    
    # Helper to convert index to word (list of 8 ints)
    def idx_to_word(idx):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        return w

    # Precompute neighbors
    for idx in range(total_words):
        w = idx_to_word(idx)
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
        neighbors_np[idx] = np.array(nbrs, dtype=np.int32)

    # 1. Greedy Construction
    uncovered_mask = np.ones(total_words, dtype=bool)
    code_set = set()
    code_list = []
    
    while np.any(uncovered_mask) and time.time() < deadline - 20:
        gains = np.array([np.sum(uncovered_mask[neighbors_np[i]]) for i in range(total_words)])
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        code_set.add(best_idx)
        code_list.append(best_idx)
        uncovered_mask[neighbors_np[best_idx]] = False

    # Ensure validity
    if np.any(uncovered_mask):
        holes = np.where(uncovered_mask)[0]
        for h in holes:
            if h not in code_set:
                code_set.add(h)
                code_list.append(h)
                uncovered_mask[neighbors_np[h]] = False

    # 2. Local Search: Minimize
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code_list:
        cover_count[neighbors_np[c]] += 1

    def is_redundant(c):
        return np.min(cover_count[neighbors_np[c]]) >= 2

    def remove_redundant(max_rounds=5):
        changed = False
        for _ in range(max_rounds):
            progress = False
            for i in range(len(code_list) - 1, -1, -1):
                if time.time() > deadline - 15:
                    return changed
                c = code_list[i]
                if is_redundant(c):
                    code_list.pop(i)
                    code_set.discard(c)
                    cover_count[neighbors_np[c]] -= 1
                    changed = True
                    progress = True
            if not progress:
                break
        return changed

    # Initial cleanup
    remove_redundant()

    # Local Search Loop
    last_improvement = time.time()
    no_improve_count = 0
    
    while time.time() < deadline - 15:
        # Try to remove redundant words
        if remove_redundant():
            last_improvement = time.time()
            no_improve_count = 0
            continue
            
        no_improve_count += 1
        
        # 1. Try 1-Swap with better victim selection
        # Score victims by damage (number of neighbors with cover_count == 1)
        # Prefer victims with low damage, and among those, prefer those where the hole is "easy" to fill
        
        if len(code_list) > 50:
            # Sample a larger pool for better victim selection
            victims = random.sample(code_list, 50)
        else:
            victims = code_list[:]
            random.shuffle(victims)
        
        best_victim = -1
        best_hole = -1
        best_victim_damage = 999
        best_hole_score = -1  # Score of how easy the hole is to fill
        
        for c in victims:
            nbrs_c = neighbors_np[c]
            holes_mask = cover_count[nbrs_c] == 1
            damage = np.sum(holes_mask)
            
            if damage < best_victim_damage:
                best_victim_damage = damage
                best_victim = c
                if damage == 1:
                    hole_idx = np.where(holes_mask)[0]
                    if len(hole_idx) > 0:
                        best_hole = nbrs_c[hole_idx[0]]
                        # Score the hole: how many candidate replacements are there?
                        # Candidates are neighbors of the hole not in code_set
                        cand_count = sum(1 for x in neighbors_np[best_hole] if x not in code_set)
                        best_hole_score = cand_count
                        # If we find a victim with damage 1 and good hole, break early
                        if cand_count >= 3:
                            break
                elif damage == 0:
                    break

        if best_victim != -1 and best_victim_damage == 1:
            c = best_victim
            h = best_hole
            
            # Temporarily remove c
            cover_count[neighbors_np[c]] -= 1
            code_list.remove(c)
            code_set.discard(c)
            
            # Find replacement for h
            cand_pool = [x for x in neighbors_np[h] if x not in code_set]
            
            if cand_pool:
                best_cand = -1
                best_score = -1
                
                for cand in cand_pool:
                    # Score: number of neighbors of cand that are already covered
                    # This indicates overlap, which helps future removals
                    score = np.sum(cover_count[neighbors_np[cand]] >= 1)
                    # Bonus if cand makes other words redundant
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                
                if best_cand != -1:
                    code_list.append(best_cand)
                    code_set.add(best_cand)
                    cover_count[neighbors_np[best_cand]] += 1
                    last_improvement = time.time()
                    no_improve_count = 0
                    continue
            
            # Revert if no replacement found
            code_list.append(c)
            code_set.add(c)
            cover_count[neighbors_np[c]] += 1

        # 2. Try 2-Swap if stuck
        if no_improve_count > 3:
            found_improvement = False
            for _ in range(10):  # Try more random pairs
                if time.time() > deadline - 15:
                    break
                if len(code_list) < 2:
                    break
                i1, i2 = random.sample(range(len(code_list)), 2)
                c1 = code_list[i1]
                c2 = code_list[i2]
                
                # Temporarily remove both
                cover_count[neighbors_np[c1]] -= 1
                cover_count[neighbors_np[c2]] -= 1
                
                holes_mask = cover_count == 0
                holes = np.where(holes_mask)[0]
                
                if len(holes) <= 2:
                    cand_set = set()
                    for h in holes:
                        for nb in neighbors_np[h]:
                            if nb not in code_set:
                                cand_set.add(nb)
                    
                    cand_list = list(cand_set)
                    
                    if len(holes) == 0:
                        # Keep them removed
                        last_improvement = time.time()
                        no_improve_count = 0
                        found_improvement = True
                        break
                    elif len(holes) == 1:
                        h = holes[0]
                        best_cand = -1
                        best_score = -1
                        for cand in cand_list:
                            if h in neighbors_np[cand]:
                                score = np.sum(cover_count[neighbors_np[cand]] >= 1)
                                if score > best_score:
                                    best_score = score
                                    best_cand = cand
                        if best_cand != -1:
                            code_list.append(best_cand)
                            code_set.add(best_cand)
                            cover_count[neighbors_np[best_cand]] += 1
                            last_improvement = time.time()
                            no_improve_count = 0
                            found_improvement = True
                            break
                    elif len(holes) == 2:
                        h1, h2 = holes
                        cands_h1 = [c for c in cand_list if h1 in neighbors_np[c]]
                        cands_h2 = [c for c in cand_list if h2 in neighbors_np[c]]
                        
                        best_pair = None
                        best_pair_score = -1
                        
                        if len(cands_h1) * len(cands_h2) < 200:
                            for c1_new in cands_h1:
                                for c2_new in cands_h2:
                                    if c1_new == c2_new:
                                        if h1 in neighbors_np[c1_new] and h2 in neighbors_np[c1_new]:
                                            score = np.sum(cover_count[neighbors_np[c1_new]] >= 1)
                                            if score > best_pair_score:
                                                best_pair_score = score
                                                best_pair = (c1_new, None)
                                    else:
                                        score1 = np.sum(cover_count[neighbors_np[c1_new]] >= 1)
                                        score2 = np.sum(cover_count[neighbors_np[c2_new]] >= 1)
                                        if score1 + score2 > best_pair_score:
                                            best_pair_score = score1 + score2
                                            best_pair = (c1_new, c2_new)
                        
                        if best_pair:
                            c1_new, c2_new = best_pair
                            code_list.append(c1_new)
                            code_set.add(c1_new)
                            cover_count[neighbors_np[c1_new]] += 1
                            if c2_new is not None:
                                code_list.append(c2_new)
                                code_set.add(c2_new)
                                cover_count[neighbors_np[c2_new]] += 1
                            last_improvement = time.time()
                            no_improve_count = 0
                            found_improvement = True
                            break
                
                if not found_improvement:
                    # Revert
                    code_list.extend([c1, c2])
                    code_set.update([c1, c2])
                    cover_count[neighbors_np[c1]] += 1
                    cover_count[neighbors_np[c2]] += 1

    # Final cleanup
    remove_redundant(max_rounds=10)

    # Final Validity Check
    final_cover = np.zeros(total_words, dtype=bool)
    for c in code_list:
        final_cover[neighbors_np[c]] = True
        
    if not np.all(final_cover):
        uncovered_idx = np.where(~final_cover)[0]
        for idx in uncovered_idx:
            if idx not in code_set:
                code_list.append(idx)
                code_set.add(idx)
            else:
                for nb in neighbors_np[idx]:
                    if nb not in code_set:
                        code_list.append(nb)
                        code_set.add(nb)
                        break

    return [idx_to_word(c) for c in code_list]
