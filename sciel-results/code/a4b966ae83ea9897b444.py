import time
import random
import itertools

def construct(q, n, radius):
    # Handle the specific known case with a strong algorithm
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1()
    
    # Fallback for other cases: simple greedy covering
    return _fallback_construct(q, n, radius)


def _fallback_construct(q, n, radius):
    # Simple greedy set cover for general case
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    
    if radius != 1:
        return [list(w) for w in all_words]
    
    covered = [False] * total
    codewords = []
    
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[w] = i
    
    covers = {}
    for i, w in enumerate(all_words):
        w_list = list(w)
        nbrs = [i]
        for pos in range(n):
            orig = w_list[pos]
            for val in range(q):
                if val != orig:
                    w_list[pos] = val
                    nbrs.append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig
        covers[i] = set(nbrs)
        
    while not all(covered):
        best_idx = -1
        best_count = -1
        for i in range(total):
            if all(covered[j] for j in covers[i]):
                continue
            cnt = sum(1 for j in covers[i] if not covered[j])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx == -1:
            break
        codewords.append(list(all_words[best_idx]))
        for j in covers[best_idx]:
            covered[j] = True
            
    return codewords


def _construct_ternary_8_1():
    q = 3
    n = 8
    radius = 1
    total = q ** n  # 6561
    
    import numpy as np
    
    # Build all words as a numpy array of shape (6561, 8)
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Precompute neighbors for all 6561 words
    # For ternary length 8, each word has 1 + 8*2 = 17 neighbors
    neighbors = [None] * total
    # Also precompute neighbor sets for fast membership testing
    neighbor_sets = [None] * total
    
    for idx in range(total):
        w = all_words[idx]
        nbrs = [idx]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    # Compute new index directly without tuple lookup
                    # Index = sum of digits * 3^(n-1-i)
                    new_idx = 0
                    for i in range(n):
                        d = w[i] if i != pos else val
                        new_idx = new_idx * q + d
                    nbrs.append(new_idx)
        neighbors[idx] = nbrs
        neighbor_sets[idx] = frozenset(nbrs)
    
    start_time = time.time()
    deadline = start_time + 115  # Leave 35s margin
    
    # Greedy construction with efficient uncovered tracking
    cover_count = np.zeros(total, dtype=np.int8)
    uncovered_count = total  # Number of uncovered words
    uncovered = np.ones(total, dtype=bool)
    
    code_list = []
    
    # Greedy: pick word covering most uncovered words
    while uncovered_count > 0 and time.time() < deadline - 45:
        best_idx = -1
        best_count = 0
        for idx in range(total):
            # Count uncovered neighbors
            cnt = 0
            for nbr in neighbors[idx]:
                if uncovered[nbr]:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        if best_idx == -1 or best_count == 0:
            break
        code_list.append(best_idx)
        for nbr in neighbors[best_idx]:
            if uncovered[nbr]:
                uncovered[nbr] = False
                uncovered_count -= 1
            cover_count[nbr] += 1
    
    # If not fully covered, continue greedy
    while uncovered_count > 0 and time.time() < deadline - 30:
        best_idx = -1
        best_count = 0
        for idx in range(total):
            cnt = 0
            for nbr in neighbors[idx]:
                if uncovered[nbr]:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        if best_idx == -1 or best_count == 0:
            break
        code_list.append(best_idx)
        for nbr in neighbors[best_idx]:
            if uncovered[nbr]:
                uncovered[nbr] = False
                uncovered_count -= 1
            cover_count[nbr] += 1
    
    code_set = set(code_list)
    
    best_size = len(code_list)
    best_code = list(code_list)
    
    no_improvement_count = 0
    max_no_improvement = 50
    swap_attempts_in_row = 0
    
    while time.time() < deadline - 5:
        # Phase 1: Try to remove words (multiple passes until no more removals)
        removed_any = False
        random.shuffle(code_list)
        
        for i in range(len(code_list) - 1, -1, -1):
            if time.time() > deadline - 8:
                break
            c = code_list[i]
            removable = True
            for nbr in neighbors[c]:
                if cover_count[nbr] <= 1:
                    removable = False
                    break
            if removable:
                code_list.pop(i)
                code_set.discard(c)
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                removed_any = True
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
        
        if removed_any:
            no_improvement_count = 0
            swap_attempts_in_row = 0
            continue
        
        # Phase 2: Try swaps to escape local minimum
        if len(code_list) > 440:
            idx_c = random.randint(0, len(code_list) - 1)
            c = code_list[idx_c]
            
            # Temporarily remove
            for nbr in neighbors[c]:
                cover_count[nbr] -= 1
            
            # Find uncovered words
            uncovered_words = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]
            
            if not uncovered_words:
                code_list.pop(idx_c)
                code_set.discard(c)
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
                no_improvement_count = 0
                swap_attempts_in_row = 0
                continue
            
            # Find a replacement word that covers all uncovered words
            found_swap = False
            first_uncovered = uncovered_words[0]
            for idx_cp in neighbors[first_uncovered]:
                if idx_cp == c:
                    continue
                if idx_cp in code_set:
                    continue
                # Check if idx_cp covers all other uncovered words
                cp_nbrs_set = neighbor_sets[idx_cp]
                all_covered = True
                for u in uncovered_words:
                    if u not in cp_nbrs_set:
                        all_covered = False
                        break
                if all_covered:
                    code_list[idx_c] = idx_cp
                    code_set.discard(c)
                    code_set.add(idx_cp)
                    for nbr in neighbors[idx_cp]:
                        cover_count[nbr] += 1
                    found_swap = True
                    no_improvement_count = 0
                    swap_attempts_in_row = 0
                    break
            
            if not found_swap:
                for nbr in neighbors[c]:
                    cover_count[nbr] += 1
                no_improvement_count += 1
                swap_attempts_in_row += 1
        
        if no_improvement_count > max_no_improvement:
            # Aggressive restart: remove 15% and rebuild
            if time.time() < deadline - 25:
                num_remove = max(1, len(code_list) // 7)
                to_remove = random.sample(code_list, min(num_remove, len(code_list)))
                for c in to_remove:
                    if c in code_set:
                        code_set.discard(c)
                        for nbr in neighbors[c]:
                            cover_count[nbr] -= 1
                        code_list.remove(c)
                
                # Rebuild: greedily add words to cover uncovered
                while True:
                    if time.time() > deadline - 12:
                        break
                    # Check if all covered
                    any_uncovered = False
                    for x in range(total):
                        if cover_count[x] == 0:
                            any_uncovered = True
                            break
                    if not any_uncovered:
                        break
                    
                    best_idx = -1
                    best_count = 0
                    for idx in range(total):
                        if idx in code_set:
                            continue
                        cnt = 0
                        for nbr in neighbors[idx]:
                            if cover_count[nbr] == 0:
                                cnt += 1
                        if cnt > best_count:
                            best_count = cnt
                            best_idx = idx
                    if best_idx == -1 or best_count == 0:
                        break
                    code_list.append(best_idx)
                    code_set.add(best_idx)
                    for nbr in neighbors[best_idx]:
                        cover_count[nbr] += 1
                
                no_improvement_count = 0
                swap_attempts_in_row = 0
    
    if len(best_code) < len(code_list):
        return [list(all_words[c]) for c in best_code]
    return [list(all_words[c]) for c in code_list]
