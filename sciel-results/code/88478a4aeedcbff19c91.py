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
    
    # Precompute neighbors for all 6561 words using numpy for speed
    import numpy as np
    
    # Build all words as a numpy array of shape (6561, 8)
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Build a mapping from word tuple to index
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
    
    # Precompute neighbors as a numpy array: neighbors_matrix[i] = list of indices covered by word i
    # Each word covers 17 words (itself + 16 at distance 1)
    neighbors = [None] * total
    for idx in range(total):
        w = all_words[idx]
        nbrs = [idx]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    w_new = w.copy()
                    w_new[pos] = val
                    nbrs.append(word_to_idx[tuple(w_new)])
        neighbors[idx] = nbrs
    
    # Build numpy matrix for fast coverage counting
    # coverage_matrix[i][j] = 1 if word j is in neighbors[i], else 0
    # Shape: (total, total) = (6561, 6561) - this is ~43M bytes as int8, acceptable
    cov_matrix = np.zeros((total, total), dtype=np.int8)
    for i in range(total):
        for j in neighbors[i]:
            cov_matrix[i, j] = 1
    
    # Greedy construction with numpy acceleration
    uncovered_mask = np.ones(total, dtype=np.bool_)  # True if uncovered
    uncovered_count = total
    
    code_list = []
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin for return
    
    # Greedy phase: use numpy to compute coverage counts in parallel
    while uncovered_count > 0 and time.time() < deadline - 20:
        # Compute how many uncovered words each candidate covers
        # uncovered_mask is (total,), cov_matrix is (total, total)
        # counts[i] = sum(cov_matrix[i] & uncovered_mask)
        counts = np.sum(cov_matrix & uncovered_mask, axis=1)
        
        best_idx = int(np.argmax(counts))
        best_count = int(counts[best_idx])
        
        if best_count == 0:
            break
        
        code_list.append(best_idx)
        # Update uncovered mask
        for nbr in neighbors[best_idx]:
            if uncovered_mask[nbr]:
                uncovered_mask[nbr] = False
                uncovered_count -= 1
    
    # Ensure we have a valid code
    if uncovered_count > 0:
        # Fallback: add words until covered (should rarely happen)
        while uncovered_count > 0 and time.time() < deadline - 15:
            counts = np.sum(cov_matrix & uncovered_mask, axis=1)
            best_idx = int(np.argmax(counts))
            best_count = int(counts[best_idx])
            if best_count == 0:
                break
            code_list.append(best_idx)
            for nbr in neighbors[best_idx]:
                if uncovered_mask[nbr]:
                    uncovered_mask[nbr] = False
                    uncovered_count -= 1
    
    # Now we have a valid code. Do local search to minimize size.
    code_set = set(code_list)
    
    # Recompute cover_count properly
    cover_count = [0] * total
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1
    
    best_size = len(code_list)
    best_code = list(code_list)
    
    # Local search with removal and swap moves
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 3:
        # Phase 1: Try to remove words
        removed_any = False
        random.shuffle(code_list)
        
        for i in range(len(code_list) - 1, -1, -1):
            if time.time() > deadline - 5:
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
            continue
        
        # Phase 2: Try swaps to escape local minimum
        if len(code_list) > 450:
            idx_c = random.randint(0, len(code_list) - 1)
            c = code_list[idx_c]
            
            # Temporarily remove
            for nbr in neighbors[c]:
                cover_count[nbr] -= 1
            
            # Find uncovered words
            uncovered_words = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]
            
            if not uncovered_words:
                # Shouldn't happen, but if it does, keep the removal
                code_list.pop(idx_c)
                code_set.discard(c)
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
                no_improvement_count = 0
                continue
            
            # Find a replacement word that covers all uncovered words
            found_swap = False
            if uncovered_words:
                first_uncovered = uncovered_words[0]
                # Candidates must be in neighbors[first_uncovered]
                for idx_cp in neighbors[first_uncovered]:
                    if idx_cp == c:
                        continue
                    if idx_cp in code_set:
                        continue
                    # Check if idx_cp covers all other uncovered words
                    all_covered = True
                    cp_nbrs_set = set(neighbors[idx_cp])
                    for u in uncovered_words:
                        if u not in cp_nbrs_set:
                            all_covered = False
                            break
                    if all_covered:
                        # Perform swap
                        code_list[idx_c] = idx_cp
                        code_set.discard(c)
                        code_set.add(idx_cp)
                        for nbr in neighbors[idx_cp]:
                            cover_count[nbr] += 1
                        found_swap = True
                        no_improvement_count = 0
                        break
            
            if not found_swap:
                # Revert
                for nbr in neighbors[c]:
                    cover_count[nbr] += 1
                no_improvement_count += 1
        
        if no_improvement_count > max_no_improvement:
            # Try a more aggressive restart: remove a random subset and rebuild
            if time.time() < deadline - 25:
                # Remove 10% of codewords randomly
                num_remove = max(1, len(code_list) // 10)
                to_remove = random.sample(code_list, num_remove)
                for c in to_remove:
                    if c in code_set:
                        code_set.discard(c)
                        for nbr in neighbors[c]:
                            cover_count[nbr] -= 1
                        code_list.remove(c)
                
                # Rebuild: greedily add words to cover uncovered
                # Use numpy for speed
                while time.time() < deadline - 10:
                    # Check if all covered
                    min_cover = min(cover_count)
                    if min_cover > 0:
                        break
                    counts = np.sum(cov_matrix & (cover_count == 0), axis=1)
                    best_idx = int(np.argmax(counts))
                    best_count = int(counts[best_idx])
                    if best_count == 0:
                        break
                    if best_idx not in code_set:
                        code_list.append(best_idx)
                        code_set.add(best_idx)
                        for nbr in neighbors[best_idx]:
                            cover_count[nbr] += 1
                
                no_improvement_count = 0
    
    # Return the best code found
    if len(best_code) < len(code_list):
        return [list(all_words[c]) for c in best_code]
    return [list(all_words[c]) for c in code_list]
