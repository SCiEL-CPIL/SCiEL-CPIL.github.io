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
    
    # Build a mapping from word tuple to index
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
    
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
    
    # Precompute neighbor sets for O(1) membership tests
    neighbor_sets = [set(nbrs) for nbrs in neighbors]
    
    # Greedy construction
    uncovered = [True] * total
    code_list = []
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for local search
    
    # Greedy phase
    while time.time() < deadline - 45:
        best_idx = -1
        best_count = -1
        for idx in range(total):
            cnt = sum(1 for nbr in neighbors[idx] if uncovered[nbr])
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        if best_idx == -1 or best_count == 0:
            break
        code_list.append(best_idx)
        for nbr in neighbors[best_idx]:
            uncovered[nbr] = False
    
    # Now we have a valid code. Do local search to minimize size.
    code_set = set(code_list)
    
    # Recompute cover_count properly
    cover_count = [0] * total
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1
    
    best_size = len(code_list)
    best_code = list(code_list)
    
    # Track uncovered count for fast checks
    uncovered_count = sum(1 for x in range(total) if cover_count[x] == 0)
    
    # Local search with removal and swap moves
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 3:
        # Phase 1: Try to remove words (deterministic pass)
        removed_any = False
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
            continue
        
        # Phase 2: Try multiple swaps per iteration to escape local minima
        swap_attempts = 0
        max_swap_attempts = 5
        found_swap = False
        
        while swap_attempts < max_swap_attempts and time.time() < deadline - 5:
            idx_c = random.randint(0, len(code_list) - 1)
            c = code_list[idx_c]
            
            # Temporarily remove
            for nbr in neighbors[c]:
                cover_count[nbr] -= 1
            
            # Find uncovered words (should be small)
            uncovered_words = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]
            
            if not uncovered_words:
                # Removal was safe
                code_list.pop(idx_c)
                code_set.discard(c)
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
                no_improvement_count = 0
                found_swap = True
                break
            
            # Find a replacement word that covers all uncovered words
            # Candidates must be neighbors of every uncovered word
            # Intersect neighbor sets of all uncovered words to get candidates
            candidate_set = neighbor_sets[uncovered_words[0]].copy()
            for u in uncovered_words[1:]:
                candidate_set &= neighbor_sets[u]
            
            # Remove c and existing code words from candidates
            candidate_set.discard(c)
            candidate_set -= code_set
            
            if candidate_set:
                # Try candidates, prefer those that cover more total words
                best_cp = -1
                best_cp_score = -1
                for idx_cp in candidate_set:
                    # Score: number of words it covers (to prefer high-coverage replacements)
                    score = sum(1 for nbr in neighbors[idx_cp] if cover_count[nbr] == 0)
                    if score > best_cp_score:
                        best_cp_score = score
                        best_cp = idx_cp
                
                if best_cp != -1:
                    # Perform swap
                    code_list[idx_c] = best_cp
                    code_set.discard(c)
                    code_set.add(best_cp)
                    for nbr in neighbors[best_cp]:
                        cover_count[nbr] += 1
                    no_improvement_count = 0
                    found_swap = True
                    break
            
            # Revert
            for nbr in neighbors[c]:
                cover_count[nbr] += 1
            swap_attempts += 1
        
        if not found_swap:
            no_improvement_count += 1
        
        if no_improvement_count > max_no_improvement:
            # Aggressive restart: remove random subset and rebuild
            if time.time() < deadline - 20:
                num_remove = max(1, len(code_list) // 8)
                to_remove = random.sample(code_list, num_remove)
                for c in to_remove:
                    if c in code_set:
                        code_set.discard(c)
                        for nbr in neighbors[c]:
                            cover_count[nbr] -= 1
                        code_list.remove(c)
                
                # Rebuild: greedily add words to cover uncovered
                while time.time() < deadline - 10:
                    # Check if all covered
                    any_uncovered = False
                    for x in range(total):
                        if cover_count[x] == 0:
                            any_uncovered = True
                            break
                    if not any_uncovered:
                        break
                    
                    best_idx = -1
                    best_count = -1
                    for idx in range(total):
                        if idx in code_set:
                            continue
                        cnt = sum(1 for nbr in neighbors[idx] if cover_count[nbr] == 0)
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
    
    # Return the best code found
    if len(best_code) < len(code_list):
        return [list(all_words[c]) for c in best_code]
    return [list(all_words[c]) for c in code_list]
