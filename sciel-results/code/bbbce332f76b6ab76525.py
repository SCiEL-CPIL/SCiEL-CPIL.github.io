import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin

    if n == 0:
        return [[]]
    
    # Handle small cases or specific known perfect codes
    if q == 3 and radius == 1 and n == 4:
        return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for all words as bitmasks (integers) for fast coverage operations
    neighbors_mask = [0] * total_words
    neighbors_list = []
    for i, w in enumerate(all_words):
        mask = 0
        nb = [i]
        mask |= (1 << i)  # Include self
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
                nb.append(j)
        neighbors_mask[i] = mask
        neighbors_list.append(nb)
    
    # Precompute reverse neighbors
    reverse_neighbors = [0] * total_words
    for i in range(total_words):
        mask = neighbors_mask[i]
        while mask:
            lowbit = mask & (-mask)
            j = lowbit.bit_length() - 1
            reverse_neighbors[j] |= (1 << i)
            mask ^= lowbit

    def greedy_init(seed=0):
        rng = random.Random(seed)
        uncovered_mask = (1 << total_words) - 1
        code_indices = []
        code_set_mask = 0
        
        while uncovered_mask and time.time() < deadline - 5:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                if (code_set_mask >> i) & 1:
                    continue
                cnt = (neighbors_mask[i] & uncovered_mask).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                break
                
            code_indices.append(best_idx)
            code_set_mask |= (1 << best_idx)
            uncovered_mask &= ~neighbors_mask[best_idx]
                
        return code_indices

    def direct_sum_init():
        if q == 3 and n == 6 and radius == 1:
            tetracode = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]
            code = []
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=2):
                    c = h + tail
                    code.append(word_to_idx[c])
            return code
        return greedy_init(0)
    
    def compute_uncovered_mask(active_set_mask):
        uncovered = 0
        for j in range(total_words):
            if (reverse_neighbors[j] & active_set_mask) == 0:
                uncovered |= (1 << j)
        return uncovered

    def local_search(init_code, seed=0, max_time=None):
        if max_time is None:
            max_time = deadline - time.time() - 2
        rng = random.Random(seed)
        active = list(init_code)
        active_set_mask = 0
        for c in active:
            active_set_mask |= (1 << c)
        
        coverage = np.zeros(total_words, dtype=np.int32)
        for c in active:
            for nb in neighbors_list[c]:
                coverage[nb] += 1
        
        def repair():
            nonlocal active, active_set_mask, coverage
            uncovered_mask = compute_uncovered_mask(active_set_mask)
            while uncovered_mask and time.time() < deadline - 3:
                best_idx = -1
                best_count = -1
                for i in range(total_words):
                    if (active_set_mask >> i) & 1:
                        continue
                    cnt = (neighbors_mask[i] & uncovered_mask).bit_count()
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
                if best_idx == -1 or best_count == 0:
                    break
                active.append(best_idx)
                active_set_mask |= (1 << best_idx)
                for nb in neighbors_list[best_idx]:
                    coverage[nb] += 1
                uncovered_mask = compute_uncovered_mask(active_set_mask)
        
        repair()
        
        best_size = len(active)
        best_code = list(active)
        
        # Precompute for faster 2-1 swap: for each pair (i,j) in active, 
        # compute the set of words that would become uncovered
        # and find candidates c3 that cover them
        
        while time.time() < deadline - 3:
            improved = False
            
            # Phase 1: Single removal
            removal_found = False
            indices = list(range(len(active)))
            rng.shuffle(indices)
            
            for idx in indices:
                c = active[idx]
                can_remove = True
                for nb in neighbors_list[c]:
                    if coverage[nb] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for nb in neighbors_list[c]:
                        coverage[nb] -= 1
                    active.pop(idx)
                    active_set_mask &= ~(1 << c)
                    improved = True
                    removal_found = True
                    break
            
            if removal_found:
                if len(active) < best_size:
                    best_size = len(active)
                    best_code = list(active)
                continue

            # Phase 2: 2-0 or 2-1 Swap
            swap21_found = False
            if len(active) > 1:
                pairs = list(itertools.combinations(range(len(active)), 2))
                rng.shuffle(pairs)
                
                for i, j in pairs:
                    if time.time() > deadline - 4:
                        break
                    c1 = active[i]
                    c2 = active[j]
                    
                    mask_c1 = neighbors_mask[c1]
                    mask_c2 = neighbors_mask[c2]
                    mask_union = mask_c1 | mask_c2
                    
                    # Find words that would be uncovered if we remove both c1 and c2
                    uncovered_if_removed = 0
                    for nb in neighbors_list[c1]:
                        if coverage[nb] == 1:  # Only covered by c1
                            uncovered_if_removed |= (1 << nb)
                        elif coverage[nb] == 2 and (mask_c2 >> nb) & 1:
                            # Covered by c1 and c2 only
                            uncovered_if_removed |= (1 << nb)
                    for nb in neighbors_list[c2]:
                        if coverage[nb] == 1:  # Only covered by c2
                            uncovered_if_removed |= (1 << nb)
                        elif coverage[nb] == 2 and (mask_c1 >> nb) & 1:
                            # Covered by c2 and c1 only
                            uncovered_if_removed |= (1 << nb)
                    
                    if uncovered_if_removed == 0:
                        # 2-0 swap: remove both
                        for nb in neighbors_list[c1]:
                            coverage[nb] -= 1
                        for nb in neighbors_list[c2]:
                            coverage[nb] -= 1
                        if i > j:
                            active.pop(i)
                            active.pop(j)
                        else:
                            active.pop(j)
                            active.pop(i)
                        active_set_mask &= ~(1 << c1)
                        active_set_mask &= ~(1 << c2)
                        improved = True
                        swap21_found = True
                        break
                    else:
                        # 2-1 swap: find c3 that covers uncovered_if_removed
                        best_c3 = -1
                        best_c3_score = -1
                        for c3 in range(total_words):
                            if (active_set_mask >> c3) & 1:
                                continue
                            if (neighbors_mask[c3] & uncovered_if_removed) == uncovered_if_removed:
                                # Score by how many additional words c3 covers that are already covered
                                score = 0
                                for nb in neighbors_list[c3]:
                                    if coverage[nb] > 2:
                                        score += 1
                                if score > best_c3_score:
                                    best_c3_score = score
                                    best_c3 = c3
                        if best_c3 != -1:
                            for nb in neighbors_list[c1]:
                                coverage[nb] -= 1
                            for nb in neighbors_list[c2]:
                                coverage[nb] -= 1
                            if i > j:
                                active.pop(i)
                                active.pop(j)
                            else:
                                active.pop(j)
                                active.pop(i)
                            active_set_mask &= ~(1 << c1)
                            active_set_mask &= ~(1 << c2)
                            active.append(best_c3)
                            active_set_mask |= (1 << best_c3)
                            for nb in neighbors_list[best_c3]:
                                coverage[nb] += 1
                            improved = True
                            swap21_found = True
                            break

            if swap21_found:
                if len(active) < best_size:
                    best_size = len(active)
                    best_code = list(active)
                continue

            # Phase 3: 1-1 Swap
            swap_found = False
            indices = list(range(len(active)))
            rng.shuffle(indices)
            
            for idx in indices:
                c = active[idx]
                unique_words_mask = 0
                for nb in neighbors_list[c]:
                    if coverage[nb] == 1:
                        unique_words_mask |= (1 << nb)
                
                if unique_words_mask == 0:
                    continue

                best_candidate = -1
                best_score = -1
                
                for c_prime in range(total_words):
                    if (active_set_mask >> c_prime) & 1:
                        continue
                    if (neighbors_mask[c_prime] & unique_words_mask) != unique_words_mask:
                        continue
                    score = 0
                    for nb in neighbors_list[c_prime]:
                        if coverage[nb] > 1:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_candidate = c_prime

                if best_candidate != -1:
                    for nb in neighbors_list[c]:
                        coverage[nb] -= 1
                    active.pop(idx)
                    active_set_mask &= ~(1 << c)
                    active.append(best_candidate)
                    active_set_mask |= (1 << best_candidate)
                    for nb in neighbors_list[best_candidate]:
                        coverage[nb] += 1
                    improved = True
                    swap_found = True
                    break
            
            if not improved:
                # Perturbation: remove 1-2 codewords with most unique coverage, then repair
                if time.time() < deadline - 6:
                    criticalities = []
                    for c in active:
                        count = 0
                        for nb in neighbors_list[c]:
                            if coverage[nb] == 1:
                                count += 1
                        criticalities.append((count, c))
                    criticalities.sort(reverse=True)
                    
                    perturb_count = min(rng.randint(1, 2), len(active) - 1)
                    for _ in range(perturb_count):
                        if not active:
                            break
                        _, c_to_remove = criticalities[0]
                        try:
                            idx = active.index(c_to_remove)
                        except ValueError:
                            break
                        active.pop(idx)
                        active_set_mask &= ~(1 << c_to_remove)
                        for nb in neighbors_list[c_to_remove]:
                            coverage[nb] -= 1
                        criticalities.pop(0)
                    
                    repair()
                    
                    if len(active) < best_size:
                        best_size = len(active)
                        best_code = list(active)

            if len(active) < best_size:
                best_size = len(active)
                best_code = list(active)

        return best_code

    best_code = None
    best_size = float('inf')
    
    # Use direct sum construction as primary initialization
    if q == 3 and n == 6 and radius == 1:
        ds_code = direct_sum_init()
        result = local_search(ds_code, seed=42, max_time=30)
        if len(result) < best_size:
            best_size = len(result)
            best_code = result
    
    # Try multiple greedy initializations with local search
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for seed in seeds:
        if time.time() > deadline - 15:
            break
        g_code = greedy_init(seed)
        result = local_search(g_code, seed=seed * 100 + 7, max_time=8)
        if len(result) < best_size:
            best_size = len(result)
            best_code = result
    
    # Polish the best result found
    if best_code is not None and time.time() < deadline - 10:
        for seed in [100, 200, 300, 400, 500]:
            if time.time() > deadline - 5:
                break
            result = local_search(best_code, seed=seed, max_time=5)
            if len(result) < best_size:
                best_size = len(result)
                best_code = result
    
    if best_code is None:
        best_code = greedy_init(0)

    return [list(all_words[idx]) for idx in best_code]
