import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # 1. Initial Construction: Direct sum of tetracode and all words of length n-4
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n - 4))
    
    initial_codewords = []
    for h in tetracode:
        for tail in tails:
            initial_codewords.append(h + list(tail))
            
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    space_size = q ** n
    neighbors = [[] for _ in range(space_size)]
    
    for i in range(space_size):
        w = idx_to_word(i)
        neighbors[i].append(i)
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val != original_val:
                    w[pos] = new_val
                    neighbors[i].append(word_to_idx(w))
            w[pos] = original_val
            
    # Precompute neighbor sets for fast membership
    neighbor_set = [set(nb) for nb in neighbors]
    # Precompute list of distance-1 neighbors (excluding self) for candidate generation
    dist1_neighbors = [nb for nb in neighbors]  # same as neighbors minus self, but we'll filter
    
    def compute_counts(codewords):
        cnt = np.zeros(space_size, dtype=np.int32)
        for c in codewords:
            idx_c = word_to_idx(c)
            for nb in neighbors[idx_c]:
                cnt[nb] += 1
        return cnt

    def greedy_delete(codewords, cnt, deadline):
        """Iteratively remove any codeword whose removal doesn't uncover any word."""
        current = list(codewords)
        c_cnt = cnt.copy()
        
        for _ in range(100):
            if time.time() > deadline:
                break
            removed_any = False
            # Sort by "removability" - prefer removing codewords with most redundancy
            # Actually just iterate in reverse for stability
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline:
                    break
                idx_c = word_to_idx(current[i])
                
                # Check if any neighbor has count == 1
                can_remove = True
                for nb in neighbors[idx_c]:
                    if c_cnt[nb] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    for nb in neighbors[idx_c]:
                        c_cnt[nb] -= 1
                    current.pop(i)
                    removed_any = True
                    
            if not removed_any:
                break
        
        return current, c_cnt

    def generate_candidates_near(essential, current_idx_set, max_cands):
        """Generate candidate replacement words near the essential neighbors."""
        cands = []
        seen = set()
        # For each essential neighbor, try all its distance-1 neighbors as candidates
        for ess in essential:
            for pos in range(n):
                w = idx_to_word(ess)
                orig = w[pos]
                for nv in range(q):
                    if nv == orig:
                        continue
                    w[pos] = nv
                    ci = word_to_idx(w)
                    w[pos] = orig
                    if ci not in current_idx_set and ci not in seen:
                        seen.add(ci)
                        cands.append(ci)
                        if len(cands) >= max_cands:
                            return cands
        # Also add some random candidates
        while len(cands) < max_cands:
            ci = random.randint(0, space_size - 1)
            if ci not in current_idx_set and ci not in seen:
                seen.add(ci)
                cands.append(ci)
        return cands

    def local_search(codewords, cnt, deadline):
        """Local search with removals and smart swaps."""
        current = list(codewords)
        c_cnt = cnt.copy()
        current_idx_set = set(word_to_idx(c) for c in current)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try simple removals
            for i in range(len(current)):
                if time.time() > deadline:
                    return current, c_cnt
                
                idx_c = word_to_idx(current[i])
                can_remove = True
                for nb in neighbors[idx_c]:
                    if c_cnt[nb] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    for nb in neighbors[idx_c]:
                        c_cnt[nb] -= 1
                    removed = current.pop(i)
                    current_idx_set.discard(word_to_idx(removed))
                    improved = True
                    break
            
            if improved:
                continue
            
            # Phase 2: Try swaps - replace one codeword with another
            for i in range(len(current)):
                if time.time() > deadline:
                    return current, c_cnt
                
                idx_c = word_to_idx(current[i])
                
                # Find essential neighbors (only covered by this codeword)
                essential = [nb for nb in neighbors[idx_c] if c_cnt[nb] == 1]
                
                if not essential:
                    # Can remove directly
                    for nb in neighbors[idx_c]:
                        c_cnt[nb] -= 1
                    removed = current.pop(i)
                    current_idx_set.discard(word_to_idx(removed))
                    improved = True
                    break
                
                # Generate smart candidates near essential neighbors
                cands = generate_candidates_near(essential, current_idx_set, 40)
                
                found_swap = False
                for cand_idx in cands:
                    if time.time() > deadline:
                        return current, c_cnt
                    
                    cand_nb = neighbors[cand_idx]
                    
                    # Check candidate covers all essential neighbors
                    covers = all(nb in cand_nb for nb in essential)
                    if not covers:
                        continue
                    
                    # Simulate swap: remove idx_c, add cand_idx
                    # Check no word drops to 0
                    valid = True
                    for nb in neighbors[idx_c]:
                        if c_cnt[nb] == 1 and nb not in cand_nb:
                            valid = False
                            break
                    
                    if valid:
                        # Accept swap
                        for nb in neighbors[idx_c]:
                            c_cnt[nb] -= 1
                        for nb in cand_nb:
                            c_cnt[nb] += 1
                        old_idx = word_to_idx(current[i])
                        current_idx_set.discard(old_idx)
                        current[i] = idx_to_word(cand_idx)
                        current_idx_set.add(cand_idx)
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
            if improved:
                continue
            
            # Phase 3: Try 2-for-1 swap (remove two codewords, add one)
            # This can escape local optima where no single swap works
            if len(current) > 2:
                for i in range(len(current)):
                    if time.time() > deadline:
                        return current, c_cnt
                    for j in range(i + 1, len(current)):
                        if time.time() > deadline:
                            return current, c_cnt
                        
                        idx_i = word_to_idx(current[i])
                        idx_j = word_to_idx(current[j])
                        
                        # Essential neighbors of i and j combined
                        ess_i = set(nb for nb in neighbors[idx_i] if c_cnt[nb] == 1)
                        ess_j = set(nb for nb in neighbors[idx_j] if c_cnt[nb] == 1)
                        # Only consider neighbors that would be uncovered if we remove both
                        # A word is uncovered if it's in neighbors[idx_i] with cnt==1 AND not in neighbors[idx_j]
                        # OR in neighbors[idx_j] with cnt==1 AND not in neighbors[idx_i]
                        # OR in both with cnt==1 (but then removing both uncovers it)
                        
                        # Simpler: simulate removal of both, find uncovered
                        uncovered = set()
                        for nb in neighbors[idx_i]:
                            if c_cnt[nb] == 1:
                                if nb not in neighbors[idx_j] or c_cnt[nb] - 1 < 1:
                                    uncovered.add(nb)
                            elif c_cnt[nb] == 2:
                                # After removing i, cnt becomes 1; if j also covers it, stays 1
                                if nb in neighbors[idx_j]:
                                    pass  # still covered by j
                                else:
                                    uncovered.add(nb)
                        for nb in neighbors[idx_j]:
                            if c_cnt[nb] == 1:
                                if nb not in neighbors[idx_i]:
                                    uncovered.add(nb)
                            # if c_cnt[nb] >= 2 and nb in neighbors[idx_i], after removing both:
                            # cnt decreases by 2, so if cnt was 2, becomes 0 -> uncovered
                            elif c_cnt[nb] == 2 and nb in neighbors[idx_i]:
                                uncovered.add(nb)
                        
                        if not uncovered:
                            # Can remove both
                            for nb in neighbors[idx_i]:
                                c_cnt[nb] -= 1
                            for nb in neighbors[idx_j]:
                                c_cnt[nb] -= 1
                            current.pop(j)
                            current.pop(i)
                            current_idx_set.discard(idx_i)
                            current_idx_set.discard(idx_j)
                            improved = True
                            break
                        
                        # Try to find a single replacement covering all uncovered
                        if len(uncovered) <= 3:  # only try small uncovered sets
                            ess_list = list(uncovered)
                            cands = generate_candidates_near(ess_list, current_idx_set, 30)
                            for cand_idx in cands:
                                if time.time() > deadline:
                                    return current, c_cnt
                                cand_nb = neighbors[cand_idx]
                                covers = all(nb in cand_nb for nb in uncovered)
                                if not covers:
                                    continue
                                # Simulate: remove i and j, add cand
                                valid = True
                                for nb in neighbors[idx_i]:
                                    new_cnt = c_cnt[nb] - 1
                                    if nb in neighbors[idx_j]:
                                        new_cnt -= 1
                                    if nb in cand_nb:
                                        new_cnt += 1
                                    if new_cnt < 1:
                                        valid = False
                                        break
                                if valid:
                                    for nb in neighbors[idx_j]:
                                        new_cnt = c_cnt[nb] - 1
                                        if nb in neighbors[idx_i]:
                                            pass  # already handled
                                        if nb in cand_nb:
                                            new_cnt += 1
                                        if new_cnt < 1:
                                            valid = False
                                            break
                                if valid:
                                    for nb in neighbors[idx_i]:
                                        c_cnt[nb] -= 1
                                    for nb in neighbors[idx_j]:
                                        c_cnt[nb] -= 1
                                    for nb in cand_nb:
                                        c_cnt[nb] += 1
                                    current.pop(j)
                                    current.pop(i)
                                    current.append(idx_to_word(cand_idx))
                                    current_idx_set.discard(idx_i)
                                    current_idx_set.discard(idx_j)
                                    current_idx_set.add(cand_idx)
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
        
        return current, c_cnt

    def full_optimize(codewords, deadline):
        counts = compute_counts(codewords)
        codewords, counts = greedy_delete(codewords, counts, deadline)
        codewords, counts = local_search(codewords, counts, deadline)
        codewords, counts = greedy_delete(codewords, counts, deadline)
        codewords, counts = local_search(codewords, counts, deadline)
        codewords, counts = greedy_delete(codewords, counts, deadline)
        return codewords, counts

    best_codewords = None
    best_size = float('inf')
    
    max_restarts = 15
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        seed = 42 + restart * 137
        random.seed(seed)
        
        shuffled = list(initial_codewords)
        random.shuffle(shuffled)
        
        codewords, counts = full_optimize(shuffled, deadline)
        
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords
        
        # Early exit if we hit known optimum
        if best_size <= 73:
            break
    
    if best_codewords is None:
        counts = compute_counts(initial_codewords)
        best_codewords, _ = greedy_delete(initial_codewords, counts, deadline)
    
    return best_codewords
