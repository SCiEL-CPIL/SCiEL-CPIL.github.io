import numpy as np
import itertools
import time
import random
import sys

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4:
            return tetracode
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def _construct_3_6_1():
    q = 3
    n = 6
    N = q ** n
    start_time = time.time()
    deadline = start_time + 140.0

    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    neighbors = []
    for i, c in enumerate(words):
        neigh = [i]
        for pos in range(n):
            for new_val in range(q):
                if new_val == c[pos]:
                    continue
                new_w = list(c)
                new_w[pos] = new_val
                neigh.append(word_to_idx[tuple(new_w)])
        neighbors.append(neigh)
    
    neighbors_np = [np.array(neigh, dtype=np.int32) for neigh in neighbors]
    
    # Initial Code: 81 codewords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_indices = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            initial_indices.add(word_to_idx[tuple(h + list(tail))])
            
    selected = initial_indices
    coverage = np.zeros(N, dtype=np.int32)
    for idx in selected:
        coverage[neighbors_np[idx]] += 1
        
    # Helper to check if removing idx is valid
    def can_remove(idx, cov):
        # If any neighbor has cov 1, removing idx makes it 0.
        # Check neighbors of idx
        return not np.any(cov[neighbors_np[idx]] == 1)

    # Helper to check if adding idx and removing out1, out2 is valid
    def can_swap_remove2_add1(idx_in, idx_out1, idx_out2, cov):
        # cov' = cov - e_out1 - e_out2 + e_in
        # We need min(cov') >= 1
        # Critical words are those with cov 1.
        # If a word w has cov 1, and w is in N(out1) or N(out2), it becomes 0 unless w is in N(in).
        # So for all w in (N(out1) U N(out2)) where cov[w]==1:
        #   w must be in N(in)
        
        # Get neighbors of out1 and out2
        n_out1 = neighbors_np[idx_out1]
        n_out2 = neighbors_np[idx_out2]
        n_in = neighbors_np[idx_in]
        
        # Words at risk: those in n_out1 or n_out2 with cov 1
        # We can just check min(cov + delta) >= 1
        # delta = -1 for n_out1, -1 for n_out2, +1 for n_in
        # Since N is small (729), we can just compute the new cov for the union of neighbors?
        # Union size is at most 13+13+13 = 39.
        # Let's just compute the new coverage for the affected words.
        
        affected = np.union1d(np.union1d(n_out1, n_out2), n_in)
        new_cov_affected = cov[affected].copy()
        
        # Subtract out1
        for w in n_out1:
            new_cov_affected[affected == w] -= 1
        # Subtract out2
        for w in n_out2:
            new_cov_affected[affected == w] -= 1
        # Add in
        for w in n_in:
            new_cov_affected[affected == w] += 1
            
        return np.min(new_cov_affected) >= 1

    # Phase 1: Greedy Reduction
    changed = True
    while changed and time.time() < deadline - 20.0:
        changed = False
        for idx in list(selected):
            if can_remove(idx, coverage):
                coverage[neighbors_np[idx]] -= 1
                selected.remove(idx)
                changed = True

    best_selected = set(selected)
    best_size = len(selected)
    best_cov = coverage.copy()
    
    current_selected = set(selected)
    current_cov = coverage.copy()
    current_size = len(selected)
    
    all_indices = list(range(N))
    random.seed(42)
    
    # Phase 2: Local Search
    while time.time() < deadline - 5.0:
        if current_size <= 73:
            break
            
        move_type = random.random()
        
        if move_type < 0.3:
            # Remove one
            candidates = list(current_selected)
            random.shuffle(candidates)
            removed = None
            for idx in candidates:
                if can_remove(idx, current_cov):
                    current_cov[neighbors_np[idx]] -= 1
                    current_selected.remove(idx)
                    current_size -= 1
                    removed = idx
                    break
            if removed:
                if current_size < best_size:
                    best_selected = set(current_selected)
                    best_size = current_size
                    best_cov = current_cov.copy()

        elif move_type < 0.8:
            # Swap one
            idx_out = random.choice(list(current_selected))
            idx_in = random.choice(all_indices)
            if idx_in in current_selected:
                continue
                
            # Check validity
            # New cov = cov - e_out + e_in
            # Affected: N(out) U N(in)
            n_out = neighbors_np[idx_out]
            n_in = neighbors_np[idx_in]
            affected = np.union1d(n_out, n_in)
            new_cov_affected = current_cov[affected].copy()
            
            for w in n_out:
                new_cov_affected[affected == w] -= 1
            for w in n_in:
                new_cov_affected[affected == w] += 1
                
            if np.min(new_cov_affected) >= 1:
                current_cov[affected] = new_cov_affected
                current_selected.remove(idx_out)
                current_selected.add(idx_in)
                # Size same, don't update best yet, but explore

        else:
            # Remove 2, Add 1
            if current_size < 3: continue
            candidates_out = random.sample(list(current_selected), 2)
            idx_out1, idx_out2 = candidates_out
            idx_in = random.choice(all_indices)
            if idx_in in current_selected:
                continue
                
            if can_swap_remove2_add1(idx_in, idx_out1, idx_out2, current_cov):
                # Apply
                current_cov[neighbors_np[idx_out1]] -= 1
                current_cov[neighbors_np[idx_out2]] -= 1
                current_cov[neighbors_np[idx_in]] += 1
                current_selected.remove(idx_out1)
                current_selected.remove(idx_out2)
                current_selected.add(idx_in)
                current_size -= 1
                
                if current_size < best_size:
                    best_selected = set(current_selected)
                    best_size = current_size
                    best_cov = current_cov.copy()

    # Final Polish
    final_selected = set(best_selected)
    final_cov = best_cov.copy()
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        for idx in list(final_selected):
            if can_remove(idx, final_cov):
                final_cov[neighbors_np[idx]] -= 1
                final_selected.remove(idx)
                changed = True

    codewords = [words[i] for i in final_selected]
    return codewords
