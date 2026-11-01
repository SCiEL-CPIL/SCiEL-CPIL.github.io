# EVOLVE-BLOCK-START
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
    deadline = start_time + 110.0  # Safety margin

    if n == 0:
        return [[]]

    # Specific optimization for the known hard case: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Fallback for other cases
    if q == 3 and radius == 1 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return code
    
    # Generic greedy for small n
    return greedy_covering(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    N = q ** n  # 729
    
    # Precompute all words and their neighbors (distance <= 1)
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    
    # neighbors: list of lists of indices at distance <= 1
    neighbors = [None] * N
    for i, w in enumerate(all_words):
        neigh = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                neigh.append(word_to_idx[tuple(new_w)])
        neighbors[i] = neigh
    
    # Initial code: Direct Sum (Tetracode x 3^2) -> 108 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    code_indices = [word_to_idx[tuple(w)] for w in init_code_words]
    
    # Coverage counts: how many codewords cover each position
    counts = np.zeros(N, dtype=np.int32)
    for idx in code_indices:
        for n_idx in neighbors[idx]:
            counts[n_idx] += 1
            
    active_indices = set(code_indices)
    
    # Precompute for each word, its neighbor list as a numpy array for potential speedups if needed,
    # but the list access is fast enough.
    
    def can_remove(idx, counts):
        # A word can be removed if no neighbor has count 1
        # i.e., min(counts[neighbors[idx]]) > 1
        # Since idx itself is in neighbors[idx] and counts[idx] >= 1 (if idx is in code),
        # and other neighbors might have count 1.
        # If any neighbor has count 1, removing idx would leave it uncovered (count 0).
        # So we check if min count in neighbors[idx] is > 1.
        # Note: neighbors[idx] includes idx itself. If idx is in code, counts[idx] >= 1.
        # If we remove idx, counts[neighbors[idx]] decrease by 1.
        # We need counts[neighbors[idx]] - 1 >= 1  => counts[neighbors[idx]] >= 2 for all neighbors.
        # So we check if min(counts[neighbors[idx]]) >= 2.
        min_count = min(counts[n_idx] for n_idx in neighbors[idx])
        return min_count >= 2

    def remove_idx(idx, counts, active_indices):
        for n_idx in neighbors[idx]:
            counts[n_idx] -= 1
        active_indices.remove(idx)

    def add_idx(idx, counts, active_indices):
        for n_idx in neighbors[idx]:
            counts[n_idx] += 1
        active_indices.add(idx)

    # Local Search Loop
    # We want to minimize the size of active_indices.
    # Phase 1: Pure reduction
    # Phase 2: Perturbation (add random, try to remove 2)
    
    # To avoid getting stuck in a specific local minimum, we can restart from a different random perturbation
    # or just keep perturbing.
    
    # First, do a full reduction pass
    # Sort active indices? Random order helps avoid bias.
    while time.time() < deadline:
        removed_any = False
        # Convert to list to iterate, since we are modifying the set
        current_indices = list(active_indices)
        random.shuffle(current_indices)
        
        for idx in current_indices:
            if idx not in active_indices:
                continue
            if time.time() > deadline:
                return [list(idx_to_word[i]) for i in active_indices]
            
            if can_remove(idx, counts):
                remove_idx(idx, counts, active_indices)
                removed_any = True
        
        if not removed_any:
            # Local minimum reached for reduction.
            # Try perturbation: Add a random word, then try to remove 2 words.
            # If we can remove 2, net size decreases by 1.
            # If we can remove 1, net size stays same.
            # If we can remove 0, net size increases by 1 (bad, unless we revert).
            
            # Pick a random word not in code? Or any word?
            # Picking a word not in code is usually better for exploration.
            random_idx = random.randint(0, N - 1)
            
            # Add it
            add_idx(random_idx, counts, active_indices)
            
            # Now try to remove up to 2 words
            removed_count = 0
            # We want to find removable words.
            # To maximize chance of removing 2, we should look for words that are "weakly" covered.
            # Words with low min-count in their neighborhood are more likely to be removable.
            
            # Get list of candidates that might be removable
            # We can just scan and remove the first one we find, then scan again for the second.
            
            # Optimization: Prioritize removing words that are neighbors of the added word?
            # Or just random.
            
            # Let's try to find removable words.
            # To be efficient, we can check the neighbors of the added word first, as they are more likely to have increased count,
            # but actually we want to find words that are NOT critical.
            
            # Let's just do a random shuffle scan for removable words.
            
            # To increase efficiency, we can maintain a list of "candidate" removals?
            # For now, simple scan.
            
            # First removal
            found_first = False
            indices_to_check = list(active_indices)
            random.shuffle(indices_to_check)
            
            for idx in indices_to_check:
                if idx not in active_indices:
                    continue
                if can_remove(idx, counts):
                    remove_idx(idx, counts, active_indices)
                    removed_count += 1
                    found_first = True
                    break
            
            # Second removal
            if found_first:
                indices_to_check = list(active_indices)
                random.shuffle(indices_to_check)
                for idx in indices_to_check:
                    if idx not in active_indices:
                        continue
                    if can_remove(idx, counts):
                        remove_idx(idx, counts, active_indices)
                        removed_count += 1
                        break
            
            if removed_count < 2:
                # We didn't reduce the size enough.
                # If removed_count == 0, we increased size by 1.
                # If removed_count == 1, size is unchanged.
                # It's often better to revert if we increased size, to avoid wandering into large codes.
                # But if size is unchanged, it might be a move to a different local minimum.
                
                if removed_count == 0:
                    # Revert
                    # We added random_idx. If we didn't remove it, it's still there.
                    # Wait, if we didn't remove anything, random_idx is still in code.
                    # We should remove it to get back to previous state.
                    remove_idx(random_idx, counts, active_indices)
                elif removed_count == 1:
                    # Size is same. We can keep it to explore, or revert.
                    # Keeping it allows exploration. Let's keep it.
                    pass
                    
            # If removed_count == 2, we improved. Keep it.
            
            # Check if we can do a pure reduction again?
            # The loop will continue and check for removals.
            
            # If we are stuck (removed_count < 2 repeatedly), we might want to restart?
            # For now, the loop continues.

    return [list(idx_to_word[i]) for i in active_indices]

def greedy_covering(q, n, radius, deadline):
    """
    Greedy construction of a covering code.
    """
    if n > 8:
        if q == 3:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             if n >= 4:
                 return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
             else:
                 return [[0]*n]
        return [[0]*n]
        
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    code = []
    
    def get_neighbors(c):
        neighbors = {c}
        for i in range(n):
            orig = c[i]
            for val in range(q):
                if val == orig: continue
                lst = list(c)
                lst[i] = val
                neighbors.add(tuple(lst))
        return neighbors

    while uncovered and time.time() < deadline:
        best_word = None
        best_count = -1
        
        # Greedy choice: pick word covering most uncovered
        for w in all_words:
            if w in code: continue
            cnt = 0
            for x in get_neighbors(w):
                if x in uncovered:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_word = w
                
        if best_word is None or best_count == 0:
            break
            
        code.append(list(best_word))
        uncovered.difference_update(get_neighbors(best_word))
        
    return code
# EVOLVE-BLOCK-END
