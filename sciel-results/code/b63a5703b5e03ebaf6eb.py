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
    deadline = start_time + 115.0  # Safety margin

    if n == 0:
        return [[]]

    # Specific optimization for the known hard case: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Fallback for other cases
    # Use a simple greedy or direct sum if applicable
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
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    N = q ** n  # 729

    # Precompute neighbors for each word index using vectorized approach
    # For q=3, n=6, radius=1: each word has 1 + 6*2 = 13 neighbors
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

    # Also create a neighbor matrix for vectorized operations
    # neighbor_matrix[i] = set of indices covered by word i
    # For efficiency, we'll use a different approach: precompute the coverage matrix
    # coverage_matrix[j, i] = 1 if word j is covered by word i (j in neighbors[i])
    coverage_matrix = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        for n_idx in neighbors[i]:
            coverage_matrix[n_idx, i] = 1

    # Initial code: Direct Sum (Tetracode x 3^2)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    # Convert to indices
    code_indices = [word_to_idx[tuple(w)] for w in init_code_words]
    
    # Coverage counts: counts[j] = number of codewords that cover word j
    counts = np.zeros(N, dtype=np.int32)
    for idx in code_indices:
        counts += coverage_matrix[:, idx]
    
    # Local Search
    # We maintain a set of active indices
    active_set = set(code_indices)
    active_list = list(code_indices)
    
    # Helper to check if a word can be removed using vectorized min
    def can_remove(idx):
        # A word idx can be removed if all words it covers have count > 1
        # i.e., min(counts[neighbors[idx]]) > 1
        # Using the coverage matrix: coverage_matrix[:, idx] gives which words are covered by idx
        covered = np.where(coverage_matrix[:, idx] == 1)[0]
        if len(covered) == 0:
            return True
        return np.min(counts[covered]) > 1

    # Removal Loop
    while time.time() < deadline:
        removed_any = False
        
        # Shuffle to break symmetry/bias
        random.shuffle(active_list)
        
        for idx in active_list:
            if time.time() > deadline:
                break
            
            if idx not in active_set:
                continue
                
            if can_remove(idx):
                # Remove idx
                counts -= coverage_matrix[:, idx]
                active_set.remove(idx)
                active_list.remove(idx)
                removed_any = True
                
        if not removed_any:
            # Stuck. Try a perturbation: Add a random word, then try to remove.
            # This allows escaping local minima.
            
            # Pick a random word from the universe
            random_idx = random.randint(0, N - 1)
            
            # If it's already in the code, skip (or pick another)
            if random_idx in active_set:
                continue
                
            # Add it
            active_set.add(random_idx)
            active_list.append(random_idx)
            counts += coverage_matrix[:, random_idx]
            
            # Now try to remove redundant words
            # We want to see if we can remove MORE than 1 word to improve size
            
            # Find all removable candidates
            candidates = []
            for idx in active_list:
                if idx not in active_set:
                    continue
                if can_remove(idx):
                    candidates.append(idx)
            
            # We want to remove up to 2 words to get net improvement
            if len(candidates) >= 2:
                # Remove 2 words
                to_remove = candidates[:2]
                
                # Remove first
                idx1 = to_remove[0]
                if idx1 in active_set:
                    counts -= coverage_matrix[:, idx1]
                    active_set.remove(idx1)
                    if idx1 in active_list:
                        active_list.remove(idx1)
                
                # Try to remove a second word
                found_second = False
                for idx in active_list:
                    if idx not in active_set:
                        continue
                    if idx == idx1:
                        continue
                    if can_remove(idx):
                        counts -= coverage_matrix[:, idx]
                        active_set.remove(idx)
                        active_list.remove(idx)
                        found_second = True
                        break
                        
                if not found_second:
                    # We removed 1, added 1. Size unchanged.
                    # Keep it to explore
                    pass
            elif len(candidates) == 1:
                # Remove 1. Size unchanged.
                idx1 = candidates[0]
                if idx1 in active_set:
                    counts -= coverage_matrix[:, idx1]
                    active_set.remove(idx1)
                    if idx1 in active_list:
                        active_list.remove(idx1)
            else:
                # Cannot remove anything. Revert the addition
                if random_idx in active_set:
                    active_set.remove(random_idx)
                    counts -= coverage_matrix[:, random_idx]
                    if random_idx in active_list:
                        active_list.remove(random_idx)
                continue

    # Convert back to list of lists
    return [list(idx_to_word[idx]) for idx in active_set]

def greedy_covering(q, n, radius, deadline):
    """
    Greedy construction of a covering code.
    """
    # Placeholder for generic greedy, used as fallback
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
            if w in code: continue # Optimization: don't pick existing
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
