# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from functools import lru_cache

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

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
    return _greedy_covering(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    N = q ** n  # 729
    
    # Precompute neighbors for each word index
    # neighbors[i] is a list of indices covered by word i (distance <= radius)
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
        
    best_code_size = float('inf')
    best_code_indices = []
    
    # Strategy: Multiple Greedy Constructions followed by Local Search
    # Greedy Construction:
    # 1. Initialize coverage array to 0
    # 2. While uncovered words exist:
    #    a. For each candidate c, count how many uncovered words it covers
    #    b. Pick c with max count (tie-break randomly to allow diversity)
    #    c. Mark its neighbors as covered
    #    d. Add c to code
    # 3. Local Search: Remove redundant words
    
    for restart in range(50): # Try 50 restarts
        if time.time() > deadline - 10:
            break
            
        # --- Greedy Phase ---
        covered = np.zeros(N, dtype=bool)
        code_indices = []
        
        # To speed up greedy, we can maintain a list of uncovered indices
        # But calculating scores for all 729 candidates is fast enough (729*13 ~ 10k ops)
        # We do this for ~80 steps, so ~800k ops total. Very fast.
        
        while not np.all(covered):
            best_candidate = -1
            best_score = -1
            best_candidates = []
            
            # Calculate score for all candidates
            # We can vectorize this partially or just loop
            # Loop is fine for 729 items
            for c in range(N):
                # Count uncovered neighbors
                # neighbors[c] is a list of indices
                # We need sum(1 for u in neighbors[c] if not covered[u])
                # Using numpy slicing might be faster if neighbors were arrays
                # But lists are okay.
                score = 0
                for u in neighbors[c]:
                    if not covered[u]:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_candidate = c
                    best_candidates = [c]
                elif score == best_score and score > 0:
                    best_candidates.append(c)
            
            if best_score == 0:
                # Should not happen if covered is not all True, but safety
                break
                
            # Pick one from best_candidates
            chosen = random.choice(best_candidates)
            code_indices.append(chosen)
            
            # Update coverage
            for u in neighbors[chosen]:
                covered[u] = True
                
        # --- Local Search Phase ---
        # Optimize the code found by greedy
        code_indices = _local_search_optimize(code_indices, neighbors, N, deadline)
        
        if len(code_indices) < best_code_size:
            best_code_size = len(code_indices)
            best_code_indices = list(code_indices)
            
            # If we hit the theoretical lower bound or known best, we can stop early?
            # Known best is 73. Lower bound 71.
            if best_code_size <= 73:
                # We can keep searching for 72 or 71, but 73 is a solid target.
                # Let's do a few more restarts to be sure, or break if very fast.
                pass

    return [list(idx_to_word[i]) for i in best_code_indices]

def _local_search_optimize(code_indices, neighbors, N, deadline):
    """
    Optimizes a set of codewords by removing redundant ones and performing perturbations.
    """
    active_indices = list(code_indices)
    
    # Coverage counts
    counts = np.zeros(N, dtype=np.int32)
    for idx in active_indices:
        for n_idx in neighbors[idx]:
            counts[n_idx] += 1
            
    def can_remove(idx):
        for n_idx in neighbors[idx]:
            if counts[n_idx] == 1:
                return False
        return True

    best_size = len(active_indices)
    best_code = list(active_indices)
    
    # Main loop
    while time.time() < deadline - 2:
        removed_any = False
        
        # Shuffle to break symmetry/bias
        random.shuffle(active_indices)
        
        for idx in active_indices:
            if time.time() > deadline - 2:
                break
            
            if can_remove(idx):
                # Remove idx
                for n_idx in neighbors[idx]:
                    counts[n_idx] -= 1
                active_indices.remove(idx)
                removed_any = True
                
        if len(active_indices) < best_size:
            best_size = len(active_indices)
            best_code = list(active_indices)
            
        if not removed_any:
            # Stuck. Try a perturbation: Add a random word, then try to remove.
            # This allows escaping local minima.
            
            # Pick a random word from the universe
            random_idx = random.randint(0, N - 1)
            
            # If it's already in the code, skip (or pick another)
            if random_idx in active_indices:
                continue
                
            # Add it
            active_indices.append(random_idx)
            for n_idx in neighbors[random_idx]:
                counts[n_idx] += 1
                
            # Now try to remove redundant words
            # We want to remove at least 1, preferably 2, to make progress
            candidates = []
            for idx in active_indices:
                if can_remove(idx):
                    candidates.append(idx)
            
            if len(candidates) >= 2:
                # Try to remove 2
                # Pick two distinct candidates
                idx1 = candidates[0]
                # Find a second one that is not idx1
                idx2 = -1
                for c in candidates[1:]:
                    if c != idx1:
                        idx2 = c
                        break
                
                if idx2 != -1:
                    # Remove idx1
                    for n_idx in neighbors[idx1]:
                        counts[n_idx] -= 1
                    active_indices.remove(idx1)
                    
                    # Check if idx2 is still removable
                    if can_remove(idx2):
                        for n_idx in neighbors[idx2]:
                            counts[n_idx] -= 1
                        active_indices.remove(idx2)
                    else:
                        # Revert idx1? No, we already removed it. 
                        # If we can't remove idx2, we are at size -1.
                        # This is still an improvement if best_size was larger.
                        pass
                        
            elif len(candidates) == 1:
                idx1 = candidates[0]
                for n_idx in neighbors[idx1]:
                    counts[n_idx] -= 1
                active_indices.remove(idx1)

            if len(active_indices) < best_size:
                best_size = len(active_indices)
                best_code = list(active_indices)
            
            if len(active_indices) > best_size:
                # Revert to best
                counts = np.zeros(N, dtype=np.int32)
                for idx in best_code:
                    for n_idx in neighbors[idx]:
                        counts[n_idx] += 1
                active_indices = list(best_code)
                
    return best_code

def _greedy_covering(q, n, radius, deadline):
    """
    Greedy construction of a covering code.
    """
    if n > 8:
        # Fallback: Direct sum if possible, else random
        if q == 3:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             if n >= 4:
                 return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
             else:
                 pass
        
        if n <= 8:
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
                
                if n <= 6:
                    # Cache neighbors for speed
                    neighbors_map = {w: get_neighbors(w) for w in all_words}
                    for w in all_words:
                        cnt = sum(1 for x in neighbors_map[w] if x in uncovered)
                        if cnt > best_count:
                            best_count = cnt
                            best_word = w
                else:
                    best_word = random.choice(all_words)
                    
                if best_word is None or best_count == 0:
                    break
                    
                code.append(list(best_word))
                uncovered.difference_update(get_neighbors(best_word))
                
            return code
        else:
            if q == 3 and n >= 4:
                 tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                 return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            else:
                 return [[0]*n]
# EVOLVE-BLOCK-END
