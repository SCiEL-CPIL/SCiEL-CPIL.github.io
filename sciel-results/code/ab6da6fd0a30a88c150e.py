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
        
    # Initial code: Direct Sum (Tetracode x 3^2)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    # Convert to indices
    code_indices = [word_to_idx[tuple(w)] for w in init_code_words]
    
    # Coverage counts
    counts = np.zeros(N, dtype=np.int32)
    for idx in code_indices:
        for n_idx in neighbors[idx]:
            counts[n_idx] += 1
            
    # Local Search
    # We maintain a list of active indices
    active_indices = list(code_indices)
    
    def can_remove(idx):
        for n_idx in neighbors[idx]:
            if counts[n_idx] == 1:
                return False
        return True

    # Removal Loop
    # We use a strategy of repeated removals, and if stuck, we try perturbation (add random, remove 2)
    # We also try different random orders to escape local minima.
    
    best_size = len(active_indices)
    best_code = list(active_indices)
    
    # Main loop
    while time.time() < deadline:
        removed_any = False
        
        # Shuffle to break symmetry/bias
        random.shuffle(active_indices)
        
        for idx in active_indices:
            if time.time() > deadline:
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
            removed_count = 0
            candidates = []
            for idx in active_indices:
                if can_remove(idx):
                    candidates.append(idx)
            
            if len(candidates) >= 2:
                to_remove = candidates[:2]
                
                idx1 = to_remove[0]
                for n_idx in neighbors[idx1]:
                    counts[n_idx] -= 1
                active_indices.remove(idx1)
                
                for idx in active_indices:
                    if can_remove(idx):
                        for n_idx in neighbors[idx]:
                            counts[n_idx] -= 1
                        active_indices.remove(idx)
                        removed_count = 2
                        break
                        
            elif len(candidates) == 1:
                idx1 = candidates[0]
                for n_idx in neighbors[idx1]:
                    counts[n_idx] -= 1
                active_indices.remove(idx1)
                removed_count = 1

            if len(active_indices) < best_size:
                best_size = len(active_indices)
                best_code = list(active_indices)
            
            if len(active_indices) > best_size:
                # Revert
                counts = np.zeros(N, dtype=np.int32)
                for idx in best_code:
                    for n_idx in neighbors[idx]:
                        counts[n_idx] += 1
                active_indices = list(best_code)

    return [list(idx_to_word[i]) for i in best_code]

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
