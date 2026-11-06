import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters
        # Simple greedy for general case
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = set()
            neigh_set.add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
            neighbors.append(list(neigh_set))
        
        N = q ** n
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 100.0
        
        # Simple greedy
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit: break
            
        # Reduction
        coverage_counts = np.zeros(N, dtype=int)
        for c in C:
            coverage_counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    coverage_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = [i]
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.append(word_to_idx[new_w])
        neighbors.append(neigh_set)
        
    # Convert to numpy arrays for faster indexing if needed, but lists are fine for small sizes
    # We will use lists for neighbors to avoid numpy overhead in tight loops if possible, 
    # but numpy is good for bulk operations.
    
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    best_code = []
    best_size = N
    
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts

    def reduce_code(code, counts):
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                # Check if any neighbor has count == 1
                # If min count >= 2, we can remove c
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts

    def fast_greedy():
        uncovered = np.ones(N, dtype=bool)
        pot = np.full(N, 17, dtype=int) # 1 + 8*2 = 17 neighbors
        code = []
        
        while np.any(uncovered):
            max_pot = np.max(pot)
            if max_pot == 0:
                break
            
            # Find all indices with max potential
            candidates = np.where(pot == max_pot)[0]
            # Random choice among candidates
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            code.append(best_idx)
            
            # Update uncovered and potentials
            for v in neighbors[best_idx]:
                if uncovered[v]:
                    uncovered[v] = False
                    # v is now covered. Any word w covering v loses 1 potential.
                    # Note: best_idx itself covers v, so its potential decreases too.
                    for w in neighbors[v]:
                        pot[w] -= 1
                        
        return code

    def local_search(code):
        counts = get_counts(code)
        code, counts = reduce_code(code, counts)
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Shuffle code to randomize order
            random.shuffle(code)
            
            for idx in range(len(code)):
                if time.time() - start_time > time_limit:
                    break
                
                c = code[idx]
                u_indices = np.array(neighbors[c])
                unique_mask = counts[u_indices] == 1
                unique_points = u_indices[unique_mask]
                
                if len(unique_points) == 0:
                    # Can remove
                    code.pop(idx)
                    counts[neighbors[c]] -= 1
                    improved = True
                    break
                else:
                    # Try to swap
                    if len(unique_points) > 4:
                        continue # Unlikely to find a swap
                    
                    # Find candidates covering all unique points
                    # Intersection of neighbors of unique points
                    cand_set = set(neighbors[unique_points[0]])
                    for u in unique_points[1:]:
                        cand_set &= set(neighbors[u])
                        if not cand_set:
                            break
                    
                    if not cand_set:
                        continue
                    
                    # Remove current code members
                    cand_set.difference_update(code)
                    
                    if not cand_set:
                        continue
                    
                    # Pick a random candidate
                    c_prime = random.choice(list(cand_set))
                    
                    # Tentative swap
                    old_code = code.copy()
                    old_counts = counts.copy()
                    
                    code[idx] = c_prime
                    counts[neighbors[c]] -= 1
                    counts[neighbors[c_prime]] += 1
                    
                    # Check validity (should be valid by construction of unique points)
                    # But check if any count < 1
                    if np.any(counts < 0):
                        code = old_code
                        counts = old_counts
                        continue
                    
                    # Try to reduce
                    new_code, new_counts = reduce_code(code, counts)
                    
                    if len(new_code) < len(old_code):
                        code = new_code
                        counts = new_counts
                        improved = True
                        break
                    else:
                        # Accept neutral swap to explore?
                        # Let's accept with some probability or if it changes the "fingerprint"
                        # For now, accept to escape local minima
                        code = new_code
                        counts = new_counts
                        improved = True
                        break
                        
        return code, counts

    # Main Loop: Multiple restarts
    while time.time() - start_time < time_limit:
        # 1. Greedy
        code = fast_greedy()
        
        # 2. Local Search
        code, counts = local_search(code)
        
        # 3. Update Best
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
        # Optional: If best_size is very low, we can try harder local search on best_code
        # But for now, restarts are cheap.

    # Final reduction on best_code
    best_code, _ = reduce_code(best_code, get_counts(best_code))
    
    return [list(all_words[i]) for i in best_code]
