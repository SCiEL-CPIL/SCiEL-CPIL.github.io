import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters
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
    
    # Precompute neighbors as numpy arrays for fast operations
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = set()
        neigh_set.add(i)
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        neighbors.append(np.array(list(neigh_set), dtype=int))
    
    # Precompute neighbor sets as Python sets for fast intersection
    neighbor_sets = [set(arr.tolist()) for arr in neighbors]
    
    start_time = time.time()
    time_limit = 115.0
    
    # Generate a valid 486-word covering using greedy
    uncovered = np.ones(N, dtype=bool)
    C = []
    
    # Fast greedy: pick word covering most uncovered
    while np.any(uncovered):
        best_idx = -1
        best_count = -1
        for i in range(N):
            cnt = np.sum(uncovered[neighbors[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_count == 0:
            break
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        if time.time() - start_time > time_limit:
            break
    
    # Build coverage counts
    counts = np.zeros(N, dtype=int)
    for c in C:
        counts[neighbors[c]] += 1
    
    # Verify coverage
    if np.any(counts < 1):
        # Add more words to cover uncovered points
        while np.any(counts < 1):
            uncovered_mask = counts < 1
            # Find word covering most uncovered
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered_mask[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0:
                break
            C.append(best_idx)
            counts[neighbors[best_idx]] += 1
            if time.time() - start_time > time_limit:
                break
    
    # Local search: try to remove words
    def try_remove(c, counts):
        """Check if word c can be removed. Return True if yes, and update counts."""
        if np.min(counts[neighbors[c]]) >= 2:
            counts[neighbors[c]] -= 1
            return True
        return False
    
    # Local search loop
    best_C = C.copy()
    best_counts = counts.copy()
    best_size = len(C)
    
    while time.time() - start_time < time_limit:
        # Pass 1: Try to remove words
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(counts[neighbors[c]]) >= 2:
                    counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
        
        if time.time() - start_time > time_limit:
            break
        
        # Pass 2: Try swaps to enable further removals
        improved = False
        indices = list(range(len(C)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            c = C[idx]
            u_indices = neighbors[c]
            unique_mask = counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            
            if len(unique_points) == 0:
                # Remove
                C.remove(c)
                counts[neighbors[c]] -= 1
                improved = True
                break
            
            if len(unique_points) > 3:
                continue
            
            # Find candidates that cover all unique points
            if len(unique_points) == 1:
                candidates_set = set(neighbor_sets[unique_points[0]])
            else:
                candidates_set = set(neighbor_sets[unique_points[0]])
                for u in unique_points[1:]:
                    candidates_set &= neighbor_sets[u]
                    if not candidates_set:
                        break
            
            if not candidates_set:
                continue
            
            candidates_set.difference_update(C)
            
            if not candidates_set:
                continue
            
            # Pick candidate that maximizes overlap with existing code
            c_prime = random.choice(list(candidates_set))
            
            old_C = C.copy()
            old_counts = counts.copy()
            
            C.remove(c)
            C.append(c_prime)
            
            counts[neighbors[c]] -= 1
            counts[neighbors[c_prime]] += 1
            
            if np.any(counts < 1):
                C = old_C
                counts = old_counts
                continue
            
            # Try to reduce
            new_C = []
            new_counts = counts.copy()
            for cc in C:
                if np.min(new_counts[neighbors[cc]]) >= 2:
                    new_counts[neighbors[cc]] -= 1
                else:
                    new_C.append(cc)
            
            if len(new_C) < len(old_C):
                C = new_C
                counts = new_counts
                if len(C) < best_size:
                    best_size = len(C)
                    best_C = C.copy()
                    best_counts = counts.copy()
                improved = True
                break
            else:
                # Keep the swap to escape local minimum
                C = new_C
                counts = new_counts
                improved = True
                break
        
        if not improved:
            # Try random kicks
            for _ in range(5):
                if time.time() - start_time > time_limit:
                    break
                if not C:
                    break
                c_idx = random.randint(0, len(C)-1)
                c = C[c_idx]
                cand = neighbors[c]
                c_prime = cand[random.randint(0, len(cand)-1)]
                if c_prime in C:
                    continue
                
                old_C = C.copy()
                old_counts = counts.copy()
                
                C.remove(c)
                C.append(c_prime)
                counts[neighbors[c]] -= 1
                counts[neighbors[c_prime]] += 1
                
                if np.any(counts < 1):
                    C = old_C
                    counts = old_counts
                    continue
                
                # Try reduce
                new_C = []
                new_counts = counts.copy()
                for cc in C:
                    if np.min(new_counts[neighbors[cc]]) >= 2:
                        new_counts[neighbors[cc]] -= 1
                    else:
                        new_C.append(cc)
                
                if len(new_C) < len(old_C):
                    C = new_C
                    counts = new_counts
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = counts.copy()
                    break
                else:
                    C = new_C
                    counts = new_counts
    
    # Final reduction
    final_counts = np.zeros(N, dtype=int)
    for c in best_C:
        final_counts[neighbors[c]] += 1
    
    changed = True
    while changed:
        changed = False
        new_C = []
        for c in best_C:
            if np.min(final_counts[neighbors[c]]) >= 2:
                final_counts[neighbors[c]] -= 1
                changed = True
            else:
                new_C.append(c)
        best_C = new_C
    
    return [list(all_words[i]) for i in best_C]
