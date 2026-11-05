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
    
    # Precompute neighbors as a 2D array for faster access
    # Each word has 1 + 8*2 = 17 neighbors
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_list = [i]
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_list.append(word_to_idx[new_w])
        neighbors.append(neigh_list)
    
    # Convert to numpy arrays for faster indexing
    neighbors_arr = np.array(neighbors, dtype=int)
    
    # 1. Fast Greedy Initialization with vectorized operations
    def greedy_init(seed=None, time_limit=115.0):
        if seed is not None:
            random.seed(seed)
        
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        
        while np.any(uncovered):
            if time.time() - start_time > time_limit:
                break
            
            # Vectorized: compute coverage count for each candidate
            # For each word i, count how many uncovered words it covers
            # neighbors_arr[i] gives the neighbors of word i
            # We need sum(uncovered[neighbors_arr[i]]) for all i
            
            # This is expensive to do fully vectorized, so use a hybrid approach
            # Process in batches for better cache performance
            batch_size = 1000
            best_idx = -1
            best_count = -1
            
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                # Get neighbors for this batch
                batch_neighbors = neighbors_arr[start:end]
                # Count uncovered in each batch's neighbors
                # uncovered is a boolean array, batch_neighbors[i] gives indices
                # We need to count uncovered[batch_neighbors[i]] for each i
                # This requires advanced indexing which can be slow
                # Alternative: use the fact that we're iterating anyway
                
                for j in range(end - start):
                    idx = start + j
                    if not uncovered[idx]:
                        continue  # Skip if already covered? No, we need to cover others
                    cnt = np.sum(uncovered[neighbors_arr[idx]])
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = idx
            
            if best_count == 0:
                break
            
            C.append(best_idx)
            uncovered[neighbors_arr[best_idx]] = False
        
        return C
    
    # 2. Local Search with reduction and swap
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors_arr[c]] += 1
        return counts
    
    def reduce_code(code, counts):
        """
        Reduces code by removing redundant words.
        Returns new_code and new_counts.
        """
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors_arr[c]]) >= 2:
                    # Remove c
                    current_counts[neighbors_arr[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts
    
    # 3. Local search with restarts
    def local_search(C, counts, time_limit, seed=None):
        if seed is not None:
            random.seed(seed)
        
        start_time = time.time()
        best_size = len(C)
        best_C = C.copy()
        best_counts = counts.copy()
        
        # Local Search Loop
        while time.time() - start_time < time_limit:
            # 1. Reduction Pass
            C, counts = reduce_code(C, counts)
            if len(C) < best_size:
                best_size = len(C)
                best_C = C.copy()
                best_counts = counts.copy()
                
            if time.time() - start_time > time_limit:
                break
                
            # 2. Swap Pass
            improved = False
            # Shuffle C to avoid bias
            C_shuffled = C.copy()
            random.shuffle(C_shuffled)
            
            for c in C_shuffled:
                if time.time() - start_time > time_limit:
                    break
                    
                u_indices = neighbors_arr[c]
                unique_mask = counts[u_indices] == 1
                unique_points = u_indices[unique_mask]
                
                if len(unique_points) == 0:
                    continue # Should have been removed
                    
                # If too many unique points, unlikely to find a single replacement
                if len(unique_points) > 3:
                    continue
                
                # Find candidates c' that cover all unique_points
                # Intersection of neighbors[u] for u in unique_points
                if len(unique_points) == 1:
                    candidates_set = set(neighbors_arr[unique_points[0]].tolist())
                else:
                    candidates_set = set(neighbors_arr[unique_points[0]].tolist())
                    for u in unique_points[1:]:
                        candidates_set &= set(neighbors_arr[u].tolist())
                        if not candidates_set:
                            break
                
                if not candidates_set:
                    continue
                
                # Filter candidates: must not be in C
                candidates_set.difference_update(C)
                
                if not candidates_set:
                    continue
                    
                # Pick a candidate with heuristic: maximize overlap with other codewords
                # This might create more redundancy, enabling further reduction
                best_c_prime = None
                max_overlap = -1
                c_set = set(C)
                
                for c_prime in candidates_set:
                    # Count how many neighbors of c_prime are already in C
                    overlap = len(set(neighbors_arr[c_prime].tolist()) & c_set)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_c_prime = c_prime
                
                if best_c_prime is None:
                    continue
                    
                c_prime = best_c_prime
                
                # Apply swap
                old_C = C.copy()
                old_counts = counts.copy()
                
                C.remove(c)
                C.append(c_prime)
                
                counts[neighbors_arr[c]] -= 1
                counts[neighbors_arr[c_prime]] += 1
                
                # Check validity (should be valid)
                if np.any(counts < 1):
                    C = old_C
                    counts = old_counts
                    continue
                
                # Try to reduce
                new_C, new_counts = reduce_code(C, counts)
                
                if len(new_C) < len(old_C):
                    # Improvement!
                    C = new_C
                    counts = new_counts
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = counts.copy()
                    improved = True
                    break # Restart loop
                    
                else:
                    # Keep the swap to escape local minimum
                    improved = True
                    break # Restart loop to see if this new state allows reduction
            
            if not improved:
                # No improvement in this pass, try a random perturbation
                if len(C) > 10 and random.random() < 0.3:
                    # Randomly swap a few codewords
                    for _ in range(3):
                        if time.time() - start_time > time_limit:
                            break
                        c = random.choice(C)
                        u_indices = neighbors_arr[c]
                        unique_mask = counts[u_indices] == 1
                        unique_points = u_indices[unique_mask]
                        
                        if len(unique_points) == 0:
                            continue
                        if len(unique_points) > 3:
                            continue
                        
                        if len(unique_points) == 1:
                            candidates_set = set(neighbors_arr[unique_points[0]].tolist())
                        else:
                            candidates_set = set(neighbors_arr[unique_points[0]].tolist())
                            for u in unique_points[1:]:
                                candidates_set &= set(neighbors_arr[u].tolist())
                                if not candidates_set:
                                    break
                        
                        if not candidates_set:
                            continue
                        
                        candidates_set.difference_update(C)
                        
                        if not candidates_set:
                            continue
                        
                        c_prime = random.choice(list(candidates_set))
                        
                        old_C = C.copy()
                        old_counts = counts.copy()
                        
                        C.remove(c)
                        C.append(c_prime)
                        
                        counts[neighbors_arr[c]] -= 1
                        counts[neighbors_arr[c_prime]] += 1
                        
                        if np.any(counts < 1):
                            C = old_C
                            counts = old_counts
                            continue
                        
                        new_C, new_counts = reduce_code(C, counts)
                        if len(new_C) < len(C):
                            C = new_C
                            counts = new_counts
                            if len(C) < best_size:
                                best_size = len(C)
                                best_C = C.copy()
                                best_counts = counts.copy()
        
        return best_C, best_counts
    
    # Main execution with multiple restarts
    start_time = time.time()
    time_limit = 115.0
    
    # Initial greedy
    C = greedy_init(time_limit=time_limit)
    counts = get_counts(C)
    C, counts = reduce_code(C, counts)
    
    best_C = C.copy()
    best_counts = counts.copy()
    best_size = len(C)
    
    # Multiple local search passes with different seeds
    remaining_time = time_limit - (time.time() - start_time)
    num_restarts = max(1, int(remaining_time // 10))  # ~10s per restart
    
    for restart in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5:
            break
        
        # Use the current best as starting point for local search
        C_copy = best_C.copy()
        counts_copy = best_counts.copy()
        
        time_for_this = min(remaining_time - (time.time() - start_time), 10.0)
        if time_for_this < 2.0:
            break
        
        C_new, counts_new = local_search(C_copy, counts_copy, time_for_this, seed=restart)
        
        if len(C_new) < best_size:
            best_size = len(C_new)
            best_C = C_new.copy()
            best_counts = counts_new.copy()
    
    # Final reduction
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
