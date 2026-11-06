import itertools
import numpy as np
import random
import time
import sys

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
    neighbor_sets = [set(neigh.tolist()) for neigh in neighbors]
    
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
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
                if np.min(current_counts[neighbors[c]]) >= 2:
                    # Remove c
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts

    def frequency_peel(code, counts):
        """
        Frequency-driven peeling: iteratively remove the codeword with the lowest
        average coverage count (most "exposed"), provided it is redundant.
        This leverages the idea that in a minimal covering code, coverage should
        be more uniform. Removing low-frequency centers forces the remaining set
        to compensate, potentially leading to a more uniform and smaller code.
        """
        current_code = list(code)
        current_counts = counts.copy()
        
        while True:
            if not current_code:
                break
            removed_any = False
            # Calculate average coverage for each codeword
            # A codeword c covers neighbors[c]. The "exposure" is the average count
            # over its neighbors. Lower average means it's covering "less crowded" areas.
            # We want to remove redundant codewords. A codeword is redundant if min count >= 2.
            # Among redundant codewords, which one to remove first?
            # The one with the lowest average coverage might be the "most exposed" in terms of
            # being on the fringe, or conversely, the one with highest average might be most redundant.
            # Let's try: remove redundant codeword with LOWEST average coverage first.
            # Why? If a codeword covers areas that are barely covered (low avg), removing it might
            # break coverage. But if it's redundant (min>=2), it's safe.
            # Actually, removing the one with HIGHEST average coverage (most redundant) is standard.
            # The directive says: "removes the codeword with the lowest average coverage count".
            # Let's follow the directive.
            
            # Find all redundant codewords
            redundant = []
            for i, c in enumerate(current_code):
                if np.min(current_counts[neighbors[c]]) >= 2:
                    redundant.append(i)
            
            if not redundant:
                break
                
            # Among redundant, find the one with lowest average coverage
            best_idx_to_remove = -1
            best_avg = float('inf')
            for i in redundant:
                c = current_code[i]
                avg_cov = np.mean(current_counts[neighbors[c]])
                if avg_cov < best_avg:
                    best_avg = avg_cov
                    best_idx_to_remove = i
            
            if best_idx_to_remove != -1:
                c_to_remove = current_code[best_idx_to_remove]
                current_counts[neighbors[c_to_remove]] -= 1
                current_code.pop(best_idx_to_remove)
                removed_any = True
        
        return current_code, current_counts

    # 1. Fast Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    start_time = time.time()
    time_limit = 115.0
    
    # Initialize gains
    gains = np.array([np.sum(uncovered[neighbors[i]]) for i in range(N)], dtype=int)
    
    while np.any(uncovered):
        if time.time() - start_time > time_limit:
            break
        
        best_idx = int(np.argmax(gains))
        best_count = gains[best_idx]
        if best_count == 0:
            break
        
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        
        # Update gains
        # For efficiency, recompute gains periodically or use a smarter update.
        # Recomputing is O(N*17) ~ 100k ops. With ~500 iterations, 50M ops.
        # In Python loop, this might be slow. Let's try to optimize the update.
        # A candidate i's gain decreases by |neighbors[i] ∩ neighbors[best_idx] ∩ uncovered_before|.
        # Since we just marked neighbors[best_idx] as covered, the points lost are
        # neighbors[i] ∩ neighbors[best_idx].
        # We can precompute the intersection sizes? No, too large.
        # Let's just recompute gains every K steps or use a heuristic.
        # For now, recompute if time allows, otherwise skip.
        
        # Optimization: Only recompute gains if we haven't done so recently
        # or if the change is significant.
        # Actually, let's just recompute. It's the safest.
        gains = np.array([np.sum(uncovered[neighbors[i]]) for i in range(N)], dtype=int)
            
    # 2. Local Search
    C, counts = reduce_code(C, get_counts(C))
    
    # Apply frequency peeling
    C, counts = frequency_peel(C, counts)
    
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    
    # Local Search Loop
    while time.time() - start_time < time_limit:
        # 1. Reduction Pass (Standard)
        old_size = len(C)
        C, counts = reduce_code(C, counts)
        
        # 2. Frequency Peeling Pass
        C, counts = frequency_peel(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            
        if time.time() - start_time > time_limit:
            break
            
        # 3. Swap Pass
        if len(C) == old_size:
            # No reduction happened. Try to find a move.
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
                    # Should have been removed
                    C.remove(c)
                    counts[neighbors[c]] -= 1
                    improved = True
                    break
                
                if len(unique_points) > 3:
                    continue
                
                # Find candidates c' that cover all unique_points
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
                new_C, new_counts = reduce_code(C, counts)
                new_C, new_counts = frequency_peel(new_C, new_counts)
                
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
                    # Accept neutral move to escape local minimum
                    C = new_C
                    counts = new_counts
                    improved = True
                    break
            
            if not improved:
                # Try random kicks
                for _ in range(10):
                    if time.time() - start_time > time_limit: break
                    if not C: break
                    c_idx = random.randint(0, len(C)-1)
                    c = C[c_idx]
                    cand = neighbors[c]
                    c_prime = cand[random.randint(0, len(cand)-1)]
                    if c_prime in C: continue
                    
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
                        
                    new_C, new_counts = reduce_code(C, counts)
                    new_C, new_counts = frequency_peel(new_C, new_counts)
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

    # Final Reduction with aggressive polish
    best_C, best_counts = reduce_code(best_C, get_counts(best_C))
    best_C, best_counts = frequency_peel(best_C, best_counts)
    
    # Additional polish
    polish_start = time.time()
    polish_deadline = start_time + time_limit
    
    for polish_round in range(20):
        if time.time() > polish_deadline - 5:
            break
        
        old_size = len(best_C)
        best_C, best_counts = reduce_code(best_C, best_counts)
        best_C, best_counts = frequency_peel(best_C, best_counts)
        
        if len(best_C) < old_size:
            continue
        
        improved = False
        for _ in range(50):
            if time.time() > polish_deadline - 5:
                break
            if not best_C:
                break
            
            c_idx = random.randint(0, len(best_C) - 1)
            c = best_C[c_idx]
            
            u_indices = neighbors[c]
            unique_mask = best_counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            
            if len(unique_points) == 0:
                best_C.remove(c)
                best_counts[neighbors[c]] -= 1
                improved = True
                break
            
            if len(unique_points) > 4:
                continue
            
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
            
            candidates_set.difference_update(best_C)
            if not candidates_set:
                continue
            
            c_prime = random.choice(list(candidates_set))
            
            old_C = best_C.copy()
            old_counts = best_counts.copy()
            
            best_C.remove(c)
            best_C.append(c_prime)
            best_counts[neighbors[c]] -= 1
            best_counts[neighbors[c_prime]] += 1
            
            if np.any(best_counts < 1):
                best_C = old_C
                best_counts = old_counts
                continue
            
            new_C, new_counts = reduce_code(best_C, best_counts)
            new_C, new_counts = frequency_peel(new_C, new_counts)
            
            if len(new_C) < len(old_C):
                best_C = new_C
                best_counts = new_counts
                improved = True
                break
            else:
                best_C = new_C
                best_counts = new_counts
                improved = True
                break
        
        if not improved:
            break
    
    # Final reduction pass
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    best_C, _ = frequency_peel(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
