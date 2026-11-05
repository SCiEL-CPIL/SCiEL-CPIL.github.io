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
            neigh_set = {i}
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    neigh_set.add(word_to_idx[tuple(new_w_list)])
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
    
    # Precompute neighbors
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = {i}
        for pos in range(8):
            for val in range(3):
                if val == w[pos]: continue
                new_w_list = list(w)
                new_w_list[pos] = val
                neigh_set.add(word_to_idx[tuple(new_w_list)])
        neighbors.append(np.array(list(neigh_set), dtype=int))
        
    # 1. Fast Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    start_time = time.time()
    time_limit = 115.0
    
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
            
    # 2. Local Search
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
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts

    C, counts = reduce_code(C, get_counts(C))
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    
    # Local Search Loop with Randomization
    while time.time() - start_time < time_limit:
        # 1. Reduction Pass
        C, counts = reduce_code(C, counts)
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            
        if time.time() - start_time > time_limit:
            break
            
        # 2. Swap Pass with Randomization
        improved = False
        # Try multiple random swaps to escape local minima
        for _ in range(20):
            if time.time() - start_time > time_limit:
                break
            if not C:
                break
                
            c = random.choice(C)
            u_indices = neighbors[c]
            unique_mask = counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            
            if len(unique_points) == 0:
                continue
            if len(unique_points) > 4:
                continue
                
            # Find candidates covering unique_points
            candidates_set = None
            for u in unique_points:
                if candidates_set is None:
                    candidates_set = set(neighbors[u].tolist())
                else:
                    candidates_set &= set(neighbors[u].tolist())
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
                
            new_C, new_counts = reduce_code(C, counts)
            
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
                # Revert to avoid cycling, but randomization helps escape
                C = old_C
                counts = old_counts
        
        # 3. Two-for-One Move: remove two codewords, add one
        if not improved and len(C) >= 2 and time.time() - start_time < time_limit:
            for _ in range(10):
                if time.time() - start_time > time_limit:
                    break
                if len(C) < 2:
                    break
                
                c1 = random.choice(C)
                remaining = [x for x in C if x != c1]
                if not remaining:
                    break
                c2 = random.choice(remaining)
                
                # Points uniquely covered by c1 or c2
                u1 = neighbors[c1]
                u2 = neighbors[c2]
                unique_mask1 = counts[u1] == 1
                unique_mask2 = counts[u2] == 1
                unique_points = np.union1d(u1[unique_mask1], u2[unique_mask2])
                
                if len(unique_points) == 0:
                    continue
                if len(unique_points) > 6:
                    continue
                
                # Find candidates covering all unique_points
                candidates_set = None
                for u in unique_points:
                    if candidates_set is None:
                        candidates_set = set(neighbors[u].tolist())
                    else:
                        candidates_set &= set(neighbors[u].tolist())
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
                
                C.remove(c1)
                C.remove(c2)
                C.append(c_prime)
                counts[neighbors[c1]] -= 1
                counts[neighbors[c2]] -= 1
                counts[neighbors[c_prime]] += 1
                
                if np.any(counts < 1):
                    C = old_C
                    counts = old_counts
                    continue
                
                new_C, new_counts = reduce_code(C, counts)
                
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
                    C = old_C
                    counts = old_counts
                
    # Final Reduction
    best_C, best_counts = reduce_code(best_C, get_counts(best_C))
    
    # Final Polish: exhaustive single-removal + exhaustive swap search
    if time.time() - start_time < time_limit - 5:
        # Try to remove each codeword one at a time
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in best_C:
                if np.min(best_counts[neighbors[c]]) >= 2:
                    best_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            best_C = new_C
        
        # Exhaustive swap: for each pair (c_in, c_out), check if swap improves
        C_list = list(best_C)
        C_set = set(C_list)
        improved_swap = True
        while improved_swap and time.time() - start_time < time_limit - 5:
            improved_swap = False
            for i, c in enumerate(C_list):
                if time.time() - start_time > time_limit - 5:
                    break
                u_indices = neighbors[c]
                unique_mask = best_counts[u_indices] == 1
                unique_points = u_indices[unique_mask]
                
                if len(unique_points) == 0:
                    continue
                if len(unique_points) > 4:
                    continue
                
                candidates_set = None
                for u in unique_points:
                    if candidates_set is None:
                        candidates_set = set(neighbors[u].tolist())
                    else:
                        candidates_set &= set(neighbors[u].tolist())
                        if not candidates_set:
                            break
                
                if not candidates_set:
                    continue
                
                candidates_set.difference_update(C_set)
                if not candidates_set:
                    continue
                
                # Try each candidate
                for c_prime in candidates_set:
                    old_counts = best_counts.copy()
                    best_counts[neighbors[c]] -= 1
                    best_counts[neighbors[c_prime]] += 1
                    
                    if np.any(best_counts < 1):
                        best_counts = old_counts
                        continue
                    
                    # Check if we can reduce further
                    new_C = []
                    for cc in C_list:
                        if cc == c:
                            continue
                        if np.min(best_counts[neighbors[cc]]) >= 2:
                            best_counts[neighbors[cc]] -= 1
                        else:
                            new_C.append(cc)
                    new_C.append(c_prime)
                    
                    if len(new_C) < len(C_list):
                        C_list = new_C
                        C_set = set(C_list)
                        improved_swap = True
                        break
                    else:
                        best_counts = old_counts
        
        best_C, best_counts = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
