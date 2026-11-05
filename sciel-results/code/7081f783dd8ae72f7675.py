import itertools
import numpy as np
import random
import time
from scipy.sparse import csr_matrix

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
    
    # Precompute neighbors
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
    
    # Build sparse matrix for fast greedy
    rows = []
    cols = []
    for i, neigh in enumerate(neighbors):
        rows.extend([i] * len(neigh))
        cols.extend(neigh.tolist())
    
    neighbor_matrix = csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(N, N))
    
    start_time = time.time()
    time_limit = 115.0
    
    # Helper functions
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

    def find_unique_points(c, counts):
        u_indices = neighbors[c]
        unique_mask = counts[u_indices] == 1
        return u_indices[unique_mask]

    # 1. Fast Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    
    while np.any(uncovered) and time.time() - start_time < time_limit:
        # Vectorized coverage count
        # counts[i] is number of uncovered neighbors of i
        counts = neighbor_matrix @ uncovered
        best_idx = np.argmax(counts)
        best_count = counts[best_idx]
        
        if best_count == 0:
            break
            
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        
    # Initial reduction
    C, counts = reduce_code(C, get_counts(C))
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    
    # 2. Local Search
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() - start_time < time_limit:
        # 1. Reduction Pass
        old_size = len(C)
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Try a random kick to escape local minimum
            if not C:
                break
            
            c_idx = random.randint(0, len(C)-1)
            c = C[c_idx]
            cand = neighbors[c]
            c_prime = cand[random.randint(0, len(cand)-1)]
            
            if c_prime in C:
                for _ in range(10):
                    c_prime = cand[random.randint(0, len(cand)-1)]
                    if c_prime not in C:
                        break
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
            else:
                new_C, new_counts = reduce_code(C, counts)
                C = new_C
                counts = new_counts
                no_improve_count = 0
            continue
        
        if time.time() - start_time > time_limit:
            break
        
        # 2. Swap Pass
        if len(C) == old_size:
            improved = False
            indices = list(range(len(C)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() - start_time > time_limit:
                    break
                c = C[idx]
                unique_points = find_unique_points(c, counts)
                
                if len(unique_points) == 0:
                    C.remove(c)
                    counts[neighbors[c]] -= 1
                    improved = True
                    break
                
                if len(unique_points) > 3:
                    continue
                
                # Find candidates that cover all unique points
                if len(unique_points) == 1:
                    candidates_set = set(neighbors[unique_points[0]].tolist())
                else:
                    candidates_set = set(neighbors[unique_points[0]].tolist())
                    for u in unique_points[1:]:
                        candidates_set &= set(neighbors[u].tolist())
                        if not candidates_set:
                            break
                
                if not candidates_set:
                    continue
                
                candidates_set.difference_update(C)
                
                if not candidates_set:
                    continue
                
                best_cand = None
                max_overlap = -1
                cand_list = list(candidates_set)
                if len(cand_list) > 10:
                    random.shuffle(cand_list)
                    cand_list = cand_list[:10]
                
                for cp in cand_list:
                    overlap = np.sum(counts[neighbors[cp]] > 0)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_cand = cp
                
                if best_cand is None:
                    best_cand = cand_list[0]
                
                c_prime = best_cand
                
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
                    if no_improve_count > 5:
                        C = new_C
                        counts = new_counts
                        improved = True
                        break
            if not improved:
                # Try 2-for-1 swap
                if time.time() - start_time < time_limit:
                    improved2 = False
                    c1_idx = random.randint(0, len(C)-1)
                    c2_idx = random.randint(0, len(C)-1)
                    if c1_idx == c2_idx:
                        continue
                    c1 = C[c1_idx]
                    c2 = C[c2_idx]
                    
                    u1 = find_unique_points(c1, counts)
                    u2 = find_unique_points(c2, counts)
                    
                    if len(u1) + len(u2) > 4:
                        continue
                    
                    all_unique = list(set(u1.tolist()) | set(u2.tolist()))
                    
                    if not all_unique:
                        continue
                    
                    if len(all_unique) == 1:
                        candidates_set = set(neighbors[all_unique[0]].tolist())
                    else:
                        candidates_set = set(neighbors[all_unique[0]].tolist())
                        for u in all_unique[1:]:
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
                        improved2 = True
                        break

    # Final Reduction
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
