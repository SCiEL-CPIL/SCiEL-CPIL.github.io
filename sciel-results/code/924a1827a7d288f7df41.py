import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback
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

    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    
    # Precompute neighbors
    neighbors = []
    for i in range(N):
        w = all_words[i]
        neigh_set = {i}
        for pos in range(8):
            val = w[pos]
            for new_val in range(3):
                if new_val == val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                idx = 0
                for k in range(8):
                    idx = idx * 3 + new_w[k]
                neigh_set.add(idx)
        neighbors.append(list(neigh_set))
        
    neighbors_np = [np.array(neigh, dtype=int) for neigh in neighbors]

    # 1. Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    start_time = time.time()
    time_limit = 115.0
    
    while np.any(uncovered):
        best_idx = -1
        best_count = -1
        for i in range(N):
            cnt = np.sum(uncovered[neighbors_np[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_count == 0:
            break
        C.append(best_idx)
        uncovered[neighbors_np[best_idx]] = False
        if time.time() - start_time > time_limit:
            break
            
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors_np[c]] += 1
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
                if np.all(current_counts[neighbors_np[c]] >= 2):
                    current_counts[neighbors_np[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts

    C, counts = reduce_code(C, get_counts(C))
    best_C = C.copy()
    best_size = len(C)
    
    # Local Search
    # We will iterate, trying to improve.
    # To avoid getting stuck, we will try multiple random candidates for swaps.
    # We will also allow "neutral" swaps (size unchanged) to explore the state space,
    # but we keep the best solution found.
    
    while time.time() - start_time < time_limit:
        improved = False
        # Shuffle codewords to try in random order
        indices = list(range(len(C)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            c = C[idx]
            u_indices = neighbors_np[c]
            unique_mask = counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            
            if len(unique_points) == 0:
                C.remove(c)
                counts[neighbors_np[c]] -= 1
                improved = True
                # Restart the index loop as C changed
                indices = list(range(len(C)))
                random.shuffle(indices)
                break
            
            if len(unique_points) > 3:
                continue
            
            # Find candidates
            if len(unique_points) == 1:
                candidates_set = set(neighbors_np[unique_points[0]].tolist())
            else:
                candidates_set = set(neighbors_np[unique_points[0]].tolist())
                for u in unique_points[1:]:
                    candidates_set &= set(neighbors_np[u].tolist())
                    if not candidates_set:
                        break
            
            if not candidates_set:
                continue
            
            candidates_set.difference_update(C)
            
            if not candidates_set:
                continue
            
            candidates_list = list(candidates_set)
            random.shuffle(candidates_list)
            
            # Try up to 10 random candidates
            for c_prime in candidates_list[:10]:
                old_C = C.copy()
                old_counts = counts.copy()
                
                C.remove(c)
                C.append(c_prime)
                
                counts[neighbors_np[c]] -= 1
                counts[neighbors_np[c_prime]] += 1
                
                if np.any(counts < 1):
                    C = old_C
                    counts = old_counts
                    continue
                
                new_C, new_counts = reduce_code(C, counts)
                
                if len(new_C) < len(old_C):
                    C = new_C
                    counts = new_counts
                    improved = True
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = counts.copy()
                    # Restart
                    indices = list(range(len(C)))
                    random.shuffle(indices)
                    break
                else:
                    # Neutral move: keep it to explore?
                    # If we keep it, we might find a path to improvement.
                    # But we must not cycle.
                    # Let's accept the first neutral move if we haven't found an improvement in this pass.
                    # Actually, let's just revert and try the next candidate.
                    C = old_C
                    counts = old_counts
        
        if not improved:
            # No improvement found in this pass.
            # Perform a random kick to escape local minimum.
            # Pick a random codeword and swap it with a random neighbor.
            # This might increase size temporarily, but reduce_code will fix it if possible.
            # If it doesn't fix it, we revert.
            if C:
                c_idx = random.randint(0, len(C)-1)
                c = C[c_idx]
                c_prime = neighbors_np[c][random.randint(0, len(neighbors_np[c])-1)]
                
                if c_prime not in C:
                    old_C = C.copy()
                    old_counts = counts.copy()
                    
                    C.remove(c)
                    C.append(c_prime)
                    counts[neighbors_np[c]] -= 1
                    counts[neighbors_np[c_prime]] += 1
                    
                    if np.any(counts < 1):
                        C = old_C
                        counts = old_counts
                    else:
                        new_C, new_counts = reduce_code(C, counts)
                        # Accept if size is not larger
                        if len(new_C) <= len(old_C):
                            C = new_C
                            counts = new_counts
                            if len(C) < best_size:
                                best_size = len(C)
                                best_C = C.copy()
                                best_counts = counts.copy()

    best_C, _ = reduce_code(best_C, get_counts(best_C))
    return [list(all_words[i]) for i in best_C]
