import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
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

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as numpy arrays
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
    
    # Build a boolean mask matrix for fast coverage counting
    # cover_mask[i] is a boolean array of length N, True where neighbors[i] is True
    cover_mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        cover_mask[i, neighbors[i]] = True
    
    start_time = time.time()
    time_limit = 115.0
    
    # Generate initial code using greedy
    uncovered = np.ones(N, dtype=bool)
    C = []
    
    # Greedy initialization - pick word covering most uncovered
    # Use a faster approach: maintain coverage counts
    remaining = np.ones(N, dtype=bool)
    while np.any(remaining):
        best_idx = -1
        best_count = -1
        for i in range(N):
            if not remaining[i]:
                continue
            cnt = np.sum(remaining[neighbors[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_count == 0:
            break
        C.append(best_idx)
        remaining[neighbors[best_idx]] = False
        if time.time() - start_time > time_limit:
            break
    
    # Convert C to set for fast lookup
    C_set = set(C)
    
    # Compute coverage counts
    counts = np.zeros(N, dtype=int)
    for c in C:
        counts[neighbors[c]] += 1
    
    # Reduction function
    def reduce_code(code, counts):
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
    
    # Initial reduction
    C, counts = reduce_code(C, counts)
    C_set = set(C)
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    
    # Local search
    # We'll use a simulated-annealing-like approach with kicks
    
    def try_swap_remove(c, C_list, C_set, counts):
        """Try to remove c and see if we can cover its unique points with other codewords."""
        u_indices = neighbors[c]
        unique_mask = counts[u_indices] == 1
        unique_points = u_indices[unique_mask]
        
        if len(unique_points) == 0:
            # Can remove directly
            C_list.remove(c)
            C_set.discard(c)
            counts[neighbors[c]] -= 1
            return True, C_list, C_set, counts
        
        if len(unique_points) > 3:
            return False, C_list, C_set, counts
        
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
            return False, C_list, C_set, counts
        
        candidates_set.difference_update(C_set)
        
        if not candidates_set:
            return False, C_list, C_set, counts
        
        # Try each candidate
        for c_prime in candidates_set:
            # Check if swapping enables reduction
            old_C = C_list.copy()
            old_counts = counts.copy()
            
            C_list.remove(c)
            C_set.discard(c)
            C_list.append(c_prime)
            C_set.add(c_prime)
            counts[neighbors[c]] -= 1
            counts[neighbors[c_prime]] += 1
            
            if np.any(counts < 1):
                C_list = old_C
                C_set = set(old_C)
                counts = old_counts
                continue
            
            new_C, new_counts = reduce_code(C_list, counts)
            
            if len(new_C) < len(old_C):
                C_set = set(new_C)
                return True, new_C, C_set, new_counts
            else:
                C_list = old_C
                C_set = set(old_C)
                counts = old_counts
        
        return False, C_list, C_set, counts
    
    def random_kick(C_list, C_set, counts):
        """Perform a random swap to escape local minima."""
        if not C_list:
            return C_list, C_set, counts, False
        
        c_idx = random.randint(0, len(C_list) - 1)
        c = C_list[c_idx]
        cand_arr = neighbors[c]
        c_prime = cand_arr[random.randint(0, len(cand_arr) - 1)]
        
        if c_prime in C_set:
            return C_list, C_set, counts, False
        
        old_C = C_list.copy()
        old_counts = counts.copy()
        
        C_list.remove(c)
        C_set.discard(c)
        C_list.append(c_prime)
        C_set.add(c_prime)
        counts[neighbors[c]] -= 1
        counts[neighbors[c_prime]] += 1
        
        if np.any(counts < 1):
            C_list = old_C
            C_set = set(old_C)
            counts = old_counts
            return C_list, C_set, counts, False
        
        new_C, new_counts = reduce_code(C_list, counts)
        C_set = set(new_C)
        return new_C, C_set, new_counts, True
    
    # Main local search loop
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        # Try to find a removable codeword
        found_improvement = False
        indices = list(range(len(C)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            c = C[idx]
            
            # Check if c can be removed
            u_indices = neighbors[c]
            unique_mask = counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            
            if len(unique_points) == 0:
                # Direct removal
                C.remove(c)
                C_set.discard(c)
                counts[neighbors[c]] -= 1
                found_improvement = True
                break
            
            if len(unique_points) > 3:
                continue
            
            # Try swap
            success, C, C_set, counts = try_swap_remove(c, C, C_set, counts)
            if success:
                found_improvement = True
                break
        
        if found_improvement:
            if len(C) < best_size:
                best_size = len(C)
                best_C = C.copy()
                best_counts = counts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Try random kicks to escape local minima
            if no_improve_count >= max_no_improve or random.random() < 0.3:
                C, C_set, counts, kicked = random_kick(C, C_set, counts)
                if kicked:
                    no_improve_count = 0
                    # Try to reduce after kick
                    C, counts = reduce_code(C, counts)
                    C_set = set(C)
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = counts.copy()
    
    # Final reduction
    final_counts = np.zeros(N, dtype=int)
    for c in best_C:
        final_counts[neighbors[c]] += 1
    best_C, _ = reduce_code(best_C, final_counts)
    
    return [list(all_words[i]) for i in best_C]
