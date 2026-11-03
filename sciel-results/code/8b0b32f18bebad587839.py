import random
import time
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
    # Pre-compute neighbors for radius 1
    # neighbors[idx] = np.array of indices within distance <= 1
    neighbors = []
    weights = [q ** (n - 1 - i) for i in range(n)]
    
    for idx in range(total_words):
        w = [0] * n
        temp = idx
        for d in range(n-1, -1, -1):
            w[d] = temp % q
            temp //= q
            
        neigh = [idx]
        for i in range(n):
            for delta in range(1, q):
                new_val = (w[i] + delta) % q
                new_idx = idx + (new_val - w[i]) * weights[i]
                neigh.append(new_idx)
        neighbors.append(np.array(neigh, dtype=np.int32))

    # Pre-compute neighbor sets for fast subset checks?
    # We need to check if Crit(c) is subset of N(w).
    # Crit(c) is small. N(w) is small.
    # We can just iterate over Crit(c) and check membership in N(w).
    # Membership in N(w) can be checked via a boolean mask or set.
    # Let's use a boolean mask for N(w) for O(1) lookup?
    # 2187 x 2187 bool matrix is fine.
    neighbor_mask = np.zeros((total_words, total_words), dtype=bool)
    for idx in range(total_words):
        neighbor_mask[idx, neighbors[idx]] = True

    def idx_to_word(idx):
        w = [0] * n
        for d in range(n-1, -1, -1):
            w[d] = idx % q
            idx //= q
        return w

    def greedy_construct(random_seed):
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.int8)
        code = []
        remaining_count = total_words
        
        # M matrix for gain calculation
        # M[idx, j] = 1 if j in neighbors[idx]
        M = np.zeros((total_words, total_words), dtype=np.int8)
        for idx in range(total_words):
            M[idx, neighbors[idx]] = 1

        while remaining_count > 0:
            gains = (M @ uncovered).astype(np.int32)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[np.random.randint(len(candidates))]
            
            code.append(chosen_idx)
            nb_idx = neighbors[chosen_idx]
            num_covered = np.sum(uncovered[nb_idx])
            uncovered[nb_idx] = 0
            remaining_count -= num_covered
            
        return code

    def local_search(code):
        # code is a list of int indices
        code_set = set(code)
        coverage = np.zeros(total_words, dtype=np.int32)
        for idx in code:
            coverage[neighbors[idx]] += 1
        
        improved = True
        max_iters = 2000
        iter_count = 0
        no_swap_count = 0
        max_no_swap = 50
        
        while improved and iter_count < max_iters:
            iter_count += 1
            improved = False
            
            # 1. Greedy Removal Phase
            # Find all removable words
            removable = []
            for i, idx in enumerate(code):
                if np.min(coverage[neighbors[idx]]) > 1:
                    removable.append(i)
            
            if removable:
                # Remove a random removable word
                rem_idx = random.choice(removable)
                idx_to_remove = code[rem_idx]
                coverage[neighbors[idx_to_remove]] -= 1
                code_set.remove(idx_to_remove)
                del code[rem_idx]
                improved = True
                no_swap_count = 0 # Reset counter
                continue
            
            # 2. Random Valid Swap Phase
            # Try to find a valid swap c -> w
            # Condition: Crit(c) subset of N(w)
            # Crit(c) = {x in N(c) | coverage[x] == 1}
            
            found_swap = False
            # Try a few random swaps
            for _ in range(10):
                if not code:
                    break
                c = random.choice(code)
                crit_c = neighbors[c][coverage[neighbors[c]] == 1]
                
                if len(crit_c) == 0:
                    # Should have been removed
                    continue
                
                # Pick a random w not in code
                # To avoid O(N) check for "not in code", we can pick random indices and check set
                # Or maintain a list of non-code words?
                # Set membership is O(1) average.
                
                # Heuristic: Pick w from neighbors of some crit word?
                # w must cover all crit_c.
                # So w must be in intersection of N(x) for x in crit_c.
                # Let's just pick random w and check.
                
                w = random.randint(0, total_words - 1)
                if w in code_set:
                    continue
                
                # Check if N(w) covers all crit_c
                # i.e., for all x in crit_c, neighbor_mask[w, x] is True
                # Vectorized check
                if np.all(neighbor_mask[w, crit_c]):
                    # Valid swap
                    # Update coverage
                    coverage[neighbors[c]] -= 1
                    coverage[neighbors[w]] += 1
                    code_set.remove(c)
                    code_set.add(w)
                    code[code.index(c)] = w
                    found_swap = True
                    improved = True
                    no_swap_count = 0
                    break
            
            if found_swap:
                continue
            
            # 3. Perturbation if stuck
            no_swap_count += 1
            if no_swap_count > max_no_swap:
                # Add a random word to break the local minimum
                # This increases size, but might allow larger reductions later
                w_add = random.randint(0, total_words - 1)
                if w_add not in code_set:
                    code.append(w_add)
                    code_set.add(w_add)
                    coverage[neighbors[w_add]] += 1
                    no_swap_count = 0
                    improved = True
                    continue
                
            if not improved:
                break
                
        return code

    # Main Iterated Local Search
    start_time = time.time()
    best_code = None
    best_size = float('inf')
    seed = 42
    
    # Run multiple iterations
    while time.time() - start_time < 115:
        # Generate random greedy cover
        code = greedy_construct(seed)
        seed += 1
        
        # Local search
        code = local_search(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
    if best_code is None:
        code = greedy_construct(0)
        best_code = code
        best_size = len(code)

    # Convert to list of lists
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
