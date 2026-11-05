import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback: Simple greedy with NumPy
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = {i}
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    nw = list(w)
                    nw[pos] = val
                    neigh_set.add(word_to_idx[tuple(nw)])
            neighbors.append(list(neigh_set))
        
        N = q ** n
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 100.0
        
        while np.any(uncovered) and time.time() - start_time < time_limit:
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
            
        # Reduce
        counts = np.zeros(N, dtype=int)
        for c in C:
            counts[neighbors[c]] += 1
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
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as indices and bitsets
    neighbors_idx = []
    neighbors_bitset = [0] * N
    
    for i, w in enumerate(all_words):
        idx_set = {i}
        bset = 1 << i
        for pos in range(8):
            orig_val = w[pos]
            for val in range(3):
                if val == orig_val:
                    continue
                nw = list(w)
                nw[pos] = val
                new_idx = word_to_idx[tuple(nw)]
                idx_set.add(new_idx)
                bset |= (1 << new_idx)
        neighbors_idx.append(list(idx_set))
        neighbors_bitset[i] = bset
    
    full_mask = (1 << N) - 1
    
    # --- Phase 1: Fast Greedy using Bitsets ---
    def greedy_fill(covered_mask, current_code, time_limit, start_time):
        code = list(current_code)
        covered = covered_mask
        # We need to update counts or just keep going until full coverage
        # For greedy, we just want to maximize new coverage.
        
        # If current_code is empty, covered is 0
        # We need to be careful: if we start with some code, we should ensure it's valid?
        # Usually we start empty.
        
        while covered != full_mask:
            if time.time() - start_time > time_limit:
                break
            
            uncovered = full_mask ^ covered
            best_idx = -1
            best_gain = -1
            
            # Optimize: Only check candidates that might add something?
            # With N=6561, checking all is ~6561 bitwise ops. ~10-20ms.
            # We can do this.
            for i in range(N):
                # If i is already in code, gain is 0 (or rather, we don't add duplicates)
                # Actually, adding a duplicate is useless.
                if i in code: 
                    continue
                
                gain = bin(uncovered & neighbors_bitset[i]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_gain == 0:
                # Should not happen if covered != full_mask and graph is connected
                break
                
            code.append(best_idx)
            covered |= neighbors_bitset[best_idx]
            
        return code, covered

    start_time = time.time()
    time_limit = 110.0
    
    # Initial Greedy
    C, covered_mask = greedy_fill(0, [], time_limit, start_time)
    
    # Convert to counts for reduction
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors_idx[c]] += 1
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
                if np.min(current_counts[neighbors_idx[c]]) >= 2:
                    current_counts[neighbors_idx[c]] -= 1
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
    
    no_improvement_count = 0
    max_no_improvement = 20 # Lower threshold to trigger perturbations earlier
    
    while time.time() - start_time < time_limit:
        # 1. Reduction Pass
        C, counts = reduce_code(C, counts)
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        if time.time() - start_time > time_limit:
            break
        
        # 2. Swap Pass
        improved = False
        swaps_made = 0
        max_swaps_per_iter = 10 # Increased swaps
        
        for c in C[:]:
            if time.time() - start_time > time_limit or swaps_made >= max_swaps_per_iter:
                break
                
            u_indices = neighbors_idx[c]
            unique_mask = counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            
            if len(unique_points) == 0:
                continue
            
            # If too many unique points, hard to replace
            if len(unique_points) > 4:
                continue
            
            # Find candidates
            if len(unique_points) == 1:
                candidates_set = set(neighbors_idx[unique_points[0]])
            else:
                candidates_set = set(neighbors_idx[unique_points[0]])
                for u in unique_points[1:]:
                    candidates_set &= set(neighbors_idx[u])
                    if not candidates_set:
                        break
            
            if not candidates_set:
                continue
            
            candidates_set.difference_update(C)
            
            if not candidates_set:
                continue
            
            # Select best candidate
            best_c_prime = None
            max_score = -1
            
            # Score: number of currently covered points this candidate covers (overlap)
            # Higher overlap means less "waste" or more stability
            for c_prime in candidates_set:
                # Quick score: count how many of c_prime's neighbors are already covered
                # This is expensive. Use a proxy: random pick?
                # Let's try the one with highest index? No.
                # Let's try multiple random candidates from the set
                pass
            
            # To keep it fast, pick a random candidate, or the one with highest "density"
            # Let's pick the one that covers the most *total* points (not just unique)
            # This is hard to compute fast.
            
            # Strategy: Pick 3 random candidates, apply swap, keep if valid (it always is if unique covered)
            # Wait, if we cover unique points, the swap is valid.
            # We just want to avoid bad structural changes.
            # Let's pick the candidate that appears in the most other codewords' neighborhoods?
            # Too slow.
            
            # Let's just pick the first one, or a random one.
            # Randomization helps.
            c_prime = random.choice(list(candidates_set))
            
            old_C = C.copy()
            old_counts = counts.copy()
            
            C.remove(c)
            C.append(c_prime)
            
            counts[neighbors_idx[c]] -= 1
            counts[neighbors_idx[c_prime]] += 1
            
            if np.any(counts < 1):
                # This shouldn't happen if we covered all unique points of c
                # But maybe c_prime was covering some points exclusively?
                # If c_prime was in C, we removed it? No, c_prime not in C.
                # So counts[neighbors_idx[c_prime]] increases.
                # counts[neighbors_idx[c]] decreases.
                # Points only covered by c are covered by c_prime (by construction).
                # So min count should be >= 1.
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
                    no_improvement_count = 0
                improved = True
                swaps_made += 1
            else:
                # Keep neutral swap to diversify
                C = new_C
                counts = new_counts
                swaps_made += 1
        
        # 3. Perturbation
        if not improved and no_improvement_count > max_no_improvement:
            # Remove 4 random codewords
            if len(C) >= 4:
                to_remove = random.sample(C, 4)
                for c in to_remove:
                    C.remove(c)
                    counts[neighbors_idx[c]] -= 1
                
                # Re-greedy fill using bitsets
                # We need to pass current covered mask
                # Recompute covered mask from counts?
                covered_mask = 0
                for c in C:
                    covered_mask |= neighbors_bitset[c]
                
                # Run greedy fill starting from current state
                C, covered_mask = greedy_fill(covered_mask, C, time_limit, start_time)
                
                # Recompute counts
                counts = get_counts(C)
                C, counts = reduce_code(C, counts)
                
                no_improvement_count = 0
    
    # Final Reduction
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
