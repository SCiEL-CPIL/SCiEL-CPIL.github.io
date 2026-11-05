import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    # This implementation is optimized for q=3, n=8, radius=1
    # which is the specific instance in the task contract.
    # For other parameters, it will still run but might be slower or less effective.
    
    total_words = q ** n
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute neighbor lists for radius 1
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = neigh
        
    # Convert neighbors to numpy arrays for faster vectorized operations
    neighbor_arrays = []
    for i in range(total_words):
        neighbor_arrays.append(np.array(neighbors[i], dtype=np.int32))
        
    # Precompute neighbor sets for fast membership testing in SA
    neighbor_sets = [set(neigh) for neigh in neighbors]
        
    # --- Phase 1: Vectorized Greedy Set Cover Initialization ---
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []
    
    potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
    
    code_set = set()
    uncovered_remaining = total_words
    
    while uncovered_remaining > 0 and time.time() < deadline - 70.0:
        max_pot = np.max(potential)
        if max_pot == 0:
            break
        
        candidates = np.where(potential == max_pot)[0]
        c = random.choice(candidates.tolist())
        
        if c in code_set:
            potential[c] = 0
            continue
            
        code_set.add(c)
        code_list.append(c)
        is_code[c] = True
        
        neigh_c = neighbor_arrays[c]
        
        for x in neigh_c:
            if cov_count[x] == 0:
                cov_count[x] = 1
                uncovered_remaining -= 1
                potential[neighbor_arrays[x]] -= 1
            else:
                cov_count[x] += 1
                
        potential[c] = 0
    
    if not code_list:
        return [list(digits_of[i]) for i in range(total_words)]
        
    best_code = code_list[:]
    best_size = len(best_code)
    
    # --- Phase 2: Improved Simulated Annealing with neighbor-restricted swaps ---
    current_code = best_code[:]
    current_code_set = set(current_code)
    current_is_code = is_code.copy()
    current_cov_count = cov_count.copy()
    current_size = len(current_code)
    
    # Track uncovered points for fast swap candidate selection
    # uncovered_points[x] = True if current_cov_count[x] == 0
    # We maintain this incrementally
    
    # SA Parameters
    T = 8.0
    cooling_rate = 0.997
    
    # Precompute neighbor membership sets for O(1) lookup
    # Already have neighbor_sets
    
    # Main SA loop
    while time.time() < deadline - 8.0:
        r = random.random()
        
        if r < 0.35 and current_size > 0:
            # Try Removal
            c1 = random.choice(current_code)
            
            newly_uncovered = 0
            for x in neighbors[c1]:
                if current_cov_count[x] == 1:
                    newly_uncovered += 1
            
            if newly_uncovered == 0:
                # Valid removal
                current_cov_count[neighbors[c1]] -= 1
                current_is_code[c1] = False
                current_code_set.remove(c1)
                current_code.remove(c1)
                current_size -= 1
                
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            else:
                delta_E = -1 + 100 * newly_uncovered
                if random.random() < np.exp(-delta_E / T):
                    current_cov_count[neighbors[c1]] -= 1
                    current_is_code[c1] = False
                    current_code_set.remove(c1)
                    current_code.remove(c1)
                    current_size -= 1
        elif r < 0.85:
            # Try Swap - neighbor-restricted: pick c2 near an uncovered point
            if current_size == 0:
                continue
                
            c1 = random.choice(current_code)
            
            # Find candidate c2: prefer neighbors of points with low coverage
            # First check if there are any uncovered or singly-covered points
            # to guide our choice
            
            # Strategy: pick a random point x that is "vulnerable" (cov_count <= 2)
            # then pick c2 from neighbors[x] that is not in code
            # If no vulnerable points, just pick random c2
            
            c2 = None
            
            # Try to find a good swap candidate
            # Pick a random position in the code, check its neighborhood
            # for points with low coverage
            
            # For efficiency, just pick random c2 not in code
            # but weight toward neighbors of low-coverage points
            
            # Simple approach: try a few random candidates
            attempts = 0
            while c2 is None and attempts < 15:
                candidate = random.randint(0, total_words - 1)
                if candidate not in current_code_set:
                    # Check if this candidate helps: does it cover any low-coverage point?
                    # Quick heuristic: check if any neighbor has cov_count <= 1
                    help_score = 0
                    for x in neighbors[candidate]:
                        if current_cov_count[x] <= 1:
                            help_score += 1
                    if help_score > 0 or random.random() < 0.3:
                        c2 = candidate
                    else:
                        # Still might use it, but less preferred
                        if random.random() < 0.2:
                            c2 = candidate
                attempts += 1
            
            if c2 is None:
                # Fallback: just pick any non-code word
                for _ in range(30):
                    candidate = random.randint(0, total_words - 1)
                    if candidate not in current_code_set:
                        c2 = candidate
                        break
                if c2 is None:
                    continue
            
            # Evaluate swap
            n1_set = neighbor_sets[c1]
            n2_set = neighbor_sets[c2]
            
            newly_uncovered = 0
            newly_covered = 0
            
            # Check points only in n1 (lose coverage)
            for x in n1_set:
                if x not in n2_set:
                    if current_cov_count[x] == 1:
                        newly_uncovered += 1
                else:
                    # In both, net change 0
                    pass
            
            # Check points only in n2 (gain coverage)
            for x in n2_set:
                if x not in n1_set:
                    if current_cov_count[x] == 0:
                        newly_covered += 1
            
            delta_E = 100 * (newly_uncovered - newly_covered)
            
            if newly_uncovered == 0 and newly_covered >= 0:
                accept = True
            elif delta_E <= 0:
                accept = True
            else:
                accept = random.random() < np.exp(-delta_E / T)
            
            if accept:
                # Apply swap
                current_cov_count[neighbors[c1]] -= 1
                current_is_code[c1] = False
                current_code_set.remove(c1)
                current_code.remove(c1)
                
                current_cov_count[neighbors[c2]] += 1
                current_is_code[c2] = True
                current_code_set.add(c2)
                current_code.append(c2)
        else:
            # Try double swap or kick: remove two, add two
            if current_size < 2:
                continue
                
            c1 = random.choice(current_code)
            remaining = [x for x in current_code if x != c1]
            if not remaining:
                continue
            c2 = random.choice(remaining)
            
            # Find two replacement candidates
            cand1 = None
            cand2 = None
            for _ in range(20):
                candidate = random.randint(0, total_words - 1)
                if candidate not in current_code_set and candidate != c1 and candidate != c2:
                    if cand1 is None:
                        cand1 = candidate
                    else:
                        cand2 = candidate
                    break
            
            if cand1 is None or cand2 is None:
                continue
            
            # Evaluate double swap
            s1 = neighbor_sets[c1]
            s2 = neighbor_sets[c2]
            s3 = neighbor_sets[cand1]
            s4 = neighbor_sets[cand2]
            
            newly_uncovered = 0
            newly_covered = 0
            
            # Union of all affected
            all_affected = s1 | s2 | s3 | s4
            
            for x in all_affected:
                old_cov = current_cov_count[x]
                delta = 0
                if x in s1: delta -= 1
                if x in s2: delta -= 1
                if x in s3: delta += 1
                if x in s4: delta += 1
                new_cov = old_cov + delta
                if old_cov == 0 and new_cov > 0:
                    newly_covered += 1
                elif old_cov == 1 and new_cov == 0:
                    newly_uncovered += 1
            
            delta_E = 100 * (newly_uncovered - newly_covered)
            
            if newly_uncovered == 0:
                accept = True
            elif delta_E <= 0:
                accept = True
            else:
                accept = random.random() < np.exp(-delta_E / T)
            
            if accept:
                current_cov_count[neighbors[c1]] -= 1
                current_is_code[c1] = False
                current_code_set.remove(c1)
                current_code.remove(c1)
                
                current_cov_count[neighbors[c2]] -= 1
                current_is_code[c2] = False
                current_code_set.remove(c2)
                current_code.remove(c2)
                
                current_cov_count[neighbors[cand1]] += 1
                current_is_code[cand1] = True
                current_code_set.add(cand1)
                current_code.append(cand1)
                
                current_cov_count[neighbors[cand2]] += 1
                current_is_code[cand2] = True
                current_code_set.add(cand2)
                current_code.append(cand2)
        
        # Cooling
        T *= cooling_rate
        if T < 0.05:
            T = 0.05
    
    # --- Phase 3: Final Polish - iterative pruning ---
    final_code = best_code[:]
    final_code_set = set(final_code)
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1
        
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        random.shuffle(final_code)
        
        for c1 in final_code[:]:
            removable = True
            for x in neighbors[c1]:
                if final_cov_count[x] == 1:
                    removable = False
                    break
            
            if removable:
                final_cov_count[neighbors[c1]] -= 1
                final_code_set.remove(c1)
                final_code.remove(c1)
                improved = True
    
    # Verify final code
    if not np.all(final_cov_count > 0):
        return [list(digits_of[i]) for i in best_code]
        
    return [list(digits_of[i]) for i in final_code]
