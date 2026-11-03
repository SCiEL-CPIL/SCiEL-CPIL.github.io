import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (ball of radius 1)
    # neighbors[i] is a list of indices j such that d(i, j) <= 1
    neighbors = [[] for _ in range(total_words)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i) # Self
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_word = w[:pos] + (new_val,) + w[pos+1:]
                j = word_to_idx[new_word]
                neighbors[i].append(j)
                
    # Helper to get index
    def get_idx(word):
        return word_to_idx[word]

    # 1. Greedy Initialization with Coverage Counts
    # We will run a few greedy passes to get a good start
    
    best_initial_code = None
    best_initial_size = float('inf')
    
    for run in range(5): # 5 greedy runs
        uncovered = [1] * total_words # 1 means uncovered, 0 means covered
        uncovered_count = total_words
        code = []
        code_set = set()
        
        # Symmetry breaking: Fix first codeword to all zeros
        first_word = (0,) * n
        first_idx = get_idx(first_word)
        code.append(first_word)
        code_set.add(first_word)
        
        # Update coverage
        for v in neighbors[first_idx]:
            if uncovered[v]:
                uncovered[v] = 0
                uncovered_count -= 1
                
        while uncovered_count > 0 and time.time() < deadline - 30.0:
            best_idx = -1
            best_gain = 0
            
            # Find word that covers most uncovered
            # Optimization: Only check words not in code
            for i in range(total_words):
                if all_words[i] in code_set:
                    continue
                
                gain = 0
                for v in neighbors[i]:
                    if uncovered[v]:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    
            if best_idx == -1:
                break
                
            best_word = all_words[best_idx]
            code.append(best_word)
            code_set.add(best_word)
            
            for v in neighbors[best_idx]:
                if uncovered[v]:
                    uncovered[v] = 0
                    uncovered_count -= 1

        if len(code) < best_initial_size:
            best_initial_size = len(code)
            best_initial_code = list(code)
            # Store the coverage state? No, we will re-init counts for SA.

    if best_initial_code is None:
        # Fallback if something went wrong (shouldn't happen)
        best_initial_code = [(0,)*n]
        
    code = best_initial_code
    code_set = set(code)
    
    # Initialize coverage counts for SA
    counts = [0] * total_words
    for w in code:
        idx = get_idx(w)
        for v in neighbors[idx]:
            counts[v] += 1
            
    uncovered_count = sum(1 for c in counts if c == 0)
    
    # 2. Local Search / Simulated Annealing with Coverage Counts
    PENALTY = 1000.0 # High penalty to ensure validity
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    best_code = list(code)
    best_size = len(code)
    best_uncovered = uncovered_count
    
    all_indices = list(range(total_words))
    
    while time.time() < deadline:
        if not code:
            break
            
        if random.random() < 0.5:
            # Remove move
            idx_to_remove = random.randint(0, len(code) - 1)
            word_to_remove = code[idx_to_remove]
            idx_r = get_idx(word_to_remove)
            
            # Calculate change
            newly_uncovered = 0
            affected = []
            for v in neighbors[idx_r]:
                if counts[v] == 1:
                    # Will become 0
                    newly_uncovered += 1
                affected.append(v)
                
            new_size = len(code) - 1
            new_uncovered = uncovered_count + newly_uncovered
            new_cost = new_size + PENALTY * new_uncovered
            old_cost = len(code) + PENALTY * uncovered_count
            
            delta_cost = new_cost - old_cost
            
            if delta_cost <= 0 or random.random() < np.exp(-delta_cost / T):
                # Accept
                for v in affected:
                    counts[v] -= 1
                code.pop(idx_to_remove)
                code_set.discard(word_to_remove)
                uncovered_count = new_uncovered
                
                if uncovered_count == 0 and len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                    best_uncovered = 0
                elif uncovered_count == 0 and len(code) == best_size:
                    # Optional: keep the "best" structure? Not necessary for size.
                    pass
        else:
            # Swap move
            idx_to_swap = random.randint(0, len(code) - 1)
            old_word = code[idx_to_swap]
            idx_old = get_idx(old_word)
            
            new_word_idx = random.choice(all_indices)
            new_word = all_words[new_word_idx]
            
            if new_word in code_set:
                continue
                
            idx_new = new_word_idx
            
            # Calculate change
            # We remove old, add new.
            # Words affected are in neighbors[idx_old] U neighbors[idx_new]
            # We need to know how many words become uncovered (count 0 -> 0? No, count 1 -> 0)
            # and how many become covered (count 0 -> 1).
            
            # Efficient way:
            # 1. Decrement counts for neighbors of old.
            # 2. Increment counts for neighbors of new.
            # 3. Count how many changed from 0 to 1 (covered) and 1 to 0 (uncovered).
            
            # To avoid modifying and reverting, we can calculate delta directly.
            # But sets might overlap.
            
            # Let's just do the modify-check-revert pattern as it's safe and fast enough with lists.
            
            # Store old counts for affected nodes
            affected_nodes = set(neighbors[idx_old]) | set(neighbors[idx_new])
            old_counts = {v: counts[v] for v in affected_nodes}
            
            # Apply changes
            for v in neighbors[idx_old]:
                counts[v] -= 1
            for v in neighbors[idx_new]:
                counts[v] += 1
                
            # Calculate new uncovered
            new_uncovered = 0
            for v in affected_nodes:
                if counts[v] == 0:
                    new_uncovered += 1
            
            # Old uncovered in this set?
            # We know global uncovered_count.
            # Words that were uncovered (count 0) and are still uncovered (count 0) remain.
            # Words that were uncovered (count 0) and now covered (count > 0) decrease uncovered.
            # Words that were covered (count > 0) and now uncovered (count 0) increase uncovered.
            
            # Let's just recompute uncovered_count for affected nodes?
            # No, we can track delta.
            
            delta_uncovered = 0
            for v in affected_nodes:
                if old_counts[v] == 0 and counts[v] > 0:
                    delta_uncovered -= 1 # Covered
                elif old_counts[v] > 0 and counts[v] == 0:
                    delta_uncovered += 1 # Uncovered
                
            new_uncovered_total = uncovered_count + delta_uncovered
            new_size = len(code)
            
            new_cost = new_size + PENALTY * new_uncovered_total
            old_cost = len(code) + PENALTY * uncovered_count
            
            delta_cost = new_cost - old_cost
            
            if delta_cost <= 0 or random.random() < np.exp(-delta_cost / T):
                # Accept
                code[idx_to_swap] = new_word
                code_set.discard(old_word)
                code_set.add(new_word)
                uncovered_count = new_uncovered_total
                
                if uncovered_count == 0 and len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                    best_uncovered = 0
            else:
                # Revert
                for v in affected_nodes:
                    counts[v] = old_counts[v]
                    
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Final local search on best_code
    # Re-init counts for best_code
    final_code = list(best_code)
    final_set = set(final_code)
    
    counts = [0] * total_words
    for w in final_code:
        idx = get_idx(w)
        for v in neighbors[idx]:
            counts[v] += 1
            
    uncovered_count = sum(1 for c in counts if c == 0)
    
    # Greedy removal
    improved = True
    while improved:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            w = final_code[i]
            idx = get_idx(w)
            
            # Check if removing w leaves any word uncovered
            # i.e., if any v in neighbors[idx] has counts[v] == 1
            can_remove = True
            for v in neighbors[idx]:
                if counts[v] == 1:
                    can_remove = False
                    break
                    
            if can_remove:
                # Remove
                for v in neighbors[idx]:
                    counts[v] -= 1
                final_code.pop(i)
                final_set.discard(w)
                uncovered_count -= 0 # No change in uncovered count if valid removal
                improved = True
                break # Restart loop as indices changed
                
    return [list(w) for w in final_code]
