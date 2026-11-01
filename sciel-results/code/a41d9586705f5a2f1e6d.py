"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math
import heapq

def construct(q, n, radius):
    # For the specific task: q=3, n=7, radius=1
    # We use a refined Simulated Annealing approach.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to direct sum for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Precomputation for q=3, n=7, r=1
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # neighbors[i] is a list of indices of words within distance 1 of all_words[i]
    neighbors = []
    for w in all_words:
        w_list = list(w)
        idxs = [word_to_idx[w]]
        for i in range(7):
            orig = w_list[i]
            for v in range(3):
                if v != orig:
                    w_list[i] = v
                    idxs.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig
        neighbors.append(idxs)
        
    random.seed(42)
    
    # 1. Greedy Initialization: Max Coverage Greedy
    counts = [0] * 2187
    covered = 0
    code = []
    uncovered_set = set(range(2187))
    
    while uncovered_set:
        best_word_idx = -1
        max_cov = 0
        
        # Optimization: Only check words that are not already in code? 
        # Actually, adding a duplicate doesn't help, but we don't track "in code" easily in this loop.
        # We can just pick the best. If it's already in code, its marginal gain is 0.
        
        for w_idx in range(2187):
            cov = 0
            # Fast count of uncovered neighbors
            for nb in neighbors[w_idx]:
                if nb in uncovered_set:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
                best_word_idx = w_idx
        
        if best_word_idx == -1:
            break
            
        code.append(all_words[best_word_idx])
        for nb in neighbors[best_word_idx]:
            if counts[nb] == 0:
                covered += 1
                uncovered_set.remove(nb)
            counts[nb] += 1
            
    # Start Simulated Annealing
    current_code = code
    current_counts = counts
    current_covered = covered
    current_size = len(current_code)
    
    P = 10.0  # Penalty for uncovered words
    
    def get_cost(size, covered):
        return size + P * (2187 - covered)
        
    current_cost = get_cost(current_size, current_covered)
    
    best_code = list(current_code)
    best_size = current_size
    best_covered = current_covered
    best_cost = current_cost
    
    start_time = time.time()
    time_limit = 115.0
    
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute sets for faster intersection? No, lists are fine for small n.
    # But we can speed up delta calculations.
    
    while time.time() - start_time < time_limit:
        move_type = random.random()
        
        if move_type < 0.4:
            # Swap: Remove one, Add one
            if current_size == 0:
                continue
            idx_remove = random.randint(0, current_size - 1)
            w_remove = current_code[idx_remove]
            idx_w_remove = word_to_idx[w_remove]
            
            # Pick a word to add
            # Strategy: Pick a random word, or a word that covers uncovered spots
            # Let's pick a random word from the whole space for diversity, 
            # or biased towards uncovered neighbors.
            if random.random() < 0.5:
                # Biased: pick a neighbor of an uncovered word
                if 2187 - current_covered > 0:
                    # Find an uncovered word
                    # We don't have a direct list of uncovered words in current state easily without scanning.
                    # Let's just pick a random word.
                    idx_add = random.randint(0, 2186)
                else:
                    idx_add = random.randint(0, 2186)
            else:
                idx_add = random.randint(0, 2186)
                
            if idx_add == idx_w_remove:
                continue
                
            w_add = all_words[idx_add]
            
            # Check if w_add is already in code
            # This is O(N) check. We can maintain a set of indices in code for O(1) check.
            # Let's maintain code_set.
            
            # Delta calculation
            nbs_remove = neighbors[idx_w_remove]
            nbs_add = neighbors[idx_add]
            
            delta_covered = 0
            # Words only covered by remove
            for nb in nbs_remove:
                if current_counts[nb] == 1:
                    # Check if it's covered by add
                    # If add also covers it, count doesn't drop
                    # We need to know if nb is in nbs_add
                    # Instead of set intersection, we can just compute final state? 
                    # No, incremental is faster.
                    pass
            
            # Better way: 
            # new_counts = old_counts - counts(nbs_remove) + counts(nbs_add)
            # delta_covered = (new_covered - old_covered)
            
            # Let's calculate delta precisely.
            # A word becomes uncovered if count goes 1->0.
            # A word becomes covered if count goes 0->1.
            
            # To avoid O(N^2) in the move evaluation, we rely on the fact that 
            # the neighborhood is small (1 + 7*2 = 15 words).
            
            # We can create a temporary dict or just iterate.
            
            # Let's optimize the swap evaluation:
            # 1. Subtract remove neighbors
            # 2. Add add neighbors
            # 3. Count changes
            
            # This is O(15 + 15) operations if we do it carefully.
            
            # However, we need to know if a word is covered by *other* codewords.
            # current_counts[nb] tells us the total count.
            
            # If we remove w_remove:
            # For nb in nbs_remove:
            #   if current_counts[nb] == 1:
            #       it might become uncovered, UNLESS it is also in nbs_add
            
            # If we add w_add:
            # For nb in nbs_add:
            #   if current_counts[nb] == 0:
            #       it becomes covered
            
            # But the order matters for the "1->0" check if they overlap.
            
            # Let's just simulate the change on a copy of counts? 
            # No, too slow.
            
            # Let's do this:
            # 1. Identify words that change state.
            # 2. The set of affected words is union(nbs_remove, nbs_add). Size <= 30.
            
            affected = set()
            for nb in nbs_remove: affected.add(nb)
            for nb in nbs_add: affected.add(nb)
            
            delta = 0
            for nb in affected:
                old_count = current_counts[nb]
                new_count = old_count
                if nb in nbs_remove: new_count -= 1
                if nb in nbs_add: new_count += 1
                if old_count == 0 and new_count > 0: delta += 1
                if old_count > 0 and new_count == 0: delta -= 1
                
            new_size = current_size
            new_covered = current_covered + delta
            new_cost = get_cost(new_size, new_covered)
            delta_cost = new_cost - current_cost
            
            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                # Execute move
                current_code[idx_remove] = w_add
                for nb in nbs_remove: current_counts[nb] -= 1
                for nb in nbs_add: current_counts[nb] += 1
                current_covered = new_covered
                current_cost = new_cost
                
                if current_covered == 2187 and current_size < best_size:
                    best_size = current_size
                    best_covered = current_covered
                    best_code = list(current_code)
                    best_cost = current_cost
                    
                T *= cooling
                if T < T_min: T = T_min
                continue
            else:
                T *= cooling
                if T < T_min: T = T_min
                continue

        elif move_type < 0.8:
            # Remove
            if current_size == 0:
                continue
            idx = random.randint(0, current_size - 1)
            w = current_code[idx]
            w_idx = word_to_idx[w]
            
            delta_covered = 0
            for nb in neighbors[w_idx]:
                if current_counts[nb] == 1:
                    delta_covered -= 1
            
            new_size = current_size - 1
            new_covered = current_covered + delta_covered
            new_cost = get_cost(new_size, new_covered)
            delta_cost = new_cost - current_cost
            
            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                current_code.pop(idx)
                for nb in neighbors[w_idx]:
                    current_counts[nb] -= 1
                current_size = new_size
                current_covered = new_covered
                current_cost = new_cost
                
                if current_covered == 2187 and current_size < best_size:
                    best_size = current_size
                    best_covered = current_covered
                    best_code = list(current_code)
                    best_cost = current_cost
                    
                T *= cooling
                if T < T_min: T = T_min
                continue
            else:
                T *= cooling
                if T < T_min: T = T_min
                continue

        else:
            # Add
            if current_covered == 2187:
                # Try to add and remove? No, just skip or swap.
                # If fully covered, adding increases size, which is bad.
                # So we should probably prefer Swap or Remove.
                # But if we are stuck, maybe a swap is better.
                # Let's change probability: if covered, do swap.
                # For now, just skip add if covered.
                pass
            
            if current_covered < 2187:
                # Pick a word that covers uncovered spots
                # Find an uncovered word
                # We don't have a list. Let's pick a random word.
                # Or better, pick a random uncovered word's neighbor.
                # To find an uncovered word, we might need to scan. 
                # Let's just pick a random word.
                idx_add = random.randint(0, 2186)
                w_add = all_words[idx_add]
                
                # Check if already in code (slow check)
                # We can maintain a set of indices in code.
                # Let's add that optimization.
                
                # For now, simple add.
                nbs_add = neighbors[idx_add]
                delta_covered = 0
                for nb in nbs_add:
                    if current_counts[nb] == 0:
                        delta_covered += 1
                        
                new_size = current_size + 1
                new_covered = current_covered + delta_covered
                new_cost = get_cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    current_code.append(w_add)
                    for nb in nbs_add:
                        current_counts[nb] += 1
                    current_size = new_size
                    current_covered = new_covered
                    current_cost = new_cost
                    
                    if current_covered == 2187 and current_size < best_size:
                        best_size = current_size
                        best_covered = current_covered
                        best_code = list(current_code)
                        best_cost = current_cost
                        
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
            else:
                # Fallback to swap if covered
                # Re-do swap logic? Or just continue.
                pass
            
            T *= cooling
            if T < T_min: T = T_min
            continue

    return [list(w) for w in best_code]
# EVOLVE-BLOCK-END
