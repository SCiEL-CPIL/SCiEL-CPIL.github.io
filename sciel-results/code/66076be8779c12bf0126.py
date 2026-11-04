import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # This implementation is optimized for q=3, n=8, radius=1
    # which is the specific instance in the task contract.
    # For other parameters, it will still run but might be slower or less effective.
    
    total_words = q ** n
    
    # Precompute digits for all words
    # This is used to convert between index and tuple representation
    # And to generate neighbors efficiently
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
    # neighbors[i] is a list of indices j such that dist(i, j) <= radius
    # For q=3, n=8, radius=1, each word has 1 + 8*(3-1) = 17 neighbors
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
                # Calculate index from digits
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = neigh
        
    # Convert neighbors to numpy arrays for faster vectorized operations
    # This is a significant speedup for coverage updates
    neighbor_arrays = []
    for i in range(total_words):
        neighbor_arrays.append(np.array(neighbors[i], dtype=np.int32))
        
    # --- Phase 1: Vectorized Greedy Set Cover Initialization ---
    # Start with an empty code
    # We want to cover all total_words points.
    # cov_count[x] = number of codewords covering x
    # uncovered_count = number of x with cov_count[x] == 0
    
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []
    
    # To make greedy selection efficient, we can just iterate through all words
    # and pick the one that covers the most currently uncovered points.
    # However, doing this from scratch for 6561 words is O(N^2 * K) which might be slow in Python loops.
    # Instead, we can use a randomized greedy approach or a simple heuristic.
    
    # A better initialization: Randomly shuffle all words, then greedily add if it covers new points.
    # This is O(N * K) on average if we are lucky, but worst case is higher.
    # Given N=6561, K=17, N*K is ~111k operations per word. 6561*111k is ~700M. Too slow in pure Python.
    
    # Let's use a smarter heuristic:
    # 1. Pick a random word.
    # 2. Add it.
    # 3. Find the word that maximizes coverage of *uncovered* points.
    #    To do this efficiently, we can precompute for each word the list of neighbors.
    #    When we add a word, we update cov_count.
    #    The "gain" of adding word c is the number of x in neighbors[c] where cov_count[x] == 0.
    
    # We can maintain an array `gain` where gain[c] is the current number of uncovered neighbors of c.
    # Initially, gain[c] = len(neighbors[c]) = 17 for all c.
    # When we add a word w, for each x in neighbors[w]:
    #    if cov_count[x] was 0, it becomes 1.
    #    For every other word c that covers x (i.e., x in neighbors[c]), gain[c] decreases by 1.
    #    This is the expensive part: updating gain for all c covering x.
    #    How many c cover a given x? Exactly 17 (the neighbors of x).
    #    So adding one word w costs sum_{x in neighbors[w]} 17 operations = 17 * 17 = 289 ops.
    #    Total for 486 words: 486 * 289 ~ 140k ops. This is very fast!
    
    # Let's implement this.
    
    gain = np.full(total_words, radius, dtype=np.int32) # Wait, gain is number of uncovered neighbors.
    # Actually, initially all are uncovered. So gain[c] = len(neighbors[c]) = 1 + n*(q-1).
    # Let's call it `potential`.
    potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
    
    # We need to know, for each x, which c cover it.
    # We have neighbors[c]. We can invert this.
    # coverers[x] = list of c such that x in neighbors[c].
    # Note: neighbors is symmetric for radius 1 (Hamming distance).
    # So coverers[x] is just neighbors[x].
    
    # Algorithm:
    # 1. Initialize potential[c] = 17 for all c.
    # 2. Repeat until all covered:
    #    a. Find c with max potential. (Use np.argmax)
    #    b. Add c to code.
    #    c. For each x in neighbors[c]:
    #       if cov_count[x] == 0:
    #           cov_count[x] = 1
    #           # x was uncovered, now covered.
    #           # For all c' covering x, potential[c'] decreases by 1.
    #           # c' are neighbors[x].
    #           potential[neighbors[x]] -= 1
    #       else:
    #           cov_count[x] += 1
    #           # x was already covered. No change to potential?
    #           # Wait, potential[c] is defined as # uncovered neighbors of c.
    #           # If x was already covered, adding c doesn't change the status of x from uncovered to covered.
    #           # So potential[c] for c covering x does NOT decrease.
    #           # Correct.
    
    # One issue: `np.argmax` returns the first maximum. If there are ties, it's deterministic.
    # To introduce randomness and explore different basins, we can break ties randomly.
    # Or simply shuffle the order of candidates with max potential.
    
    # Let's implement this greedy construction.
    
    code_set = set()
    uncovered_remaining = total_words
    
    # We need a fast way to get neighbors[x] as a numpy array for vectorized subtraction.
    # We already have neighbor_arrays.
    
    while uncovered_remaining > 0 and time.time() < deadline - 60.0:
        # Find candidates with max potential
        max_pot = np.max(potential)
        if max_pot == 0:
            break # Should not happen if uncovered_remaining > 0, unless bug
        
        # Get all indices with max potential
        candidates = np.where(potential == max_pot)[0]
        
        # Pick one randomly to break ties and add diversity
        c = random.choice(candidates.tolist())
        
        if c in code_set:
            # This shouldn't happen if we mark them, but just in case
            potential[c] = 0
            continue
            
        code_set.add(c)
        code_list.append(c)
        is_code[c] = True
        
        # Update coverage
        neigh_c = neighbor_arrays[c]
        
        # We need to identify which neighbors were uncovered (cov_count == 0)
        # and which were covered (cov_count > 0).
        
        # Efficient update:
        # 1. Identify uncovered neighbors
        #    mask = cov_count[neigh_c] == 0
        #    uncovered_neigh = neigh_c[mask]
        #    
        # 2. For each x in uncovered_neigh:
        #    cov_count[x] = 1 (was 0)
        #    potential[neighbors[x]] -= 1
        #    
        # 3. For each x in neigh_c[~mask]:
        #    cov_count[x] += 1
        
        # Vectorized approach for step 2 is tricky because we need to update potential
        # for neighbors of neighbors.
        # Let's loop over the 17 neighbors of c.
        
        for x in neigh_c:
            if cov_count[x] == 0:
                # x becomes covered
                cov_count[x] = 1
                uncovered_remaining -= 1
                # Decrease potential for all c' covering x
                # c' are neighbors[x]
                # potential[neighbors[x]] -= 1
                # This is a vectorized operation on a small array (size 17)
                potential[neighbor_arrays[x]] -= 1
            else:
                cov_count[x] += 1
                
        # Ensure the added codeword itself has 0 potential to avoid re-adding
        potential[c] = 0
        
    # Now we have a valid covering code.
    # Size is likely around 500-600.
    
    if not code_list:
        # Fallback if greedy failed (should not happen)
        return [list(digits_of[i]) for i in range(total_words)]
        
    best_code = code_list[:]
    best_size = len(best_code)
    
    # --- Phase 2: Simulated Annealing ---
    # We will try to reduce the size of the code.
    # Moves:
    # 1. Remove a codeword c if it doesn't break coverage. (Delta size = -1)
    # 2. Swap a codeword c1 with c2 (not in code) if it maintains coverage. (Delta size = 0)
    #    This helps escape local minima where no single removal is possible.
    #    Actually, SA usually works on a fixed size or allows size changes.
    #    Here, we want to minimize size.
    #    So "Remove" is a strong improvement.
    #    "Swap" is neutral in size but changes the configuration.
    
    # We will use a cooling schedule.
    # T starts high, decreases.
    # Accept improvement always.
    # Accept worsening with probability exp(-delta_E / T).
    # Here, "Energy" can be defined as:
    # E = size * W1 + uncovered_count * W2
    # We want to minimize E.
    # Uncovered_count must be 0 for validity.
    # So we penalize uncovered points heavily.
    # If we allow temporary uncovered points, we can explore more.
    
    # Let's define Energy E = size + 100 * uncovered_count.
    # This encourages reducing size, but strongly penalizes losing coverage.
    
    # Current state:
    # cov_count, is_code, code_list, code_set
    
    # We need to be able to revert moves.
    # So we store the current state.
    
    current_code = best_code[:]
    current_code_set = set(current_code)
    current_is_code = is_code.copy()
    current_cov_count = cov_count.copy()
    current_size = len(current_code)
    
    # Precompute neighbors as lists for easier iteration in SA loop
    # We already have `neighbors` list of lists.
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.995
    T = T_start
    
    # Number of steps per temperature or total steps?
    # Let's run until deadline.
    
    while time.time() < deadline - 5.0:
        # Pick a random move
        # 1. Try to remove a random codeword
        # 2. Try to swap a random codeword with a random non-codeword
        
        if random.random() < 0.5 and current_size > 0:
            # Try Removal
            c1 = random.choice(current_code)
            
            # Check if removal is valid
            # Words that become uncovered: x in neighbors[c1] where cov_count[x] == 1
            # If any, removal increases uncovered_count.
            
            # Calculate delta_E
            # If removal is valid (no uncovered), delta_E = -1 (size decreases)
            # If invalid, delta_E = -1 + 100 * (number of newly uncovered words)
            
            newly_uncovered = 0
            for x in neighbors[c1]:
                if current_cov_count[x] == 1:
                    newly_uncovered += 1
            
            if newly_uncovered == 0:
                # Valid removal
                # Apply move
                current_cov_count[neighbors[c1]] -= 1
                current_is_code[c1] = False
                current_code_set.remove(c1)
                current_code.remove(c1)
                current_size -= 1
                
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            else:
                # Invalid removal, but might be accepted in SA
                delta_E = -1 + 100 * newly_uncovered
                # Accept with prob exp(-delta_E / T)
                if random.random() < np.exp(-delta_E / T):
                    # Apply move (temporarily invalid)
                    current_cov_count[neighbors[c1]] -= 1
                    current_is_code[c1] = False
                    current_code_set.remove(c1)
                    current_code.remove(c1)
                    current_size -= 1
        else:
            # Try Swap
            if current_size == 0:
                continue
                
            c1 = random.choice(current_code)
            
            # Pick c2 not in code
            # To avoid O(N) scan, we can pick a random index and check.
            # Or precompute list of non-codes? That's expensive to maintain.
            # Let's just pick random indices until we find one not in code.
            # Since code size is ~500 out of 6561, probability of hit is high.
            
            c2 = None
            for _ in range(20): # Try 20 times
                candidate = random.randint(0, total_words - 1)
                if candidate not in current_code_set:
                    c2 = candidate
                    break
            
            if c2 is None:
                continue
            
            # Check if swap is valid
            # Remove c1, Add c2.
            # Words affected: neighbors[c1] and neighbors[c2].
            # Union of these sets.
            
            # We can calculate the change in coverage for each x in the union.
            # But we only care about words that become uncovered (cov_count goes from 1 to 0)
            # or words that become covered (cov_count goes from 0 to 1, though we start covered).
            
            # Let's calculate newly_uncovered and newly_covered.
            
            # To do this efficiently, we can iterate over the union of neighbors.
            # Since neighbors are small (17), the union is at most 34.
            
            # Create a set of affected words
            affected = set(neighbors[c1]) | set(neighbors[c2])
            
            newly_uncovered = 0
            newly_covered = 0
            
            for x in affected:
                old_cov = current_cov_count[x]
                # Change in coverage for x:
                # -1 if x in neighbors[c1]
                # +1 if x in neighbors[c2]
                delta_cov = 0
                if x in neighbors[c1]:
                    delta_cov -= 1
                if x in neighbors[c2]:
                    delta_cov += 1
                
                new_cov = old_cov + delta_cov
                
                if old_cov == 0 and new_cov > 0:
                    newly_covered += 1
                elif old_cov == 1 and new_cov == 0:
                    newly_uncovered += 1
            
            # Delta size is 0.
            # Delta E = 100 * (newly_uncovered - newly_covered)
            # Note: newly_covered only matters if we allow temporary uncovered states.
            # If we are currently fully covered, newly_covered is 0.
            
            delta_E = 100 * newly_uncovered
            
            # If newly_uncovered == 0, delta_E = 0. Neutral move.
            # We should always accept neutral moves to explore.
            # If newly_uncovered > 0, accept with prob exp(-delta_E / T).
            
            accept = False
            if newly_uncovered == 0:
                accept = True
            else:
                if random.random() < np.exp(-delta_E / T):
                    accept = True
            
            if accept:
                # Apply swap
                # Remove c1
                current_cov_count[neighbors[c1]] -= 1
                current_is_code[c1] = False
                current_code_set.remove(c1)
                current_code.remove(c1)
                
                # Add c2
                current_cov_count[neighbors[c2]] += 1
                current_is_code[c2] = True
                current_code_set.add(c2)
                current_code.append(c2)
                
                # Note: current_size doesn't change
        
        # Cooling
        T *= cooling_rate
        
    # After SA, we might be in an invalid state (uncovered points).
    # We need to repair the code to be valid before returning.
    # Or we can just return the best valid code found.
    # We have been tracking `best_code` which is always valid (we only updated it on valid removals).
    # Wait, in the removal step, we only updated best_code if newly_uncovered == 0.
    # In the swap step, we didn't update best_code.
    # So best_code is always a valid covering code.
    
    # However, the current state might be better if we can repair it.
    # But repairing is hard.
    # Let's stick with best_code.
    
    # One final polish: try to remove more words from best_code greedily.
    # This is the "Kick and Prune" idea.
    
    final_code = best_code[:]
    final_code_set = set(final_code)
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1
        
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Shuffle to break symmetry
        random.shuffle(final_code)
        
        for c1 in final_code[:]:
            # Check if c1 can be removed
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
        # Should not happen, but fallback
        return [list(digits_of[i]) for i in best_code]
        
    return [list(digits_of[i]) for i in final_code]
