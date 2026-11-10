import numpy as np
import random
import time
import math

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        total_words = 3 ** n
        # Precompute neighbors
        neighbors = [[] for _ in range(total_words)]
        for i in range(total_words):
            digits = [0] * n
            val = i
            for j in range(n):
                digits[j] = val % 3
                val //= 3
            
            neigh = [i]
            for pos in range(n):
                orig = digits[pos]
                for new_val in range(3):
                    if new_val == orig:
                        continue
                    digits[pos] = new_val
                    ni = 0
                    for k in range(n-1, -1, -1):
                        ni = ni * 3 + digits[k]
                    neigh.append(ni)
                digits[pos] = orig
            neighbors[i] = neigh

        # 1. Initial Greedy Construction
        uncovered = np.ones(total_words, dtype=bool)
        counts = np.full(total_words, 17, dtype=np.int32)
        code = []
        
        while uncovered.any():
            best_idx = np.argmax(counts)
            code.append(best_idx)
            for v in neighbors[best_idx]:
                if uncovered[v]:
                    uncovered[v] = False
                    for u in neighbors[v]:
                        counts[u] -= 1
                        
        # 2. Simulated Annealing
        code_set = set(code)
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            for v in neighbors[c]:
                coverage_count[v] += 1
                
        uncovered_count = 0
        for v in range(total_words):
            if coverage_count[v] == 0:
                uncovered_count += 1
                
        M = 100.0 # Penalty for uncovered words
        start_time = time.time()
        max_time = 115.0
        
        T = 100.0
        cooling_rate = 0.999 # Tune this
        min_T = 0.1
        
        # Helper to calculate cost
        def get_cost(size, unc):
            return size + M * unc
            
        current_size = len(code)
        current_unc = uncovered_count
        current_cost = get_cost(current_size, current_unc)
        
        while T > min_T and (time.time() - start_time) < max_time:
            # Number of iterations per temperature step
            # Estimate: 10000 moves per step
            # Adjust based on remaining time?
            # Let's just run a fixed number of moves per loop iteration
            # and update T at the end of the block?
            # Or update T every move? Too slow.
            # Let's do 1000 moves, then cool.
            
            moves_per_step = 1000
            for _ in range(moves_per_step):
                # Choose move
                r = random.random()
                if r < 0.5:
                    # Remove
                    if current_size == 0:
                        continue
                    idx = random.randint(0, current_size - 1)
                    c = code[idx]
                    
                    # Apply
                    for v in neighbors[c]:
                        coverage_count[v] -= 1
                        if coverage_count[v] == 0:
                            uncovered_count += 1
                            
                    new_size = current_size - 1
                    new_unc = uncovered_count
                    new_cost = get_cost(new_size, new_unc)
                    delta = new_cost - current_cost
                    
                    if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                        # Accept
                        code.pop(idx)
                        code_set.discard(c)
                        current_size = new_size
                        current_unc = new_unc
                        current_cost = new_cost
                    else:
                        # Revert
                        for v in neighbors[c]:
                            coverage_count[v] += 1
                            if coverage_count[v] == 1:
                                uncovered_count -= 1
                elif r < 0.8:
                    # Add
                    # Pick a candidate
                    # If uncovered > 0, try to pick one that covers uncovered
                    # Else random
                    if uncovered_count > 0:
                        # Greedy-ish pick: random from neighbors of an uncovered word?
                        # Or just random, SA will find it
                        # Let's try: pick random word from space
                        # To be smarter: pick random from list of words that are neighbors of uncovered?
                        # That's expensive.
                        # Just random.
                        c = random.randint(0, total_words - 1)
                    else:
                        c = random.randint(0, total_words - 1)
                        
                    if c in code_set:
                        continue
                        
                    # Apply
                    for v in neighbors[c]:
                        old_c = coverage_count[v]
                        coverage_count[v] += 1
                        if old_c == 0:
                            uncovered_count -= 1
                            
                    new_size = current_size + 1
                    new_unc = uncovered_count
                    new_cost = get_cost(new_size, new_unc)
                    delta = new_cost - current_cost
                    
                    if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                        # Accept
                        code.append(c)
                        code_set.add(c)
                        current_size = new_size
                        current_unc = new_unc
                        current_cost = new_cost
                    else:
                        # Revert
                        for v in neighbors[c]:
                            coverage_count[v] -= 1
                            if coverage_count[v] == 0:
                                uncovered_count += 1
                else:
                    # Swap: Remove one, Add one
                    if current_size == 0:
                        continue
                    idx_out = random.randint(0, current_size - 1)
                    c_out = code[idx_out]
                    
                    # Pick c_in
                    c_in = random.randint(0, total_words - 1)
                    if c_in == c_out or c_in in code_set:
                        continue
                        
                    # Apply Remove
                    for v in neighbors[c_out]:
                        coverage_count[v] -= 1
                        if coverage_count[v] == 0:
                            uncovered_count += 1
                            
                    # Apply Add
                    for v in neighbors[c_in]:
                        old_c = coverage_count[v]
                        coverage_count[v] += 1
                        if old_c == 0:
                            uncovered_count -= 1
                            
                    new_size = current_size # Size unchanged
                    new_unc = uncovered_count
                    new_cost = get_cost(new_size, new_unc)
                    delta = new_cost - current_cost
                    
                    if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                        # Accept
                        code[idx_out] = c_in
                        code_set.discard(c_out)
                        code_set.add(c_in)
                        current_unc = new_unc
                        current_cost = new_cost
                    else:
                        # Revert
                        for v in neighbors[c_in]:
                            coverage_count[v] -= 1
                            if coverage_count[v] == 0:
                                uncovered_count += 1
                        for v in neighbors[c_out]:
                            coverage_count[v] += 1
                            if coverage_count[v] == 1:
                                uncovered_count -= 1
                            
            T *= cooling_rate
            if T < min_T:
                break

        # 3. Final Greedy Deletion (Hill Climbing)
        # Ensure we are in a local minimum
        improved = True
        while improved and (time.time() - start_time) < max_time:
            improved = False
            random.shuffle(code)
            new_code = []
            for c in code:
                removable = True
                for v in neighbors[c]:
                    if coverage_count[v] == 1:
                        removable = False
                        break
                if removable:
                    for v in neighbors[c]:
                        coverage_count[v] -= 1
                    improved = True
                else:
                    new_code.append(c)
            code = new_code
            # Update code_set? Not strictly needed if we don't use it, 
            # but good practice. 
            # Actually, we don't use code_set in this phase.

        # Convert back to list of lists
        def int_to_digits(val, base, length):
            res = [0] * length
            for i in range(length):
                res[i] = val % base
                val //= base
            return res

        return [int_to_digits(c, q, n) for c in code]
    else:
        # Fallback
        total_words = q ** n
        def int_to_list(val, base, length):
            res = [0] * length
            for i in range(length):
                res[i] = val % base
                val //= base
            return res

        def list_to_int(lst, base):
            val = 0
            for i in range(len(lst)-1, -1, -1):
                val = val * base + lst[i]
            return val

        neighbors = []
        for i in range(total_words):
            w = int_to_list(i, q, n)
            neigh = {i}
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    w[pos] = new_val
                    ni = list_to_int(w, q)
                    neigh.add(ni)
                w[pos] = orig
            neighbors.append(neigh)

        uncovered = set(range(total_words))
        code = []
        while uncovered:
            best_c = -1
            best_count = -1
            for c in uncovered:
                count = len(neighbors[c] & uncovered)
                if count > best_count:
                    best_count = count
                    best_c = c
            code.append(best_c)
            uncovered = uncovered - neighbors[best_c]
        return [int_to_list(c, q, n) for c in code]
