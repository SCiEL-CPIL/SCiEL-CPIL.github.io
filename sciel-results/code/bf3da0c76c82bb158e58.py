import random
import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    N = 1000
    max_val = N
    
    # Helper to calculate penalty from scratch (for validation)
    def get_penalty(sums):
        pen = 0
        for v in sums.values():
            if v > 2:
                pen += v - 2
        return pen
    
    # Fast greedy initialization with given order
    def greedy_init(order):
        chosen = []
        counts = defaultdict(int)
        chosen_set = set()
        for x in order:
            if x in chosen_set:
                continue
            # Check if adding x keeps all counts <= 2
            valid = True
            # Check 2x
            if counts.get(2 * x, 0) >= 2:
                valid = False
            else:
                # Check x + y for all y in chosen
                for y in chosen:
                    if counts.get(x + y, 0) >= 2:
                        valid = False
                        break
            
            if valid:
                chosen.append(x)
                chosen_set.add(x)
                counts[2 * x] = counts.get(2 * x, 0) + 1
                for y in chosen:
                    if y != x:
                        counts[x + y] = counts.get(x + y, 0) + 1
        return chosen, counts, chosen_set
    
    # Try multiple diverse initializations
    best_S = []
    best_counts = defaultdict(int)
    best_set = set()
    best_size = 0
    best_penalty = float('inf')
    
    # Strategy 1: Increasing order
    S1, C1, Set1 = greedy_init(list(range(1, max_val + 1)))
    p1 = get_penalty(C1)
    if len(S1) > best_size or (len(S1) == best_size and p1 < best_penalty):
        best_S, best_counts, best_set, best_size, best_penalty = S1, C1, Set1, len(S1), p1
    
    # Strategy 2: Decreasing order
    S2, C2, Set2 = greedy_init(list(range(max_val, 0, -1)))
    p2 = get_penalty(C2)
    if len(S2) > best_size or (len(S2) == best_size and p2 < best_penalty):
        best_S, best_counts, best_set, best_size, best_penalty = S2, C2, Set2, len(S2), p2
    
    # Strategy 3-8: Random orders with different seeds
    for seed in range(6):
        if time.time() > deadline - 10:
            break
        random.seed(seed * 137 + 42)
        order = list(range(1, max_val + 1))
        random.shuffle(order)
        S_r, C_r, Set_r = greedy_init(order)
        p_r = get_penalty(C_r)
        if len(S_r) > best_size or (len(S_r) == best_size and p_r < best_penalty):
            best_S, best_counts, best_set, best_size, best_penalty = S_r, C_r, Set_r, len(S_r), p_r
    
    # If no valid solution found, use the best we have
    if best_penalty > 0:
        # We'll work with the best solution and try to fix it
        pass
    
    # Simulated Annealing refinement
    S = list(best_S)
    sums = dict(best_counts)
    S_set = set(S)
    current_size = len(S)
    current_penalty = get_penalty(sums)
    current_score = current_size * 1000 - current_penalty
    
    # Track best valid solution
    best_valid_size = best_size if best_penalty == 0 else 0
    best_valid_set = set(S) if best_penalty == 0 else set()
    
    # SA parameters
    T = 50.0
    cooling = 0.99995
    min_T = 0.1
    
    all_elements = list(range(1, max_val + 1))
    
    # Precompute for faster lookups
    def try_insert(x):
        """Try to insert x into S. Returns (delta_penalty, added_sums) if valid check passes structurally."""
        added_sums = [x + y for y in S] + [2 * x]
        delta_pen = 0
        for s in added_sums:
            c = sums.get(s, 0)
            if c >= 2:
                delta_pen += 1
        return delta_pen, added_sums
    
    def try_delete(x):
        """Calculate delta penalty for deleting x."""
        # Sums that were added when x was inserted: x+y for y in S\{x}, and 2x
        removed_sums = [x + y for y in S if y != x] + [2 * x]
        delta_pen = 0
        for s in removed_sums:
            c = sums.get(s, 0)
            if c >= 3:
                delta_pen -= 1
        return delta_pen, removed_sums
    
    def try_swap(x, y):
        """Calculate delta penalty for swapping x with y (x in S, y not in S)."""
        # Remove x: sums x+z for z in S\{x}, and 2x
        sums_remove = [x + z for z in S if z != x] + [2 * x]
        # Add y: sums y+z for z in S\{x}, and 2y
        sums_add = [y + z for z in S if z != x] + [2 * y]
        
        delta_pen = 0
        # For removals
        for s in sums_remove:
            c = sums.get(s, 0)
            if c >= 3:
                delta_pen -= 1
        # For additions
        for s in sums_add:
            c = sums.get(s, 0)
            if c >= 2:
                delta_pen += 1
        
        return delta_pen, sums_remove, sums_add
    
    steps = 0
    while time.time() < deadline - 2:
        if T < min_T:
            # Reheat occasionally
            if random.random() < 0.05:
                T = 10.0
            else:
                T = min_T
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Insert
            if len(S) >= max_val:
                continue
            # Pick random candidate not in S
            # Sample a few and pick first not in S
            candidates = []
            for _ in range(20):
                x = random.randint(1, max_val)
                if x not in S_set:
                    candidates.append(x)
                    if len(candidates) >= 5:
                        break
            if not candidates:
                # Fallback: find any not in S
                x = random.randint(1, max_val)
                if x in S_set:
                    continue
                candidates = [x]
            
            x = random.choice(candidates)
            
            delta_pen, added_sums = try_insert(x)
            new_penalty = current_penalty + delta_pen
            new_size = current_size + 1
            new_score = new_size * 1000 - new_penalty
            
            delta_E = current_score - new_score
            if delta_E < 0 or (T > 0 and random.random() < math.exp(-delta_E / T)):
                # Accept
                S.append(x)
                S_set.add(x)
                for s in added_sums:
                    sums[s] = sums.get(s, 0) + 1
                current_score = new_score
                current_size = new_size
                current_penalty = new_penalty
                
                if new_penalty == 0 and new_size > best_valid_size:
                    best_valid_size = new_size
                    best_valid_set = set(S)
                    
        elif move_type < 0.6:
            # Delete
            if not S:
                continue
            x = random.choice(S)
            idx = S.index(x)
            
            delta_pen, removed_sums = try_delete(x)
            new_penalty = current_penalty + delta_pen
            new_size = current_size - 1
            new_score = new_size * 1000 - new_penalty
            
            delta_E = current_score - new_score
            if delta_E < 0 or (T > 0 and random.random() < math.exp(-delta_E / T)):
                # Accept
                S.pop(idx)
                S_set.discard(x)
                for s in removed_sums:
                    sums[s] -= 1
                    if sums[s] == 0:
                        del sums[s]
                current_score = new_score
                current_size = new_size
                current_penalty = new_penalty
            # If rejected, state unchanged (we didn't modify yet)
            
        else:
            # Swap
            if not S:
                continue
            x = random.choice(S)
            # Pick y not in S
            y = None
            for _ in range(20):
                y = random.randint(1, max_val)
                if y not in S_set:
                    break
            if y is None or y == x:
                continue
            
            delta_pen, sums_remove, sums_add = try_swap(x, y)
            new_penalty = current_penalty + delta_pen
            new_size = current_size
            new_score = new_size * 1000 - new_penalty
            
            delta_E = current_score - new_score
            if delta_E < 0 or (T > 0 and random.random() < math.exp(-delta_E / T)):
                # Accept: perform swap
                idx = S.index(x)
                S[idx] = y
                S_set.discard(x)
                S_set.add(y)
                
                for s in sums_remove:
                    sums[s] -= 1
                    if sums[s] == 0:
                        del sums[s]
                for s in sums_add:
                    sums[s] = sums.get(s, 0) + 1
                
                current_score = new_score
                current_penalty = new_penalty
                
                if new_penalty == 0 and new_size > best_valid_size:
                    best_valid_size = new_size
                    best_valid_set = set(S)
            
        steps += 1
        T *= cooling
    
    # Return the best valid solution
    if best_valid_size > 0:
        return list(best_valid_set)
    else:
        # Fallback: return the best we have (might have penalty)
        # Try to clean up by removing elements that cause violations
        S_clean = list(S)
        counts_clean = dict(sums)
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for i in range(len(S_clean)):
                x = S_clean[i]
                # Check if removing x reduces penalty
                removed_sums = [x + y for y in S_clean if y != x] + [2 * x]
                delta_pen = 0
                for s in removed_sums:
                    c = counts_clean.get(s, 0)
                    if c >= 3:
                        delta_pen -= 1
                if delta_pen < 0:
                    # Remove x
                    S_clean.pop(i)
                    for s in removed_sums:
                        counts_clean[s] -= 1
                        if counts_clean[s] == 0:
                            del counts_clean[s]
                    changed = True
                    break
        return S_clean
# EVOLVE-BLOCK-END
