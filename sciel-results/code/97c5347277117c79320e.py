import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    N_MAX = 192
    K_MAX = 48
    
    def time_left():
        return deadline - time.time()
    
    def is_valid_set(selected):
        """Check if the selected set is valid."""
        n = len(selected)
        for i, a in enumerate(selected):
            others = [x for j, x in enumerate(selected) if j != i]
            if len(others) < 3:
                continue
            # If a > sum of largest 3 others, it's safe
            others_sorted = sorted(others, reverse=True)
            if a > sum(others_sorted[:3]):
                continue
            # Check all triples
            for b, c, d in combinations(others, 3):
                if (b + c + d) % a == 0:
                    return False
        return True

    def can_add(selected, candidate):
        """Check if candidate can be added to selected without violating constraints."""
        n = len(selected)
        
        # 1. candidate must not divide sum of any 3 from selected
        if n >= 3:
            for b, c, d in combinations(selected, 3):
                if (b + c + d) % candidate == 0:
                    return False
        
        # 2. No a in selected should divide (candidate + b + c) for any b, c in selected \ {a}
        for i, a in enumerate(selected):
            # Need b, c from selected excluding a
            others = [x for j, x in enumerate(selected) if j != i]
            if len(others) < 2:
                continue
            # (candidate + b + c) % a == 0  =>  (b + c) % a == (-candidate) % a
            target = (-candidate) % a
            # Check if any pair in others sums to target mod a
            # Optimization: if a > candidate + max(others)*2, skip? No, just iterate.
            # For speed, we can precompute pair sums mod a? 
            # Since n is small, O(n^2) is fine.
            for j in range(len(others)):
                b = others[j]
                for k in range(j+1, len(others)):
                    c = others[k]
                    if (b + c) % a == target:
                        return False
        return True

    def greedy_build(order, max_size=K_MAX):
        """Greedy construction following a specific order."""
        selected = []
        for x in order:
            if len(selected) >= max_size:
                break
            if time_left() < 0.5:
                break
            if can_add(selected, x):
                selected.append(x)
        return selected

    def local_search_improve(selected, max_iters=200):
        """Try to improve the set by adding elements or swapping."""
        current = selected[:]
        current_len = len(current)
        improved = True
        iterations = 0
        
        while improved and current_len < K_MAX and iterations < max_iters and time_left() > 1.0:
            improved = False
            iterations += 1
            
            # 1. Try to add any missing element
            missing = [x for x in range(1, N_MAX + 1) if x not in current]
            # Sort by some heuristic? Larger numbers are often safer to add?
            # Or just random order.
            random.shuffle(missing)
            
            added = False
            for x in missing:
                if can_add(current, x):
                    current.append(x)
                    current_len += 1
                    improved = True
                    added = True
                    if current_len >= K_MAX:
                        break
            
            if added:
                continue
                
            # 2. If no add, try swaps (remove one, add one) to change structure
            # This might allow a subsequent add.
            # Pick a random element to remove
            if current:
                idx = random.randint(0, len(current) - 1)
                removed = current[idx]
                temp_current = current[:idx] + current[idx+1:]
                
                # Try to find a replacement that allows an add later
                # Or just find any valid replacement
                candidates = [x for x in range(1, N_MAX + 1) if x not in temp_current and x != removed]
                random.shuffle(candidates)
                
                best_replacement = None
                best_new_len = len(temp_current)
                
                for x in candidates[:50]: # Limit candidates for speed
                    if can_add(temp_current, x):
                        temp_with_x = temp_current + [x]
                        # Check if we can add anything else to temp_with_x
                        # This is expensive, so maybe just accept the swap if it's valid
                        # and hope for the best? 
                        # Better: just accept valid swap.
                        best_replacement = x
                        break
                
                if best_replacement is not None:
                    current = temp_current + [best_replacement]
                    # current_len stays same, but structure changed
                    # Force another iteration to try adding
                    improved = True 
                    # Don't increment current_len here as it's a swap
                    # But we need to ensure we don't loop infinitely on swaps
                    # The 'added' flag logic handles adds. Swaps just reshuffle.
        
        return current

    best_set = []
    best_len = 0
    
    # Strategy: Multiple random restarts with different orderings
    # Ordering heuristic: 
    # 1. Descending (large numbers first) - often works well for divisibility constraints
    # 2. Random
    # 3. Ascending (small numbers first) - usually bad, but maybe?
    # 4. Prime-heavy?
    
    num_restarts = 50
    for trial in range(num_restarts):
        if time_left() < 5.0:
            break
        
        random.seed(42 + trial)
        
        if trial % 3 == 0:
            # Descending order with noise
            order = list(range(N_MAX, 0, -1))
            # Add small noise to break ties
            noisy_order = [(x, random.uniform(-5, 5)) for x in order]
            noisy_order.sort(key=lambda t: -(t[0] + t[1]))
            order = [x for x, _ in noisy_order]
        elif trial % 3 == 1:
            # Random order
            order = list(range(1, N_MAX + 1))
            random.shuffle(order)
        else:
            # Descending, but prioritize numbers with few divisors? 
            # Or just descending again with different noise
            order = list(range(N_MAX, 0, -1))
            noisy_order = [(x, random.uniform(-10, 10)) for x in order]
            noisy_order.sort(key=lambda t: -(t[0] + t[1]))
            order = [x for x, _ in noisy_order]
            
        current_set = greedy_build(order)
        
        if len(current_set) > best_len:
            best_set = current_set[:]
            best_len = len(current_set)
            
        if best_len >= K_MAX:
            break
        
        # Local search improvement
        if time_left() > 10.0:
            improved_set = local_search_improve(current_set)
            if len(improved_set) > best_len:
                best_set = improved_set[:]
                best_len = len(improved_set)
                
                if best_len >= K_MAX:
                    break

    # Final polish: try to add more to the best set found
    if best_len < K_MAX and time_left() > 2.0:
        # Try adding missing numbers in descending order
        missing = [x for x in range(N_MAX, 0, -1) if x not in best_set]
        for x in missing:
            if best_len >= K_MAX:
                break
            if can_add(best_set, x):
                best_set.append(x)
                best_len += 1

    return best_set[:K_MAX]
