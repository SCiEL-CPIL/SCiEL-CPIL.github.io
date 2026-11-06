"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Phase 1: Greedy construction with numpy sum-count array
    # sum_counts[s] = number of pairs (a,b) in chosen with a<=b and a+b=s
    N = 1000
    MAX_SUM = 2 * N  # max possible sum is 2000
    sum_counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    chosen = []
    chosen_set = set()
    
    for x in range(1, N + 1):
        # Check if adding x would violate B2[2] constraint
        # New sums: x + y for all y in chosen, plus 2*x
        # We need to check that no sum exceeds count 2
        valid = True
        for y in chosen:
            s = x + y
            if sum_counts[s] >= 2:
                valid = False
                break
        if valid:
            s = 2 * x
            if sum_counts[s] >= 2:
                valid = False
        if valid:
            chosen.append(x)
            chosen_set.add(x)
            # Update sum counts
            for y in chosen[:-1]:  # all previous elements
                sum_counts[x + y] += 1
            sum_counts[2 * x] += 1
    
    if time.time() > deadline - 5:
        return chosen
    
    # Phase 2: Local search - try to improve by swapping elements
    # Try to add elements that were skipped, possibly removing others
    # Use simulated annealing style approach
    
    best_size = len(chosen)
    current_chosen = list(chosen)
    current_sum_counts = sum_counts.copy()
    
    # Build a list of candidates (elements 1-1000 not in current set)
    candidates = [x for x in range(1, N + 1) if x not in chosen_set]
    
    # Try simple improvements: for each candidate, check if we can add it
    # (may need to remove some existing elements)
    # First, try direct additions
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        random.shuffle(candidates)
        for x in candidates:
            if x in chosen_set:
                continue
            # Check if x can be added directly
            valid = True
            for y in current_chosen:
                s = x + y
                if current_sum_counts[s] >= 2:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if current_sum_counts[s] >= 2:
                    valid = False
            if valid:
                # Add x
                current_chosen.append(x)
                chosen_set.add(x)
                for y in current_chosen[:-1]:
                    current_sum_counts[x + y] += 1
                current_sum_counts[2 * x] += 1
                improved = True
                if len(current_chosen) > best_size:
                    best_size = len(current_chosen)
    
    if time.time() > deadline - 5:
        return current_chosen
    
    # Phase 3: Simulated annealing - try swaps
    # Remove one element, try to add two new ones
    best_chosen = list(current_chosen)
    best_counts = current_sum_counts.copy()
    
    # Precompute: for each element in current_chosen, what sums does it contribute to
    # and what other elements are involved
    
    # Try local moves: remove element r, add element a (and possibly b)
    # Score = len(set) - penalty for constraint violations (should be 0)
    
    sa_iterations = 0
    max_sa = 50000
    temperature = 1.0
    cooling = 0.999
    
    for iteration in range(max_sa):
        if time.time() > deadline - 3:
            break
        
        if len(current_chosen) == 0:
            break
        
        # Pick a random element to potentially remove
        r_idx = random.randint(0, len(current_chosen) - 1)
        r = current_chosen[r_idx]
        
        # Try to find 1-2 replacements
        # First try: just add one element (net gain 0, but might enable more later)
        # Better: remove r, try to add two elements a, b such that set grows by 1
        
        # For efficiency, try a simple move: remove r, add a random candidate
        # This keeps size same but might lead to better configuration
        
        # Actually, let's try: remove r, then greedily try to add back as many as possible
        # including r itself and new elements
        
        # Simpler approach: try pairwise swaps
        # Remove r, try to add a new element a (not in set)
        # Net change: 0 in size, but different structure
        
        # Let's try a different strategy: 
        # Try to add a new element by removing the minimum number of existing elements
        # that conflict with it
        
        # For each candidate not in set, find which existing elements conflict
        # If we can remove k elements and add 1, net change is 1-k
        # We want 1-k > 0, i.e., k=0 (direct add, already tried)
        # Or we accept k=1 (swap, net 0) to improve structure
        
        # Let's do swaps: remove r, add a
        a_candidates = [x for x in candidates if x not in chosen_set]
        if not a_candidates:
            continue
        
        a = random.choice(a_candidates)
        
        # Check if removing r and adding a is valid
        # After removing r, check if a can be added
        # Sums involving r that will be removed: r + y for all y in current_chosen (y != r), and 2*r
        # Sums involving a that will be added: a + y for all y in (current_chosen - {r}), and 2*a
        
        # This is complex to check quickly. Let's just do it directly for now.
        # Since we have limited time, let's try a simpler improvement:
        # Try to add elements in a different order using backtracking
        
        # Alternative: restart greedy with different orderings
        break  # Skip SA for now, try other improvements
    
    # Phase 4: Try greedy with different orderings
    # Use the remaining time to try multiple greedy passes with shuffled/perturbed orders
    orders_tried = 0
    max_orders = 50
    
    for order_seed in range(max_orders):
        if time.time() > deadline - 3:
            break
        
        random.seed(order_seed)
        
        # Create a perturbed order
        order = list(range(1, N + 1))
        
        # Mix strategies: sorted, reverse, random, biased
        strategy = order_seed % 4
        if strategy == 0:
            pass  # natural order
        elif strategy == 1:
            order.reverse()
        elif strategy == 2:
            random.shuffle(order)
        else:
            # Biased: prefer numbers that are far apart
            # Use a golden ratio based ordering or similar
            golden = 1.618
            order = sorted(range(1, N + 1), key=lambda x: (x * golden) % 1)
        
        # Greedy with this order
        sc = np.zeros(MAX_SUM + 1, dtype=np.int32)
        ch = []
        ch_set = set()
        
        for x in order:
            valid = True
            for y in ch:
                s = x + y
                if sc[s] >= 2:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if sc[s] >= 2:
                    valid = False
            if valid:
                ch.append(x)
                ch_set.add(x)
                for y in ch[:-1]:
                    sc[x + y] += 1
                sc[2 * x] += 1
        
        if len(ch) > best_size:
            best_size = len(ch)
            best_chosen = list(ch)
            best_counts = sc.copy()
            current_chosen = list(ch)
            current_sum_counts = sc.copy()
            # Rebuild chosen_set
            chosen_set = set(ch)
            # Rebuild candidates
            candidates = [x for x in range(1, N + 1) if x not in chosen_set]
    
    # Phase 5: Final local search on best solution
    if time.time() < deadline - 5:
        # Try to add any remaining candidates to best_chosen
        best_set = set(best_chosen)
        best_sc = best_counts.copy()
        
        for x in range(1, N + 1):
            if x in best_set:
                continue
            valid = True
            for y in best_chosen:
                s = x + y
                if best_sc[s] >= 2:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if best_sc[s] >= 2:
                    valid = False
            if valid:
                best_chosen.append(x)
                best_set.add(x)
                for y in best_chosen[:-1]:
                    best_sc[x + y] += 1
                best_sc[2 * x] += 1
    
    return best_chosen
# EVOLVE-BLOCK-END
