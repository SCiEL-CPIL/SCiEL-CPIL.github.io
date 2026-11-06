def construct():
    import time
    import random
    from math import gcd
    from fractions import Fraction
    
    start_time = time.time()
    deadline = 88.0
    
    denominators = list(range(2, 49))
    n = 47
    
    # Work with exact fractions for precision
    # We want to minimize (1 - sum) where sum < 1
    # Equivalent to maximizing sum subject to sum < 1
    
    def current_sum(idx_set):
        s = Fraction(0, 1)
        for i in idx_set:
            s += Fraction(1, denominators[i])
        return s
    
    def deficit(idx_set):
        s = current_sum(idx_set)
        return 1 - s  # positive since s < 1
    
    # Phase 1: Greedy construction - start with all items, remove those that make sum >= 1
    # Actually, let's start empty and add items greedily by value (1/d)
    
    # Better approach: Start with a random feasible solution, then improve via local search
    # Feasible means sum < 1
    
    # First, find a good starting point using greedy
    # Sort by 1/d descending (i.e., d ascending)
    order = list(range(n))
    
    # Try multiple random initial solutions
    best_def = None
    best_set = None
    
    # Greedy: add items in order of 1/d (largest first), skip if would exceed 1
    greedy_set = set()
    gsum = Fraction(0, 1)
    for i in order:
        val = Fraction(1, denominators[i])
        if gsum + val < 1:
            gsum += val
            greedy_set.add(i)
    
    if greedy_set:
        d = deficit(greedy_set)
        best_def = d
        best_set = set(greedy_set)
    
    # Also try: start with all, remove smallest until sum < 1
    all_set = set(range(n))
    asum = sum(Fraction(1, denominators[i]) for i in all_set)
    if asum >= 1:
        # Remove items with smallest 1/d (largest d) until sum < 1
        for i in sorted(order, key=lambda x: denominators[x], reverse=True):
            if asum < 1:
                break
            asum -= Fraction(1, denominators[i])
            all_set.discard(i)
        if all_set:
            d = deficit(all_set)
            if best_def is None or d < best_def:
                best_def = d
                best_set = set(all_set)
    
    # Local search: try adding/removing/swapping elements
    if best_set is None:
        # Fallback
        return [0]
    
    current = set(best_set)
    curr_def = best_def
    
    # Precompute values
    vals = [Fraction(1, denominators[i]) for i in range(n)]
    
    # Local search with simulated annealing
    rng = random.Random(42)
    
    def try_add_remove_swap(current_set, curr_def, max_iter=5000):
        """Try local improvements"""
        best_set = set(current_set)
        best_def = curr_def
        curr_set = set(current_set)
        curr_d = curr_def
        
        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            move_type = rng.random()
            
            if move_type < 0.3:
                # Try adding an element not in set
                candidates = [i for i in range(n) if i not in curr_set]
                if not candidates:
                    continue
                i = rng.choice(candidates)
                new_sum = sum(vals[j] for j in curr_set) + vals[i]
                if new_sum < 1:
                    new_d = 1 - new_sum
                    if new_d < curr_d:
                        curr_set.add(i)
                        curr_d = new_d
                        if curr_d < best_def:
                            best_def = curr_d
                            best_set = set(curr_set)
                    # Accept worse with probability (simplified SA)
                    elif rng.random() < 0.05:
                        curr_set.add(i)
                        curr_d = new_d
                else:
                    # Can't add, try removing something and adding this
                    if curr_set:
                        j = rng.choice(list(curr_set))
                        new_sum2 = sum(vals[k] for k in curr_set) - vals[j] + vals[i]
                        if new_sum2 < 1:
                            new_d2 = 1 - new_sum2
                            if new_d2 < curr_d:
                                curr_set.remove(j)
                                curr_set.add(i)
                                curr_d = new_d2
                                if curr_d < best_def:
                                    best_def = curr_d
                                    best_set = set(curr_set)
                            elif rng.random() < 0.03:
                                curr_set.remove(j)
                                curr_set.add(i)
                                curr_d = new_d2
            
            elif move_type < 0.6:
                # Try removing an element
                if curr_set:
                    i = rng.choice(list(curr_set))
                    new_sum = sum(vals[j] for j in curr_set) - vals[i]
                    new_d = 1 - new_sum
                    # Removing always increases deficit, so only if it enables better add later
                    # For now, skip pure removals unless we're stuck
                    if rng.random() < 0.02:
                        curr_set.discard(i)
                        curr_d = new_d
            
            else:
                # Try swap: remove one, add another
                if curr_set:
                    i = rng.choice(list(curr_set))
                    candidates = [k for k in range(n) if k not in curr_set]
                    if candidates:
                        j = rng.choice(candidates)
                        new_sum = sum(vals[k] for k in curr_set) - vals[i] + vals[j]
                        if new_sum < 1:
                            new_d = 1 - new_sum
                            if new_d < curr_d:
                                curr_set.remove(i)
                                curr_set.add(j)
                                curr_d = new_d
                                if curr_d < best_def:
                                    best_def = curr_d
                                    best_set = set(curr_set)
                            elif rng.random() < 0.04:
                                curr_set.remove(i)
                                curr_set.add(j)
                                curr_d = new_d
        
        return best_set, best_def
    
    # Run several rounds of local search
    rounds = 0
    while time.time() < deadline - 5:
        current, curr_def = try_add_remove_swap(current, curr_def, max_iter=3000)
        rounds += 1
        if curr_def < best_def:
            best_def = curr_def
            best_set = set(current)
        
        # Occasionally restart from a perturbed version
        if rounds % 3 == 0 and time.time() < deadline - 10:
            # Perturb best solution
            perturbed = set(best_set)
            for _ in range(rng.randint(1, 3)):
                if perturbed:
                    i = rng.choice(list(perturbed))
                    perturbed.discard(i)
                candidates = [k for k in range(n) if k not in perturbed]
                if candidates:
                    j = rng.choice(candidates)
                    new_sum = sum(vals[k] for k in perturbed) + vals[j]
                    if new_sum < 1:
                        perturbed.add(j)
            if perturbed:
                pd = deficit(perturbed)
                if pd < curr_def:
                    current = set(perturbed)
                    curr_def = pd
    
    # Final verification
    final_sum = sum(vals[i] for i in best_set)
    assert final_sum < 1, f"Sum {final_sum} is not < 1"
    assert len(best_set) > 0
    
    return sorted(list(best_set))
