"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import random
    
    def verify(S):
        """Check if no subset of S sums to 200."""
        S = list(S)
        reachable = {0}
        for x in S:
            if x > 200:
                continue
            new = {s + x for s in reachable if s + x <= 200}
            reachable |= new
        return 200 not in reachable

    def greedy_greedy(order):
        """Greedy: process elements in given order, add if doesn't create sum 200."""
        selected = []
        reachable = {0}
        for x in order:
            new_sums = {s + x for s in reachable if s + x <= 200}
            if 200 not in new_sums:
                selected.append(x)
                reachable |= new_sums
        return selected

    # 1. Generate initial solution using multiple greedy strategies
    best = set()
    
    # Descending
    desc = list(range(100, 0, -1))
    best = set(greedy_greedy(desc))
    
    # Ascending
    asc = list(range(1, 101))
    cand = greedy_greedy(asc)
    if len(cand) > len(best):
        best = set(cand)
        
    # Random shuffles
    rng = random.Random(42)
    for _ in range(100):
        shuffled = list(range(1, 101))
        rng.shuffle(shuffled)
        cand = greedy_greedy(shuffled)
        if len(cand) > len(best):
            best = set(cand)

    # 2. Local Search Improvement
    # Try to add elements from the complement
    # Or swap elements to allow adding more?
    # Simple approach: Try to add any element from [1,100]\best that keeps validity.
    # If we can add one, we increase size.
    # We might need to remove some to add others, but let's start with simple addition.
    
    improved = True
    while improved:
        improved = False
        # Sort candidates by some heuristic (e.g., larger first, or random)
        # To maximize size, we should try to add elements that "cost" the least in terms of blocking other elements.
        # However, simple addition is safe.
        
        # Try adding elements from the missing set
        missing = [x for x in range(1, 101) if x not in best]
        # Shuffle missing to avoid bias
        rng.shuffle(missing)
        
        for x in missing:
            new_set = set(best)
            new_set.add(x)
            if verify(new_set):
                best = new_set
                improved = True
                # Break and restart loop to try more additions
                break

    # 3. Advanced Local Search: Swap
    # If we can't add any, try to swap: remove y, add x, where x > y or x allows more additions later.
    # Given time limits, simple addition might be sufficient if greedy is not optimal.
    # But greedy descending often gets stuck.
    # Let's try a few random restarts of the local search.
    
    best_size = len(best)
    
    for _ in range(10):
        # Restart from a random greedy solution
        shuffled = list(range(1, 101))
        rng.shuffle(shuffled)
        current = set(greedy_greedy(shuffled))
        
        # Local search: Add
        improved = True
        while improved:
            improved = False
            missing = [x for x in range(1, 101) if x not in current]
            rng.shuffle(missing)
            for x in missing:
                new_set = set(current)
                new_set.add(x)
                if verify(new_set):
                    current = new_set
                    improved = True
                    break
        
        if len(current) > best_size:
            best = current
            best_size = len(best)

    return list(best)
# EVOLVE-BLOCK-END
