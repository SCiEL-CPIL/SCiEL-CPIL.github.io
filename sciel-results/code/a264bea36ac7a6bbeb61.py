import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_instance(S, time_budget):
        """Solve a single instance within the given time budget."""
        S_set = set(S)
        n = len(S)
        
        # Create index mapping for faster lookups
        val_to_idx = {v: i for i, v in enumerate(S)}
        
        # Precompute conflict information using bitsets
        # conflict_mask[i] has bit j set if S[i] + S[j] is in S
        conflict_mask = [0] * n
        for i in range(n):
            for j in range(n):
                s = S[i] + S[j]
                if s in val_to_idx:
                    conflict_mask[i] |= (1 << val_to_idx[s])
        
        # Also compute: for each element x, which pairs (a,b) have a+b=x
        # This helps with removal checks
        sum_to_pairs = {}
        for i in range(n):
            for j in range(n):
                s = S[i] + S[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    if k not in sum_to_pairs:
                        sum_to_pairs[k] = []
                    sum_to_pairs[k].append((i, j))
        
        odds = [x for x in S if x % 2 != 0]
        evens = [x for x in S if x % 2 == 0]
        
        best_set = set(odds)
        best_size = len(best_set)
        
        rng = Random(42)
        
        def can_add_fast(x, current_set):
            """Fast check if x can be added to current_set."""
            if x not in S_set:
                return False
            for a in current_set:
                if a + x in current_set:
                    return False
                if x - a in current_set:
                    return False
            if 2 * x in current_set:
                return False
            return True
        
        def local_search_with_swaps(current_set, max_iters=50):
            """Local search with 1-flip and 2-swap operations."""
            current = set(current_set)
            for _ in range(max_iters):
                if time.time() > time_budget:
                    break
                
                improved = False
                elements = list(current)
                rng.shuffle(elements)
                
                # Try 1-flip: remove one element
                for u in elements:
                    if time.time() > time_budget:
                        break
                    test = set(current)
                    test.discard(u)
                    
                    # Try to add back any element
                    for e in S:
                        if e not in test and can_add_fast(e, test):
                            test.add(e)
                    
                    if len(test) > len(current):
                        current = test
                        improved = True
                        break
                
                if improved:
                    continue
                
                # Try 2-swap: remove two, add two
                for i in range(len(elements)):
                    if time.time() > time_budget:
                        break
                    for j in range(i + 1, len(elements)):
                        if time.time() > time_budget:
                            break
                        u1, u2 = elements[i], elements[j]
                        test = set(current)
                        test.discard(u1)
                        test.discard(u2)
                        
                        # Try to add back any two elements
                        added = 0
                        for e in S:
                            if e not in test and can_add_fast(e, test):
                                test.add(e)
                                added += 1
                                if added >= 2:
                                    break
                        
                        if len(test) > len(current):
                            current = test
                            improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    break
            
            return current
        
        # Multiple restarts with different strategies
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() > time_budget:
                break
            
            current_set = set(odds)
            
            # Different initialization strategies
            if r % 3 == 0:
                # Start with odds, greedily add evens
                shuffled_evens = evens[:]
                rng.shuffle(shuffled_evens)
                for e in shuffled_evens:
                    if time.time() > time_budget:
                        break
                    if can_add_fast(e, current_set):
                        current_set.add(e)
            elif r % 3 == 1:
                # Start with large absolute values
                sorted_by_abs = sorted(S, key=lambda x: abs(x), reverse=True)
                for e in sorted_by_abs:
                    if time.time() > time_budget:
                        break
                    if can_add_fast(e, current_set):
                        current_set.add(e)
            else:
                # Random initialization
                random_subset = S[:]
                rng.shuffle(random_subset)
                for e in random_subset:
                    if time.time() > time_budget:
                        break
                    if can_add_fast(e, current_set):
                        current_set.add(e)
            
            # Local search with swaps
            current_set = local_search_with_swaps(current_set, max_iters=30)
            
            if len(current_set) > best_size:
                best_set = current_set
                best_size = len(current_set)
        
        return list(best_set)

    results = []
    remaining_time = deadline - time.time()
    time_per_instance = remaining_time / 3
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        
        # Fallback if time is tight
        if time.time() > deadline - 10:
            results.append([x for x in S if x % 2])
            continue
        
        instance_budget = min(time_per_instance, deadline - time.time() - 5)
        res = solve_instance(S, instance_budget)
        
        # Verify
        if not is_sum_free(res):
            # Fallback to parity
            res = [x for x in S if x % 2]
        
        results.append(res)
        
    return results
