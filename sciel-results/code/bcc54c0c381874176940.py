"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free_subset(S, subset):
        """Check if subset is sum-free within the context of S."""
        sub_set = set(subset)
        for a in subset:
            for b in subset:
                c = a + b
                if c in sub_set:
                    return False
        return True

    def greedy_sum_free(S, max_time=None):
        """Greedy algorithm to find a large sum-free subset."""
        S_set = set(S)
        # Try multiple orderings: by absolute value, by value, random
        best_result = []
        
        # Order 1: largest absolute values first (often works well)
        for ordering in [
            sorted(S, key=lambda x: -abs(x)),
            sorted(S, key=lambda x: -x),
            sorted(S, key=lambda x: x),
        ]:
            if max_time and time.time() > max_time:
                break
                
            subset = []
            subset_set = set()
            
            for x in ordering:
                # Check if adding x keeps sum-free
                ok = True
                for y in subset:
                    # x + y in subset?
                    if x + y in subset_set:
                        ok = False
                        break
                    # y + x in subset? (same as above)
                    # 2*y == x? i.e., x - 2*y == 0, or y + y = x
                    if y + y == x:
                        ok = False
                        break
                    # x + x in subset? (checked when y=x, but x not yet in subset)
                    # Actually we need: for any a,b in new subset, a+b not in new subset
                    # New subset = subset ∪ {x}
                    # Pairs to check: (x,x), (x,y) for y in subset, (y,x) for y in subset
                    # (x,x): 2x in subset?
                    if 2 * x in subset_set:
                        ok = False
                        break
                    # (x,y) and (y,x): x+y in subset?
                    if x + y in subset_set:
                        ok = False
                        break
                
                if ok:
                    subset.append(x)
                    subset_set.add(x)
            
            if len(subset) > len(best_result):
                best_result = subset
        
        return best_result

    def local_search_improve(S, initial_subset, max_time=None):
        """Try to improve by swapping elements."""
        S_set = set(S)
        current = list(initial_subset)
        current_set = set(current)
        
        # Precompute conflicts: for each element, which other elements it conflicts with
        # Conflict means: adding element would violate sum-free
        # Or removing one element allows adding another
        
        improved = True
        while improved and (max_time is None or time.time() < max_time):
            improved = False
            
            # Try to add elements not in current
            candidates = [x for x in S if x not in current_set]
            random.shuffle(candidates)
            
            for x in candidates:
                # Check if x can be added
                ok = True
                for y in current:
                    if x + y in current_set:
                        ok = False
                        break
                    if y + y == x:
                        ok = False
                        break
                    if 2 * x in current_set:
                        ok = False
                        break
                
                if ok:
                    current.append(x)
                    current_set.add(x)
                    improved = True
            
            # Try to swap: remove one, add another
            if not improved:
                for i in range(len(current)):
                    if max_time and time.time() > max_time:
                        break
                    removed = current[i]
                    new_set = current_set - {removed}
                    
                    candidates = [x for x in S if x not in new_set and x != removed]
                    random.shuffle(candidates)
                    
                    for x in candidates:
                        # Check if x can be added to new_set
                        ok = True
                        for y in new_set:
                            if x + y in new_set:
                                ok = False
                                break
                            if y + y == x:
                                ok = False
                                break
                            if 2 * x in new_set:
                                ok = False
                                break
                        
                        if ok:
                            current[i] = x
                            current_set = new_set | {x}
                            improved = True
                            break
                    if improved:
                        break
        
        return current

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        S = instance(size, seed)
        
        # Start with greedy
        best = greedy_sum_free(S, max_time=deadline - 5)
        
        # Try local search improvement
        remaining = deadline - 5 - (time.time() - start_time)
        if remaining > 1:
            improved = local_search_improve(S, best, max_time=start_time + remaining)
            if len(improved) > len(best):
                best = improved
        
        results.append(best)
        
        # Check deadline
        if time.time() > deadline - 10:
            # Fill remaining with greedy
            for size2, seed2 in instances[len(results):]:
                S2 = instance(size2, seed2)
                results.append(greedy_sum_free(S2))
            break
    
    # Ensure all three results are present
    while len(results) < 3:
        results.append([])
    
    return results
# EVOLVE-BLOCK-END
