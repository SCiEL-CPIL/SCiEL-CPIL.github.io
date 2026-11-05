"""Improved B2[2] set construction using greedy with backtracking and randomization."""
import time
import random
import sys

# Deadline guard: keep a safety margin before 150s wall-clock limit
DEADLINE = 135.0  # seconds

def _check_validity(chosen):
    """Check if chosen is a valid B2[2] set (every sum has at most 2 representations)."""
    counts = {}
    for i, a in enumerate(chosen):
        for j in range(i, len(chosen)):
            b = chosen[j]
            s = a + b
            counts[s] = counts.get(s, 0) + 1
            if counts[s] > 2:
                return False
    return True

def _greedy_build(order, time_limit):
    """Greedy build in the given order, returning the set if valid."""
    chosen = []
    counts = {}
    start_time = time.time()
    
    for x in order:
        if time.time() > time_limit:
            break
        # Check if adding x would violate the B2[2] condition
        valid = True
        new_sums = []
        for y in chosen:
            s = x + y
            new_sums.append(s)
        new_sums.append(2 * x)
        
        # Check each new sum
        for s in new_sums:
            if counts.get(s, 0) + 1 > 2:
                valid = False
                break
        
        if valid:
            chosen.append(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
    
    return chosen

def _local_search(chosen, time_limit):
    """Try to improve the set by swapping elements or adding new ones."""
    if not chosen:
        return chosen
    
    best = list(chosen)
    best_size = len(best)
    start_time = time.time()
    
    # Build current counts
    def build_counts(lst):
        c = {}
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                s = lst[i] + lst[j]
                c[s] = c.get(s, 0) + 1
        return c
    
    # Try adding elements not in the set
    current_set = set(best)
    counts = build_counts(best)
    
    # Try adding one element at a time
    for x in range(1, 1001):
        if time.time() > time_limit:
            break
        if x in current_set:
            continue
        
        # Check if we can add x
        can_add = True
        new_sums = []
        for y in best:
            s = x + y
            new_sums.append(s)
        new_sums.append(2 * x)
        
        for s in new_sums:
            if counts.get(s, 0) + 1 > 2:
                can_add = False
                break
        
        if can_add:
            best.append(x)
            current_set.add(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
            best_size = len(best)
    
    # Try swaps: remove one element, add another
    for remove_idx in range(len(best)):
        if time.time() > time_limit:
            break
        removed = best[remove_idx]
        temp = best[:remove_idx] + best[remove_idx+1:]
        temp_set = set(temp)
        temp_counts = build_counts(temp)
        
        # Try to add a different element
        best_add = None
        best_add_sums = None
        for x in range(1, 1001):
            if time.time() > time_limit:
                break
            if x in temp_set:
                continue
            
            can_add = True
            new_sums = []
            for y in temp:
                s = x + y
                new_sums.append(s)
            new_sums.append(2 * x)
            
            for s in new_sums:
                if temp_counts.get(s, 0) + 1 > 2:
                    can_add = False
                    break
            
            if can_add:
                best_add = x
                best_add_sums = new_sums
                break
        
        if best_add is not None:
            new_best = temp + [best_add]
            if len(new_best) > best_size:
                best = new_best
                best_size = len(best)
                current_set = set(best)
                counts = build_counts(best)
    
    return best

def construct():
    """Construct a B2[2] set in {1,...,1000} maximizing size."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Strategy 1: Simple greedy in increasing order
    best_set = _greedy_build(range(1, 1001), deadline - 5)
    best_size = len(best_set)
    
    if time.time() > deadline - 10:
        return best_set
    
    # Strategy 2: Try different random orderings
    rng = random.Random(42)
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        # Create a random ordering with some bias
        order = list(range(1, 1001))
        # Shuffle with a bias towards smaller numbers first (they're more "valuable")
        for i in range(len(order) - 1, 0, -1):
            # With some probability, keep the smaller number earlier
            if rng.random() < 0.3:
                j = rng.randint(0, i)
                if order[j] < order[i]:
                    order[j], order[i] = order[i], order[j]
            else:
                order[i] = rng.randint(1, 1000)
        
        # Deduplicate while maintaining order preference
        seen = set()
        unique_order = []
        for x in order:
            if x not in seen:
                seen.add(x)
                unique_order.append(x)
        
        result = _greedy_build(unique_order, deadline - 10)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)
    
    if time.time() > deadline - 5:
        return best_set
    
    # Strategy 3: Local search to improve the best set found
    improved = _local_search(best_set, deadline - 2)
    if len(improved) > best_size:
        best_set = improved
        best_size = len(best_set)
    
    return best_set

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
    print(f"First 20: {result[:20]}")
