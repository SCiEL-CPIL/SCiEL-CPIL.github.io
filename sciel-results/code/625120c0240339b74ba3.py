import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    # First, get a good baseline with greedy
    best_set = _greedy_baseline()
    
    # Try to improve with backtracking
    if time.time() < deadline - 5:
        improved = _backtracking_improve(best_set, deadline)
        if len(improved) > len(best_set):
            best_set = improved
    
    return best_set

def _greedy_baseline():
    """Original greedy approach with multiple orderings."""
    best_set = []
    mask = (1 << 201) - 1
    
    orders = []
    orders.append(list(range(100, 0, -1)))
    orders.append(list(range(1, 101)))
    
    base = list(range(1, 101))
    for _ in range(50):
        random.shuffle(base)
        orders.append(base[:])
        
    orders.append(sorted(range(1, 101), key=lambda x: x % 2))
    orders.append(sorted(range(1, 101), key=lambda x: x % 3))
    orders.append(sorted(range(1, 101), key=lambda x: x % 5))
    
    for order in orders:
        selected = []
        reachable_bits = 1
        
        for x in order:
            if (reachable_bits >> (200 - x)) & 1:
                continue
            else:
                selected.append(x)
                new_bits = (reachable_bits << x) & mask
                reachable_bits |= new_bits
                
        if len(selected) > len(best_set):
            best_set = selected
            
    return best_set

def _backtracking_improve(initial_set, deadline):
    """Try to add more elements from [1,100] not in initial_set using backtracking."""
    initial_set = set(initial_set)
    remaining = [x for x in range(1, 101) if x not in initial_set]
    
    # Sort remaining by value descending (larger numbers are "safer" to add)
    remaining.sort(reverse=True)
    
    mask = (1 << 201) - 1
    
    # Compute reachable sums for the initial set
    reachable_bits = 1
    for x in initial_set:
        new_bits = (reachable_bits << x) & mask
        reachable_bits |= new_bits
    
    best_set = set(initial_set)
    best_size = len(initial_set)
    
    # Backtracking search with forward checking
    def backtrack(index, current_set, current_reachable):
        nonlocal best_set, best_size
        
        if time.time() > deadline - 2:
            return
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
        
        if index >= len(remaining):
            return
        
        # Try including remaining[index]
        x = remaining[index]
        if not ((current_reachable >> (200 - x)) & 1):
            # Safe to include
            new_reachable = current_reachable | ((current_reachable << x) & mask)
            current_set.add(x)
            backtrack(index + 1, current_set, new_reachable)
            current_set.remove(x)
        
        # Try excluding remaining[index]
        backtrack(index + 1, current_set, current_reachable)
    
    backtrack(0, set(initial_set), reachable_bits)
    
    return list(best_set)
