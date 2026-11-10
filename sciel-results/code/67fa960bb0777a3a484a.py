"""Improved B2[2] set construction for {1..1000}."""
import time
import random
from collections import Counter, defaultdict

N = 1000
MAX_REP = 2  # at most two representations a+b with a<=b

def _sum_counts(chosen):
    """Compute the count of each sum a+b with a<=b from chosen."""
    counts = {}
    for i, a in enumerate(chosen):
        for j in range(i, len(chosen)):
            b = chosen[j]
            s = a + b
            counts[s] = counts.get(s, 0) + 1
    return counts

def _greedy_build(order):
    """Build a B2[2] set by greedily adding elements in the given order."""
    chosen = []
    counts = {}
    for x in order:
        # Check if adding x keeps all sums <= MAX_REP
        # New sums: x + y for y in chosen
        new_sums = []
        for y in chosen:
            a, b = (x, y) if x <= y else (y, x)
            s = a + b
            new_sums.append(s)
        new_sums.append(2 * x)
        
        # Check if any sum would exceed MAX_REP
        ok = True
        for s in new_sums:
            if counts.get(s, 0) + 1 > MAX_REP:
                ok = False
                break
        
        if ok:
            chosen.append(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
    
    return chosen

def _local_search(chosen, counts, max_steps=5000, deadline=None):
    """Improve a B2[2] set via local search: try insertions and swaps."""
    chosen = list(chosen)
    counts = dict(counts)
    chosen_set = set(chosen)
    
    def try_insert(x):
        """Try to insert x into the set. Returns True if successful."""
        if x in chosen_set:
            return False
        
        # Check validity
        for y in chosen:
            a, b = (x, y) if x <= y else (y, x)
            s = a + b
            if counts.get(s, 0) + 1 > MAX_REP:
                return False
        
        # Check 2*x
        s = 2 * x
        if counts.get(s, 0) + 1 > MAX_REP:
            return False
            
        # Insert
        chosen.append(x)
        chosen_set.add(x)
        for y in chosen[:-1]: # y is the previous elements
             a, b = (x, y) if x <= y else (y, x)
             s = a + b
             counts[s] = counts.get(s, 0) + 1
        counts[2*x] = counts.get(2*x, 0) + 1
        return True

    def try_remove(x):
        """Remove x from the set."""
        # Remove from set
        chosen_set.remove(x)
        
        # Find index to remove from list
        try:
            idx = chosen.index(x)
            chosen.pop(idx)
        except ValueError:
            return # Should not happen
        
        # Update counts
        for y in chosen:
            a, b = (x, y) if x <= y else (y, x)
            s = a + b
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        
        s = 2 * x
        counts[s] -= 1
        if counts[s] == 0:
            del counts[s]

    def try_swap(remove_x, add_y):
        """Try to swap remove_x for add_y."""
        if add_y in chosen_set:
            return False
        
        # Check if removing remove_x and adding add_y is valid
        # 1. Temporarily remove remove_x from counts logic? 
        #    It's faster to check the net change.
        #    Sums involving remove_x: (remove_x, y) for y in chosen, y != remove_x, and 2*remove_x.
        #    Sums involving add_y: (add_y, y) for y in chosen, y != remove_x, and 2*add_y.
        
        # Let's just simulate removal and addition on a copy of counts? 
        # No, that's O(k). 
        # Let's check constraints directly.
        
        # For every sum s, new_count = old_count - delta_remove + delta_add
        # delta_remove[s] is 1 if s is a sum involving remove_x, 0 otherwise.
        # delta_add[s] is 1 if s is a sum involving add_y, 0 otherwise.
        
        # We need new_count <= MAX_REP.
        # Also new_count >= 0 (which is guaranteed if we only remove existing sums).
        
        # Optimization: 
        # 1. Check if adding add_y to (chosen - {remove_x}) is valid.
        #    This is equivalent to: for all y in chosen \ {remove_x}, count[(add_y+y)] + count[2*add_y] <= MAX_REP - (1 if remove_x involved in that sum? No.)
        
        # Let's do it safely but efficiently.
        # 1. Compute sums for add_y with all y in chosen.
        #    If y == remove_x, this sum is irrelevant for the "after" state, but it exists in the "before" state?
        #    No, in the "after" state, remove_x is gone.
        
        # Let's just use the net change dictionary approach, it's O(k).
        net_changes = {}
        
        # Sums to remove (involving remove_x)
        for y in chosen:
            if y == remove_x: continue
            a, b = (remove_x, y) if remove_x <= y else (y, remove_x)
            s = a + b
            net_changes[s] = net_changes.get(s, 0) - 1
        s_rx = 2 * remove_x
        net_changes[s_rx] = net_changes.get(s_rx, 0) - 1
        
        # Sums to add (involving add_y)
        # Note: if add_y == remove_x, we skip, but we already checked add_y not in set.
        for y in chosen:
            if y == remove_x: continue
            a, b = (add_y, y) if add_y <= y else (y, add_y)
            s = a + b
            net_changes[s] = net_changes.get(s, 0) + 1
        s_ay = 2 * add_y
        net_changes[s_ay] = net_changes.get(s_ay, 0) + 1
        
        # Check validity
        for s, delta in net_changes.items():
            new_count = counts.get(s, 0) + delta
            if new_count < 0 or new_count > MAX_REP:
                return False
        
        # Perform swap
        try_remove(remove_x)
        try_insert(add_y)
        return True

    improved = True
    step = 0
    while improved and step < max_steps:
        if deadline and time.time() > deadline:
            break
        improved = False
        step += 1
        
        # Try insertions first
        # Iterate over all possible candidates
        # To speed up, we can shuffle candidates or just iterate 1..N
        # Since we maintain chosen_set, check is O(1).
        for x in range(1, N + 1):
            if x not in chosen_set:
                if try_insert(x):
                    improved = True
                    if deadline and time.time() > deadline:
                        break
        if deadline and time.time() > deadline:
            break
        
        # Try swaps if no improvement from insertions
        if not improved:
            # Randomize the order of removals to escape local minima
            remove_candidates = list(chosen)
            random.shuffle(remove_candidates)
            
            for x in remove_candidates:
                if deadline and time.time() > deadline:
                    break
                # Try swapping x with some y
                # We can try all y, or a subset
                # To be fast, try all y not in set
                for y in range(1, N + 1):
                    if y not in chosen_set:
                        if try_swap(x, y):
                            improved = True
                            break # Restart insertion phase
                if improved:
                    break
                if deadline and time.time() > deadline:
                    break
    
    return chosen, counts

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_size = 0
    best_chosen = []
    
    # Multiple greedy strategies with different orderings
    strategies = []
    
    # 1. Ascending order
    strategies.append(list(range(1, N + 1)))
    
    # 2. Descending order
    strategies.append(list(range(N, 0, -1)))
    
    # 3. Random shuffles
    rng = random.Random(42)
    for _ in range(20):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        strategies.append(order)
    
    # Run greedy for each strategy
    for order in strategies:
        if time.time() > deadline:
            break
        chosen = _greedy_build(order)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_chosen = list(chosen)
    
    # Local search to improve
    if best_chosen:
        counts = _sum_counts(best_chosen)
        best_chosen, counts = _local_search(best_chosen, counts, max_steps=5000, deadline=deadline)
    
    # Additional local search restarts
    # Use remaining time for more restarts
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        # Start from a random greedy solution
        order = list(range(1, N + 1))
        random.Random(restart_count * 1000).shuffle(order)
        chosen = _greedy_build(order)
        
        # Only proceed if this greedy solution is promising or we have time
        # Actually, even if smaller, local search might improve it significantly?
        # Usually, starting from a larger set is better.
        # Let's just run local search on the current best if we have time, 
        # or on new random starts.
        
        # Strategy: 
        # 1. Keep polishing the best found so far.
        # 2. Try new random starts.
        
        # Let's alternate: 1 polish, 1 new start.
        if restart_count % 2 == 0:
             # Polish current best
             if best_chosen:
                 counts = _sum_counts(best_chosen)
                 new_chosen, _ = _local_search(best_chosen, counts, max_steps=2000, deadline=deadline)
                 if len(new_chosen) > best_size:
                     best_size = len(new_chosen)
                     best_chosen = new_chosen
        else:
             # New start
             counts = _sum_counts(chosen)
             new_chosen, _ = _local_search(chosen, counts, max_steps=2000, deadline=deadline)
             if len(new_chosen) > best_size:
                 best_size = len(new_chosen)
                 best_chosen = new_chosen
        
    # Final polish
    if best_chosen and time.time() < deadline - 5:
        counts = _sum_counts(best_chosen)
        best_chosen, _ = _local_search(best_chosen, counts, max_steps=5000, deadline=deadline)
    
    return sorted(best_chosen)
