"""Combined program: greedy + integer verification + randomized restarts."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    timeout = 58.0  # Safety margin for 60s limit (90s kill, but 60s per eval)
    
    universe = list(range(1, 23))
    
    def is_dissociated(S):
        """Check if all subset sums of reciprocals of S are distinct."""
        if not S:
            return True
        sums = {Fraction(0)}
        for x in S:
            val = Fraction(1, x)
            # Check collision before updating
            for s in sums:
                if (s + val) in sums:
                    return False
            # Update sums
            new_sums = [s + val for s in sums]
            sums.update(new_sums)
        return True

    def try_add(S, x):
        """Check if S union {x} is dissociated."""
        return is_dissociated(S + [x])

    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy with different deterministic orders
    # Order by descending value (large numbers first = small reciprocals)
    orders = [
        sorted(universe, reverse=True),
        sorted(universe),
        [n for n in universe if n >= 12],  # Large numbers only
        [n for n in universe if n % 2 == 0],  # Evens
        [n for n in universe if n % 2 == 1],  # Odds
    ]
    
    for order in orders:
        if time.time() - start_time > timeout:
            break
        current_set = []
        for x in order:
            if try_add(current_set, x):
                current_set.append(x)
        
        # Try to expand with remaining elements
        remaining = [x for x in universe if x not in current_set]
        added = True
        while added and time.time() - start_time < timeout:
            added = False
            for x in remaining[:]:
                if try_add(current_set, x):
                    current_set.append(x)
                    remaining.remove(x)
                    added = True
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Strategy 2: Randomized restarts
    random.seed(42)
    restart_count = 0
    while time.time() - start_time < timeout:
        restart_count += 1
        order = universe[:]
        random.shuffle(order)
        
        current_set = []
        for x in order:
            if try_add(current_set, x):
                current_set.append(x)
        
        # Try to expand with remaining elements
        remaining = [x for x in universe if x not in current_set]
        added = True
        while added and time.time() - start_time < timeout:
            added = False
            for x in remaining[:]:
                if try_add(current_set, x):
                    current_set.append(x)
                    remaining.remove(x)
                    added = True
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            # If we found a very large set, we can stop early
            if best_size >= 15:
                break
    
    # Fallback to a known valid set if search fails
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
# EVOLVE-BLOCK-END
