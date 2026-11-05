import time
import random
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin for 90s limit
    
    # The universe of candidates
    candidates = list(range(1, 23))
    
    best_set = []
    best_size = 0
    
    # Precompute LCM of 1..22 for integer scaling
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, 23):
        L = lcm(L, i)
        
    # Integer values: v[i] = L // i
    v = {i: L // i for i in range(1, 23)}
    
    def check_collision(sums_set, new_val):
        """Check if adding new_val causes any collision."""
        for s in sums_set:
            if (s + new_val) in sums_set:
                return True
        return False

    def try_greedy(order):
        """Try to build a set by greedily adding elements in the given order."""
        current_set = []
        sums = {0}
        
        for x in order:
            val = v[x]
            if not check_collision(sums, val):
                current_set.append(x)
                # Update sums
                new_sums = {s + val for s in sums}
                sums.update(new_sums)
        return current_set

    def backtrack_search(remaining, current_set, sums, time_limit):
        """Backtracking search to find a large dissociated set."""
        nonlocal best_set, best_size
        
        if time.time() > time_limit:
            return
        
        if not remaining:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return
        
        # Pruning: if we can't beat best_size, return
        if len(current_set) + len(remaining) <= best_size:
            return
        
        # Heuristic: Sort remaining by some criterion? 
        # We'll just pick the first one for now, but we can try different orderings outside.
        
        first = remaining[0]
        rest = remaining[1:]
        val = v[first]
        
        # Try including first
        if not check_collision(sums, val):
            new_sums = sums | {s + val for s in sums}
            backtrack_search(rest, current_set + [first], new_sums, time_limit)
        
        # Try excluding first
        backtrack_search(rest, current_set, sums, time_limit)
    
    # Strategy 1: Greedy with different orders to get a good initial best_set
    orders = [
        sorted(candidates, reverse=True), # Large denominators first (small values)
        sorted(candidates),               # Small denominators first (large values)
        [n for n in candidates if n % 2 == 0],
        [n for n in candidates if n % 2 != 0],
        [n for n in candidates if all(n % p != 0 for p in range(2, int(n**0.5)+1))], # Primes
        [n for n in candidates if not all(n % p != 0 for p in range(2, int(n**0.5)+1))], # Composites
    ]
    
    for order in orders:
        if time.time() - start_time > timeout:
            break
        s = try_greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
    
    # Strategy 2: Randomized greedy
    for _ in range(50):
        if time.time() - start_time > timeout:
            break
        shuffled = candidates[:]
        random.shuffle(shuffled)
        s = try_greedy(shuffled)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
    
    # Strategy 3: Backtracking to improve
    # We run backtracking from scratch with different orderings
    # and also from the current best set
    
    # Ordering 1: Descending indices (small values first)
    # This is often good because small values are less likely to collide?
    # Actually, large values (small indices) are more restrictive.
    # Let's try both.
    
    def run_backtrack_with_order(order, time_limit):
        nonlocal best_set, best_size
        backtrack_search(order[:], [], {0}, time_limit)
        
    # Try backtracking with descending order (22, 21, ...)
    if time.time() - start_time < timeout - 5:
        run_backtrack_with_order(sorted(candidates, reverse=True), time.time() + (timeout - (time.time() - start_time) - 5))
        
    # Try backtracking with ascending order (1, 2, ...)
    if time.time() - start_time < timeout - 5:
        run_backtrack_with_order(sorted(candidates), time.time() + (timeout - (time.time() - start_time) - 5))
        
    # Try to extend the best set found so far
    if best_set:
        best_set_set = set(best_set)
        remaining = [x for x in candidates if x not in best_set_set]
        
        # Compute sums for best_set
        current_sums = {0}
        for x in best_set:
            val = v[x]
            current_sums = current_sums | {s + val for s in current_sums}
        
        # Backtrack on remaining elements
        def extend_backtrack(remaining, current_set, sums, time_limit):
            nonlocal best_set, best_size
            
            if time.time() > time_limit:
                return
            
            if not remaining:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                return
            
            # Pruning
            if len(current_set) + len(remaining) <= best_size:
                return
            
            first = remaining[0]
            rest = remaining[1:]
            val = v[first]
            
            # Try including first
            if not check_collision(sums, val):
                new_sums = sums | {s + val for s in sums}
                extend_backtrack(rest, current_set + [first], new_sums, time_limit)
            
            # Try excluding first
            extend_backtrack(rest, current_set, sums, time_limit)
            
        # Use remaining time for extension
        time_left = timeout - (time.time() - start_time)
        if time_left > 5:
            extend_backtrack(remaining, best_set[:], current_sums, start_time + timeout)
    
    if not best_set:
        best_set = [22]  # Fallback

    return best_set
# EVOLVE-BLOCK-END
