import time
import math

def construct():
    deadline = time.time() + 55  # Safety margin
    
    # Precompute LCM of 1..24
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for i in range(1, 25):
        L = lcm(L, i)
        
    # val[i] = L // i
    val = [0] * 25
    for i in range(1, 25):
        val[i] = L // i

    best_set = []
    best_size = 0
    
    def is_valid(set_elements):
        """
        Check if set_elements is reciprocal-subsum-free.
        For each x in set_elements, v[x] must not be in subset sums of {v[y] for y in set_elements, y != x}.
        """
        n = len(set_elements)
        if n == 0:
            return True
        vals = [val[x] for x in set_elements]
        
        for i in range(n):
            target = vals[i]
            # Get others
            others = vals[:i] + vals[i+1:]
            
            # Filter others to those <= target
            relevant_others = [v for v in others if v <= target]
            
            # Check if target is in subset sums of relevant_others
            sums = {0}
            for v in relevant_others:
                new_sums = set()
                for s in sums:
                    ns = s + v
                    if ns == target:
                        return False
                    if ns < target:
                        new_sums.add(ns)
                sums |= new_sums
            if target in sums:
                return False
                
        return True

    # Greedy initialization
    greedy_set = []
    greedy_sums = {0}
    for x in range(24, 0, -1):
        v_x = val[x]
        # Quick check: is v_x in greedy_sums?
        if v_x not in greedy_sums:
            new_set = greedy_set + [x]
            # Full validation
            if is_valid(new_set):
                greedy_set = new_set
                # Update greedy_sums for next iteration's quick check
                greedy_sums = greedy_sums | {s + v_x for s in greedy_sums}
    
    best_set = greedy_set
    best_size = len(greedy_set)

    def branch_and_bound(next_x, current_set, current_sums, current_size):
        nonlocal best_set, best_size, deadline
        
        if time.time() >= deadline:
            return
        
        # Pruning
        if current_size + next_x - 1 <= best_size:
            return
            
        if next_x < 1:
            return
            
        x = next_x
        v_x = val[x]
        
        # Option 1: Include x
        # Quick check: is v_x in current_sums?
        if v_x not in current_sums:
            new_set = current_set + [x]
            # Full validation
            if is_valid(new_set):
                new_size = current_size + 1
                if new_size > best_size:
                    best_size = new_size
                    best_set = new_set
                # Update sums for recursion
                new_sums = current_sums | {s + v_x for s in current_sums}
                branch_and_bound(x - 1, new_set, new_sums, new_size)
        
        # Option 2: Exclude x
        branch_and_bound(x - 1, current_set, current_sums, current_size)
        
    branch_and_bound(24, [], {0}, 0)
    
    if not best_set:
        best_set = [24]
        
    return best_set
