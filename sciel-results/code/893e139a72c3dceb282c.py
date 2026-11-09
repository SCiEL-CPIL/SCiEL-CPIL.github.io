import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for 60s limit

    max_val = 4096
    max_sum = 3 * max_val

    best_set = []
    best_size = 0

    # Fast collision check using a set of used sums
    def try_add_fast(x, current_set, used_sums):
        """Try to add x to current_set, checking all triples involving x."""
        n = len(current_set)
        
        # Check (x,x,x)
        s = 3 * x
        if s in used_sums:
            return False
        
        # Check (x,x,y) for y in current_set
        for y in current_set:
            s = 2 * x + y
            if s in used_sums:
                return False
        
        # Check (x,y,y) for y in current_set
        for y in current_set:
            s = x + 2 * y
            if s in used_sums:
                return False
        
        # Check (x,y,z) for y < z in current_set
        for i in range(n):
            y = current_set[i]
            for j in range(i + 1, n):
                z = current_set[j]
                s = x + y + z
                if s in used_sums:
                    return False
        
        # All checks passed, update used_sums
        used_sums.add(3 * x)
        for y in current_set:
            used_sums.add(2 * x + y)
            used_sums.add(x + 2 * y)
        for i in range(n):
            y = current_set[i]
            for j in range(i + 1, n):
                z = current_set[j]
                used_sums.add(x + y + z)
        
        return True

    def greedy_build(order, deadline):
        """Build a set using greedy approach with given order."""
        current_set = []
        used_sums = set()
        
        for x in order:
            if time.time() > deadline - 1.0:
                break
            if try_add_fast(x, current_set, used_sums):
                current_set.append(x)
        
        return current_set

    def local_search(initial_set, deadline, max_iterations=200):
        """Try to improve the set by swapping elements."""
        current_set = sorted(initial_set)
        used_sums = set()
        
        # Rebuild used_sums from current_set
        n = len(current_set)
        for i in range(n):
            a = current_set[i]
            for j in range(i, n):
                b = current_set[j]
                for k in range(j, n):
                    c = current_set[k]
                    used_sums.add(a + b + c)
        
        best_size = len(current_set)
        best_set = current_set[:]
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 1.0:
                break
            
            improved = False
            n = len(current_set)
            
            # Try removing each element and replacing with a better one
            for i in range(n):
                if time.time() > deadline - 1.0:
                    break
                
                removed = current_set[i]
                # Rebuild without this element
                temp_set = current_set[:i] + current_set[i+1:]
                temp_sums = set()
                m = len(temp_set)
                for ii in range(m):
                    a = temp_set[ii]
                    for jj in range(ii, m):
                        b = temp_set[jj]
                        for kk in range(jj, m):
                            c = temp_set[kk]
                            temp_sums.add(a + b + c)
                
                # Try to find a replacement
                for candidate in range(1, max_val + 1):
                    if candidate == removed:
                        continue
                    if candidate in temp_set:
                        continue
                    
                    if try_add_fast(candidate, temp_set, temp_sums):
                        new_set = temp_set + [candidate]
                        if len(new_set) > best_size:
                            best_size = len(new_set)
                            best_set = new_set
                        # Keep the improved set for further search
                        current_set = new_set
                        used_sums = temp_sums | {3*candidate}
                        for y in temp_set:
                            used_sums.add(2*candidate + y)
                            used_sums.add(candidate + 2*y)
                        for ii in range(len(temp_set)):
                            y = temp_set[ii]
                            for jj in range(ii+1, len(temp_set)):
                                z = temp_set[jj]
                                used_sums.add(candidate + y + z)
                        improved = True
                        break
                # If no improvement found for this position, continue
                if not improved:
                    # Restore original
                    pass
            if not improved:
                break
        
        return best_set

    # Generate diverse ordering strategies
    candidates = list(range(1, max_val + 1))
    
    orders = []
    
    # 1. Pure ascending
    orders.append(candidates[:])
    
    # 2. Descending (large numbers first)
    orders.append(candidates[::-1])
    
    # 3-6. Modular orderings
    for mod in [3, 5, 7, 11]:
        c_mod = sorted(candidates, key=lambda x: (x % mod, x))
        orders.append(c_mod)
    
    # 7-10. Random shuffles with fixed seeds for reproducibility
    rng = random.Random(42)
    for _ in range(4):
        c = candidates[:]
        rng.shuffle(c)
        orders.append(c)
    
    # 11-14. Biased random: favor certain ranges
    for bias_start, bias_end in [(1, 100), (100, 500), (500, 2000), (2000, 4096)]:
        c = candidates[:]
        rng.shuffle(c)
        # Move elements in bias range to front
        biased = [x for x in c if bias_start <= x <= bias_end]
        rest = [x for x in c if not (bias_start <= x <= bias_end)]
        orders.append(biased + rest)
    
    # Run greedy for each order
    for idx, order in enumerate(orders):
        if time.time() > deadline - 3.0:
            break
        
        result = greedy_build(order, deadline)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
            
            # If we have a good set, try local search
            if best_size >= 35 and time.time() < deadline - 10.0:
                improved = local_search(best_set, deadline)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_set = improved[:]
    
    return best_set
