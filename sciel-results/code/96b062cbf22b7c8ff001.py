"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    numbers = list(range(2, 257))
    n = len(numbers)
    
    # Precompute conflict graph: two numbers conflict if lcm(a,b) <= 256
    # Build adjacency: conflict[i] = set of indices j that conflict with i
    conflict = [set() for _ in range(n)]
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Greedy initial solution: sort by 1/num descending (num ascending), add if no conflict
    selected_indices = []
    for x in sorted(range(n), key=lambda i: numbers[i]):
        if all(j not in selected_indices for j in conflict[x]):
            selected_indices.append(x)
    
    best_sum = sum(Fraction(1, numbers[i]) for i in selected_indices)
    best_set = set(selected_indices)
    
    # Local search: try adding one element, removing one, or swapping
    # We'll do multiple rounds of local search
    current_set = set(selected_indices)
    current_sum = best_sum
    
    def try_improve(current_set, current_sum, time_limit):
        """Try to improve the current set via local search."""
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # Try adding each non-member
            for i in range(n):
                if i in current_set:
                    continue
                # Check if i conflicts with any member
                if not conflict[i].intersection(current_set):
                    new_sum = current_sum + Fraction(1, numbers[i])
                    if new_sum > current_sum:
                        current_set.add(i)
                        current_sum = new_sum
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try removing one element and adding another (swap)
            members = list(current_set)
            for i in range(n):
                if i in current_set:
                    continue
                # For each member, check if removing it allows adding i
                for m in members:
                    if m not in current_set:
                        continue
                    # Check if i conflicts with (current_set - {m})
                    remaining = current_set - {m}
                    if not conflict[i].intersection(remaining):
                        new_sum = current_sum - Fraction(1, numbers[m]) + Fraction(1, numbers[i])
                        if new_sum > current_sum:
                            current_set.discard(m)
                            current_set.add(i)
                            current_sum = new_sum
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try removing one element (only if it frees up space for something better)
            # This is handled by the swap above, but let's also try pure removals
            # followed by additions in the next iteration
            for m in list(current_set):
                # Try removing m and see if we can add something
                new_set = current_set - {m}
                # Try to add the best possible element
                best_add = None
                best_add_val = Fraction(0)
                for i in range(n):
                    if i in new_set:
                        continue
                    if not conflict[i].intersection(new_set):
                        val = Fraction(1, numbers[i])
                        if val > best_add_val:
                            best_add_val = val
                            best_add = i
                if best_add is not None:
                    new_sum = current_sum - Fraction(1, numbers[m]) + best_add_val
                    if new_sum > current_sum:
                        current_set = new_set
                        current_set.add(best_add)
                        current_sum = new_sum
                        improved = True
                        break
        
        return current_set, current_sum
    
    # Run local search with time limit
    time_limit = deadline - 5.0
    current_set, current_sum = try_improve(current_set, current_sum, time_limit)
    
    if current_sum > best_sum:
        best_sum = current_sum
        best_set = current_set
    
    # Additional refinement: try different initial orderings
    if (time.time() - start_time) < deadline - 10.0:
        # Try greedy with different sort orders
        for sort_key in [lambda i: numbers[i], 
                         lambda i: -len(conflict[i]),
                         lambda i: numbers[i] * len(conflict[i])]:
            if (time.time() - start_time) > deadline - 10.0:
                break
            
            selected_indices = []
            for x in sorted(range(n), key=sort_key):
                if all(j not in selected_indices for j in conflict[x]):
                    selected_indices.append(x)
            
            current_set = set(selected_indices)
            current_sum = sum(Fraction(1, numbers[i]) for i in current_set)
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set
            
            # Local search from this starting point
            time_limit = min(deadline - 5.0, time.time() + 5.0)
            current_set, current_sum = try_improve(current_set, current_sum, time_limit)
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set
    
    # Convert back to numbers
    result = [numbers[i] for i in best_set]
    
    if not result:
        result = [256]
    
    return result
# EVOLVE-BLOCK-END
