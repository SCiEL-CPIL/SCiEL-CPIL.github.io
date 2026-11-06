"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def construct():
    """
    Find a nonempty subset of [1,22] with maximum cardinality such that
    all reciprocal subset sums (including empty sum 0) are distinct.
    """
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Compute LCM of 1..22 to convert reciprocal sums to integer sums
    L = reduce(lcm, range(1, 23))
    
    # For each element a in [1,22], its contribution to the integer sum is L // a
    vals = {a: L // a for a in range(1, 23)}
    
    # Sort elements by their value (L//a) in descending order
    elements = sorted(range(1, 23), key=lambda a: vals[a], reverse=True)
    
    def check_add(sums, v):
        """
        Check if adding value v to a set with existing subset sums 'sums' 
        keeps all subset sums distinct.
        New sums would be sums ∪ {s+v for s in sums}.
        Collision iff there exist s1, s2 in sums with s1 + v = s2, i.e., v = s2 - s1.
        Equivalently: (sums + v) intersects sums.
        Returns (True, new_sums) if valid, (False, None) otherwise.
        """
        shifted = {s + v for s in sums}
        if shifted & sums:
            return (False, None)
        return (True, sums | shifted)
    
    def greedy_build(order):
        """Build a dissociated set by greedily adding elements in the given order."""
        current = []
        sums = {0}
        for a in order:
            v = vals[a]
            ok, new_sums = check_add(sums, v)
            if ok:
                current.append(a)
                sums = new_sums
        return current, sums
    
    best_subset = []
    best_sums = {0}
    
    # Strategy 1: Greedy in descending value order
    current, sums = greedy_build(elements)
    if len(current) > len(best_subset):
        best_subset = list(current)
        best_sums = sums
    
    # Strategy 2: Greedy in ascending value order
    current, sums = greedy_build(elements[::-1])
    if len(current) > len(best_subset):
        best_subset = list(current)
        best_sums = sums
    
    # Strategy 3: Try to extend the best set by trying to add remaining elements
    remaining = [a for a in elements if a not in best_subset]
    for a in remaining:
        if time.time() > deadline - 5:
            break
        v = vals[a]
        ok, new_sums = check_add(best_sums, v)
        if ok:
            best_subset.append(a)
            best_sums = new_sums
    
    # Strategy 4: Backtracking search for larger sets
    # Use the incremental sum tracking for efficiency
    def backtrack(start_idx, current_subset, current_sums, target_size):
        """Try to build a dissociated set of target_size starting from start_idx."""
        if time.time() > deadline - 3:
            return None, None
        
        if len(current_subset) == target_size:
            return list(current_subset), current_sums
        
        remaining_needed = target_size - len(current_subset)
        available = 22 - start_idx
        
        if available < remaining_needed:
            return None, None
        
        for i in range(start_idx, 22):
            if time.time() > deadline - 3:
                return None, None
            
            a = elements[i]
            v = vals[a]
            ok, new_sums = check_add(current_sums, v)
            if ok:
                current_subset.append(a)
                result, result_sums = backtrack(i + 1, current_subset, new_sums, target_size)
                current_subset.pop()
                if result is not None:
                    return result, result_sums
        
        return None, None
    
    greedy_size = len(best_subset)
    
    # Try to find larger sets via backtracking
    for target_size in range(22, greedy_size, -1):
        if time.time() > deadline - 5:
            break
        
        result, result_sums = backtrack(0, [], {0}, target_size)
        if result is not None:
            best_subset = result
            best_sums = result_sums
            break
    
    # Strategy 5: Local search - try swapping elements to improve
    if time.time() < deadline - 10 and len(best_subset) < 22:
        import random
        random.seed(42)
        
        for _ in range(200):
            if time.time() > deadline - 3:
                break
            
            # Try a random greedy ordering
            shuffled = elements[:]
            random.shuffle(shuffled)
            current, sums = greedy_build(shuffled)
            
            if len(current) > len(best_subset):
                best_subset = list(current)
                best_sums = sums
    
    # Ensure we have at least one element
    if not best_subset:
        best_subset = [1]
    
    return best_subset

# EVOLVE-BLOCK-END
