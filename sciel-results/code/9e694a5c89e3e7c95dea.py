import time
import random
import numpy as np
from itertools import combinations_with_replacement

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0  # safety margin before 90s kill
    
    N = 4096
    MAX_K = 80
    seed = 42
    
    def check_validity(arr):
        """Check if all unordered triple sums (with repetition) are unique."""
        n = len(arr)
        if n < 3:
            return True
        seen = set()
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s = arr[i] + arr[j] + arr[k]
                    if s in seen:
                        return False
                    seen.add(s)
        return True
    
    def count_conflicts(arr):
        """Count the number of duplicate triple sums (0 means valid)."""
        n = len(arr)
        if n < 3:
            return 0
        seen = set()
        conflicts = 0
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s = arr[i] + arr[j] + arr[k]
                    if s in seen:
                        conflicts += 1
                    else:
                        seen.add(s)
        return conflicts
    
    def greedy_build(start_vals, order, max_size=MAX_K):
        """
        Greedily build a set by adding candidates from 'order' if they don't create conflicts.
        start_vals: initial list of values
        order: list of candidate values to try in order
        Returns the constructed list.
        """
        selected = list(start_vals)
        selected_set = set(selected)
        
        # Precompute current triple sums for fast incremental check
        # For a new candidate x, the new triple sums are:
        # x+x+x, x+x+y, x+y+y, x+y+z for y,z in selected (y<=z)
        # We need to check these against existing sums and among themselves.
        
        # For efficiency, we'll recompute conflicts incrementally
        # But for simplicity in the first pass, let's use a fast check
        
        for x in order:
            if x in selected_set:
                continue
            if len(selected) >= max_size:
                break
            
            # Check if adding x creates any conflicts
            # New triples involving x:
            # (x, x, x)
            # (x, x, y) for y in selected
            # (x, y, y) for y in selected
            # (x, y, z) for y <= z in selected
            
            # We need to check all new sums against existing sums and each other
            new_sums = []
            new_sums.append(3 * x)
            for y in selected:
                new_sums.append(2 * x + y)
                new_sums.append(x + 2 * y)
            for i in range(len(selected)):
                for j in range(i, len(selected)):
                    new_sums.append(x + selected[i] + selected[j])
            
            # Check for duplicates within new_sums
            if len(new_sums) != len(set(new_sums)):
                continue
            
            # Check against existing triple sums
            # We need to compute existing sums or maintain them
            # For now, let's just do a full check if the set is small enough
            # Otherwise, we'll use a faster method
            
            candidate = selected + [x]
            if check_validity(candidate):
                selected.append(x)
                selected_set.add(x)
        
        return selected
    
    def fast_greedy_with_order(order, max_size=MAX_K):
        """
        Fast greedy that maintains a set of triple sums for incremental checking.
        """
        selected = []
        selected_set = set()
        triple_sums = set()
        
        for x in order:
            if x in selected_set:
                continue
            if len(selected) >= max_size:
                break
            
            # Compute new triple sums involving x
            new_sums = []
            new_sums.append(3 * x)
            for y in selected:
                new_sums.append(2 * x + y)
                new_sums.append(x + 2 * y)
            for i in range(len(selected)):
                for j in range(i, len(selected)):
                    new_sums.append(x + selected[i] + selected[j])
            
            # Check for duplicates within new_sums
            if len(new_sums) != len(set(new_sums)):
                continue
            
            # Check against existing triple_sums
            conflict = False
            for s in new_sums:
                if s in triple_sums:
                    conflict = True
                    break
            
            if not conflict:
                selected.append(x)
                selected_set.add(x)
                triple_sums.update(new_sums)
        
        return selected
    
    def local_improvement(arr, max_iter=5000):
        """
        Try to improve the set by swapping elements to reduce conflicts.
        """
        arr = list(arr)
        n = len(arr)
        best_conflicts = count_conflicts(arr)
        
        for _ in range(max_iter):
            if time.time() > DEADLINE:
                break
            if best_conflicts == 0:
                break
            
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    # Try swapping arr[i] and arr[j] with some other values
                    # This is expensive, so let's try a simpler approach:
                    # Try removing one element and adding another
                    pass
            
            # Simpler local search: try replacing one element
            for i in range(n):
                if time.time() > DEADLINE:
                    break
                old_val = arr[i]
                # Try replacing with a value not in the set
                for new_val in random.sample(range(1, N + 1), min(50, N)):
                    if new_val in arr:
                        continue
                    arr[i] = new_val
                    conflicts = count_conflicts(arr)
                    if conflicts < best_conflicts:
                        best_conflicts = conflicts
                        improved = True
                        break
                    else:
                        arr[i] = old_val
                if improved:
                    break
            if not improved:
                break
        
        return arr
    
    # Strategy 1: Use base-representation-inspired ordering
    # Numbers with large gaps in their base-B representation tend to have unique sums
    # Try different bases
    
    best_result = []
    
    # Try greedy with different orderings
    # Order 1: Natural order
    if len(best_result) < MAX_K:
        result = fast_greedy_with_order(range(1, N + 1), MAX_K)
        if len(result) > len(best_result):
            best_result = result
    
    if time.time() > DEADLINE:
        return best_result
    
    # Order 2: Powers of a base (e.g., base 4 or 5)
    for base in [4, 5, 6, 7, 8]:
        if time.time() > DEADLINE:
            break
        order = []
        val = 1
        while val <= N:
            order.append(val)
            val *= base
        # Add some more values
        for x in range(1, N + 1):
            if x not in order:
                order.append(x)
        
        result = fast_greedy_with_order(order, MAX_K)
        if len(result) > len(best_result):
            best_result = result
    
    if time.time() > DEADLINE:
        return best_result
    
    # Try random restarts with different seeds
    rng = random.Random(seed)
    for trial in range(20):
        if time.time() > DEADLINE:
            break
        
        # Create a random ordering with bias towards larger numbers
        # or use a structured approach
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        result = fast_greedy_with_order(order, MAX_K)
        if len(result) > len(best_result):
            best_result = result
    
    if time.time() > DEADLINE:
        return best_result
    
    # Try to improve the best result with local search
    if len(best_result) < MAX_K:
        improved = local_improvement(best_result, max_iter=2000)
        if len(improved) > len(best_result):
            best_result = improved
        elif count_conflicts(improved) == 0 and count_conflicts(best_result) > 0:
            best_result = improved
    
    if time.time() > DEADLINE:
        return best_result
    
    # Try more structured approaches
    # Use a mix of small and large numbers
    for base in [3, 4, 5]:
        if time.time() > DEADLINE:
            break
        order = []
        # Add powers
        val = 1
        while val <= N:
            order.append(val)
            val *= base
        # Add multiples
        for mult in range(1, 10):
            val = mult
            while val <= N:
                if val not in order:
                    order.append(val)
                val *= base
        
        result = fast_greedy_with_order(order, MAX_K)
        if len(result) > len(best_result):
            best_result = result
    
    if time.time() > DEADLINE:
        return best_result
    
    # Final local improvement
    if len(best_result) < MAX_K and count_conflicts(best_result) > 0:
        improved = local_improvement(best_result, max_iter=3000)
        if count_conflicts(improved) < count_conflicts(best_result):
            best_result = improved
    
    # Verify the result
    if best_result and not check_validity(best_result):
        # Fallback to a simple valid set
        best_result = fast_greedy_with_order(range(1, N + 1), 10)
    
    return best_result
